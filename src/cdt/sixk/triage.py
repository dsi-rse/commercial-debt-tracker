"""Two-stage triage selecting Form 6-K snippets worth extracting from.

Stage 1 is a TF-IDF linear SVM over 400-token windows, run at a threshold tuned
for recall rather than precision: 0.332 admits 5.8% of windows and holds 95.4%
of relevant ones, at 35.4% precision. That is deliberately loose. Tightening it
costs recall faster than it buys precision, and recall lost here is invisible
downstream.

Stage 2 hands every admitted window from one filing to an LLM *together* and
asks which to keep. Grouping by filing is the whole point: whether a window
merely repeats a sibling cannot be judged from the window alone. It answers two
questions, and they fail differently:

* Does the window state a concrete attribute of a specific instrument? Rejecting
  one that does not is a precision gain and costs nothing.
* Is every attribute it states already covered by a window being kept? Dropping
  a genuine duplicate is a precision gain; dropping one that added an attribute
  is silent data loss, so the rule is deliberately conservative and every
  duplicate ruling is reported.

Measured on 88 filings against 500 hand-labelled windows: precision 35.4% ->
70.9%, and filing-level recall 100% (53 of 53 filings holding a relevant window
still hold one). Window-level recall falls to 74.6%, which is the metric moving
in the wrong direction *by design* -- consolidating siblings is the job.

Stage 2 costs about $0.25 per 1,000 6-K filings on ``openai/gpt-5.6-luna`` and
removes 47.5% of the windows stage 1 admits, so it returns roughly 76x its cost
in avoided extraction. See ``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

from cdt import settings
from cdt.classifier.core import load_training_artifacts, score_model

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

#: Stage-1 cutoff, used only when an artifact carries no threshold of its own.
#: Prefer :func:`load_stage1_model`, which reads it from the artifact: the
#: threshold is a property of one fitted model, and a constant here would drift
#: away from the model it was calibrated against.
DEFAULT_STAGE1_THRESHOLD = 0.332

#: Stage-2 model. Cheap is the point: it reads only what stage 1 admits.
#: Sourced from settings so it is overridable by ``SIXK_TRIAGE_MODEL``.
DEFAULT_STAGE2_MODEL = settings.SIXK_TRIAGE_MODEL

#: Attempts per filing, matching the extractor's ``DEFAULT_MAX_ATTEMPTS``.
DEFAULT_MAX_ATTEMPTS = 3

#: Filename of the stage-1 artifact inside its model directory, matching the
#: 8-K classifier's layout so both load the same way.
MODEL_FILENAME = "model.pkl"

SYSTEM_PROMPT = """\
You triage snippets from a single SEC Form 6-K filing for a commercial debt
tracker. A later extraction stage will pull structured debt-instrument records
from whatever you keep, so keeping junk costs money and dropping a snippet that
carried a detail loses data permanently.

KEEP a snippet if it states at least one concrete attribute of a specific debt
instrument: a principal or nominal amount, an interest rate or margin, a
maturity or repayment schedule, a named lender/holder/trustee, security or a
guarantee, or a named series or facility. The borrower or issuer may be the
filer rather than named in the snippet.

DROP a snippet if it has no such attribute. In particular drop:
- aggregates and portfolio metrics: "total debt was $47.9 million", "weighted
  average maturity 11.33 years", net-debt ratios, maturity tables broken out by
  liability category rather than by instrument
- templates whose values are still blank: "February [ - ], 2026", "$[____]"
- mechanics with no terms: conversion arithmetic, redemption-notice contents,
  paying-agent duties, restricted-securities legends, insurance covenants
- accounting inputs that are not instrument terms: an IRR or discount rate used
  to fair-value an instrument is not its coupon (a stated face value IS an
  attribute)
- equity: shares, warrants, options, buybacks, preferred paying dividends on a
  share series
- deferred acquisition consideration with no lender and no facility
- inline XBRL tag blocks, cover pages, boilerplate, tables of contents

THEN deduplicate, but only under this rule:

  Drop a snippet as a duplicate ONLY IF every attribute it states already
  appears in a snippet you are keeping.

If a snippet repeats an instrument a kept snippet already covers but adds any
attribute the kept one lacks -- a rate the other omitted, the security, a
maturity, a party -- KEEP IT. Several snippets describing one instrument from
different angles are normal and all of them are wanted. When unsure whether an
attribute is genuinely new, keep the snippet.

Return JSON only:
{"keep": [<ids>],
 "drop": [{"id": <id>, "reason": "no_details"} or
          {"id": <id>, "reason": "duplicate", "covered_by": <kept id>,
           "attributes": "<the attributes you judged already present>"}]}
Every id given to you must appear exactly once across "keep" and "drop".
"""


class SupportsChatCompletion(Protocol):
    """Protocol for chat-capable clients, matching the extractor's."""

    async def complete(
        self: Self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Return one chat completion as plain text."""


@dataclass(frozen=True)
class Snippet:
    """One stage-1 window offered to stage 2."""

    snippet_id: str
    text: str
    score: float


@dataclass
class FilingVerdict:
    """Stage-2 outcome for one filing."""

    accession_number: str
    kept: list[str] = field(default_factory=list)
    dropped_no_details: list[str] = field(default_factory=list)
    dropped_duplicate: list[tuple[str, str]] = field(default_factory=list)
    attempts: int = 1
    error: str | None = None


def default_model_dir(data_dir: Path | None = None) -> Path:
    """Return the default stage-1 artifact directory.

    Mirrors :func:`cdt.classifier.core.default_model_dir`: the path is derived
    from ``DATA_DIR`` rather than configured separately, so a deployment moves
    both classifiers by moving one variable.

    Args:
        data_dir: Root to resolve against; defaults to ``settings.DATA_DIR``.

    Returns:
        Directory expected to hold :data:`MODEL_FILENAME`.
    """
    return (
        (data_dir or settings.DATA_DIR) / "models" / "sixk" / "stage1-tfidf-linear-svc"
    )


def load_stage1_model(model_dir: Path | None = None) -> tuple[object, float]:
    """Load the stage-1 pipeline and the threshold calibrated with it.

    Delegates to :func:`cdt.classifier.core.load_training_artifacts`, so the 6-K
    and 8-K artifacts have the same on-disk contract. The threshold travels with
    the model rather than living in code, because it is a property of one fitted
    pipeline: refitting moves the score scale, and a hard-coded cutoff would
    quietly stop meaning what it meant.

    Args:
        model_dir: Directory holding ``model.pkl`` and ``metadata.json``;
            defaults to :func:`default_model_dir`.

    Returns:
        The fitted pipeline and its threshold.

    Raises:
        FileNotFoundError: If the artifact is absent.
    """
    resolved = model_dir or default_model_dir()
    if not (resolved / MODEL_FILENAME).exists():
        raise FileNotFoundError(f"no stage-1 model at {resolved / MODEL_FILENAME}")
    model, threshold, _ = load_training_artifacts(resolved)
    return model, threshold


def stage1_admit(
    model: object,
    snippets: Sequence[tuple[str, str]],
    *,
    threshold: float = DEFAULT_STAGE1_THRESHOLD,
) -> list[Snippet]:
    """Score windows and keep those at or above the threshold.

    Args:
        model: Fitted stage-1 pipeline.
        snippets: ``(snippet_id, text)`` pairs for one or more filings.
        threshold: Stage-1 cutoff.

    Returns:
        Admitted snippets, in the order given.
    """
    if not snippets:
        return []
    scores = score_model(model, [text for _, text in snippets])  # type: ignore[arg-type]
    return [
        Snippet(snippet_id=snippet_id, text=text, score=float(score))
        for (snippet_id, text), score in zip(snippets, scores, strict=True)
        if score >= threshold
    ]


def validate_verdict(verdict: object, expected: int) -> list[str]:
    """Check that a stage-2 verdict partitions the snippet ids exactly once.

    Args:
        verdict: Parsed model output.
        expected: Number of snippets sent.

    Returns:
        Human-readable failures; empty when the verdict is well formed.

    >>> validate_verdict({"keep": [1], "drop": [{"id": 2, "reason": "no_details"}]}, 2)
    []
    >>> validate_verdict({"keep": [1], "drop": []}, 2)
    ['no verdict for snippet 2']
    >>> validate_verdict({"keep": [1], "drop": [{"id": 1, "reason": "no_details"}]}, 1)
    ['snippet 1 appears in both keep and drop']
    """
    if not isinstance(verdict, dict):
        return ["response was not a JSON object"]
    keep = {int(value) for value in verdict.get("keep", []) if _is_index(value)}
    entries = [
        entry
        for entry in verdict.get("drop", [])
        if isinstance(entry, dict) and _is_index(entry.get("id"))
    ]
    drop = {int(entry["id"]) for entry in entries}
    failures = [
        f"snippet {index} appears in both keep and drop"
        for index in sorted(keep & drop)
    ]
    failures += [
        f"no verdict for snippet {index}"
        for index in range(1, expected + 1)
        if index not in keep | drop
    ]
    failures += [
        f"snippet {index} does not exist; ids run 1 to {expected}"
        for index in sorted((keep | drop) - set(range(1, expected + 1)))
    ]
    failures += [
        f"snippet {entry['id']} dropped as a duplicate without a covered_by id"
        for entry in entries
        if entry.get("reason") == "duplicate" and not _is_index(entry.get("covered_by"))
    ]
    return failures


def _is_index(value: object) -> bool:
    """Return whether a value is usable as a 1-based snippet index.

    Args:
        value: Candidate value from model output.

    Returns:
        ``True`` when it parses as a positive integer.

    >>> _is_index(3), _is_index("3"), _is_index("x"), _is_index(None)
    (True, True, False, False)
    """
    return str(value).strip().isdigit()


def build_retry_message(failures: list[str], expected: int) -> str:
    """Build the corrective turn after a validation failure.

    Args:
        failures: Output of :func:`validate_verdict`.
        expected: Number of snippets sent.

    Returns:
        The user message to append before re-asking.
    """
    listed = "\n".join(f"- {failure}" for failure in failures)
    return (
        f"That response was not usable:\n{listed}\n\n"
        f"Return the JSON again. Every id from 1 to {expected} must appear "
        "exactly once, across either keep or drop, and every duplicate must "
        "name the kept snippet it is covered by. Do not change any ruling you "
        "already made correctly."
    )


async def triage_filing(
    client: SupportsChatCompletion,
    accession_number: str,
    snippets: Sequence[Snippet],
    *,
    model: str = DEFAULT_STAGE2_MODEL,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> FilingVerdict:
    """Ask stage 2 which of one filing's admitted snippets to keep.

    Retries on a verdict that does not partition the ids, feeding the failures
    back alongside the rejected answer, which mirrors how the extractor recovers
    from a failed validation.

    Args:
        client: Chat client.
        accession_number: Filing the snippets came from.
        snippets: Stage-1 admitted snippets for that filing.
        model: Stage-2 model slug.
        max_attempts: Attempts before giving up.

    Returns:
        The verdict. On failure every snippet is kept and ``error`` is set, so a
        stage-2 outage degrades to stage-1 behaviour rather than losing data.
    """
    body = "\n\n".join(
        f"--- snippet {index} ---\n{snippet.text}"
        for index, snippet in enumerate(snippets, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = await client.complete(
                messages=messages, model=model, reasoning_effort=""
            )
        except Exception as error:  # noqa: BLE001 - one filing must not stop a run
            LOGGER.warning("stage 2 call failed for %s: %r", accession_number, error)
            return FilingVerdict(
                accession_number=accession_number,
                kept=[snippet.snippet_id for snippet in snippets],
                attempts=attempt,
                error=repr(error),
            )
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError as error:
            verdict, failures = None, [f"response was not valid JSON: {error}"]
        else:
            failures = validate_verdict(verdict, len(snippets))
        if not failures:
            return _to_filing_verdict(accession_number, snippets, verdict, attempt)
        messages = [
            *messages,
            {"role": "assistant", "content": text},
            {"role": "user", "content": build_retry_message(failures, len(snippets))},
        ]
    LOGGER.warning(
        "stage 2 gave up on %s after %s attempts: %s",
        accession_number,
        max_attempts,
        failures,
    )
    return FilingVerdict(
        accession_number=accession_number,
        kept=[snippet.snippet_id for snippet in snippets],
        attempts=max_attempts,
        error=f"validation failed: {failures}",
    )


def _to_filing_verdict(
    accession_number: str,
    snippets: Sequence[Snippet],
    verdict: dict,
    attempts: int,
) -> FilingVerdict:
    """Convert a validated verdict into snippet ids.

    Args:
        accession_number: Filing the snippets came from.
        snippets: Snippets sent, in the order they were numbered.
        verdict: A verdict that has passed :func:`validate_verdict`.
        attempts: Attempts taken.

    Returns:
        The verdict with model indices resolved to snippet ids.
    """
    result = FilingVerdict(accession_number=accession_number, attempts=attempts)
    keep = {int(value) for value in verdict.get("keep", []) if _is_index(value)}
    for index, snippet in enumerate(snippets, start=1):
        if index in keep:
            result.kept.append(snippet.snippet_id)
    for entry in verdict.get("drop", []):
        if not isinstance(entry, dict) or not _is_index(entry.get("id")):
            continue
        snippet_id = snippets[int(entry["id"]) - 1].snippet_id
        if entry.get("reason") == "duplicate":
            covered = snippets[int(entry["covered_by"]) - 1].snippet_id
            result.dropped_duplicate.append((snippet_id, covered))
        else:
            result.dropped_no_details.append(snippet_id)
    return result
