"""The 6-K classify stage: score a filing's windows, prune them, persist snippets.

The 6-K counterpart of the 8-K item classifier. Reads the ``sixk-windows`` spans
the segment stage wrote and writes ``sixk-snippets``, whose rows carry the
classified-item columns plus :data:`SIXK_EXTRA_COLUMNS`, so the extractor reads
both genres with one projection.

Per filing: rebuild each window over the text it was cut from
(:func:`cdt.segmenter.sixk.window_from_span`), admit windows with stage 1,
expand and merge the admitted ones
(:func:`cdt.segmenter.sixk.expand_admitted_windows`), then judge them with
stage 2. One row is written per snippet sent to stage 2, kept, dropped or
failed, with its verdict; windows stage 1 rejected are not persisted. Design
notes are in ``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pandas as pd

from cdt import settings
from cdt.classifier.core import (
    CLASSIFIED_ITEM_COLUMNS,
    LABEL_FAILED,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    VERDICT_DROPPED_DUPLICATE,
    VERDICT_DROPPED_NO_DETAILS,
    VERDICT_FAILED,
    VERDICT_KEPT,
    RetryResult,
    retry_failed_classifications,
)
from cdt.classifier.triage import (
    FilingVerdict,
    Snippet,
    SupportsChatCompletion,
    load_stage1_model,
    stage1_admit,
    triage_filing,
)
from cdt.datasets import (
    SIXK_DOCUMENT_DATASET_NAME,
    SIXK_SNIPPET_DATASET_NAME,
    SIXK_WINDOW_DATASET_NAME,
    dataset_root,
    date_shard_partition_path,
    document_shard,
    month_partition_path,
    resolve_artifact_root,
    source_partition_month,
)
from cdt.ingest.core import DOCUMENT_COLUMNS
from cdt.partition_stage import PartitionOutput, run_partition_stage
from cdt.segmenter.core import document_text_for_record, ensure_s3_client
from cdt.segmenter.sixk import (
    SIXK_WINDOW_COLUMNS,
    TextWindow,
    body_digest,
    expand_admitted_windows,
    gated_body,
    prose_documents,
    window_from_span,
)
from cdt.shared import get_logger
from cdt.storage.objects import artifact_exists
from cdt.storage.tables import read_table

LOGGER = get_logger(__name__)

#: Completion-registry and run-manifest name of the 6-K classify stage.
STAGE_NAME = "sixk-classify"
#: How many filings' stage-2 calls are in flight at once. One call per filing,
#: so this bounds provider concurrency for the whole stage.
DEFAULT_CONCURRENCY = 4
#: Columns beyond the classified-item contract, for auditing stage-2 decisions
#: and tracing a snippet to its span of the flattened document. The extractor
#: does not read them.
SIXK_EXTRA_COLUMNS = [
    "sixk_window_start",
    "sixk_window_end",
    "sixk_token_count",
    "sixk_duplicate_of",
    # Comma-separated indices of the admitted windows this row's text covers
    # (several when adjacent windows merged).
    "sixk_member_windows",
]
SIXK_SNIPPET_COLUMNS = [*CLASSIFIED_ITEM_COLUMNS, *SIXK_EXTRA_COLUMNS]
SIXK_SNIPPET_INTEGER_COLUMNS = [
    "start_line",
    "end_line",
    "section_char_count",
    "sixk_window_start",
    "sixk_window_end",
    "sixk_token_count",
]
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OPENAI = "openai"


def sixk_snippets_root(
    artifact_root: str | Path | None = None,
) -> str:
    """Return the canonical 6-K snippets dataset root."""
    return dataset_root(SIXK_SNIPPET_DATASET_NAME, artifact_root=artifact_root)


def snippet_id_for(
    accession_number: str, document_index: int, window_index: int
) -> str:
    """Build the stage-2 snippet id: what the model sees and answers about."""
    return f"{accession_number}:{document_index}:{window_index}"


def item_id_for(
    accession_number: str, document_index: int, start: int, end: int
) -> str:
    """Build a snippet's extractor-facing item id from the span its text covers.

    Disjoint from 8-K item ids (``{accession}-{item}`` with a dotted item
    number), so both genres can share a mentions partition keyed by item id.
    Naming the span, not a window index, means the id changes whenever the
    snippet's text does; the extractor skips ids it has already finished, so a
    regrouped snippet must not keep its old id.
    """
    return f"{accession_number}-6K-{document_index}-{start}-{end}"


class OpenRouterTextClient:
    """Adapt ``extractor.llm.OpenRouterChatClient`` to return text only."""

    def __init__(self: Self, *, api_key: str | None = None) -> None:
        """Initialize the underlying extractor client."""
        from cdt.extractor.llm import OpenRouterChatClient

        self._inner = OpenRouterChatClient(api_key=api_key)

    async def complete(
        self: Self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Return one completion's text."""
        result = await self._inner.complete(
            messages=messages, model=model, reasoning_effort=reasoning_effort
        )
        return result.text


class OpenAIChatClient:
    """Chat client for the OpenAI API, selected by ``SIXK_TRIAGE_PROVIDER=openai``."""

    def __init__(self: Self, *, api_key: str | None = None) -> None:
        """Initialize the OpenAI client.

        Raises:
            RuntimeError: If neither ``api_key`` nor ``OPENAI_API_KEY`` is set.
        """
        from openai import AsyncOpenAI

        key = api_key or settings.OPENAI_API_KEY
        if not key:
            msg = "OPENAI_API_KEY is required for SIXK_TRIAGE_PROVIDER=openai."
            raise RuntimeError(msg)
        self._client = AsyncOpenAI(api_key=key, timeout=300.0, max_retries=3)

    async def complete(
        self: Self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Return one completion's text."""
        # Settings carry OpenRouter slugs; the OpenAI API wants the bare id.
        bare_model = model.split("/", 1)[-1]
        kwargs: dict[str, object] = {"model": bare_model, "messages": messages}
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort
        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        return response.choices[0].message.content or ""


def default_triage_client() -> SupportsChatCompletion:
    """Build the stage-2 client for ``settings.SIXK_TRIAGE_PROVIDER``.

    Raises:
        ValueError: If the provider is neither ``openrouter`` nor ``openai``.
    """
    provider = (settings.SIXK_TRIAGE_PROVIDER or PROVIDER_OPENROUTER).strip().lower()
    if provider == PROVIDER_OPENROUTER:
        return OpenRouterTextClient()
    if provider == PROVIDER_OPENAI:
        return OpenAIChatClient()
    msg = (
        f"SIXK_TRIAGE_PROVIDER={provider!r} is not recognised; "
        f"expected {PROVIDER_OPENROUTER!r} or {PROVIDER_OPENAI!r}."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class _Candidate:
    """One window of one document of one filing, before stage 1 sees it."""

    snippet_id: str
    document_index: int
    document_type: str
    window: TextWindow


@dataclass(frozen=True)
class _SentSnippet:
    """One snippet as stage 2 receives it: an admitted window plus context.

    ``candidate`` is the group's first member; its snippet id is the row's
    ``item``. The row's ``item_id`` comes from ``window``'s span instead (see
    :func:`item_id_for`).
    """

    snippet: Snippet
    candidate: _Candidate
    window: TextWindow
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class _FilingPlan:
    """A filing's document row and the windows the gate left of it."""

    document: dict[str, object]
    candidates: list[_Candidate]


#: What to do when stored spans no longer match their text. A plain segment
#: run re-windows every documents partition that changed; --force is needed
#: only when the segmenter itself changed, and re-windows every partition.
RESEGMENT_ADVICE = (
    "Run `cdt segment --genres 6-K`, adding --force only if the segmenter "
    "code changed."
)


class StaleSegmentationError(RuntimeError):
    """Window spans no longer index the text their source submission yields."""


@dataclass(frozen=True)
class TriageOutput:
    """Snippet rows for a set of filings, and the filings that could not be judged."""

    rows: pd.DataFrame
    #: Accessions whose stored spans do not match the rebuilt text (see
    #: :func:`_candidates_for_filing`); none of their rows are in ``rows``.
    stale_accessions: tuple[str, ...] = ()
    #: Accessions whose stage-2 call failed on the provider or transport;
    #: their rows are in ``rows`` as failed, but the caller holds the partition.
    infrastructure_failed_accessions: tuple[str, ...] = ()
    #: Accession -> error for every other filing stage 2 could not judge
    #: (validation exhausted, a rejected request); its rows are failed.
    failures: dict[str, str] = field(default_factory=dict)


def triage_windows(
    windows: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    model_dir: Path | None = None,
    artifacts: tuple[object, float] | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
    renew: Callable[[], None] | None = None,
) -> TriageOutput:
    """Score, expand and prune in-memory 6-K window spans.

    ``windows`` are span rows (SIXK_WINDOW_COLUMNS); ``documents`` the 6-K
    documents rows they were cut from, which supply each filing's text and
    metadata. ``artifacts`` is a pre-loaded ``(model, threshold)`` pair; when
    ``None`` the model is loaded from ``model_dir``. ``client`` defaults to
    :func:`default_triage_client`, built only if a stage-2 call is needed. A
    filing whose spans no longer match its text is reported in
    ``stale_accessions`` and not judged; the others are. ``renew`` is called
    after each stage-2 call returns.
    """
    if windows.empty:
        return TriageOutput(rows=_empty_snippets())

    model, threshold = artifacts or load_stage1_model(model_dir)
    documents_by_accession = {
        str(record["accession_number"]): record
        for record in documents.to_dict("records")
    }
    resolved_s3_client = ensure_s3_client(
        s3_client, list(documents_by_accession.values())
    )

    plans: list[_FilingPlan] = []
    stale: list[str] = []
    for accession_number, filing_windows in windows.groupby(
        "accession_number", sort=True
    ):
        document = documents_by_accession.get(str(accession_number))
        candidates = (
            None
            if document is None
            else _candidates_for_filing(
                document,
                filing_windows,
                s3_client=resolved_s3_client,
            )
        )
        if candidates is None:
            stale.append(str(accession_number))
            continue
        plans.append(_FilingPlan(document=document, candidates=candidates))
    if stale:
        LOGGER.error(
            "6-K window spans do not match their source text: accessions=%s. %s",
            ",".join(stale),
            RESEGMENT_ADVICE,
        )
    windowed = sum(len(plan.candidates) for plan in plans)

    admitted_by_filing = [
        stage1_admit(
            model,
            [
                (candidate.snippet_id, candidate.window.text)
                for candidate in plan.candidates
            ],
            threshold=threshold,
        )
        for plan in plans
    ]
    admitted_total = sum(len(admitted) for admitted in admitted_by_filing)
    sent_by_filing = [
        _expand_admitted(plan, admitted)
        for plan, admitted in zip(plans, admitted_by_filing, strict=True)
    ]
    sent_total = sum(len(sent) for sent in sent_by_filing)
    if admitted_total == 0:
        LOGGER.info(
            "6-K triage: filings=%s windows=%s stage1_admitted=0 (no stage-2 call)",
            len(plans),
            windowed,
        )
        return TriageOutput(rows=_empty_snippets(), stale_accessions=tuple(stale))

    verdicts = _judge_filings(
        plans,
        sent_by_filing,
        client=client,
        concurrency=concurrency,
        max_attempts=max_attempts,
        renew=renew,
    )
    rows = [
        row
        for plan, sent, verdict in zip(plans, sent_by_filing, verdicts, strict=True)
        for row in _snippet_rows(plan, sent, verdict)
    ]
    table = _normalize_snippets(pd.DataFrame(rows, columns=SIXK_SNIPPET_COLUMNS))
    infrastructure_failed = tuple(
        verdict.accession_number
        for verdict in verdicts
        if verdict is not None and verdict.infrastructure_error
    )
    failures = {
        verdict.accession_number: verdict.error
        for verdict in verdicts
        if verdict is not None
        and verdict.error is not None
        and not verdict.infrastructure_error
    }
    kept = int(table["relevance"].fillna(False).sum())
    failed = int(table["label"].eq(LABEL_FAILED).sum())
    LOGGER.info(
        "6-K triage: filings=%s windows=%s stage1_admitted=%s "
        "snippets_sent=%s kept=%s dropped=%s failed=%s failed_filings=%s "
        "infrastructure_failed_filings=%s",
        len(plans),
        windowed,
        admitted_total,
        sent_total,
        kept,
        sent_total - kept - failed,
        failed,
        len(failures),
        len(infrastructure_failed),
    )
    return TriageOutput(
        rows=table,
        stale_accessions=tuple(stale),
        infrastructure_failed_accessions=infrastructure_failed,
        failures=failures,
    )


def _candidates_for_filing(
    document: dict[str, object],
    windows: pd.DataFrame,
    *,
    s3_client: object | None,
) -> list[_Candidate] | None:
    """Rebuild one filing's stored window spans over its submission's text.

    Returns None when any span's document is missing from the submission, is
    now gated out, or yields text whose digest differs from ``source_sha256``:
    the spans were cut from different text, so none of them can be trusted.
    """
    accession_number = str(document["accession_number"])
    prose = prose_documents(document_text_for_record(document, s3_client=s3_client))
    candidates: list[_Candidate] = []
    for document_index, document_windows in windows.groupby(
        "document_index", sort=True
    ):
        index = int(document_index)
        body = gated_body(prose[index].text) if index < len(prose) else None
        expected = set(document_windows["source_sha256"])
        if body is None or expected != {body_digest(body)}:
            return None
        for row in document_windows.sort_values("window").itertuples(index=False):
            candidates.append(
                _Candidate(
                    snippet_id=snippet_id_for(accession_number, index, int(row.window)),
                    document_index=index,
                    document_type=str(row.document_type),
                    window=window_from_span(
                        body,
                        index=int(row.window),
                        start=int(row.start),
                        end=int(row.end),
                        token_count=int(row.token_count),
                    ),
                )
            )
    return candidates


def _expand_admitted(plan: _FilingPlan, admitted: list[Snippet]) -> list[_SentSnippet]:
    """Expand and merge one filing's admitted windows, document by document.

    Each result takes its snippet id from the group's first member and its
    score from the group's highest-scoring member.
    """
    by_snippet_id = {candidate.snippet_id: candidate for candidate in plan.candidates}
    scores = {snippet.snippet_id: snippet.score for snippet in admitted}
    by_document: dict[int, list[_Candidate]] = {}
    for snippet in admitted:
        candidate = by_snippet_id[snippet.snippet_id]
        by_document.setdefault(candidate.document_index, []).append(candidate)

    sent: list[_SentSnippet] = []
    for _, candidates in sorted(by_document.items()):
        by_window_index = {
            candidate.window.index: candidate for candidate in candidates
        }
        for expanded in expand_admitted_windows(
            [candidate.window for candidate in candidates]
        ):
            first = by_window_index[expanded.member_indices[0]]
            sent.append(
                _SentSnippet(
                    snippet=Snippet(
                        snippet_id=first.snippet_id,
                        text=expanded.window.text,
                        score=max(
                            scores[by_window_index[index].snippet_id]
                            for index in expanded.member_indices
                        ),
                    ),
                    candidate=first,
                    window=expanded.window,
                    member_indices=expanded.member_indices,
                )
            )
    return sent


def _judge_filings(
    plans: Sequence[_FilingPlan],
    sent_by_filing: Sequence[list[_SentSnippet]],
    *,
    client: SupportsChatCompletion | None,
    concurrency: int,
    max_attempts: int | None,
    renew: Callable[[], None] | None = None,
) -> list[FilingVerdict | None]:
    """Run stage 2 concurrently for every filing with snippets to send.

    At most ``concurrency`` calls are in flight; ``renew`` is called as each
    returns. Returns one verdict per plan, ``None`` for a filing with nothing
    sent.
    """
    # Built here so a batch that makes no call needs no API key.
    resolved_client = client or default_triage_client()
    semaphore = asyncio.Semaphore(concurrency)
    kwargs = {} if max_attempts is None else {"max_attempts": max_attempts}

    async def judge(
        plan: _FilingPlan, sent: list[_SentSnippet]
    ) -> FilingVerdict | None:
        if not sent:
            return None
        async with semaphore:
            verdict = await triage_filing(
                resolved_client,
                str(plan.document["accession_number"]),
                [item.snippet for item in sent],
                **kwargs,  # type: ignore[arg-type]
            )
        if renew is not None:
            renew()
        return verdict

    async def judge_all() -> list[FilingVerdict | None]:
        return list(
            await asyncio.gather(
                *(
                    judge(plan, sent)
                    for plan, sent in zip(plans, sent_by_filing, strict=True)
                )
            )
        )

    return asyncio.run(judge_all())


def _snippet_rows(
    plan: _FilingPlan,
    sent: list[_SentSnippet],
    verdict: FilingVerdict | None,
) -> list[dict[str, object]]:
    """Build one persisted row per snippet sent to stage 2; all failed if it failed."""
    if not sent or verdict is None:
        return []
    kept = set(verdict.kept)
    no_details = set(verdict.dropped_no_details)
    duplicates = dict(verdict.dropped_duplicate)
    rows: list[dict[str, object]] = []
    for item in sent:
        snippet = item.snippet
        if verdict.error is not None:
            resolved = VERDICT_FAILED
        elif snippet.snippet_id in kept:
            resolved = VERDICT_KEPT
        elif snippet.snippet_id in duplicates:
            resolved = VERDICT_DROPPED_DUPLICATE
        elif snippet.snippet_id in no_details:
            resolved = VERDICT_DROPPED_NO_DETAILS
        else:
            # Unreachable through triage_filing; never mark an unjudged
            # snippet relevant.
            resolved = VERDICT_DROPPED_NO_DETAILS
        rows.append(_snippet_row(plan.document, item, resolved, duplicates))
    return rows


def _snippet_row(
    document: dict[str, object],
    item: _SentSnippet,
    resolved_verdict: str,
    duplicates: dict[str, str],
) -> dict[str, object]:
    candidate = item.candidate
    snippet = item.snippet
    if resolved_verdict == VERDICT_KEPT:
        label = LABEL_RELEVANT
    elif resolved_verdict == VERDICT_FAILED:
        label = LABEL_FAILED
    else:
        label = LABEL_IRRELEVANT
    accession_number = str(document["accession_number"])
    return {
        "item_id": item_id_for(
            accession_number,
            candidate.document_index,
            item.window.start,
            item.window.end,
        ),
        "item": candidate.snippet_id,
        "accession_number": accession_number,
        "cik": document.get("cik"),
        "company_name": document.get("company_name"),
        "url": document.get("url"),
        "text": item.window.text,
        "date": document.get("date"),
        "resource_uri": document.get("resource_uri"),
        # 8-K itemizer fields with no 6-K analogue; duplicates are recorded in
        # verdict and sixk_duplicate_of instead.
        "item_information": None,
        "extraction_status": None,
        "duplicate_resolution": None,
        # The document's own <TYPE>: 6-K, EX-99.1, ...
        "section_heading": candidate.document_type,
        # Snippets are character spans, recorded in the sixk_* columns.
        "start_line": None,
        "end_line": None,
        "section_char_count": len(item.window.text),
        # Same vocabulary as the 8-K classifier.
        "label": label,
        "relevance": label == LABEL_RELEVANT,
        "classification_score": snippet.score,
        "verdict": resolved_verdict,
        # The expanded span, matching `text`.
        "sixk_window_start": item.window.start,
        "sixk_window_end": item.window.end,
        "sixk_token_count": item.window.token_count,
        "sixk_duplicate_of": duplicates.get(snippet.snippet_id),
        "sixk_member_windows": ",".join(str(index) for index in item.member_indices),
    }


def _empty_snippets() -> pd.DataFrame:
    return pd.DataFrame(columns=SIXK_SNIPPET_COLUMNS)


def _normalize_snippets(table: pd.DataFrame) -> pd.DataFrame:
    """Coerce snippet columns to Parquet-friendly dtypes."""
    if table.empty:
        return table.reindex(columns=SIXK_SNIPPET_COLUMNS)
    normalized = table.reindex(columns=SIXK_SNIPPET_COLUMNS).copy()
    for column in SIXK_SNIPPET_INTEGER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(
            "Int64"
        )
    normalized["relevance"] = normalized["relevance"].astype(bool)
    return normalized


#: The window columns a filing's done key covers: a filing whose spans or
#: source text change is judged again.
_DONE_KEY_COLUMNS = [
    "document_index",
    "window",
    "start",
    "end",
    "token_count",
    "source_sha256",
]


def filing_done_key(accession_number: str, filing_windows: pd.DataFrame) -> str:
    """Return the done-set key of one filing's windows: its accession and their digest."""
    spans = sorted(
        tuple(str(value) for value in row)
        for row in filing_windows[_DONE_KEY_COLUMNS].itertuples(index=False)
    )
    digest = hashlib.sha256(json.dumps(spans).encode("utf-8")).hexdigest()[:16]
    return f"{accession_number}@{digest}"


def filing_done_keys(windows: pd.DataFrame) -> dict[str, str]:
    """Return each filing's done-set key (:func:`filing_done_key`), by accession."""
    return {
        str(accession_number): filing_done_key(str(accession_number), filing_windows)
        for accession_number, filing_windows in windows.groupby(
            "accession_number", sort=True
        )
    }


class _WindowPartitionTriage:
    """Triage one ``sixk-windows`` month file, optionally only some filings.

    A filing whose done key (:func:`filing_done_key`) is in the month's done
    set keeps its snippet rows from the ``sixk-snippets`` month file and makes
    no stage-2 call. The others are read from the 6-K documents partitions
    their date and accession place them in, and judged. The stage-1 model is
    loaded on the first call. A month holding a filing whose spans no longer
    match its text is held whole. When a stage-2 call fails on the provider or
    transport, the month is written with that filing's earlier rows kept and
    left pending, and its done set names only the filings judged, so the next
    run judges only that filing again. Filings that failed for any other
    reason are not in the done set either. :meth:`report` logs and raises for
    what was held.
    """

    def __init__(
        self: Self,
        *,
        artifact_root: str,
        model_dir: Path | None,
        client: SupportsChatCompletion | None,
        s3_client: object | None,
        concurrency: int,
        max_attempts: int | None,
    ) -> None:
        """Hold the stage settings; nothing is loaded until the first call.

        Raises:
            ValueError: If ``concurrency`` is not positive.
        """
        if concurrency <= 0:
            msg = f"concurrency must be positive, got {concurrency}"
            raise ValueError(msg)
        self.artifact_root = artifact_root
        self.model_dir = model_dir
        self.client = client
        self.s3_client = s3_client
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.artifacts: tuple[object, float] | None = None
        self.stale_partitions: list[str] = []
        self.infrastructure_held: list[str] = []

    def _documents_for(self: Self, windows: pd.DataFrame) -> pd.DataFrame:
        """Read the 6-K documents rows of the filings ``windows`` covers."""
        if windows.empty:
            return pd.DataFrame(columns=DOCUMENT_COLUMNS)
        filings = windows[["date", "accession_number"]].drop_duplicates()
        frames: list[pd.DataFrame] = []
        for (filing_date, shard), group in filings.groupby(
            [
                filings["date"].astype(str),
                filings["accession_number"].astype(str).map(document_shard),
            ],
            sort=True,
        ):
            documents_path = date_shard_partition_path(
                SIXK_DOCUMENT_DATASET_NAME,
                partition_date=str(filing_date),
                shard=str(shard),
                artifact_root=self.artifact_root,
            )
            if not artifact_exists(documents_path):
                continue
            documents = read_table(documents_path, DOCUMENT_COLUMNS).reindex(
                columns=DOCUMENT_COLUMNS
            )
            wanted = set(group["accession_number"].astype(str))
            frames.append(
                documents.loc[documents["accession_number"].astype(str).isin(wanted)]
            )
        populated = [frame for frame in frames if not frame.empty]
        if not populated:
            return pd.DataFrame(columns=DOCUMENT_COLUMNS)
        return pd.concat(populated, ignore_index=True)

    def _earlier_rows(
        self: Self, partition: dict[str, str], accessions: set[str]
    ) -> pd.DataFrame:
        """Return the stored snippet rows of ``accessions`` in the source's month."""
        if not accessions:
            return _empty_snippets()
        snippets_path = month_partition_path(
            SIXK_SNIPPET_DATASET_NAME,
            month=source_partition_month(partition),
            artifact_root=self.artifact_root,
        )
        stored = read_table(snippets_path, SIXK_SNIPPET_COLUMNS).reindex(
            columns=SIXK_SNIPPET_COLUMNS
        )
        return stored.loc[stored["accession_number"].astype(str).isin(accessions)]

    def __call__(
        self: Self,
        source_path: str,
        partition: dict[str, str],
        accessions: frozenset[str] | None = None,
        done_keys: frozenset[str] = frozenset(),
        renew: Callable[[], None] | None = None,
    ) -> PartitionOutput:
        """Triage the month's filings, or only ``accessions``, skipping done ones.

        A filing that could not be judged this run, because its spans are
        stale or its stage-2 call failed on the provider, keeps its earlier
        rows and stays out of the done set, and the month is held; every
        other filing's rows and done key are written.
        """
        if self.artifacts is None:
            self.artifacts = load_stage1_model(self.model_dir)
            LOGGER.info("Stage-1 threshold %.3f", self.artifacts[1])
        windows = read_table(source_path, SIXK_WINDOW_COLUMNS).reindex(
            columns=SIXK_WINDOW_COLUMNS
        )
        if accessions is not None:
            windows = windows.loc[
                windows["accession_number"].astype(str).isin(accessions)
            ]
        keys = filing_done_keys(windows)
        carried = {accession for accession, key in keys.items() if key in done_keys}
        to_judge = windows.loc[~windows["accession_number"].astype(str).isin(carried)]
        documents = self._documents_for(to_judge)
        self.s3_client = ensure_s3_client(self.s3_client, documents.to_dict("records"))
        output = triage_windows(
            to_judge,
            documents,
            artifacts=self.artifacts,
            client=self.client,
            s3_client=self.s3_client,
            concurrency=self.concurrency,
            max_attempts=self.max_attempts,
            renew=renew,
        )
        source_rows = windows["accession_number"].nunique()
        if output.stale_accessions:
            self.stale_partitions.append(source_path)
        if output.infrastructure_failed_accessions:
            self.infrastructure_held.append(source_path)
        held = set(output.stale_accessions) | set(
            output.infrastructure_failed_accessions
        )
        judged = output.rows.loc[
            ~output.rows["accession_number"].astype(str).isin(held)
        ]
        earlier = self._earlier_rows(partition, carried | held)
        frames = [frame for frame in (earlier, judged) if not frame.empty]
        rows = (
            _normalize_snippets(pd.concat(frames, ignore_index=True))
            if frames
            else _empty_snippets()
        )
        done = frozenset(
            key
            for accession, key in keys.items()
            if accession not in held and accession not in output.failures
        )
        if carried:
            LOGGER.info(
                "6-K triage: %s filing(s) of %s already judged; kept their rows",
                len(carried),
                source_path,
            )
        return PartitionOutput(
            rows=rows,
            source_rows=source_rows,
            complete=not held,
            failures=output.failures,
            done_keys=done,
        )

    def report(self: Self) -> None:
        """Log the partitions held after provider failures; raise for stale ones.

        Raises:
            StaleSegmentationError: If any partition was held for stale spans.
        """
        if self.infrastructure_held:
            LOGGER.warning(
                "%s 6-K windows partition(s) left pending after stage-2 provider "
                "failures; the next run triages them again: %s",
                len(self.infrastructure_held),
                ", ".join(self.infrastructure_held),
            )
        if self.stale_partitions:
            msg = (
                f"{len(self.stale_partitions)} 6-K windows partition(s) hold spans "
                f"that no longer match their source text and were left pending. "
                f"{RESEGMENT_ADVICE} Partitions: " + ", ".join(self.stale_partitions)
            )
            raise StaleSegmentationError(msg)


def triage_pending_windows(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    model_dir: Path | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Triage pending ``sixk-windows`` month files into ``sixk-snippets`` month files.

    A windows month is pending when its fingerprint changed since it was last
    classified, or always with ``force``. Its filings already judged with the
    same windows keep their rows; the rest cost one LLM call per filing with
    admitted windows (``force`` judges every filing). A filing whose spans no
    longer match its text, or whose stage-2 call failed on the provider or
    transport, keeps its earlier rows and leaves the month pending, so the
    next run judges it again rather than sending its snippets to extraction
    untriaged; the month's other filings are written and recorded as done.
    ``renew`` is also called as stage-2 calls return, so one long month keeps
    the lease. A filing stage 2
    could not judge for any other reason (its answers failed validation every
    attempt, or the request was rejected) is written as failed and listed in
    ``failures/sixk-classify/failures.json`` for :func:`retry_failed_triage`.
    Returns the snippet rows written this run.

    Raises:
        ValueError: If ``concurrency`` is not positive.
        StaleSegmentationError: After every other partition is processed, if
            any partition was left pending for stale spans.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    triage = _WindowPartitionTriage(
        artifact_root=resolved_root,
        model_dir=model_dir,
        client=client,
        s3_client=s3_client,
        concurrency=concurrency,
        max_attempts=max_attempts,
    )
    result = run_partition_stage(
        STAGE_NAME,
        source_dataset=SIXK_WINDOW_DATASET_NAME,
        output_dataset=SIXK_SNIPPET_DATASET_NAME,
        output_columns=SIXK_SNIPPET_COLUMNS,
        process=lambda source_path, partition, done, renew: triage(
            source_path, partition, None, done, renew
        ),
        artifact_root=resolved_root,
        force=force,
        renew=renew,
        manifest_extra={
            "concurrency": concurrency,
            "stage2_model": settings.SIXK_TRIAGE_MODEL,
            "stage2_provider": settings.SIXK_TRIAGE_PROVIDER,
        },
    )
    triage.report()
    return result.rows


def retry_failed_triage(
    *,
    artifact_root: str | Path | None = None,
    model_dir: Path | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
    renew: Callable[[], None] | None = None,
) -> RetryResult:
    """Triage again the 6-K filings in this stage's failure list.

    See :func:`cdt.classifier.core.retry_failed_classifications`.

    Raises:
        ValueError: If ``concurrency`` is not positive.
        StaleSegmentationError: After every other partition is retried, if any
            was held for stale spans.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    triage = _WindowPartitionTriage(
        artifact_root=resolved_root,
        model_dir=model_dir,
        client=client,
        s3_client=s3_client,
        concurrency=concurrency,
        max_attempts=max_attempts,
    )
    result = retry_failed_classifications(
        STAGE_NAME,
        source_dataset=SIXK_WINDOW_DATASET_NAME,
        output_dataset=SIXK_SNIPPET_DATASET_NAME,
        output_columns=SIXK_SNIPPET_COLUMNS,
        classify_partition=triage,
        artifact_root=resolved_root,
        renew=renew,
    )
    triage.report()
    return result
