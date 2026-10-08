"""The 6-K classify stage: score a filing's windows, prune them, persist snippets.

The 6-K counterpart of the 8-K item classifier. Reads the ``sixk-windows`` spans
the segment stage wrote and writes ``sixk-snippets``, whose rows carry the
classified-item columns plus :data:`SIXK_EXTRA_COLUMNS`, so the extractor reads
both genres with one projection.

Per filing: rebuild each window over the text it was cut from
(:func:`cdt.segmenter.sixk.window_from_span`), admit windows with stage 1,
expand and merge the admitted ones
(:func:`cdt.segmenter.sixk.expand_admitted_windows`), then judge them with
stage 2. One row is written per snippet stage 2 judged, kept or dropped, with
its verdict; windows stage 1 rejected are not persisted. Design notes are in
``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd

from cdt import settings
from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
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
    resolve_artifact_root,
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
#: Verdict vocabulary recorded per row, in ``sixk_verdict``.
VERDICT_KEPT = "kept"
#: Stage 2 failed for the filing, so the stage-1 admission stands.
VERDICT_KEPT_DEGRADED = "kept_degraded"
VERDICT_DROPPED_NO_DETAILS = "dropped_no_details"
VERDICT_DROPPED_DUPLICATE = "dropped_duplicate"
KEPT_VERDICTS = frozenset({VERDICT_KEPT, VERDICT_KEPT_DEGRADED})
#: Columns beyond the classified-item contract, for auditing stage-2 decisions
#: and tracing a snippet to its span of the flattened document. The extractor
#: does not read them.
SIXK_EXTRA_COLUMNS = [
    "sixk_window_start",
    "sixk_window_end",
    "sixk_token_count",
    "sixk_verdict",
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
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical 6-K snippets dataset root."""
    return dataset_root(
        SIXK_SNIPPET_DATASET_NAME, artifact_root=artifact_root, data_dir=data_dir
    )


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
    #: their rows are in ``rows`` as ``kept_degraded``.
    infrastructure_failed_accessions: tuple[str, ...] = ()


def triage_windows(
    windows: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    artifacts: tuple[object, float] | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
) -> TriageOutput:
    """Score, expand and prune in-memory 6-K window spans.

    ``windows`` are span rows (SIXK_WINDOW_COLUMNS); ``documents`` the 6-K
    documents rows they were cut from, which supply each filing's text and
    metadata. ``artifacts`` is a pre-loaded ``(model, threshold)`` pair; when
    ``None`` the model is loaded from ``model_dir``. ``client`` defaults to
    :func:`default_triage_client`, built only if a stage-2 call is needed.
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
                data_dir=data_dir,
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
        # The partition is held whole, so stage 2 would be paid for nothing.
        return TriageOutput(rows=_empty_snippets(), stale_accessions=tuple(stale))
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
    )
    rows = [
        row
        for plan, sent, verdict in zip(plans, sent_by_filing, verdicts, strict=True)
        for row in _snippet_rows(plan, sent, verdict)
    ]
    table = _normalize_snippets(pd.DataFrame(rows, columns=SIXK_SNIPPET_COLUMNS))
    degraded = sum(1 for verdict in verdicts if verdict is not None and verdict.error)
    infrastructure_failed = tuple(
        verdict.accession_number
        for verdict in verdicts
        if verdict is not None and verdict.infrastructure_error
    )
    kept = int(table["relevance"].fillna(False).sum())
    LOGGER.info(
        "6-K triage: filings=%s windows=%s stage1_admitted=%s "
        "snippets_sent=%s kept=%s dropped=%s degraded_filings=%s "
        "infrastructure_failed_filings=%s",
        len(plans),
        windowed,
        admitted_total,
        sent_total,
        kept,
        sent_total - kept,
        degraded,
        len(infrastructure_failed),
    )
    return TriageOutput(
        rows=table,
        stale_accessions=tuple(stale),
        infrastructure_failed_accessions=infrastructure_failed,
    )


def _candidates_for_filing(
    document: dict[str, object],
    windows: pd.DataFrame,
    *,
    data_dir: Path | None,
    s3_client: object | None,
) -> list[_Candidate] | None:
    """Rebuild one filing's stored window spans over its submission's text.

    Returns None when any span's document is missing from the submission, is
    now gated out, or yields text whose digest differs from ``source_sha256``:
    the spans were cut from different text, so none of them can be trusted.
    """
    accession_number = str(document["accession_number"])
    prose = prose_documents(
        document_text_for_record(document, data_dir=data_dir, s3_client=s3_client)
    )
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
) -> list[FilingVerdict | None]:
    """Run stage 2 concurrently for every filing with snippets to send.

    At most ``concurrency`` calls are in flight. Returns one verdict per plan,
    ``None`` for a filing with nothing sent.
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
            return await triage_filing(
                resolved_client,
                str(plan.document["accession_number"]),
                [item.snippet for item in sent],
                **kwargs,  # type: ignore[arg-type]
            )

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
    """Build one persisted row per snippet stage 2 judged."""
    if not sent or verdict is None:
        return []
    kept = set(verdict.kept)
    no_details = set(verdict.dropped_no_details)
    duplicates = dict(verdict.dropped_duplicate)
    rows: list[dict[str, object]] = []
    for item in sent:
        snippet = item.snippet
        if snippet.snippet_id in kept:
            resolved = VERDICT_KEPT_DEGRADED if verdict.error else VERDICT_KEPT
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
    relevant = resolved_verdict in KEPT_VERDICTS
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
        # sixk_verdict instead.
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
        "label": "relevant" if relevant else "irrelevant",
        "relevance": relevant,
        "classification_score": snippet.score,
        # The expanded span, matching `text`.
        "sixk_window_start": item.window.start,
        "sixk_window_end": item.window.end,
        "sixk_token_count": item.window.token_count,
        "sixk_verdict": resolved_verdict,
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


def triage_pending_windows(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    model_dir: Path | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Triage pending ``sixk-windows`` partitions into ``sixk-snippets`` partitions.

    A windows partition is pending when its fingerprint changed since it was
    last classified, or always with ``force``, and is recomputed whole, at one
    LLM call per filing with admitted windows. Each partition's filings are read
    from the 6-K documents partition of the same date and shard. A partition
    holding a filing whose spans no longer match its text is left pending and
    unwritten, as is one where a stage-2 call failed on the provider or
    transport, so the next run triages it again rather than sending its
    snippets to extraction untriaged. A stage-2 call whose answers fail
    validation every attempt still keeps its snippets. Returns the snippet
    rows written this run.

    Raises:
        ValueError: If ``concurrency`` is not positive.
        StaleSegmentationError: After every other partition is processed, if
            any partition was left pending for stale spans.
    """
    if concurrency <= 0:
        msg = f"concurrency must be positive, got {concurrency}"
        raise ValueError(msg)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    artifacts: tuple[object, float] | None = None
    shared_client = s3_client
    stale_partitions: list[str] = []
    infrastructure_held: list[str] = []

    def process(source_path: str, partition: dict[str, str]) -> PartitionOutput:
        nonlocal artifacts, shared_client
        if artifacts is None:
            artifacts = load_stage1_model(model_dir)
            LOGGER.info("Stage-1 threshold %.3f", artifacts[1])
        windows = read_table(source_path, SIXK_WINDOW_COLUMNS).reindex(
            columns=SIXK_WINDOW_COLUMNS
        )
        documents_path = date_shard_partition_path(
            SIXK_DOCUMENT_DATASET_NAME,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        documents = (
            read_table(documents_path, DOCUMENT_COLUMNS).reindex(
                columns=DOCUMENT_COLUMNS
            )
            if artifact_exists(documents_path)
            else pd.DataFrame(columns=DOCUMENT_COLUMNS)
        )
        shared_client = ensure_s3_client(shared_client, documents.to_dict("records"))
        output = triage_windows(
            windows,
            documents,
            data_dir=data_dir,
            artifacts=artifacts,
            client=client,
            s3_client=shared_client,
            concurrency=concurrency,
            max_attempts=max_attempts,
        )
        if output.stale_accessions:
            stale_partitions.append(source_path)
        elif output.infrastructure_failed_accessions:
            infrastructure_held.append(source_path)
        return PartitionOutput(
            rows=output.rows,
            source_rows=windows["accession_number"].nunique(),
            complete=not (
                output.stale_accessions or output.infrastructure_failed_accessions
            ),
        )

    result = run_partition_stage(
        STAGE_NAME,
        source_dataset=SIXK_WINDOW_DATASET_NAME,
        output_dataset=SIXK_SNIPPET_DATASET_NAME,
        output_columns=SIXK_SNIPPET_COLUMNS,
        process=process,
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
        renew=renew,
        manifest_extra={
            "concurrency": concurrency,
            "stage2_model": settings.SIXK_TRIAGE_MODEL,
            "stage2_provider": settings.SIXK_TRIAGE_PROVIDER,
        },
    )
    if infrastructure_held:
        LOGGER.warning(
            "%s 6-K windows partition(s) left pending after stage-2 provider "
            "failures; the next run triages them again: %s",
            len(infrastructure_held),
            ", ".join(infrastructure_held),
        )
    if stale_partitions:
        msg = (
            f"{len(stale_partitions)} 6-K windows partition(s) hold spans that "
            f"no longer match their source text and were left pending. "
            f"{RESEGMENT_ADVICE} Partitions: " + ", ".join(stale_partitions)
        )
        raise StaleSegmentationError(msg)
    return result.rows
