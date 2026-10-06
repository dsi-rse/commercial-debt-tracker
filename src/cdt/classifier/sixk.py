"""The 6-K triage stage: window a filing, score it, prune it, persist it.

The 6-K counterpart of itemize → classify. Reads the 6-K documents dataset and
writes ``sixk-snippets``, whose rows carry the classified-item columns plus
:data:`SIXK_EXTRA_COLUMNS`, so the extractor reads both genres with one
projection.

Per filing: window each prose document (:func:`cdt.segmenter.sixk.prepare_filing`),
admit windows with stage 1, expand and merge the admitted ones
(:func:`cdt.segmenter.sixk.expand_admitted_windows`), then judge them with stage 2. One
row is written per snippet stage 2 judged, kept or dropped, with its verdict;
windows stage 1 rejected are not persisted. Design notes are in
``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
from cdt.completion import (
    CompletedPartition,
    completion_registry_path,
    pending_source_partitions,
    save_completion_registry,
)
from cdt.datasets import (
    SIXK_DOCUMENT_DATASET_NAME,
    SIXK_SNIPPET_DATASET_NAME,
    dataset_root,
    date_shard_partition_path,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
)
from cdt.ingest.core import DOCUMENT_COLUMNS
from cdt.segmenter.core import document_text_for_record, ensure_s3_client
from cdt.segmenter.sixk import (
    TextWindow,
    expand_admitted_windows,
    prepare_filing,
    prose_documents,
)
from cdt.shared import get_logger
from cdt.storage.objects import write_json_artifact
from cdt.storage.tables import read_table, write_partition_table

LOGGER = get_logger(__name__)

STAGE_NAME = "sixk"
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


def triage_documents(
    documents: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    artifacts: tuple[object, float] | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
) -> pd.DataFrame:
    """Window, score and prune in-memory 6-K document rows.

    ``artifacts`` is a pre-loaded ``(model, threshold)`` pair; when ``None`` the
    model is loaded from ``model_dir``. ``client`` defaults to
    :func:`default_triage_client`, built only if a stage-2 call is needed.
    Returns one row per judged snippet; empty when nothing passes stage 1.
    """
    if documents.empty:
        return _empty_snippets()

    model, threshold = artifacts or load_stage1_model(model_dir)
    records = documents.to_dict("records")
    resolved_s3_client = ensure_s3_client(s3_client, records)

    plans: list[_FilingPlan] = []
    windowed = 0
    for record in records:
        candidates = _candidates_for_filing(
            record, data_dir=data_dir, s3_client=resolved_s3_client
        )
        windowed += len(candidates)
        plans.append(_FilingPlan(document=record, candidates=candidates))

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
            "6-K triage: filings=%s gated_windows=%s stage1_admitted=0 (no stage-2 call)",
            len(plans),
            windowed,
        )
        return _empty_snippets()

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
    LOGGER.info(
        "6-K triage: filings=%s gated_windows=%s stage1_admitted=%s "
        "snippets_sent=%s kept=%s dropped=%s degraded_filings=%s",
        len(plans),
        windowed,
        admitted_total,
        sent_total,
        int(table["relevance"].fillna(False).sum()),
        sent_total - int(table["relevance"].fillna(False).sum()),
        degraded,
    )
    return table


def _candidates_for_filing(
    document: dict[str, object],
    *,
    data_dir: Path | None,
    s3_client: object | None,
) -> list[_Candidate]:
    """Split, flatten, gate and window one filing's submission."""
    accession_number = str(document["accession_number"])
    submission = document_text_for_record(
        document, data_dir=data_dir, s3_client=s3_client
    )
    candidates: list[_Candidate] = []
    for document_index, prose in enumerate(prose_documents(submission)):
        for window in prepare_filing(prose.text):
            candidates.append(
                _Candidate(
                    snippet_id=snippet_id_for(
                        accession_number, document_index, window.index
                    ),
                    document_index=document_index,
                    document_type=prose.document_type,
                    window=window,
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


def triage_pending_documents(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    model_dir: Path | None = None,
    client: SupportsChatCompletion | None = None,
    s3_client: object | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int | None = None,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Triage pending 6-K document partitions into snippet partitions.

    A documents partition is pending when its fingerprint changed since it was
    last triaged, or always with ``force``. Writes a snippets partition for each
    one that yields rows, the completion registry and a run manifest; ``renew``
    is called at each batch boundary. Returns the concatenated snippets.

    Raises:
        ValueError: If ``batch_size`` or ``concurrency`` is not positive.
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if concurrency <= 0:
        msg = f"concurrency must be positive, got {concurrency}"
        raise ValueError(msg)

    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    processed_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    visited_document_paths: set[str] = set()
    empty_partitions = 0
    total_documents = 0
    # A partition ingest merged rows into is recomputed whole, at one LLM call
    # per filing with admitted windows.
    pending_with_fingerprints, registry = pending_source_partitions(
        STAGE_NAME,
        SIXK_DOCUMENT_DATASET_NAME,
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
    )
    pending_document_paths = [path for path, _ in pending_with_fingerprints]
    source_fingerprints = dict(pending_with_fingerprints)

    artifacts: tuple[object, float] | None = None
    if pending_document_paths:
        model, threshold = load_stage1_model(model_dir)
        artifacts = (model, threshold)
        LOGGER.info("Stage-1 threshold %.3f", threshold)

    total_partitions = len(pending_document_paths)
    for chunk_start in range(0, total_partitions, batch_size):
        chunk_paths = pending_document_paths[chunk_start : chunk_start + batch_size]
        for partition_index, document_path in enumerate(
            chunk_paths, start=chunk_start + 1
        ):
            partition = parse_date_shard_partition(document_path)
            partition_label = f"date={partition['date']} shard={partition['shard']}"
            partition_start = perf_counter()
            visited_document_paths.add(document_path)
            documents = read_table(document_path, DOCUMENT_COLUMNS).reindex(
                columns=DOCUMENT_COLUMNS
            )
            total_documents += len(documents)
            snippets = triage_documents(
                documents,
                data_dir=data_dir,
                model_dir=model_dir,
                artifacts=artifacts,
                client=client,
                s3_client=s3_client,
                concurrency=concurrency,
                max_attempts=max_attempts,
            )
            if snippets.empty:
                empty_partitions += 1
            else:
                write_partition_table(
                    sixk_snippets_root(resolved_root, data_dir=data_dir),
                    partition={"date": partition["date"], "shard": partition["shard"]},
                    table=snippets.reindex(columns=SIXK_SNIPPET_COLUMNS),
                )
                processed_frames.append(snippets)
                partitions_written.append(
                    date_shard_partition_path(
                        SIXK_SNIPPET_DATASET_NAME,
                        partition_date=partition["date"],
                        shard=partition["shard"],
                        artifact_root=resolved_root,
                        data_dir=data_dir,
                    )
                )
            LOGGER.info(
                "6-K triage partition complete: %s progress=%s/%s filings=%s "
                "snippets=%s relevant=%s wrote_output=%s elapsed=%.1fs",
                partition_label,
                partition_index,
                total_partitions,
                len(documents),
                len(snippets),
                int(snippets["relevance"].fillna(False).sum())
                if not snippets.empty
                else 0,
                not snippets.empty,
                perf_counter() - partition_start,
            )

        # Persist completion and renew the writer lease per batch, so an
        # interruption keeps finished batches and a long run keeps its lease.
        for document_path in chunk_paths:
            registry[document_path] = CompletedPartition(
                fingerprint=source_fingerprints.get(document_path)
            )
        save_completion_registry(
            STAGE_NAME,
            registry,
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        if renew is not None:
            renew()

    save_completion_registry(
        STAGE_NAME, registry, artifact_root=resolved_root, data_dir=data_dir
    )
    write_json_artifact(
        run_manifest_path(
            STAGE_NAME, "latest", artifact_root=resolved_root, data_dir=data_dir
        ),
        {
            "artifact_root": resolved_root,
            "stage": STAGE_NAME,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "force": force,
            "stage2_model": settings.SIXK_TRIAGE_MODEL,
            "stage2_provider": settings.SIXK_TRIAGE_PROVIDER,
            "documents_processed": total_documents,
            "partitions_visited": sorted(visited_document_paths),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "completion_registry": completion_registry_path(
                STAGE_NAME, artifact_root=resolved_root, data_dir=data_dir
            ),
        },
    )
    LOGGER.info(
        "6-K triage complete: documents=%s partitions_written=%s",
        total_documents,
        len(partitions_written),
    )
    if not processed_frames:
        return _empty_snippets()
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=SIXK_SNIPPET_COLUMNS
    )
