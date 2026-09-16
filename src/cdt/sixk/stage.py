"""The 6-K triage stage: window a filing, score it, prune it, persist it.

Where the 8-K path runs itemize → classify, the 6-K path runs this one stage,
because a 6-K has no items to itemize and nothing for the item classifier to
classify. It reads the 6-K documents dataset and writes ``sixk-snippets``,
whose rows carry the classified-item columns verbatim — so the extractor reads
both genres with one projection and no per-source knowledge.

Its own dataset rather than a shared ``classifications`` one: classify already
owns ``classifications/date=D/shard=S/part-0000.parquet`` and rewrites it whole,
so a second writer there would need merge-on-write with genre-scoped row
replacement and an ordering hazard between two completion registries. One
writer per dataset is the invariant the file-native design leans on.

Between the two stages, an admitted window is expanded backwards into the
context the 400-token crop cut off (#172), and windows whose expansions run
into each other merge. Expansion happens *after* stage 1 and never before:
``WINDOW_TOKENS`` is what the stage-1 model was fitted on and what its
threshold was calibrated against, so widening the text it scores would
invalidate both.

Every window stage 1 admits is accounted for, kept or dropped, with the
stage-2 verdict on the row. The unit of a row is the snippet stage 2 judged,
not the window stage 1 admitted, because merging makes those differ: a merged
snippet is one row listing its members in ``sixk_member_windows``. One row per
member instead would carry the merged text more than once, and the extractor
reads rows — it would pay for the same text twice, which is the cost merging
exists to avoid. Dropped snippets are the mechanism rather than a side
effect — stage 2 exists to consolidate siblings — and it is a non-deterministic
LLM, so its decisions have to be auditable after the fact. Windows stage 1
rejected are not persisted: it admits 5.8% of them, and storing the rest would
grow the dataset ~17x to record that nothing happened.
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
from cdt.datasets import (
    SIXK_SNIPPET_DATASET_NAME,
    CompletedPartition,
    completion_registry_path,
    dataset_root,
    date_shard_partition_path,
    parse_date_shard_partition,
    pending_source_partitions,
    resolve_artifact_root,
    run_manifest_path,
    save_completion_registry,
)
from cdt.ingest import DOCUMENT_COLUMNS, SIXK_DOCUMENT_DATASET_NAME
from cdt.itemizer.core import document_text_for_record, ensure_s3_client
from cdt.shared import get_logger
from cdt.sixk.documents import prose_documents
from cdt.sixk.triage import (
    FilingVerdict,
    Snippet,
    SupportsChatCompletion,
    load_stage1_model,
    stage1_admit,
    triage_filing,
)
from cdt.sixk.windows import TextWindow, expand_admitted_windows, prepare_filing
from cdt.storage import read_table, write_json_artifact, write_partition_table

LOGGER = get_logger(__name__)

STAGE_NAME = "sixk"
#: How many filings' stage-2 calls are in flight at once. One call per filing,
#: so this bounds provider concurrency for the whole stage.
DEFAULT_CONCURRENCY = 4
#: Verdict vocabulary recorded per row, in ``sixk_verdict``.
VERDICT_KEPT = "kept"
#: Stage 2 could not be reached or would not answer, so stage-1 output stands.
VERDICT_KEPT_DEGRADED = "kept_degraded"
VERDICT_DROPPED_NO_DETAILS = "dropped_no_details"
VERDICT_DROPPED_DUPLICATE = "dropped_duplicate"
KEPT_VERDICTS = frozenset({VERDICT_KEPT, VERDICT_KEPT_DEGRADED})
#: Columns beyond the classified-item contract. The extractor projects the
#: shared columns and never sees these; they exist so a stage-2 decision can be
#: audited and a snippet traced back to its span of the flattened document.
SIXK_EXTRA_COLUMNS = [
    "sixk_window_start",
    "sixk_window_end",
    "sixk_token_count",
    "sixk_verdict",
    "sixk_duplicate_of",
    # Comma-separated window indices this row's text answers for: several when
    # adjacent admitted windows merged, one otherwise. With the row's own
    # accession and document index (both in `item`), it names every window
    # stage 1 admitted, which is what keeps admissions auditable now that a row
    # is a stage-2 snippet rather than a window.
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


def item_id_for(accession_number: str, document_index: int, window_index: int) -> str:
    """Build a snippet's extractor-facing item id.

    Disjoint from 8-K item ids by construction (those are
    ``{accession}-{item}`` with a dotted item number), which is what lets both
    genres write mentions into one partition, merged by item id.
    """
    return f"{accession_number}-6K-{document_index}-{window_index}"


class OpenRouterTextClient:
    """The extractor's OpenRouter client, reduced to returning text.

    ``extractor.core.OpenRouterChatClient`` returns a ``CompletionResult``
    carrying provider metadata the extractor records per attempt;
    ``SupportsChatCompletion`` here wants the text alone. Adapting is better
    than a second client: request shaping, timeouts and the sampling rules for
    reasoning models stay in one place.
    """

    def __init__(self: Self, *, api_key: str | None = None) -> None:
        """Initialize the underlying extractor client."""
        from cdt.extractor.core import OpenRouterChatClient

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
    """Chat client talking to the OpenAI API directly.

    Stage 2 is priced for volume and the shared OpenRouter account has run out
    of credit before — and OpenRouter reserves an estimated maximum cost per
    in-flight request, so it fails first under exactly this stage's shape (many
    concurrent long-prompt calls). Selecting a provider is then a setting change
    rather than a blocked run.
    """

    def __init__(self: Self, *, api_key: str | None = None) -> None:
        """Initialize the OpenAI client."""
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
    """Build the stage-2 client the configured provider asks for."""
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

    ``candidate`` is the first member's, which is what gives the row its
    identity — ``expand_admitted_windows`` indexes a merged window by its first
    member, so a group is named by the earliest window in it and no two groups
    can claim the same name.
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

    ``artifacts`` is a pre-loaded ``(model, threshold)`` pair; callers looping
    over partitions pass it so the pickle is deserialized once per run rather
    than once per partition, as the 8-K classifier does (#76).
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
        # prepare_filing gates on debt vocabulary per document and windows what
        # survives; the gate is why most filings produce nothing here.
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
    """Give each admitted window back the context its crop cut off (#172).

    Grouped by document before expanding, because offsets only mean anything
    within the text they index into: expanding across two documents would
    splice unrelated text together, and `expand_admitted_windows` rejects it.
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
                        # The strongest admission in the group. Stage 2 never
                        # reads it; it is persisted as the row's
                        # classification_score, where the weakest member's
                        # would understate why the text was sent at all.
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
    """Run stage 2 for every filing with admitted windows, bounded in flight.

    One event loop for the whole batch, and a semaphore rather than a serial
    loop: stage 2 sees a whole filing at once, so calls are independent and a
    partition's worth of them is the natural unit of concurrency.
    """
    # Built here, not by the caller: a batch whose filings all failed the gate
    # makes no call, and must not need an API key to find that out.
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
            # validate_verdict requires the answer to partition the ids, so
            # this is unreachable through triage_filing; be explicit rather
            # than silently mark an unjudged window relevant.
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
            accession_number, candidate.document_index, item.window.index
        ),
        "item": candidate.snippet_id,
        "accession_number": accession_number,
        "cik": document.get("cik"),
        "company_name": document.get("company_name"),
        "url": document.get("url"),
        # What the extractor reads: the expanded, possibly merged text, not
        # the crop stage 1 scored.
        "text": item.window.text,
        "date": document.get("date"),
        # Kept so a snippet can be traced to the submission it came from.
        "resource_uri": document.get("resource_uri"),
        # 8-K itemizer fields with no 6-K analogue: a window has no declared
        # item information, no per-item extraction status, and its duplicate
        # decision is stage 2's, recorded in sixk_verdict.
        "item_information": None,
        "extraction_status": None,
        "duplicate_resolution": None,
        # The document's own <TYPE>: 6-K, EX-99.1, ...
        "section_heading": candidate.document_type,
        # Windows are character spans, not line ranges, so the line columns
        # stay null and the span goes in the sixk_* columns below.
        "start_line": None,
        "end_line": None,
        "section_char_count": len(item.window.text),
        # Same vocabulary as the 8-K classifier, so one consumer reading
        # `label` across both genres sees one set of values.
        "label": "relevant" if relevant else "irrelevant",
        "relevance": relevant,
        "classification_score": snippet.score,
        # The expanded span, so the row's offsets and its text agree; the
        # admitted crops inside it are named by sixk_member_windows.
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
    """Triage pending 6-K document partitions into snippet partitions."""
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
    # Fingerprint-keyed selection, like itemize and classify: a documents
    # partition ingest merged new rows into is pending again and recomputed
    # whole. Unlike those two this stage costs an LLM call per filing, so the
    # recompute is not free — but a partition only changes when ingest actually
    # merged something into it, and stage 2 is ~$0.25 per 1,000 filings (#62).
    pending_with_fingerprints, registry = pending_source_partitions(
        STAGE_NAME,
        SIXK_DOCUMENT_DATASET_NAME,
        SIXK_SNIPPET_DATASET_NAME,
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

        # Persist completion at every batch boundary and renew the lease there,
        # for the reasons itemize and classify do (#111, #88).
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
