"""File-native orchestration for the full CDT pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Self

import pandas as pd

from cdt.classifier import classify_pending_items, default_model_dir
from cdt.datasets import normalize_cik, resolve_artifact_root
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    extract_pending_items,
    extracted_tables_path,
    mentions_root,
)
from cdt.ingest import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_BUCKET,
    DEFAULT_S3_PREFIX,
    IngestConfig,
    IngestRunResult,
    default_failure_file,
    run_ingest_pipeline,
)
from cdt.ingest import DEFAULT_BATCH_SIZE as DEFAULT_INGEST_BATCH_SIZE
from cdt.itemizer import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    itemize_pending_documents,
    items_root,
)
from cdt.matcher import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    debt_instruments_root,
    match_pending_mentions,
    mention_cluster_edges_root,
)
from cdt.matcher.core import MATCHER_SCHEMA_VERSION, apply_lineage_inference_pass
from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    artifact_tree_digest,
    coerce_dataset_text,
    count_table_rows,
    delete_artifact,
    join_artifact_path,
    list_artifacts,
    read_dataset,
    read_json_artifact,
    read_text_artifact,
    write_json_artifact,
    write_table,
)

FINAL_OUTPUT_TABLES: dict[str, Callable[[str | Path | None], str]] = {
    "items": items_root,
    "debt-instruments": debt_instruments_root,
    "debt-instrument-mentions": mentions_root,
    "mention-cluster-edges": mention_cluster_edges_root,
}

ALL_TIME_START_DATE = date(1994, 1, 1)
# Daily mode re-scans this many days back (ending yesterday) so late-arriving
# or since-repaired scraper manifests are picked up instead of falling outside
# a moved-on one-day window forever (#90).
DAILY_LOOKBACK_DAYS = 5
DEFAULT_STAGE_BATCH_SIZE = 100
PIPELINE_MODES = ("daily", "historical")
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single CDT pipeline invocation."""

    mode: str
    cik_file: ArtifactPath
    bucket: str = DEFAULT_BUCKET
    start_date: date | None = None
    end_date: date | None = None
    data_dir: Path | None = None
    artifact_root: ArtifactPath | None = None
    final_database_root: ArtifactPath | None = None
    force: bool = False
    download: bool = False
    failure_file: ArtifactPath | None = None
    aws_profile: str = DEFAULT_AWS_PROFILE
    s3_prefix: str = DEFAULT_S3_PREFIX
    ingest_batch_size: int = DEFAULT_INGEST_BATCH_SIZE
    itemize_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    classify_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    extract_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    match_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    item_numbers: tuple[str, ...] = POTENTIALLY_RELEVANT_ITEM_NUMBERS
    classifier_model_dir: Path | None = None
    extractor_model: str = DEFAULT_MODEL
    extractor_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    extractor_max_attempts: int = DEFAULT_MAX_ATTEMPTS
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN


@dataclass(frozen=True)
class PipelineRunResult:
    """Summary of one end-to-end pipeline run."""

    mode: str
    start_date: date
    end_date: date
    ingest: IngestRunResult
    itemized_rows: int
    classified_rows: int
    extracted_rows: int
    matched_rows: int
    debt_instrument_rows: int
    classifier_model_dir: Path
    artifact_root: str
    extractor_run_path: str


class PipelineOrchestrator:
    """Run the full CDT pipeline with structured logging."""

    def __init__(self: Self, config: PipelineConfig) -> None:
        """Initialize the orchestrator."""
        self.config = config
        self.logger = get_logger(type(self).__name__)

    def _log_banner(self: Self, message: str) -> None:
        self.logger.info("=" * 60)
        self.logger.info(message)
        self.logger.info("=" * 60)

    def _log_config(self: Self, resolved_start: date, resolved_end: date) -> None:
        config_values = asdict(self.config)
        config_values["start_date"] = resolved_start
        config_values["end_date"] = resolved_end
        for key, value in config_values.items():
            self.logger.info("%s: %s", key, value)

    def _log_stage_start(self: Self, stage_name: str, **details: object) -> None:
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        self.logger.info(
            "Starting stage: %s%s",
            stage_name,
            f" | {detail_text}" if detail_text else "",
        )

    def _log_stage_complete(self: Self, stage_name: str, **details: object) -> None:
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        self.logger.info(
            "Completed stage: %s%s",
            stage_name,
            f" | {detail_text}" if detail_text else "",
        )

    def _setup(self: Self) -> tuple[date, date, set[str], str]:
        """Resolve dates, CIKs, and the artifact root and emit the run banner."""
        resolved_start, resolved_end = resolve_mode_dates(
            self.config.mode,
            self.config.start_date,
            self.config.end_date,
        )
        ciks = read_cik_file(self.config.cik_file)
        resolved_artifact_root = resolve_artifact_root(
            self.config.artifact_root,
            data_dir=self.config.data_dir,
        )
        self._log_banner(
            f"Starting pipeline | mode={self.config.mode} | cik_file={self.config.cik_file}"
        )
        self._log_config(resolved_start, resolved_end)
        return resolved_start, resolved_end, ciks, resolved_artifact_root

    def _renew(self: Self, renew: Callable[[], None] | None) -> None:
        """Extend the caller's writer lease at a stage boundary.

        Historical runs outlast the lease TTL by hours; renewing between stages
        keeps the run from being stolen mid-write, and the hook raises
        LeaseLostError if it already was (#89).
        """
        if renew is not None:
            renew()

    def _ingest_itemize_classify(
        self: Self,
        resolved_start: date,
        resolved_end: date,
        ciks: set[str],
        resolved_artifact_root: str,
        renew: Callable[[], None] | None = None,
    ) -> tuple[IngestRunResult, pd.DataFrame, pd.DataFrame]:
        """Run ingest → itemize → classify and return their results."""
        self._log_stage_start(
            "ingest",
            batch_size=self.config.ingest_batch_size,
            download=self.config.download,
        )
        ingest_table, ingest_result = run_ingest_pipeline(
            IngestConfig(
                mode=self.config.mode,
                bucket=self.config.bucket,
                cik_file=Path(str(self.config.cik_file)),
                start_date=resolved_start,
                end_date=resolved_end,
                data_dir=self.config.data_dir,
                output_root=resolved_artifact_root,
                force=self.config.force,
                batch_size=self.config.ingest_batch_size,
                download=self.config.download,
                failure_file=self.config.failure_file
                or default_failure_file(
                    resolved_artifact_root,
                    data_dir=self.config.data_dir,
                ),
                aws_profile=self.config.aws_profile,
                s3_prefix=self.config.s3_prefix,
            ),
            ciks=ciks,
        )
        del ingest_table
        self._log_stage_complete(
            "ingest",
            rows=ingest_result.total_rows,
            candidates=ingest_result.candidates_seen,
            partitions=len(ingest_result.document_partitions),
            failures=ingest_result.failures,
        )
        self._renew(renew)

        self._log_stage_start(
            "itemize",
            batch_size=self.config.itemize_batch_size,
        )
        items = itemize_pending_documents(
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.itemize_batch_size,
            force=self.config.force,
            item_numbers=self.config.item_numbers,
            renew=renew,
        )
        self._log_stage_complete("itemize", rows=len(items))
        self._renew(renew)

        self._log_stage_start(
            "classify",
            batch_size=self.config.classify_batch_size,
        )
        classified = classify_pending_items(
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            model_dir=self.config.classifier_model_dir,
            batch_size=self.config.classify_batch_size,
            force=self.config.force,
            renew=renew,
        )
        self._log_stage_complete("classify", rows=len(classified))
        return ingest_result, items, classified

    def run_prepare(self: Self, renew: Callable[[], None] | None = None) -> str:
        """Run only ingest → itemize → classify; return the artifact root.

        Used by the deployed ``daily`` orchestrator, which defers the expensive
        extract stage to the asynchronous batch poller.
        """
        resolved_start, resolved_end, ciks, resolved_artifact_root = self._setup()
        self._ingest_itemize_classify(
            resolved_start, resolved_end, ciks, resolved_artifact_root, renew
        )
        return resolved_artifact_root

    def run(self: Self, renew: Callable[[], None] | None = None) -> PipelineRunResult:
        """Execute the full CDT pipeline."""
        resolved_start, resolved_end, ciks, resolved_artifact_root = self._setup()
        start_time = datetime.now()
        ingest_result, items, classified = self._ingest_itemize_classify(
            resolved_start, resolved_end, ciks, resolved_artifact_root, renew
        )
        self._renew(renew)

        self._log_stage_start(
            "extract",
            batch_size=self.config.extract_batch_size,
            model=self.config.extractor_model,
        )
        extracted = extract_pending_items(
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.extract_batch_size,
            force=self.config.force,
            model=self.config.extractor_model,
            reasoning_effort=self.config.extractor_reasoning_effort,
            max_attempts=self.config.extractor_max_attempts,
        )
        self._log_stage_complete("extract", rows=len(extracted))
        self._renew(renew)

        self._log_stage_start(
            "match",
            batch_size=self.config.match_batch_size,
        )
        matched = match_pending_mentions(
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.match_batch_size,
            force=self.config.force,
            strong_match_threshold=self.config.strong_match_threshold,
            loose_match_threshold=self.config.loose_match_threshold,
            ambiguity_margin=self.config.ambiguity_margin,
            renew=renew,
        )
        self._log_stage_complete(
            "match",
            edge_rows=len(matched["debt_instrument_mentions"]),
            debt_instruments=len(matched["debt_instrument"]),
        )
        member_edge_rows = matched["debt_instrument_mentions"]
        matched_mentions = (
            int((member_edge_rows["edge_type"] == "member").sum())
            if "edge_type" in member_edge_rows
            else len(member_edge_rows)
        )

        result = PipelineRunResult(
            mode=self.config.mode,
            start_date=resolved_start,
            end_date=resolved_end,
            ingest=ingest_result,
            itemized_rows=len(items),
            classified_rows=len(classified),
            extracted_rows=len(extracted),
            matched_rows=matched_mentions,
            debt_instrument_rows=len(matched["debt_instrument"]),
            classifier_model_dir=self.config.classifier_model_dir
            or default_model_dir(self.config.data_dir),
            artifact_root=resolved_artifact_root,
            extractor_run_path=extracted_tables_path(
                resolved_artifact_root,
                data_dir=self.config.data_dir,
            ),
        )
        # One shared tail rather than a second copy: see finalize_after_match.
        finalize_after_match(
            matched["debt_instrument"],
            artifact_root=resolved_artifact_root,
            final_database_root=self.config.final_database_root,
            data_dir=self.config.data_dir,
            force=self.config.force,
            renew=renew,
            log_stage_start=self._log_stage_start,
            log_stage_complete=self._log_stage_complete,
        )
        elapsed = datetime.now() - start_time
        self._log_banner(f"Pipeline completed successfully in {elapsed}")
        return result


def run_pipeline(
    config: PipelineConfig, *, renew: Callable[[], None] | None = None
) -> PipelineRunResult:
    """Run the full CDT pipeline for the provided config."""
    return PipelineOrchestrator(config).run(renew)


def run_prepare_stages(
    config: PipelineConfig, *, renew: Callable[[], None] | None = None
) -> str:
    """Run ingest → itemize → classify for a config; return the artifact root."""
    return PipelineOrchestrator(config).run_prepare(renew)


def run_match_and_finalize(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    batch_size: int = DEFAULT_STAGE_BATCH_SIZE,
    force: bool = False,
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    renew: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Run match on existing mentions and rewrite final snapshots.

    Both stages are deterministic and idempotent, so this is safe to run
    repeatedly: the daily orchestrator calls it to keep snapshots fresh while a
    batch extract job is still in flight, and the poller calls it when a job
    completes. ``renew`` extends the caller's writer lease per matched shard
    and before the snapshot rewrite — this is the longest phase, and it must
    not keep publishing on a lease another run has stolen (#89).
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    tables = match_pending_mentions(
        artifact_root=resolved_root,
        data_dir=data_dir,
        batch_size=batch_size,
        force=force,
        strong_match_threshold=strong_match_threshold,
        loose_match_threshold=loose_match_threshold,
        ambiguity_margin=ambiguity_margin,
        renew=renew,
    )
    return finalize_after_match(
        tables["debt_instrument"],
        artifact_root=resolved_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force=force,
        renew=renew,
    )


def read_cik_file(path: ArtifactPath) -> set[str]:
    """Read a one-CIK-per-line file from local storage or S3."""
    return {
        line.strip() for line in read_text_artifact(path).splitlines() if line.strip()
    }


def resolve_mode_dates(
    mode: str,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    """Resolve mode-specific dates for ingest-like commands.

    Daily defaults to a rolling lookback window ending yesterday, not a single
    day: a manifest the scraper writes (or repairs) after CDT's morning pass
    would otherwise never be scanned again — a permanent, unobservable gap
    (#90). Ingest dedups by accession, so the re-scan costs only LIST/GET
    requests, and the fingerprint registries propagate late merges downstream.
    """
    if mode not in PIPELINE_MODES:
        msg = f"unsupported mode {mode!r}"
        raise ValueError(msg)
    if mode == "historical":
        return start_date or ALL_TIME_START_DATE, end_date or date.today()
    if start_date is None and end_date is None:
        today = date.today()
        yesterday = today.fromordinal(today.toordinal() - 1)
        lookback_start = today.fromordinal(today.toordinal() - DAILY_LOOKBACK_DAYS)
        return lookback_start, yesterday
    if start_date is None:
        msg = "--start-date is required when --end-date is provided"
        raise ValueError(msg)
    if end_date is None:
        msg = "--end-date is required when --start-date is provided"
        raise ValueError(msg)
    return start_date, end_date


# A table shrinking below this fraction of its previously published row count
# blocks the publish (unless forced): the likeliest causes are a bug or a
# half-built artifact root, not a legitimate mass deletion of filings.
FINAL_SNAPSHOT_GUARD_RATIO = 0.5


#: Pointer key holding the digest of the source partitions a generation was
#: built from. Absent on any pointer written before the gate existed, which is
#: read as "unknown" and publishes once to record one.
PUBLISH_SOURCE_DIGEST_KEY = "source_digest"


def publish_source_digest(
    artifact_root: ArtifactPath, *, data_dir: Path | None = None
) -> str:
    """Digest every partition a publish would read, from one LIST per root.

    This mirrors what ``pending_source_partitions`` does to decide a partition
    needs reprocessing (#62) — compare a stored source version — but through
    ``artifact_content_versions``, which is byte-based on both backends. The
    registries' mtime-based version would be wrong here: the matcher rewrites
    every shard on every run, almost always to identical content, so an
    mtime-based digest would differ every time and the gate would never fire.

    On S3 this is one LIST per root and the ETag comes along with it.
    """
    return artifact_tree_digest(
        _publish_source_roots(artifact_root, data_dir=data_dir), suffix=".parquet"
    )


def _publish_source_roots(
    artifact_root: ArtifactPath, *, data_dir: Path | None
) -> list[str]:
    """Every dataset root a publish reads, deduplicated, in table order."""
    roots: list[str] = []
    for entry in FINAL_OUTPUT_TABLES.values():
        # One dataset per published table here. On dev, ``items`` unions the
        # itemizer's root with the 6-K snippets root (#172), so both shapes are
        # accepted: a digest that silently stopped covering a root would skip
        # publishes it should not, and that merge would not conflict.
        dataset_root_fns = entry if isinstance(entry, tuple) else (entry,)
        for dataset_root_fn in dataset_root_fns:
            root = dataset_root_fn(artifact_root, data_dir=data_dir)
            if root not in roots:
                roots.append(root)
    return roots


def publish_would_republish_nothing(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force: bool = False,
) -> bool:
    """Return whether a publish would re-read the whole corpus to change nothing.

    The publish is the most expensive thing in the pipeline and it is bounded by
    request count, not bytes: measured in production, publishing a delta of 14
    documents took 25 minutes and 21,214 sequential GETs at ~70 ms each. It pays
    that regardless of whether the run produced anything, and it pays it twice
    per batch cycle, because both ``run_batch_backend`` and ``run_poll``
    finalize.

    The question is whether anything the publish *reads* has changed since the
    generation the pointer names, so that is what this asks — a digest of the
    five source dataset roots against the one recorded when that generation was
    written. An earlier version of this gate asked instead whether match had
    produced instruments, which is a different question with a much narrower
    answer: ``match_pending_mentions`` rewrites and returns every shard's full
    instrument table rather than a delta, so that frame is empty only on a
    corpus that has never produced a single instrument. It would never have
    fired on the 542-instrument root #110 was measured against, and it could not
    see ``items`` at all — which itemize and the 6-K triage write *before* match
    runs, and which is one of the four published tables.

    Three things stop this from being a way to never publish. ``force``
    overrides it. A pointer with no recorded digest — one written before this
    existed, or none at all — publishes, which records one for next time. And a
    final database root missing any of its four ``latest.parquet`` objects
    publishes regardless: skipping there would mean a freshly pointed output
    root stayed empty until someone happened to pass ``--force``. That check is
    four HEAD requests against objects the publish would write anyway.

    Note that a run which crashed between writing a dataset and publishing it no
    longer needs ``--force`` to recover: the datasets moved, so the digest moved,
    so the next run publishes on its own. That matters because ``force`` is the
    pipeline-wide flag, and passing it also disables ``_guard_against_shrinkage``
    — the one protection a post-crash republish most wants.
    """
    if force:
        return False
    if final_database_root is None:
        # Nothing to publish to: ``write_final_output_tables`` returns before it
        # reads anything, so the answer is the same and this way it costs no
        # listing to find out.
        return True
    pointer_path = final_pointer_path(artifact_root)
    if not artifact_exists(pointer_path):
        return False
    pointer = read_json_artifact(pointer_path)
    recorded = (
        pointer.get(PUBLISH_SOURCE_DIGEST_KEY) if isinstance(pointer, dict) else None
    )
    if not recorded:
        LOGGER.info(
            "Publishing: %s records no source digest, so whether the sources "
            "have moved since it was written is unknown.",
            pointer_path,
        )
        return False
    if recorded != publish_source_digest(artifact_root, data_dir=data_dir):
        return False
    if not all(
        artifact_exists(
            join_artifact_path(str(final_database_root), table_name, "latest.parquet")
        )
        for table_name in FINAL_OUTPUT_TABLES
    ):
        LOGGER.info(
            "Publishing despite unchanged sources: %s has no complete published "
            "generation yet.",
            final_database_root,
        )
        return False
    LOGGER.info(
        "Skipping final publish: no partition under the published datasets has "
        "changed since generation %s, so the published snapshot is already "
        "current. Use --force to publish anyway.",
        pointer.get("run_id", "unknown"),
    )
    return True


def _ignore_stage(*args: object, **kwargs: object) -> None:
    """Swallow a stage log line: only the orchestrator reports stages."""


def finalize_after_match(
    matched_instruments: pd.DataFrame,
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force: bool = False,
    renew: Callable[[], None] | None = None,
    log_stage_start: Callable[..., None] = _ignore_stage,
    log_stage_complete: Callable[..., None] = _ignore_stage,
) -> dict[str, str]:
    """Run the lineage post-pass and the publish: the tail every path shares.

    One copy rather than two. #170 is what the second copy costs: the lineage
    pass was wired into one of three entry points and production published 537
    of 542 instruments as lineage heads, because `cdt pipeline` and the live
    extractor backend silently skipped it. Both paths reach the same three
    steps here, so a fix to any of them cannot land on only one again.

    Amendment lineage spans filings, so it can only be derived once every shard
    has matched, which makes it a post-pass over the whole corpus. It re-derives
    every pointer it ever inferred (#204) and renews the lease as it goes; it is
    skipped only when match produced nothing at all, since it would then read
    three empty datasets to write none.

    ``renew`` extends the caller's writer lease before each long step. This is
    the longest phase of a run, and it must not keep publishing on a lease
    another run has stolen (#89).
    """
    if not matched_instruments.empty:
        if renew is not None:
            renew()
        log_stage_start("infer-lineage")
        lineage_stats = apply_lineage_inference_pass(
            artifact_root, data_dir=data_dir, renew=renew
        )
        log_stage_complete("infer-lineage", **lineage_stats)
    log_stage_start("finalize", output_root=final_database_root)
    if publish_would_republish_nothing(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force=force,
    ):
        log_stage_complete("finalize", tables=0, output_root=final_database_root)
        return {}
    if renew is not None:
        renew()
    final_outputs = write_final_output_tables(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force=force,
    )
    log_stage_complete(
        "finalize", tables=len(final_outputs), output_root=final_database_root
    )
    return final_outputs


def final_snapshots_root(artifact_root: ArtifactPath) -> str:
    """Return the root for consistent snapshot generations and their pointer.

    Deliberately under the artifact root, not the final database root: the
    final database prefix is a parquet-only contract surface for downstream
    consumers, so the control metadata (``latest.json``) and the immutable
    generation copies live with the pipeline's other artifacts instead.
    """
    return join_artifact_path(str(artifact_root), "final-snapshots")


def final_pointer_path(artifact_root: ArtifactPath) -> str:
    """Return the path of the atomic latest.json snapshot pointer."""
    return join_artifact_path(final_snapshots_root(artifact_root), "latest.json")


def write_final_output_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Publish the final tables: guarded, with an atomic generation pointer (#91).

    Writing four independent ``<table>/latest.parquet`` objects in a loop can
    never be consistent as a set: a consumer polling mid-loop reads mixed
    generations (mentions referencing instruments that did not exist yet), a
    crash between writes leaves that state published permanently, and an
    accidentally empty dataset silently clobbers a good snapshot with zero rows.

    So every publish first lands whole under an immutable
    ``final-snapshots/snapshot=<run_id>/`` prefix beneath the *artifact* root,
    and a single ``latest.json`` pointer there — the object consumers wanting
    a consistent four-table generation should resolve — is replaced as the
    last, atomic step, carrying the run id, schema version, and per-table row
    counts. Only then are the per-table ``<table>/latest.parquet`` objects
    under the final database root refreshed: that prefix stays parquet-only
    (its contract), with each object individually atomic but the set not
    consistent mid-publish. Unless ``force``, the publish refuses when a
    previously non-empty table would become empty or shrink below
    FINAL_SNAPSHOT_GUARD_RATIO of its prior row count. Generations other than
    the current and prior one are pruned.
    """
    if final_database_root is None:
        return {}

    pointer_path = final_pointer_path(artifact_root)
    previous: dict[str, object] = {}
    if artifact_exists(pointer_path):
        payload = read_json_artifact(pointer_path)
        if isinstance(payload, dict):
            previous = payload

    # Placeholder text is nulled once, up front, so the immutable snapshot and
    # the parquet-only database root publish the same normalized values.
    tables = {
        table_name: normalize_snapshot_text(
            read_dataset(dataset_root_fn(artifact_root, data_dir=data_dir))
        )
        for table_name, dataset_root_fn in FINAL_OUTPUT_TABLES.items()
    }
    # Guard against what is actually published, not the pointer: the pointer
    # lives with the artifact root, so a half-built or freshly-pointed artifact
    # root has no pointer — exactly the case that must not clobber a good
    # database. Footer metadata gives the counts without decoding columns.
    previous_counts = {
        table_name: rows
        for table_name in FINAL_OUTPUT_TABLES
        if (
            rows := count_table_rows(
                join_artifact_path(
                    str(final_database_root), table_name, "latest.parquet"
                )
            )
        )
        is not None
    }
    _guard_against_shrinkage(tables, previous_counts, force=force)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots_root = final_snapshots_root(artifact_root)
    snapshot_prefix = join_artifact_path(snapshots_root, f"snapshot={run_id}")
    written_paths: dict[str, str] = {}
    pointer_tables: dict[str, object] = {}
    for table_name, table in tables.items():
        snapshot_path = join_artifact_path(snapshot_prefix, f"{table_name}.parquet")
        written_paths[table_name] = write_table(snapshot_path, table)
        pointer_tables[table_name] = {
            "path": written_paths[table_name],
            "rows": len(table),
        }
    write_json_artifact(
        pointer_path,
        {
            "run_id": run_id,
            "written_at": datetime.now(UTC).isoformat(),
            "schema_version": MATCHER_SCHEMA_VERSION,
            "tables": pointer_tables,
            # What this generation was built from, so the next run can tell
            # whether re-reading the corpus would change anything. Recorded
            # after the reads above and before the pointer flips, and the
            # publish writes nothing under the source roots, so it describes
            # exactly the bytes these tables came from.
            PUBLISH_SOURCE_DIGEST_KEY: publish_source_digest(
                artifact_root, data_dir=data_dir
            ),
        },
    )

    # The parquet-only contract surface, refreshed after the pointer so the
    # consistent generation is always resolvable first.
    for table_name, table in tables.items():
        write_table(
            join_artifact_path(str(final_database_root), table_name, "latest.parquet"),
            table,
        )

    _prune_old_snapshots(
        snapshots_root,
        keep_run_ids={run_id, str(previous.get("run_id", ""))},
    )
    return written_paths


def normalize_snapshot_text(table: pd.DataFrame) -> pd.DataFrame:
    """Return one snapshot table with placeholder text values replaced by nulls.

    Partitions written before a text column existed, or by a stage that
    stringified a missing value, carry literal text such as ``nan``. Dashboard
    consumers read these snapshots directly, so they are normalized on the way
    out instead of rendering the placeholder.
    """
    if table.empty:
        return table
    normalized = table.copy()
    for column in normalized.columns:
        if normalized[column].dtype != object:
            continue
        normalized[column] = normalized[column].map(normalize_snapshot_cell)
    if "cik" in normalized.columns:
        # Partitions written before #153 carry unpadded CIKs; published
        # snapshots always carry SEC's canonical zero-padded form.
        normalized["cik"] = normalized["cik"].map(
            lambda value: normalize_cik(value) if isinstance(value, str) else value
        )
    return normalized


def normalize_snapshot_cell(value: object) -> object:
    """Null one placeholder string, leaving non-text values at their own type.

    An object column can hold booleans when older partitions predate the column,
    so only text cells are coerced. Mapping everything through the text helper
    would publish `True` as the string `"True"`.
    """
    if isinstance(value, str):
        return coerce_dataset_text(value)
    return value


def _guard_against_shrinkage(
    tables: dict[str, pd.DataFrame],
    previous_counts: dict[str, int],
    *,
    force: bool,
) -> None:
    """Refuse to publish a snapshot that looks like data loss, unless forced."""
    regressions: list[str] = []
    for table_name, table in tables.items():
        prior_rows = previous_counts.get(table_name, 0)
        if prior_rows <= 0:
            continue
        if len(table) < prior_rows * FINAL_SNAPSHOT_GUARD_RATIO:
            regressions.append(f"{table_name}: {prior_rows} -> {len(table)} rows")
    if not regressions:
        return
    if force:
        LOGGER.warning(
            "Publishing snapshot despite row-count regressions (forced): %s",
            "; ".join(regressions),
        )
        return
    msg = (
        "Refusing to publish a final snapshot with large row-count regressions "
        f"({'; '.join(regressions)}). This usually means a bug or a half-built "
        "artifact root; re-run with force=True to publish anyway."
    )
    raise ValueError(msg)


def _prune_old_snapshots(snapshots_root: str, *, keep_run_ids: set[str]) -> None:
    """Delete snapshot generations other than the current and prior one.

    Two generations stay readable so a consumer that resolved the previous
    pointer moments ago can still finish reading it.
    """
    keep_prefixes = tuple(f"snapshot={run_id}/" for run_id in keep_run_ids if run_id)
    for path in list_artifacts(snapshots_root, suffix=".parquet"):
        if not any(prefix in path for prefix in keep_prefixes):
            delete_artifact(path)
