"""File-native orchestration for the full CDT pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Self

import pandas as pd

from cdt.classifier import classify_pending_items, default_model_dir
from cdt.datasets import resolve_artifact_root
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
from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    coerce_dataset_text,
    join_artifact_path,
    read_dataset,
    read_text_artifact,
    write_table,
)

FINAL_OUTPUT_TABLES: dict[str, Callable[[str | Path | None], str]] = {
    "items": items_root,
    "debt-instruments": debt_instruments_root,
    "debt-instrument-mentions": mentions_root,
    "mention-cluster-edges": mention_cluster_edges_root,
}

ALL_TIME_START_DATE = date(1994, 1, 1)
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

    def run(self: Self) -> PipelineRunResult:
        """Execute the full CDT pipeline."""
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
        start_time = datetime.now()

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
        )
        self._log_stage_complete("itemize", rows=len(items))

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
        )
        self._log_stage_complete("classify", rows=len(classified))

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
        self._log_stage_start(
            "finalize",
            output_root=self.config.final_database_root,
        )
        final_outputs = write_final_output_tables(
            artifact_root=resolved_artifact_root,
            final_database_root=self.config.final_database_root,
            data_dir=self.config.data_dir,
        )
        self._log_stage_complete(
            "finalize",
            tables=len(final_outputs),
            output_root=self.config.final_database_root,
        )
        elapsed = datetime.now() - start_time
        self._log_banner(f"Pipeline completed successfully in {elapsed}")
        return result


def run_pipeline(config: PipelineConfig) -> PipelineRunResult:
    """Run the full CDT pipeline for the provided config."""
    return PipelineOrchestrator(config).run()


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
    """Resolve mode-specific dates for ingest-like commands."""
    if mode not in PIPELINE_MODES:
        msg = f"unsupported mode {mode!r}"
        raise ValueError(msg)
    if mode == "historical":
        return start_date or ALL_TIME_START_DATE, end_date or date.today()
    if start_date is None and end_date is None:
        yesterday = date.today().fromordinal(date.today().toordinal() - 1)
        return yesterday, yesterday
    if start_date is None:
        msg = "--start-date is required when --end-date is provided"
        raise ValueError(msg)
    if end_date is None:
        msg = "--end-date is required when --start-date is provided"
        raise ValueError(msg)
    return start_date, end_date


def write_final_output_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
) -> dict[str, str]:
    """Materialize final latest parquet snapshots for downstream database loads."""
    if final_database_root is None:
        return {}

    written_paths: dict[str, str] = {}
    for table_name, dataset_root_fn in FINAL_OUTPUT_TABLES.items():
        table = read_dataset(dataset_root_fn(artifact_root, data_dir=data_dir))
        output_path = join_artifact_path(
            final_database_root, table_name, "latest.parquet"
        )
        written_paths[table_name] = write_table(
            output_path, normalize_snapshot_text(table)
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
        normalized[column] = normalized[column].map(coerce_dataset_text)
    return normalized
