"""File-native orchestration for the full CDT pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Self

import pandas as pd

from cdt.classifier.core import default_model_dir
from cdt.classifier.eightk import classify_pending_items
from cdt.classifier.sixk import DEFAULT_CONCURRENCY as SIXK_DEFAULT_CONCURRENCY
from cdt.classifier.sixk import triage_pending_documents
from cdt.datasets import (
    GENRE_6K,
    GENRE_8K,
    failure_registry_path,
    resolve_artifact_root,
)
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    extract_pending_items,
    extracted_tables_path,
)
from cdt.ingest.core import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_BUCKET,
    DEFAULT_S3_PREFIX,
    SIXK_DOCUMENT_DATASET_NAME,
    SIXK_FORM_TYPES,
    IngestConfig,
    IngestRunResult,
)
from cdt.ingest.core import DEFAULT_BATCH_SIZE as DEFAULT_INGEST_BATCH_SIZE
from cdt.ingest.eightk import acquire_eightk_documents
from cdt.ingest.sixk import acquire_scraped_sixk_documents
from cdt.matcher import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    match_pending_mentions,
)
from cdt.publish import finalize_after_match
from cdt.segmenter.eightk import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    itemize_pending_documents,
)
from cdt.shared import get_logger
from cdt.storage.objects import ArtifactPath, read_text_artifact

#: Genres the CLI entry points prepare unless `--genres` narrows them.
DEFAULT_GENRES: tuple[str, ...] = (GENRE_8K, GENRE_6K)
GENRES = DEFAULT_GENRES


ALL_TIME_START_DATE = date(1994, 1, 1)
# Daily mode re-scans this many days back, ending yesterday, so late or
# repaired scraper manifests are still picked up.
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
    #: Which genres to prepare. 8-K only when built in code, because the 6-K
    #: chain scrapes and calls a paid model; the CLIs pass DEFAULT_GENRES.
    genres: tuple[str, ...] = (GENRE_8K,)
    #: CIKs for the 6-K genre; None means `cik_file`.
    sixk_cik_file: ArtifactPath | None = None
    sixk_form_types: tuple[str, ...] = SIXK_FORM_TYPES
    sixk_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    sixk_concurrency: int = SIXK_DEFAULT_CONCURRENCY


@dataclass(frozen=True)
class PipelineRunResult:
    """Summary of one end-to-end pipeline run."""

    mode: str
    start_date: date
    end_date: date
    #: None when the run did not prepare that genre; a genre that ran and
    #: found nothing has a result with zeroes.
    ingest: IngestRunResult | None
    itemized_rows: int
    classified_rows: int
    extracted_rows: int
    matched_rows: int
    debt_instrument_rows: int
    classifier_model_dir: Path
    artifact_root: str
    extractor_run_path: str
    genres: tuple[str, ...] = DEFAULT_GENRES
    sixk_ingest: IngestRunResult | None = None
    sixk_snippet_rows: int = 0


@dataclass
class _PrepareOutcome:
    """What the prepare phases produced, per genre.

    An ``*_ingest`` of None means that genre's chain did not run.
    """

    ingest: IngestRunResult | None = None
    items: pd.DataFrame = field(default_factory=pd.DataFrame)
    classified: pd.DataFrame = field(default_factory=pd.DataFrame)
    sixk_ingest: IngestRunResult | None = None
    snippets: pd.DataFrame = field(default_factory=pd.DataFrame)


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
        # A config built in code skips the CLI's parsing; validate here.
        normalize_genres(self.config.genres)
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
        """Extend the caller's writer lease at a stage boundary; no-op if None.

        The hook raises LeaseLostError if the lease was already stolen.
        """
        if renew is not None:
            renew()

    def _prepare_genres(
        self: Self,
        resolved_start: date,
        resolved_end: date,
        ciks: set[str],
        resolved_artifact_root: str,
        renew: Callable[[], None] | None = None,
    ) -> _PrepareOutcome:
        """Prepare every genre this run asked for, in genre order.

        Each genre writes its own documents and classification datasets, so a
        failing genre cannot corrupt the other's. The chains run one after the
        other: a failure in the 8-K chain stops the run before 6-K starts.
        """
        outcome = _PrepareOutcome()
        if GENRE_8K in self.config.genres:
            outcome.ingest, outcome.items, outcome.classified = (
                self._ingest_itemize_classify(
                    resolved_start,
                    resolved_end,
                    ciks,
                    resolved_artifact_root,
                    renew,
                )
            )
            self._renew(renew)
        else:
            self.logger.info("Skipping the 8-K chain: genres=%s", self.config.genres)
        if GENRE_6K in self.config.genres:
            outcome.sixk_ingest, outcome.snippets = self._ingest_and_triage_sixk(
                resolved_start,
                resolved_end,
                resolved_artifact_root,
                renew,
            )
        else:
            self.logger.info("Skipping the 6-K chain: genres=%s", self.config.genres)
        return outcome

    def _ingest_and_triage_sixk(
        self: Self,
        resolved_start: date,
        resolved_end: date,
        resolved_artifact_root: str,
        renew: Callable[[], None] | None = None,
    ) -> tuple[IngestRunResult, pd.DataFrame]:
        """Run the 6-K chain: acquire filings, then triage them into snippets."""
        sixk_ciks = read_cik_file(self.config.sixk_cik_file or self.config.cik_file)
        self._log_stage_start(
            "ingest-sixk",
            batch_size=self.config.ingest_batch_size,
            forms=",".join(self.config.sixk_form_types),
            ciks=len(sixk_ciks),
        )
        _, sixk_ingest = acquire_scraped_sixk_documents(
            IngestConfig(
                mode=self.config.mode,
                bucket=self.config.bucket,
                cik_file=Path(str(self.config.sixk_cik_file or self.config.cik_file)),
                start_date=resolved_start,
                end_date=resolved_end,
                data_dir=self.config.data_dir,
                output_root=resolved_artifact_root,
                force=self.config.force,
                batch_size=self.config.ingest_batch_size,
                # Never `download`: a 6-K row points at the mirrored
                # submission; inlining bodies makes every read pay for them.
                failure_file=self.config.failure_file
                or failure_registry_path(
                    "ingest",
                    artifact_root=resolved_artifact_root,
                    data_dir=self.config.data_dir,
                ),
                aws_profile=self.config.aws_profile,
                s3_prefix=self.config.s3_prefix,
                form_types=self.config.sixk_form_types,
                dataset_name=SIXK_DOCUMENT_DATASET_NAME,
            ),
            ciks=sixk_ciks,
        )
        self._log_stage_complete(
            "ingest-sixk",
            rows=sixk_ingest.total_rows,
            candidates=sixk_ingest.candidates_seen,
            partitions=len(sixk_ingest.document_partitions),
            failures=sixk_ingest.failures,
        )
        self._renew(renew)

        self._log_stage_start(
            "sixk",
            batch_size=self.config.sixk_batch_size,
            concurrency=self.config.sixk_concurrency,
        )
        snippets = triage_pending_documents(
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.sixk_batch_size,
            force=self.config.force,
            concurrency=self.config.sixk_concurrency,
            renew=renew,
        )
        self._log_stage_complete("sixk", rows=len(snippets))
        return sixk_ingest, snippets

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
        ingest_table, ingest_result = acquire_eightk_documents(
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
                or failure_registry_path(
                    "ingest",
                    artifact_root=resolved_artifact_root,
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
        """Run only the prepare stages of each genre; return the artifact root."""
        resolved_start, resolved_end, ciks, resolved_artifact_root = self._setup()
        self._prepare_genres(
            resolved_start, resolved_end, ciks, resolved_artifact_root, renew
        )
        return resolved_artifact_root

    def run(self: Self, renew: Callable[[], None] | None = None) -> PipelineRunResult:
        """Execute the full CDT pipeline."""
        resolved_start, resolved_end, ciks, resolved_artifact_root = self._setup()
        start_time = datetime.now()
        prepared = self._prepare_genres(
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
            ingest=prepared.ingest,
            itemized_rows=len(prepared.items),
            classified_rows=len(prepared.classified),
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
            genres=self.config.genres,
            sixk_ingest=prepared.sixk_ingest,
            sixk_snippet_rows=len(prepared.snippets),
        )
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
    """Run only the prepare stages for a config; return the artifact root."""
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
    """Run match on existing mentions, then finalize; idempotent.

    ``renew`` extends the caller's writer lease per matched shard and before
    the publish, so a stolen lease cannot keep publishing.

    Returns:
        Published table name -> snapshot path; empty when nothing was published.
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


def normalize_genres(values: str | Sequence[str]) -> tuple[str, ...]:
    """Parse a genre selection into GENRES order, case-insensitively, deduplicated.

    Accepts a comma-separated string or a sequence.

    Raises:
        ValueError: If the selection is empty or names an unknown genre.
    """
    if isinstance(values, str):
        requested = [value.strip() for value in values.split(",")]
    else:
        requested = [str(value).strip() for value in values]
    selected = {value.upper() for value in requested if value}
    if not selected:
        msg = f"no genres selected; expected one or more of {', '.join(GENRES)}"
        raise ValueError(msg)
    unknown = sorted(selected - {genre.upper() for genre in GENRES})
    if unknown:
        msg = (
            f"unknown genre(s) {', '.join(unknown)}; "
            f"expected one or more of {', '.join(GENRES)}"
        )
        raise ValueError(msg)
    return tuple(genre for genre in GENRES if genre.upper() in selected)


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
    """Resolve a mode's ``(start, end)`` dates.

    Historical fills a missing start with ALL_TIME_START_DATE and a missing end
    with today. Daily with neither date gives the DAILY_LOOKBACK_DAYS window
    ending yesterday; daily with only one of them is an error.

    Raises:
        ValueError: On an unknown mode, or daily with only one date given.
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
