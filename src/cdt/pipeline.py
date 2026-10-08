"""The full CDT pipeline: every genre's prepare chain, then extract, match and publish."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Self

import pandas as pd

from cdt.classifier.eightk import classify_pending_items
from cdt.classifier.sixk import DEFAULT_CONCURRENCY as SIXK_DEFAULT_CONCURRENCY
from cdt.classifier.sixk import triage_pending_windows
from cdt.datasets import (
    GENRE_6K,
    GENRE_8K,
    GENRES,
    failure_registry_path,
    resolve_artifact_root,
)
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_REASONING_EFFORT,
    extract_pending_items,
    extracted_tables_path,
)
from cdt.ingest.core import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_BUCKET,
    DEFAULT_S3_PREFIX,
    IngestConfig,
    IngestRunResult,
)
from cdt.ingest.core import DEFAULT_BATCH_SIZE as DEFAULT_INGEST_BATCH_SIZE
from cdt.ingest.genres import ingest_genre
from cdt.lease import LeaseLostError
from cdt.matcher import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    match_pending_mentions,
)
from cdt.publish import finalize_after_match
from cdt.segmenter.eightk import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    segment_pending_eightk_documents,
)
from cdt.segmenter.sixk import segment_pending_sixk_documents
from cdt.shared import get_logger
from cdt.storage.objects import ArtifactPath, read_text_artifact

#: Genres the CLI entry points prepare unless `--genres` narrows them: all of them.
DEFAULT_GENRES: tuple[str, ...] = tuple(GENRES)


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
    segment_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    classify_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    extract_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    match_batch_size: int = DEFAULT_STAGE_BATCH_SIZE
    item_numbers: tuple[str, ...] = POTENTIALLY_RELEVANT_ITEM_NUMBERS
    #: The 8-K item classifier's artifact directory; None is the committed one.
    classifier_model_dir: Path | None = None
    #: The 6-K stage-1 artifact directory; None is the committed one.
    sixk_model_dir: Path | None = None
    #: Filings whose 6-K stage-2 calls may be in flight at once.
    sixk_concurrency: int = SIXK_DEFAULT_CONCURRENCY
    #: None resolves to the EXTRACTOR_MODEL setting when extraction runs.
    extractor_model: str | None = None
    extractor_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    extractor_max_attempts: int = DEFAULT_MAX_ATTEMPTS
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN
    #: Which genres to prepare. 8-K only when built in code, because the 6-K
    #: chain scrapes and calls a paid model; the CLI passes DEFAULT_GENRES.
    genres: tuple[str, ...] = (GENRE_8K,)


@dataclass(frozen=True)
class GenreResult:
    """What one genre's prepare chain produced: ingest, segment, classify."""

    ingest: IngestRunResult
    segmented_rows: int
    classified_rows: int


@dataclass(frozen=True)
class PipelineRunResult:
    """Summary of one end-to-end pipeline run."""

    mode: str
    start_date: date
    end_date: date
    extracted_rows: int
    matched_rows: int
    debt_instrument_rows: int
    artifact_root: str
    extractor_run_path: str
    genres: tuple[str, ...] = DEFAULT_GENRES
    #: Per genre whose chain completed; a genre that was not selected, or
    #: whose chain failed, has no entry.
    genre_results: dict[str, GenreResult] = field(default_factory=dict)
    #: Genres whose prepare chain failed; extract, match and publish still ran
    #: over the rest, and the caller reports the run as failed.
    failed_genres: tuple[str, ...] = ()


@dataclass
class _PrepareOutcome:
    """What the prepare phases produced, per genre."""

    genre_results: dict[str, GenreResult] = field(default_factory=dict)
    #: Genres whose chain raised; the others still ran.
    failed_genres: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrepareResult:
    """What a prepare-only run produced: its artifact root, and any failed genre."""

    artifact_root: str
    failed_genres: tuple[str, ...] = ()


def segment_genre(
    genre: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    batch_size: int = DEFAULT_STAGE_BATCH_SIZE,
    force: bool = False,
    item_numbers: tuple[str, ...] = POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Run one genre's segment stage: 8-K items, or 6-K window spans.

    ``item_numbers`` applies to 8-K only. Returns the rows written.

    Raises:
        ValueError: If ``genre`` is not one of GENRES.
    """
    if genre == GENRE_8K:
        return segment_pending_eightk_documents(
            artifact_root=artifact_root,
            data_dir=data_dir,
            batch_size=batch_size,
            force=force,
            item_numbers=item_numbers,
            renew=renew,
        )
    if genre == GENRE_6K:
        return segment_pending_sixk_documents(
            artifact_root=artifact_root,
            data_dir=data_dir,
            batch_size=batch_size,
            force=force,
            renew=renew,
        )
    raise ValueError(_unknown_genre(genre))


def classify_genre(
    genre: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    batch_size: int = DEFAULT_STAGE_BATCH_SIZE,
    force: bool = False,
    model_dir: Path | None = None,
    sixk_model_dir: Path | None = None,
    sixk_concurrency: int = SIXK_DEFAULT_CONCURRENCY,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Run one genre's classify stage: 8-K item relevance, or 6-K triage.

    ``model_dir`` is the 8-K classifier's, ``sixk_model_dir`` and
    ``sixk_concurrency`` the 6-K triage's. Returns the rows written.

    Raises:
        ValueError: If ``genre`` is not one of GENRES.
    """
    if genre == GENRE_8K:
        return classify_pending_items(
            artifact_root=artifact_root,
            data_dir=data_dir,
            model_dir=model_dir,
            batch_size=batch_size,
            force=force,
            renew=renew,
        )
    if genre == GENRE_6K:
        return triage_pending_windows(
            artifact_root=artifact_root,
            data_dir=data_dir,
            batch_size=batch_size,
            force=force,
            model_dir=sixk_model_dir,
            concurrency=sixk_concurrency,
            renew=renew,
        )
    raise ValueError(_unknown_genre(genre))


def _unknown_genre(genre: str) -> str:
    return f"unknown genre {genre!r}; expected one of {', '.join(GENRES)}"


class Pipeline:
    """Run the full CDT pipeline with structured logging."""

    def __init__(self: Self, config: PipelineConfig) -> None:
        """Hold the run's config."""
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

    def _ingest_config(
        self: Self,
        resolved_start: date,
        resolved_end: date,
        resolved_artifact_root: str,
    ) -> IngestConfig:
        """Return this run's ingest settings; ``ingest_genre`` narrows them per genre."""
        return IngestConfig(
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
        )

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

        Each genre writes its own documents, segment and classification
        datasets, so a failing genre cannot corrupt the other's. A genre whose
        chain raises is logged and recorded in ``failed_genres`` and the next
        genre still runs; its partitions stay pending for the next run.
        ``LeaseLostError`` is not caught: a run that lost its lease must stop
        writing.
        """
        outcome = _PrepareOutcome()
        for genre in GENRES:
            if genre not in self.config.genres:
                self.logger.info(
                    "Skipping the %s chain: genres=%s", genre, self.config.genres
                )
                continue
            try:
                outcome.genre_results[genre] = self._prepare_genre(
                    genre,
                    resolved_start,
                    resolved_end,
                    ciks,
                    resolved_artifact_root,
                    renew,
                )
            except LeaseLostError:
                raise
            except Exception:
                self.logger.exception("Genre prepare failed: genre=%s", genre)
                outcome.failed_genres.append(genre)
        return outcome

    def _prepare_genre(
        self: Self,
        genre: str,
        resolved_start: date,
        resolved_end: date,
        ciks: set[str],
        resolved_artifact_root: str,
        renew: Callable[[], None] | None,
    ) -> GenreResult:
        """Run one genre's ingest → segment → classify, renewing between stages."""
        self._log_stage_start(
            "ingest",
            genre=genre,
            batch_size=self.config.ingest_batch_size,
            ciks=len(ciks),
        )
        _, ingest_result = ingest_genre(
            genre,
            self._ingest_config(resolved_start, resolved_end, resolved_artifact_root),
            ciks=ciks,
        )
        self._log_stage_complete(
            "ingest",
            genre=genre,
            rows=ingest_result.total_rows,
            candidates=ingest_result.candidates_seen,
            partitions=len(ingest_result.document_partitions),
            failures=ingest_result.failures,
        )
        self._renew(renew)

        self._log_stage_start(
            "segment", genre=genre, batch_size=self.config.segment_batch_size
        )
        segmented = segment_genre(
            genre,
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.segment_batch_size,
            force=self.config.force,
            item_numbers=self.config.item_numbers,
            renew=renew,
        )
        self._log_stage_complete("segment", genre=genre, rows=len(segmented))
        self._renew(renew)

        self._log_stage_start(
            "classify", genre=genre, batch_size=self.config.classify_batch_size
        )
        classified = classify_genre(
            genre,
            artifact_root=resolved_artifact_root,
            data_dir=self.config.data_dir,
            batch_size=self.config.classify_batch_size,
            force=self.config.force,
            model_dir=self.config.classifier_model_dir,
            sixk_model_dir=self.config.sixk_model_dir,
            sixk_concurrency=self.config.sixk_concurrency,
            renew=renew,
        )
        self._log_stage_complete("classify", genre=genre, rows=len(classified))
        self._renew(renew)
        return GenreResult(
            ingest=ingest_result,
            segmented_rows=len(segmented),
            classified_rows=len(classified),
        )

    def run_prepare(
        self: Self, renew: Callable[[], None] | None = None
    ) -> PrepareResult:
        """Run only the prepare stages of each genre."""
        resolved_start, resolved_end, ciks, resolved_artifact_root = self._setup()
        prepared = self._prepare_genres(
            resolved_start, resolved_end, ciks, resolved_artifact_root, renew
        )
        return PrepareResult(resolved_artifact_root, tuple(prepared.failed_genres))

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
            extracted_rows=len(extracted),
            matched_rows=matched_mentions,
            debt_instrument_rows=len(matched["debt_instrument"]),
            artifact_root=resolved_artifact_root,
            extractor_run_path=extracted_tables_path(
                resolved_artifact_root,
                data_dir=self.config.data_dir,
            ),
            genres=self.config.genres,
            genre_results=prepared.genre_results,
            failed_genres=tuple(prepared.failed_genres),
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
        if prepared.failed_genres:
            self._log_banner(
                f"Pipeline finished in {elapsed} with failed genres: "
                + ",".join(prepared.failed_genres)
            )
        else:
            self._log_banner(f"Pipeline completed successfully in {elapsed}")
        return result


def run_pipeline(
    config: PipelineConfig, *, renew: Callable[[], None] | None = None
) -> PipelineRunResult:
    """Run the full CDT pipeline for the provided config."""
    return Pipeline(config).run(renew)


def run_prepare_stages(
    config: PipelineConfig, *, renew: Callable[[], None] | None = None
) -> PrepareResult:
    """Run only the prepare stages for a config."""
    return Pipeline(config).run_prepare(renew)


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
        ValueError: On an unknown mode, daily with only one date given, or
            an end date before the start date.
    """
    if mode not in PIPELINE_MODES:
        msg = f"unsupported mode {mode!r}"
        raise ValueError(msg)
    if mode == "historical":
        start, end = start_date or ALL_TIME_START_DATE, end_date or date.today()
    elif start_date is None and end_date is None:
        today = date.today()
        start = today.fromordinal(today.toordinal() - DAILY_LOOKBACK_DAYS)
        end = today.fromordinal(today.toordinal() - 1)
    elif start_date is None:
        msg = "--start-date is required when --end-date is provided"
        raise ValueError(msg)
    elif end_date is None:
        msg = "--end-date is required when --start-date is provided"
        raise ValueError(msg)
    else:
        start, end = start_date, end_date
    if end < start:
        msg = f"end date {end.isoformat()} is before start date {start.isoformat()}"
        raise ValueError(msg)
    return start, end
