"""Command-line interface for Commercial Debt Tracker."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from cdt.classifier import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_RECALL,
    classifications_root,
    classify_pending_items,
    default_model_dir,
    train_classifier_model,
)
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS as DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
)
from cdt.extractor import DEFAULT_MODEL as DEFAULT_EXTRACTOR_MODEL
from cdt.extractor import (
    DEFAULT_REASONING_EFFORT as DEFAULT_EXTRACTOR_REASONING_EFFORT,
)
from cdt.extractor import (
    ActiveJobSummary,
    describe_active_job,
    extract_pending_items,
    extracted_tables_path,
    mentions_root,
    reset_active_job,
)
from cdt.ingest import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_BUCKET,
    IngestConfig,
    default_output_root,
    documents_root,
    run_ingest_pipeline,
)
from cdt.itemizer import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    itemize_pending_documents,
    items_root,
)
from cdt.lease import PIPELINE_WRITER_LEASE, acquire_lease, release_lease
from cdt.matcher import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    debt_instruments_root,
    match_pending_mentions,
    mention_cluster_edges_root,
)
from cdt.pipeline import (
    ALL_TIME_START_DATE as PIPELINE_ALL_TIME_START_DATE,
)
from cdt.pipeline import PipelineConfig, resolve_mode_dates, run_pipeline

ALL_TIME_START_DATE = date(1994, 1, 1)
DEFAULT_BATCH_SIZE = 100


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cdt command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def add_artifact_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared artifact-root CLI argument."""
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact root as a local path or s3:// URI. Defaults to DATA_DIR.",
    )
    parser.add_argument(
        "--final-database-root",
        default=None,
        help=(
            "Optional final table root as a local path or s3:// URI. "
            "When set, writes latest.parquet files under {root}/{table}/."
        ),
    )


def add_logging_arguments(parser: argparse.ArgumentParser, *, noun: str) -> None:
    """Add quiet/log-file arguments to a parser."""
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=f"Optional path to write {noun} logs.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(prog="cdt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Index 8-K submission resources for CIKs."
    )
    add_artifact_root_argument(ingest_parser)
    ingest_parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    ingest_parser.add_argument("--download", action="store_true")
    ingest_parser.add_argument("--failure-file", default=None)
    ingest_parser.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    ingest_parser.add_argument("--s3-prefix", default="sec")
    add_logging_arguments(ingest_parser, noun="ingest")
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_mode", required=True)
    for mode_name, help_text in (
        ("daily", "Index 8-K filings from a daily date window."),
        ("historical", "Index 8-K filings from the historical scraper archive."),
    ):
        subparser = ingest_subparsers.add_parser(mode_name, help=help_text)
        subparser.add_argument(
            "cik_file", help="Local path or s3:// URI for one-CIK-per-line input."
        )
        subparser.add_argument(
            "--start-date",
            type=parse_date,
            default=None if mode_name == "daily" else ALL_TIME_START_DATE,
        )
        subparser.add_argument(
            "--end-date",
            type=parse_date,
            default=None if mode_name == "daily" else date.today(),
        )
        subparser.set_defaults(func=run_ingest)

    itemize_parser = subparsers.add_parser(
        "itemize", help="Extract 8-K item sections from document partitions."
    )
    add_artifact_root_argument(itemize_parser)
    itemize_parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    itemize_parser.add_argument("--force", action="store_true")
    itemize_parser.add_argument(
        "--item-numbers",
        type=parse_item_numbers,
        default=POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    )
    add_logging_arguments(itemize_parser, noun="itemization")
    itemize_parser.set_defaults(func=run_itemize)

    classify_parser = subparsers.add_parser(
        "classify", help="Train or run binary item relevance classification."
    )
    add_artifact_root_argument(classify_parser)
    classify_parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    classify_parser.add_argument("--force", action="store_true")
    classify_parser.add_argument("--model-dir", type=Path, default=None)
    add_logging_arguments(classify_parser, noun="classification")
    classify_parser.set_defaults(func=run_classifier)
    classify_subparsers = classify_parser.add_subparsers(dest="classify_command")
    classify_train_parser = classify_subparsers.add_parser("train")
    classify_train_parser.add_argument("--train-csv", type=Path, required=True)
    classify_train_parser.add_argument("--model-dir", type=Path, default=None)
    classify_train_parser.add_argument(
        "--target-recall", type=float, default=DEFAULT_TARGET_RECALL
    )
    classify_train_parser.add_argument(
        "--cv-splits", type=positive_int, default=DEFAULT_CV_SPLITS
    )
    classify_train_parser.add_argument(
        "--random-seed", type=int, default=DEFAULT_RANDOM_SEED
    )
    add_logging_arguments(classify_train_parser, noun="training")
    classify_train_parser.set_defaults(func=run_classifier_train)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract instrument mentions from classified item partitions."
    )
    add_artifact_root_argument(extract_parser)
    extract_parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    extract_parser.add_argument(
        "--reasoning-effort", default=DEFAULT_EXTRACTOR_REASONING_EFFORT
    )
    extract_parser.add_argument(
        "--max-attempts", type=positive_int, default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS
    )
    add_logging_arguments(extract_parser, noun="extraction")
    extract_parser.set_defaults(func=run_extractor)

    show_job_parser = subparsers.add_parser(
        "show-extract-job",
        help="Show the state of the async batch extract job (read-only).",
    )
    add_artifact_root_argument(show_job_parser)
    add_logging_arguments(show_job_parser, noun="inspection")
    show_job_parser.set_defaults(func=run_show_extract_job)

    reset_job_parser = subparsers.add_parser(
        "reset-extract-job",
        help=(
            "Clear the active batch extract job marker so the next poll tick "
            "starts fresh. Abandons any batch still in flight."
        ),
    )
    add_artifact_root_argument(reset_job_parser)
    reset_job_parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually clear the marker; without it, only reports.",
    )
    add_logging_arguments(reset_job_parser, noun="reset")
    reset_job_parser.set_defaults(func=run_reset_extract_job)

    match_parser = subparsers.add_parser(
        "match", help="Group extracted instrument mentions into debt instruments."
    )
    add_artifact_root_argument(match_parser)
    match_parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    match_parser.add_argument("--force", action="store_true")
    match_parser.add_argument(
        "--strong-match-threshold",
        type=float,
        default=DEFAULT_MEMBERSHIP_THRESHOLD,
    )
    match_parser.add_argument(
        "--loose-match-threshold",
        type=float,
        default=DEFAULT_RELATED_THRESHOLD,
    )
    match_parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=DEFAULT_AMBIGUITY_MARGIN,
    )
    add_logging_arguments(match_parser, noun="matching")
    match_parser.set_defaults(func=run_matcher)

    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the full CDT pipeline end-to-end."
    )
    add_artifact_root_argument(pipeline_parser)
    pipeline_parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.add_argument("--download", action="store_true")
    pipeline_parser.add_argument("--failure-file", default=None)
    pipeline_parser.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    pipeline_parser.add_argument("--s3-prefix", default="sec")
    pipeline_parser.add_argument(
        "--ingest-batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    pipeline_parser.add_argument(
        "--itemize-batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    pipeline_parser.add_argument(
        "--classify-batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    pipeline_parser.add_argument(
        "--extract-batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    pipeline_parser.add_argument(
        "--match-batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    pipeline_parser.add_argument(
        "--item-numbers",
        type=parse_item_numbers,
        default=POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    )
    pipeline_parser.add_argument("--model-dir", type=Path, default=None)
    pipeline_parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    pipeline_parser.add_argument(
        "--reasoning-effort", default=DEFAULT_EXTRACTOR_REASONING_EFFORT
    )
    pipeline_parser.add_argument(
        "--max-attempts", type=positive_int, default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS
    )
    pipeline_parser.add_argument(
        "--strong-match-threshold", type=float, default=DEFAULT_MEMBERSHIP_THRESHOLD
    )
    pipeline_parser.add_argument(
        "--loose-match-threshold", type=float, default=DEFAULT_RELATED_THRESHOLD
    )
    pipeline_parser.add_argument(
        "--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN
    )
    add_logging_arguments(pipeline_parser, noun="pipeline")
    pipeline_subparsers = pipeline_parser.add_subparsers(
        dest="pipeline_mode", required=True
    )
    for mode_name in ("daily", "historical"):
        subparser = pipeline_subparsers.add_parser(mode_name)
        subparser.add_argument(
            "cik_file", help="Local path or s3:// URI for one-CIK-per-line input."
        )
        subparser.add_argument(
            "--start-date",
            type=parse_date,
            default=None if mode_name == "daily" else PIPELINE_ALL_TIME_START_DATE,
        )
        subparser.add_argument(
            "--end-date",
            type=parse_date,
            default=None if mode_name == "daily" else date.today(),
        )
        subparser.set_defaults(func=run_pipeline_command)
    return parser


def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    try:
        start_date, end_date = resolve_ingest_dates(args)
        config = IngestConfig(
            mode=args.ingest_mode,
            bucket=args.bucket,
            cik_file=Path(str(args.cik_file)),
            start_date=start_date,
            end_date=end_date,
            output_root=args.artifact_root or default_output_root(),
            force=args.force,
            batch_size=args.batch_size,
            download=args.download,
            failure_file=args.failure_file,
            aws_profile=args.aws_profile,
            s3_prefix=args.s3_prefix,
        )
        logger.info(
            "Starting ingest: mode=%s bucket=%s start_date=%s end_date=%s batch_size=%s output_root=%s",
            config.mode,
            config.bucket,
            config.start_date,
            config.end_date,
            config.batch_size,
            config.output_root,
        )
        _, result = run_ingest_pipeline(config, ciks=read_cik_file(args.cik_file))
    except ValueError as exc:
        logger.error("Invalid ingest arguments: %s", exc)
        return 2
    except Exception:
        logger.exception("Ingest failed")
        return 1
    print(
        f"Indexed {result.total_rows} document rows from {result.start_date} through {result.end_date}."
    )
    print(f"Output root: {result.output_root}.")
    print(f"Documents dataset: {result.documents_root}.")
    print(f"Run manifest: {result.run_manifest}.")
    print(f"Failure registry: {result.failure_file}.")
    return 0


def run_itemize(args: argparse.Namespace) -> int:
    """Run the itemize subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    artifact_root = args.artifact_root or default_output_root()
    try:
        logger.info(
            "Starting itemization: batch_size=%s force=%s item_numbers=%s documents=%s output=%s",
            args.batch_size,
            args.force,
            ",".join(args.item_numbers),
            documents_root(artifact_root),
            items_root(artifact_root),
        )
        items = itemize_pending_documents(
            artifact_root=artifact_root,
            batch_size=args.batch_size,
            force=args.force,
            item_numbers=args.item_numbers,
        )
    except Exception:
        logger.exception("Itemization failed")
        return 1
    print(f"Itemized {len(items)} item rows.")
    print(f"Wrote item partitions to {items_root(artifact_root)}.")
    return 0


def run_pipeline_command(args: argparse.Namespace) -> int:
    """Run the full CDT pipeline."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    try:
        start_date, end_date = resolve_mode_dates(
            args.pipeline_mode,
            args.start_date,
            args.end_date,
        )
        result = run_pipeline(
            PipelineConfig(
                mode=args.pipeline_mode,
                cik_file=args.cik_file,
                bucket=args.bucket,
                start_date=start_date,
                end_date=end_date,
                artifact_root=args.artifact_root,
                final_database_root=args.final_database_root,
                force=args.force,
                download=args.download,
                failure_file=args.failure_file,
                aws_profile=args.aws_profile,
                s3_prefix=args.s3_prefix,
                ingest_batch_size=args.ingest_batch_size,
                itemize_batch_size=args.itemize_batch_size,
                classify_batch_size=args.classify_batch_size,
                extract_batch_size=args.extract_batch_size,
                match_batch_size=args.match_batch_size,
                item_numbers=args.item_numbers,
                classifier_model_dir=args.model_dir,
                extractor_model=args.model,
                extractor_reasoning_effort=args.reasoning_effort,
                extractor_max_attempts=args.max_attempts,
                strong_match_threshold=args.strong_match_threshold,
                loose_match_threshold=args.loose_match_threshold,
                ambiguity_margin=args.ambiguity_margin,
            )
        )
    except ValueError as exc:
        logger.error("Invalid pipeline arguments: %s", exc)
        return 2
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    print(f"Ran pipeline from {result.start_date} through {result.end_date}.")
    print(
        f"Ingest indexed {result.ingest.total_rows} rows, itemized {result.itemized_rows}, "
        f"classified {result.classified_rows}, extracted {result.extracted_rows}, and matched {result.matched_rows} mentions."
    )
    print(
        f"Artifact root: {result.artifact_root}. Debt instruments: {debt_instruments_root(result.artifact_root)}."
    )
    print(f"Extractor runs: {result.extractor_run_path}.")
    print(f"Failure registry: {result.ingest.failure_file}.")
    return 0


def run_classifier(args: argparse.Namespace) -> int:
    """Run the classifier inference subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    artifact_root = args.artifact_root or default_output_root()
    resolved_model_dir = args.model_dir or default_model_dir()
    try:
        logger.info(
            "Starting classification: batch_size=%s force=%s input=%s output=%s model_dir=%s",
            args.batch_size,
            args.force,
            items_root(artifact_root),
            classifications_root(artifact_root),
            resolved_model_dir,
        )
        items = classify_pending_items(
            artifact_root=artifact_root,
            batch_size=args.batch_size,
            force=args.force,
            model_dir=args.model_dir,
        )
    except Exception:
        logger.exception("Classification failed")
        return 1
    print(f"Classified {len(items)} item rows.")
    print(f"Wrote classification partitions to {classifications_root(artifact_root)}.")
    return 0


def run_classifier_train(args: argparse.Namespace) -> int:
    """Run the classifier training subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    resolved_model_dir = args.model_dir or default_model_dir()
    try:
        logger.info(
            "Starting classifier training: train_csv=%s model_dir=%s target_recall=%s cv_splits=%s random_seed=%s",
            args.train_csv,
            resolved_model_dir,
            args.target_recall,
            args.cv_splits,
            args.random_seed,
        )
        metadata = train_classifier_model(
            train_csv=args.train_csv,
            model_dir=resolved_model_dir,
            target_recall=args.target_recall,
            cv_splits=args.cv_splits,
            random_seed=args.random_seed,
        )
    except Exception:
        logger.exception("Classifier training failed")
        return 1
    print(f"Trained classifier on {metadata['training_row_count']} labeled rows.")
    print(f"Wrote model artifacts to {resolved_model_dir}.")
    return 0


def run_extractor(args: argparse.Namespace) -> int:
    """Run the LLM extractor subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    artifact_root = args.artifact_root or default_output_root()
    try:
        logger.info(
            "Starting extraction: batch_size=%s force=%s input=%s output=%s model=%s reasoning_effort=%s max_attempts=%s audit=%s",
            args.batch_size,
            args.force,
            classifications_root(artifact_root),
            mentions_root(artifact_root),
            args.model,
            args.reasoning_effort,
            args.max_attempts,
            extracted_tables_path(artifact_root),
        )
        mentions = extract_pending_items(
            artifact_root=artifact_root,
            batch_size=args.batch_size,
            force=args.force,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
        )
    except Exception:
        logger.exception("Extraction failed")
        return 1
    print(f"Extracted {len(mentions)} instrument mention rows.")
    print(f"Wrote canonical mentions to {mentions_root(artifact_root)}.")
    print(
        f"Wrote full.jsonl audit output under {extracted_tables_path(artifact_root)}."
    )
    return 0


def run_show_extract_job(args: argparse.Namespace) -> int:
    """Report the active batch extract job's state."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    artifact_root = args.artifact_root or default_output_root()
    summary: ActiveJobSummary = describe_active_job(artifact_root)
    if summary.status == "idle":
        print("No active extract job; the next poll tick will start one.")
        return 0
    if summary.status == "corrupt":
        print(f"Active job {summary.job_id} is unusable: {summary.detail}")
        print(
            "The next poll tick clears this automatically. To clear it now, run "
            "`cdt reset-extract-job --yes`."
        )
        return 1
    print(f"Active job {summary.job_id} (tick {summary.tick}):")
    for label, value in (
        ("rows", summary.total_rows),
        ("terminal rows", summary.terminal_rows),
        ("rows awaiting a request", summary.awaiting_rows),
        ("batches in flight", summary.in_flight_batches),
        ("claimed partitions", summary.claimed_partitions),
    ):
        print(f"  {label + ':':<26}{value}")
    return 0


def run_reset_extract_job(args: argparse.Namespace) -> int:
    """Clear the active batch extract job marker under the writer lease."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    artifact_root = args.artifact_root or default_output_root()
    summary: ActiveJobSummary = describe_active_job(artifact_root)
    if summary.status == "idle":
        print("No active extract job; nothing to reset.")
        return 0
    if not args.yes:
        # A corrupt job's batches.json is unreadable, so the in-flight count is
        # unknown rather than zero.
        abandoned = (
            "an unknown number of batches"
            if summary.status == "corrupt"
            else f"{summary.in_flight_batches} batch(es)"
        )
        print(f"Active job {summary.job_id} ({summary.status}).")
        print(
            f"Would clear the marker, abandoning {abandoned} still in flight. "
            "Re-run with --yes to proceed."
        )
        return 0
    # Take the same lease a poll tick holds, so a running tick cannot rewrite the
    # marker underneath the reset.
    lease = acquire_lease(artifact_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        logger.error(
            "Pipeline-writer lease is held (a poll tick is running); not resetting."
        )
        return 1
    try:
        # Pin the clear to the job shown above: if a poll tick completed it and
        # started another in between, abort instead of abandoning the new job.
        job_id = reset_active_job(artifact_root, expected_job_id=summary.job_id)
    finally:
        release_lease(lease)
    if job_id is None:
        print(
            "Not reset: the active job changed (or completed) since inspection. "
            "Re-run to inspect the current state."
        )
        return 1
    print(f"Cleared the active extract job marker for {job_id}.")
    print("The next poll tick starts a fresh job from unclaimed partitions.")
    return 0


def run_matcher(args: argparse.Namespace) -> int:
    """Run the matcher subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    artifact_root = args.artifact_root or default_output_root()
    try:
        logger.info(
            "Starting matcher: batch_size=%s force=%s input=%s mention_cluster_edges=%s debt_instruments=%s",
            args.batch_size,
            args.force,
            mentions_root(artifact_root),
            mention_cluster_edges_root(artifact_root),
            debt_instruments_root(artifact_root),
        )
        tables = match_pending_mentions(
            artifact_root=artifact_root,
            batch_size=args.batch_size,
            force=args.force,
            strong_match_threshold=args.strong_match_threshold,
            loose_match_threshold=args.loose_match_threshold,
            ambiguity_margin=args.ambiguity_margin,
        )
    except Exception:
        logger.exception("Matcher failed")
        return 1
    print(
        f"Matched {len(tables['debt_instrument_mentions'])} mention-cluster edge rows."
    )
    print(
        f"Wrote mention-cluster edges to {mention_cluster_edges_root(artifact_root)}."
    )
    print(f"Wrote debt instruments to {debt_instruments_root(artifact_root)}.")
    return 0


def configure_logging(*, quiet: bool, log_file: Path | None = None) -> None:
    """Configure CLI logging."""
    level = logging.WARNING if quiet else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def read_cik_file(path: str) -> set[str]:
    """Read a one-CIK-per-line file."""
    from cdt.pipeline import read_cik_file as _read_cik_file

    return _read_cik_file(path)


def resolve_ingest_dates(args: argparse.Namespace) -> tuple[date, date]:
    """Resolve ingest dates for the selected ingest mode."""
    return resolve_mode_dates(args.ingest_mode, args.start_date, args.end_date)


def parse_date(value: str) -> date:
    """Parse an ISO date for argparse."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"expected YYYY-MM-DD, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_item_numbers(value: str) -> tuple[str, ...]:
    """Parse a comma-separated item-number list for argparse."""
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        msg = "expected a comma-separated list of item numbers"
        raise argparse.ArgumentTypeError(msg)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
