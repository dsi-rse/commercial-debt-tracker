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
    classify_pending_items,
    default_model_dir,
    train_classifier_model,
)
from cdt.database import cdt_db_path
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS as DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
)
from cdt.extractor import (
    DEFAULT_MODEL as DEFAULT_EXTRACTOR_MODEL,
)
from cdt.extractor import (
    DEFAULT_REASONING_EFFORT as DEFAULT_EXTRACTOR_REASONING_EFFORT,
)
from cdt.extractor import (
    extract_pending_items,
    extracted_tables_path,
)
from cdt.ingest import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_BUCKET,
    IngestConfig,
    documents_db_path,
    documents_path,
    run_ingest_pipeline,
)
from cdt.itemizer import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    itemize_pending_documents,
    items_path,
)
from cdt.matcher import match_pending_mentions
from cdt.pipeline import (
    ALL_TIME_START_DATE as PIPELINE_ALL_TIME_START_DATE,
)
from cdt.pipeline import (
    PipelineConfig,
    resolve_mode_dates,
    run_pipeline,
)

ALL_TIME_START_DATE = date(1994, 1, 1)
DEFAULT_BATCH_SIZE = 100


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cdt command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(prog="cdt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Index 8-K submission resources for CIKs.",
    )
    ingest_parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket to read from. Defaults to {DEFAULT_BUCKET}.",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing rows for the requested accessions.",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of matching documents to process per batch. "
            f"Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    ingest_parser.add_argument(
        "--download",
        action="store_true",
        help="Download matched documents into Parquet batch files.",
    )
    ingest_parser.add_argument(
        "--failure-file",
        type=Path,
        default=None,
        help="Optional path to write the permanent ingest failure registry.",
    )
    ingest_parser.add_argument(
        "--aws-profile",
        default=DEFAULT_AWS_PROFILE,
        help=f"AWS profile name for S3 access. Defaults to {DEFAULT_AWS_PROFILE}.",
    )
    ingest_parser.add_argument(
        "--s3-prefix",
        default="sec",
        help="Top-level scraper prefix inside the bucket. Defaults to sec.",
    )
    ingest_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    ingest_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs for long-running ingests.",
    )

    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_mode", required=True)

    ingest_daily_parser = ingest_subparsers.add_parser(
        "daily",
        help="Index 8-K filings from a daily date window.",
    )
    ingest_daily_parser.add_argument(
        "cik_file",
        type=Path,
        help="Path to a file containing one CIK per line.",
    )
    ingest_daily_parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="First filing date to include. Defaults to yesterday when omitted.",
    )
    ingest_daily_parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="Last filing date to include. Defaults to yesterday when omitted.",
    )
    ingest_daily_parser.set_defaults(func=run_ingest)

    ingest_historical_parser = ingest_subparsers.add_parser(
        "historical",
        help="Index 8-K filings from the historical scraper archive.",
    )
    ingest_historical_parser.add_argument(
        "cik_file",
        type=Path,
        help="Path to a file containing one CIK per line.",
    )
    ingest_historical_parser.add_argument(
        "--start-date",
        type=parse_date,
        default=ALL_TIME_START_DATE,
        help="First filing date to include. Defaults to 1994-01-01.",
    )
    ingest_historical_parser.add_argument(
        "--end-date",
        type=parse_date,
        default=date.today(),
        help="Last filing date to include. Defaults to today.",
    )
    ingest_historical_parser.set_defaults(func=run_ingest)

    itemize_parser = subparsers.add_parser(
        "itemize",
        help="Extract 8-K item sections from tracked document resources.",
    )
    itemize_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of source documents to itemize per batch. "
            f"Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    itemize_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-itemize documents already marked itemized.",
    )
    itemize_parser.add_argument(
        "--item-numbers",
        type=parse_item_numbers,
        default=POTENTIALLY_RELEVANT_ITEM_NUMBERS,
        help=(
            "Comma-separated 8-K item numbers to save. "
            "Defaults to potentially relevant items only."
        ),
    )
    itemize_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    itemize_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs for long-running itemization.",
    )
    itemize_parser.set_defaults(func=run_itemize)

    classifier_parser = subparsers.add_parser(
        "classifier",
        help="Train or run the binary item relevance classifier.",
    )
    classifier_parser.set_defaults(func=run_classifier)
    classifier_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of item index rows to classify per batch. "
            f"Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    classifier_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify rows already marked classified.",
    )
    classifier_parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional path to a trained classifier artifact directory.",
    )
    classifier_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    classifier_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs for long-running classification.",
    )

    classifier_subparsers = classifier_parser.add_subparsers(dest="classifier_command")
    classifier_train_parser = classifier_subparsers.add_parser(
        "train",
        help="Train and persist the binary item relevance classifier.",
    )
    classifier_train_parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help=(
            "Training CSV path. The file must contain `text` and `label` " "columns."
        ),
    )
    classifier_train_parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional output directory for model artifacts.",
    )
    classifier_train_parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help=(
            "Minimum recall target used during threshold selection. "
            f"Defaults to {DEFAULT_TARGET_RECALL:.2f}."
        ),
    )
    classifier_train_parser.add_argument(
        "--cv-splits",
        type=positive_int,
        default=DEFAULT_CV_SPLITS,
        help=(
            "Requested number of stratified cross-validation folds. "
            f"Defaults to {DEFAULT_CV_SPLITS}."
        ),
    )
    classifier_train_parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for model training. Defaults to {DEFAULT_RANDOM_SEED}.",
    )
    classifier_train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    classifier_train_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write training logs.",
    )
    classifier_train_parser.set_defaults(func=run_classifier_train)

    extractor_parser = subparsers.add_parser(
        "extractor",
        help="Extract instrument mentions from relevant classified item rows.",
    )
    extractor_parser.set_defaults(func=run_extractor)
    extractor_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of item rows to extract per batch. "
            f"Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    extractor_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract rows already marked extracted or extraction_failed.",
    )
    extractor_parser.add_argument(
        "--model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help=f"OpenRouter model ID to use. Defaults to {DEFAULT_EXTRACTOR_MODEL}.",
    )
    extractor_parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_EXTRACTOR_REASONING_EFFORT,
        help=(
            "OpenRouter reasoning effort to request. "
            f"Defaults to {DEFAULT_EXTRACTOR_REASONING_EFFORT}."
        ),
    )
    extractor_parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
        help=(
            "Maximum prompt attempts per stage before marking the item failed. "
            f"Defaults to {DEFAULT_EXTRACTOR_MAX_ATTEMPTS}."
        ),
    )
    extractor_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    extractor_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write extraction logs.",
    )

    matcher_parser = subparsers.add_parser(
        "matcher",
        help="Group extracted instrument mentions into debt instruments.",
    )
    matcher_parser.set_defaults(func=run_matcher)
    matcher_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of pending mention rows to match per batch. "
            f"Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    matcher_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute matcher outputs for all extracted instrument mentions.",
    )
    matcher_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    matcher_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write matcher logs.",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full CDT pipeline end-to-end.",
    )
    pipeline_parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket to read from. Defaults to {DEFAULT_BUCKET}.",
    )
    pipeline_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages against already-processed rows where supported.",
    )
    pipeline_parser.add_argument(
        "--download",
        action="store_true",
        help="Download matched documents into Parquet batches during ingest.",
    )
    pipeline_parser.add_argument(
        "--failure-file",
        type=Path,
        default=None,
        help="Optional path to write the permanent ingest failure registry.",
    )
    pipeline_parser.add_argument(
        "--aws-profile",
        default=DEFAULT_AWS_PROFILE,
        help=f"AWS profile name for S3 access. Defaults to {DEFAULT_AWS_PROFILE}.",
    )
    pipeline_parser.add_argument(
        "--s3-prefix",
        default="sec",
        help="Top-level scraper prefix inside the bucket. Defaults to sec.",
    )
    pipeline_parser.add_argument(
        "--ingest-batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Documents processed per ingest batch. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    pipeline_parser.add_argument(
        "--itemize-batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Documents processed per itemize batch. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    pipeline_parser.add_argument(
        "--classify-batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Items processed per classifier batch. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    pipeline_parser.add_argument(
        "--extract-batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Items processed per extractor batch. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    pipeline_parser.add_argument(
        "--match-batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Mentions processed per matcher batch. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    pipeline_parser.add_argument(
        "--item-numbers",
        type=parse_item_numbers,
        default=POTENTIALLY_RELEVANT_ITEM_NUMBERS,
        help="Comma-separated 8-K item numbers to save during itemization.",
    )
    pipeline_parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional path to a trained classifier artifact directory.",
    )
    pipeline_parser.add_argument(
        "--model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help=f"OpenRouter model ID to use. Defaults to {DEFAULT_EXTRACTOR_MODEL}.",
    )
    pipeline_parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_EXTRACTOR_REASONING_EFFORT,
        help=(
            "OpenRouter reasoning effort to request. "
            f"Defaults to {DEFAULT_EXTRACTOR_REASONING_EFFORT}."
        ),
    )
    pipeline_parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
        help=(
            "Maximum prompt attempts per stage before marking the item failed. "
            f"Defaults to {DEFAULT_EXTRACTOR_MAX_ATTEMPTS}."
        ),
    )
    pipeline_parser.add_argument(
        "--strong-match-threshold",
        type=float,
        default=0.90,
        help="Matcher threshold for automatic direct matches. Defaults to 0.90.",
    )
    pipeline_parser.add_argument(
        "--loose-match-threshold",
        type=float,
        default=0.75,
        help="Matcher threshold for recording loose candidates. Defaults to 0.75.",
    )
    pipeline_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    pipeline_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write pipeline logs.",
    )

    pipeline_subparsers = pipeline_parser.add_subparsers(
        dest="pipeline_mode", required=True
    )
    pipeline_daily_parser = pipeline_subparsers.add_parser(
        "daily",
        help="Run the full pipeline for a daily filing window.",
    )
    pipeline_daily_parser.add_argument(
        "cik_file",
        type=Path,
        help="Path to a file containing one CIK per line.",
    )
    pipeline_daily_parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="First filing date to include. Defaults to yesterday when omitted.",
    )
    pipeline_daily_parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="Last filing date to include. Defaults to yesterday when omitted.",
    )
    pipeline_daily_parser.set_defaults(func=run_pipeline_command)

    pipeline_historical_parser = pipeline_subparsers.add_parser(
        "historical",
        help="Run the full pipeline for the historical filing range.",
    )
    pipeline_historical_parser.add_argument(
        "cik_file",
        type=Path,
        help="Path to a file containing one CIK per line.",
    )
    pipeline_historical_parser.add_argument(
        "--start-date",
        type=parse_date,
        default=PIPELINE_ALL_TIME_START_DATE,
        help="First filing date to include. Defaults to 1994-01-01.",
    )
    pipeline_historical_parser.add_argument(
        "--end-date",
        type=parse_date,
        default=date.today(),
        help="Last filing date to include. Defaults to today.",
    )
    pipeline_historical_parser.set_defaults(func=run_pipeline_command)
    return parser


def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    ciks = read_cik_file(args.cik_file)
    logger = logging.getLogger(__name__)
    try:
        start_date, end_date = resolve_ingest_dates(args)
        config = IngestConfig(
            mode=args.ingest_mode,
            bucket=args.bucket,
            cik_file=args.cik_file,
            start_date=start_date,
            end_date=end_date,
            force=args.force,
            batch_size=args.batch_size,
            download=args.download,
            failure_file=args.failure_file,
            aws_profile=args.aws_profile,
            s3_prefix=args.s3_prefix,
        )
        logger.info(
            "Starting ingest: mode=%s ciks=%s bucket=%s start_date=%s end_date=%s "
            "batch_size=%s download=%s database=%s",
            config.mode,
            len(ciks),
            config.bucket,
            config.start_date,
            config.end_date,
            config.batch_size,
            config.download,
            documents_db_path(),
        )
        _, result = run_ingest_pipeline(config, ciks=ciks)
    except ValueError as exc:
        logger.error("Invalid ingest arguments: %s", exc)
        return 2
    except Exception:
        logger.exception("Ingest failed")
        return 1
    print(
        f"Indexed {result.total_rows} document rows for {len(ciks)} CIKs "
        f"from {result.start_date} through {result.end_date}."
    )
    print(
        f"Matched {result.candidates_seen} candidate filings, skipped "
        f"{result.skipped_existing} existing accessions, and recorded {result.failures} failures."
    )
    print(f"Wrote {result.database_path}.")
    if config.download:
        print(f"Wrote document batches to {result.documents_path}.")
    else:
        print(
            f"Document batches will be written under {documents_path()} when download mode is used."
        )
    print(f"Failure registry: {result.failure_file}.")
    return 0


def run_itemize(args: argparse.Namespace) -> int:
    """Run the itemize subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    try:
        logger.info(
            "Starting itemization: batch_size=%s force=%s item_numbers=%s database=%s output=%s",
            args.batch_size,
            args.force,
            ",".join(args.item_numbers),
            documents_db_path(),
            items_path(),
        )
        items = itemize_pending_documents(
            batch_size=args.batch_size,
            force=args.force,
            item_numbers=args.item_numbers,
        )
    except Exception:
        logger.exception("Itemization failed")
        return 1
    print(f"Itemized {len(items)} item rows.")
    print(f"Wrote item batches to {items_path()}.")
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
            )
        )
    except ValueError as exc:
        logger.error("Invalid pipeline arguments: %s", exc)
        return 2
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    print(
        f"Ran pipeline for {result.ingest.ciks_count} CIKs from {result.start_date} through {result.end_date}."
    )
    print(
        f"Ingest indexed {result.ingest.total_rows} rows, itemized {result.itemized_rows}, "
        f"classified {result.classified_rows}, extracted {result.extracted_rows}, "
        f"and matched {result.matched_rows} mentions."
    )
    print(
        f"Debt instruments written: {result.debt_instrument_rows}. "
        f"Classifier model dir: {result.classifier_model_dir}."
    )
    print(f"Extractor runs: {result.extractor_run_path}.")
    print(f"Failure registry: {result.ingest.failure_file}.")
    return 0


def run_classifier(args: argparse.Namespace) -> int:
    """Run the classifier inference subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    resolved_model_dir = args.model_dir or default_model_dir()
    try:
        logger.info(
            "Starting classification: batch_size=%s force=%s database=%s model_dir=%s",
            args.batch_size,
            args.force,
            cdt_db_path(),
            resolved_model_dir,
        )
        items = classify_pending_items(
            batch_size=args.batch_size,
            force=args.force,
            model_dir=args.model_dir,
        )
    except Exception:
        logger.exception("Classification failed")
        return 1
    print(f"Classified {len(items)} item rows.")
    print(f"Updated {cdt_db_path()}.")
    return 0


def run_classifier_train(args: argparse.Namespace) -> int:
    """Run the classifier training subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    resolved_model_dir = args.model_dir or default_model_dir()
    try:
        logger.info(
            "Starting classifier training: train_csv=%s model_dir=%s "
            "target_recall=%s cv_splits=%s random_seed=%s",
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
    try:
        logger.info(
            "Starting extraction: batch_size=%s force=%s database=%s model=%s reasoning_effort=%s max_attempts=%s output=%s",
            args.batch_size,
            args.force,
            cdt_db_path(),
            args.model,
            args.reasoning_effort,
            args.max_attempts,
            extracted_tables_path(),
        )
        mentions = extract_pending_items(
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
    print(f"Updated {cdt_db_path()}.")
    print(f"Wrote full.jsonl audit output under {extracted_tables_path()}.")
    return 0


def run_matcher(args: argparse.Namespace) -> int:
    """Run the matcher subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    try:
        logger.info(
            "Starting matcher: batch_size=%s force=%s database=%s",
            args.batch_size,
            args.force,
            cdt_db_path(),
        )
        tables = match_pending_mentions(
            batch_size=args.batch_size,
            force=args.force,
        )
    except Exception:
        logger.exception("Matcher failed")
        return 1
    print(
        f"Matched {len(tables['debt_instrument_mentions'])} debt instrument mention rows."
    )
    print(f"Updated {cdt_db_path()}.")
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


def read_cik_file(path: Path) -> set[str]:
    """Read a one-CIK-per-line file."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


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
