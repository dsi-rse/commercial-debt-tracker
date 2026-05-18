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
    default_train_csv_path,
    train_classifier_model,
)
from cdt.database import cdt_db_path
from cdt.ingest import (
    DEFAULT_BUCKET,
    acquire_documents_for_date_range,
    document_batches_path,
    documents_db_path,
)
from cdt.itemizer import itemize_pending_documents, items_path

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
        "cik_file",
        type=Path,
        help="Path to a file containing one CIK per line.",
    )
    ingest_parser.add_argument(
        "--start-date",
        type=parse_date,
        default=ALL_TIME_START_DATE,
        help="First filing date to include. Defaults to 1994-01-01.",
    )
    ingest_parser.add_argument(
        "--end-date",
        type=parse_date,
        default=date.today(),
        help="Last filing date to include. Defaults to today.",
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
    ingest_parser.set_defaults(func=run_ingest)

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
        default=None,
        help="Optional training CSV path. Defaults to the shared annotation CSV.",
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
    return parser


def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    ciks = read_cik_file(args.cik_file)
    logger = logging.getLogger(__name__)
    try:
        logger.info(
            "Starting ingest: ciks=%s bucket=%s start_date=%s end_date=%s "
            "batch_size=%s download=%s database=%s",
            len(ciks),
            args.bucket,
            args.start_date,
            args.end_date,
            args.batch_size,
            args.download,
            documents_db_path(),
        )
        documents = acquire_documents_for_date_range(
            args.bucket,
            args.start_date,
            args.end_date,
            ciks,
            force=args.force,
            batch_size=args.batch_size,
            download=args.download,
        )
    except Exception:
        logger.exception("Ingest failed")
        return 1
    print(
        f"Indexed {len(documents)} document rows for {len(ciks)} CIKs "
        f"from {args.start_date} through {args.end_date}."
    )
    print(f"Wrote {documents_db_path()}.")
    if args.download:
        print(f"Wrote document batches to {document_batches_path()}.")
    return 0


def run_itemize(args: argparse.Namespace) -> int:
    """Run the itemize subcommand."""
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    try:
        logger.info(
            "Starting itemization: batch_size=%s force=%s database=%s output=%s",
            args.batch_size,
            args.force,
            documents_db_path(),
            items_path(),
        )
        items = itemize_pending_documents(
            batch_size=args.batch_size,
            force=args.force,
        )
    except Exception:
        logger.exception("Itemization failed")
        return 1
    print(f"Itemized {len(items)} item rows.")
    print(f"Wrote item batches to {items_path()}.")
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
    resolved_train_csv = args.train_csv or default_train_csv_path()
    resolved_model_dir = args.model_dir or default_model_dir()
    try:
        logger.info(
            "Starting classifier training: train_csv=%s model_dir=%s "
            "target_recall=%s cv_splits=%s random_seed=%s",
            resolved_train_csv,
            resolved_model_dir,
            args.target_recall,
            args.cv_splits,
            args.random_seed,
        )
        metadata = train_classifier_model(
            train_csv=resolved_train_csv,
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


if __name__ == "__main__":
    raise SystemExit(main())
