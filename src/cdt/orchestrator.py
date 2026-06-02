"""Dedicated orchestrator entrypoint for ECS and local batch runs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from cdt.cli import configure_logging, parse_date, positive_int
from cdt.pipeline import PipelineConfig, run_pipeline


def default_cik_file() -> str:
    """Return the default deployed CIK file path."""
    value = os.environ.get("CDT_DEFAULT_CIK_FILE")
    if not value:
        raise RuntimeError("CDT_DEFAULT_CIK_FILE is required for orchestrator runs.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator parser."""
    parser = argparse.ArgumentParser(prog="cdt-orchestrator")
    parser.add_argument("--artifact-root", default=os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument(
        "--final-database-root", default=os.environ.get("FINAL_DATABASE_ROOT")
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("BUCKET_NAME", "idi-dev-ftm2j-shared-processor-storage"),
    )
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    daily = subparsers.add_parser("daily")
    daily.add_argument("--cik-file", default=None)
    daily.add_argument("--start-date", type=parse_date, default=None)
    daily.add_argument("--end-date", type=parse_date, default=None)
    daily.add_argument("--ingest-batch-size", type=positive_int, default=100)
    daily.add_argument("--itemize-batch-size", type=positive_int, default=100)
    daily.add_argument("--classify-batch-size", type=positive_int, default=100)
    daily.add_argument("--extract-batch-size", type=positive_int, default=100)
    daily.add_argument("--match-batch-size", type=positive_int, default=100)

    historical = subparsers.add_parser("historical")
    historical.add_argument("--cik-file", default=None)
    historical.add_argument("--start-date", type=parse_date, required=True)
    historical.add_argument("--end-date", type=parse_date, required=True)
    historical.add_argument("--ingest-batch-size", type=positive_int, default=100)
    historical.add_argument("--itemize-batch-size", type=positive_int, default=100)
    historical.add_argument("--classify-batch-size", type=positive_int, default=100)
    historical.add_argument("--extract-batch-size", type=positive_int, default=100)
    historical.add_argument("--match-batch-size", type=positive_int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the orchestrator."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(quiet=args.quiet)
    cik_file = args.cik_file or default_cik_file()
    result = run_pipeline(
        PipelineConfig(
            mode=args.mode,
            cik_file=cik_file,
            bucket=args.bucket,
            aws_profile=args.aws_profile,
            start_date=args.start_date,
            end_date=args.end_date,
            artifact_root=args.artifact_root,
            final_database_root=args.final_database_root,
            force=args.force,
            ingest_batch_size=args.ingest_batch_size,
            itemize_batch_size=args.itemize_batch_size,
            classify_batch_size=args.classify_batch_size,
            extract_batch_size=args.extract_batch_size,
            match_batch_size=args.match_batch_size,
        )
    )
    print(result.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
