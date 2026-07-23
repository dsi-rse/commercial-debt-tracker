"""Dedicated orchestrator entrypoint for ECS and local batch runs.

Deployment uses two schedules:

- ``daily`` runs ingest → itemize → classify and refreshes match/final snapshots.
  With the default ``batch`` backend it does NOT run the LLM extract stage; that is
  handed to the asynchronous OpenAI Batch poller.
- ``poll`` runs hourly, advancing the OpenAI batch extract job by one tick and, when
  a job completes, re-running match + finalize. ``poll`` is the sole owner of extract
  job state, so it never races the ``daily`` schedule.

``historical`` (and ``daily --extractor-backend live``) keep the original fully
synchronous pipeline that extracts live via OpenRouter.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from cdt.cli import configure_logging, parse_date, positive_int
from cdt.datasets import resolve_artifact_root
from cdt.extractor import DEFAULT_MAX_ATTEMPTS, OpenAIBatchClient, advance_extract_job
from cdt.pipeline import (
    PipelineConfig,
    run_match_and_finalize,
    run_pipeline,
    run_prepare_stages,
)
from cdt.shared import get_logger

LOGGER = get_logger(__name__)


def default_cik_file() -> str:
    """Return the default deployed CIK file path."""
    value = os.environ.get("CDT_DEFAULT_CIK_FILE")
    if not value:
        raise RuntimeError("CDT_DEFAULT_CIK_FILE is required for orchestrator runs.")
    return value


def _add_stage_batch_size_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--ingest-batch-size", type=positive_int, default=100)
    subparser.add_argument("--itemize-batch-size", type=positive_int, default=100)
    subparser.add_argument("--classify-batch-size", type=positive_int, default=100)
    subparser.add_argument("--extract-batch-size", type=positive_int, default=100)
    subparser.add_argument("--match-batch-size", type=positive_int, default=100)


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator parser."""
    parser = argparse.ArgumentParser(prog="cdt-orchestrator")
    parser.add_argument("--artifact-root", default=os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument(
        "--final-database-root", default=os.environ.get("FINAL_DATABASE_ROOT")
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("BUCKET_NAME", "idi-dev-processor-s3"),
    )
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--extractor-backend",
        choices=("live", "batch"),
        default=os.environ.get("EXTRACTOR_BACKEND", "batch"),
        help=(
            "daily extract backend: 'batch' (default) defers extraction to the "
            "OpenAI batch poller; 'live' runs the synchronous OpenRouter pipeline."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    daily = subparsers.add_parser("daily")
    daily.add_argument("--cik-file", default=None)
    daily.add_argument("--start-date", type=parse_date, default=None)
    daily.add_argument("--end-date", type=parse_date, default=None)
    _add_stage_batch_size_arguments(daily)

    historical = subparsers.add_parser("historical")
    historical.add_argument("--cik-file", default=None)
    historical.add_argument("--start-date", type=parse_date, required=True)
    historical.add_argument("--end-date", type=parse_date, required=True)
    _add_stage_batch_size_arguments(historical)

    poll = subparsers.add_parser("poll")
    poll.add_argument("--match-batch-size", type=positive_int, default=100)
    poll.add_argument("--max-attempts", type=positive_int, default=DEFAULT_MAX_ATTEMPTS)
    poll.add_argument("--max-requests-per-batch", type=positive_int, default=None)
    return parser


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    cik_file = args.cik_file or default_cik_file()
    return PipelineConfig(
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


def run_daily_batch(args: argparse.Namespace) -> int:
    """Run daily ingest/itemize/classify plus match/finalize, deferring extract."""
    config = _pipeline_config(args)
    artifact_root = run_prepare_stages(config)
    # Keep final snapshots fresh from whatever mentions already exist; the in-flight
    # batch job (advanced by ``poll``) will refresh them again when it completes.
    run_match_and_finalize(
        artifact_root=artifact_root,
        final_database_root=args.final_database_root,
        batch_size=args.match_batch_size,
        force=args.force,
    )
    print(artifact_root)
    return 0


def run_poll(args: argparse.Namespace) -> int:
    """Advance the OpenAI batch extract job by one tick; finalize on completion."""
    resolved_root = resolve_artifact_root(args.artifact_root)
    tick_kwargs: dict[str, object] = {
        "batch_client": OpenAIBatchClient(),
        "artifact_root": resolved_root,
        "max_attempts": args.max_attempts,
        "force": args.force,
    }
    if args.max_requests_per_batch is not None:
        tick_kwargs["max_requests_per_batch"] = args.max_requests_per_batch
    result = advance_extract_job(**tick_kwargs)
    LOGGER.info(
        "Poll tick complete: status=%s job=%s folded=%s submitted=%s in_flight=%s terminal=%s",
        result.status,
        result.job_id,
        result.folded_rows,
        result.submitted_batches,
        result.in_flight_batches,
        result.terminal_rows,
    )
    if result.status == "completed":
        run_match_and_finalize(
            artifact_root=resolved_root,
            final_database_root=args.final_database_root,
            batch_size=args.match_batch_size,
            force=args.force,
        )
    print(result.status)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the orchestrator."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(quiet=args.quiet)

    if args.mode == "poll":
        return run_poll(args)
    if args.mode == "daily" and args.extractor_backend == "batch":
        return run_daily_batch(args)

    # historical, or daily with the live backend: the original synchronous pipeline.
    result = run_pipeline(_pipeline_config(args))
    print(result.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
