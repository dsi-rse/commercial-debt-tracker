"""Orchestrator entrypoint for ECS and local runs.

Modes:

- ``daily`` and ``historical`` run the prepare stages, then match and
  finalize. With the default ``batch`` backend extraction is left to ``poll``;
  ``--extractor-backend live`` runs the synchronous pipeline instead.
- ``poll`` advances the OpenAI batch extract job by one tick and, when the job
  completes, runs match and finalize.

Every mode runs under the ``pipeline-writer`` lease. ``daily`` and
``historical`` wait up to LEASE_WAIT_SECONDS for it and fail if it stays held;
``poll`` skips its tick and prints ``locked``.
"""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Callable, Sequence
from time import monotonic, sleep

from cdt.cli_support import configure_logging, parse_date, positive_int
from cdt.datasets import resolve_artifact_root
from cdt.extractor import DEFAULT_MAX_ATTEMPTS, OpenAIBatchClient, advance_extract_job
from cdt.ingest.core import DEFAULT_BUCKET
from cdt.lease import (
    PIPELINE_WRITER_LEASE,
    Lease,
    LeaseLostError,
    acquire_lease,
    release_lease,
    renewer,
)
from cdt.pipeline import (
    DEFAULT_GENRES,
    DEFAULT_STAGE_BATCH_SIZE,
    PipelineConfig,
    normalize_genres,
    run_match_and_finalize,
    run_pipeline,
    run_prepare_stages,
)
from cdt.shared import get_logger
from cdt.storage.objects import configure_s3_profile

LOGGER = get_logger(__name__)

# How long daily/historical wait for a lease held by a poll tick before failing.
LEASE_WAIT_SECONDS = 15 * 60
LEASE_POLL_SECONDS = 30

# Per-mode wall-clock deadline for the runtime watchdog: ECS has no task-level
# timeout, so a wedged run must reap itself. Override with --max-runtime-hours.
MODE_DEADLINE_HOURS: dict[str, float] = {"poll": 2, "daily": 12, "historical": 72}
# EX_SOFTWARE, so a watchdog self-reap is distinguishable from a crash.
WATCHDOG_EXIT_CODE = 70


def start_runtime_watchdog(
    mode: str,
    max_runtime_hours: float | None = None,
    *,
    exit_fn: Callable[[int], None] = os._exit,
) -> threading.Timer:
    """Arm a daemon timer that calls ``exit_fn(WATCHDOG_EXIT_CODE)`` past the deadline.

    The deadline is ``max_runtime_hours``, or MODE_DEADLINE_HOURS[mode] when
    None. The default ``os._exit`` skips ``finally`` blocks; an unreleased lease
    is recovered by its TTL.
    """
    hours = (
        max_runtime_hours
        if max_runtime_hours is not None
        else (MODE_DEADLINE_HOURS[mode])
    )

    def _expire() -> None:
        LOGGER.error(
            "Runtime watchdog expired: mode=%s exceeded %sh — the task is wedged; "
            "exiting %s so ECS reaps it and the task-failure alarm fires.",
            mode,
            hours,
            WATCHDOG_EXIT_CODE,
        )
        exit_fn(WATCHDOG_EXIT_CODE)

    timer = threading.Timer(hours * 3600, _expire)
    timer.daemon = True
    timer.start()
    return timer


def acquire_lease_with_wait(artifact_root: str, name: str) -> Lease | None:
    """Acquire the lease, retrying up to LEASE_WAIT_SECONDS; None on timeout."""
    deadline = monotonic() + LEASE_WAIT_SECONDS
    while True:
        lease = acquire_lease(artifact_root, name)
        if lease is not None:
            return lease
        if monotonic() >= deadline:
            return None
        LOGGER.info(
            "Pipeline-writer lease is held; retrying in %ss.", LEASE_POLL_SECONDS
        )
        sleep(LEASE_POLL_SECONDS)


# Prefix of the value Pulumi seeds into the SSM SecureStrings that feed these env
# vars (pulumi/infra/secrets.py). Pulumi cannot import this package, so the two
# literals are coupled by convention, like source_prefix and DEFAULT_S3_PREFIX.
_SECRET_PLACEHOLDER_PREFIX = "PLACEHOLDER-set-via-"  # noqa: S105 — a sentinel, not a credential
_SECRET_ENV_VARS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY")


def reject_placeholder_secrets() -> None:
    """Raise SystemExit if an API-key env var still holds the Pulumi placeholder."""
    stale = [
        name
        for name in _SECRET_ENV_VARS
        if os.environ.get(name, "").startswith(_SECRET_PLACEHOLDER_PREFIX)
    ]
    if stale:
        raise SystemExit(
            f"{', '.join(stale)} still hold the Pulumi placeholder value. Set the "
            "real value(s) with: aws ssm put-parameter --name "
            "/idi/<env>/cdt/secrets/<key> --type SecureString --value '<v>' "
            "--overwrite (picked up at the next task launch, no deploy needed)."
        )


def default_cik_file() -> str:
    """Return ``CDT_DEFAULT_CIK_FILE``; raise RuntimeError if it is unset."""
    value = os.environ.get("CDT_DEFAULT_CIK_FILE")
    if not value:
        raise RuntimeError("CDT_DEFAULT_CIK_FILE is required for orchestrator runs.")
    return value


def _add_stage_batch_size_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--ingest-batch-size", type=positive_int, default=DEFAULT_STAGE_BATCH_SIZE
    )
    subparser.add_argument(
        "--itemize-batch-size", type=positive_int, default=DEFAULT_STAGE_BATCH_SIZE
    )
    subparser.add_argument(
        "--classify-batch-size", type=positive_int, default=DEFAULT_STAGE_BATCH_SIZE
    )
    # None, so the batch backend can warn about an explicit value.
    subparser.add_argument(
        "--extract-batch-size",
        type=positive_int,
        default=None,
        help=(
            "rows per synchronous extract batch; applies with --extractor-backend "
            f"live only (default {DEFAULT_STAGE_BATCH_SIZE}). The batch backend "
            "chunks by --max-requests-per-batch on poll instead."
        ),
    )
    subparser.add_argument(
        "--match-batch-size", type=positive_int, default=DEFAULT_STAGE_BATCH_SIZE
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator parser."""
    parser = argparse.ArgumentParser(prog="cdt-orchestrator")
    parser.add_argument("--artifact-root", default=os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument(
        "--final-database-root", default=os.environ.get("FINAL_DATABASE_ROOT")
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("BUCKET_NAME") or DEFAULT_BUCKET,
    )
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument(
        "--genres",
        type=normalize_genres,
        default=os.environ.get("GENRES") or DEFAULT_GENRES,
        help=(
            "comma-separated filing genres to prepare (default "
            f"{','.join(DEFAULT_GENRES)}; env GENRES). A scheduled run "
            "acquires every genre unless narrowed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "reprocess partitions already in the completion registry. On poll this "
            "only applies when a new extract job is created; an already-active job "
            "keeps its claimed partitions."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help=(
            "wall-clock deadline before the run reaps itself; defaults per "
            f"mode: {MODE_DEADLINE_HOURS} (#93)"
        ),
    )
    parser.add_argument(
        "--extractor-backend",
        choices=("live", "batch"),
        default=os.environ.get("EXTRACTOR_BACKEND", "batch"),
        help=(
            "extract backend for daily and historical: 'batch' (default) defers "
            "extraction to the OpenAI batch poller; 'live' runs the synchronous "
            "OpenRouter pipeline."
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
    poll.add_argument("--max-batch-bytes", type=positive_int, default=None)
    # Caps rows claimed into one job so it cannot OOM the poll task.
    poll.add_argument("--max-rows-per-job", type=positive_int, default=None)
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
        extract_batch_size=(
            args.extract_batch_size
            if args.extract_batch_size is not None
            else DEFAULT_STAGE_BATCH_SIZE
        ),
        match_batch_size=args.match_batch_size,
        genres=(
            args.genres
            if isinstance(args.genres, tuple)
            else normalize_genres(args.genres)
        ),
    )


def run_batch_backend(args: argparse.Namespace) -> int:
    """Run the prepare stages plus match/finalize under the lease; defer extract to poll.

    Serves both ``daily`` and ``historical``. Returns the exit code: 1 when the
    lease cannot be acquired or is lost, else 0 (the artifact root is printed).
    """
    if args.extract_batch_size is not None:
        LOGGER.warning(
            "Ignoring --extract-batch-size=%s: the batch backend defers extraction "
            "to poll, which chunks by --max-requests-per-batch/--max-batch-bytes. "
            "Use --extractor-backend live to size synchronous extract batches.",
            args.extract_batch_size,
        )
    if args.force:
        LOGGER.warning(
            "--force applies to the prepare and match/finalize stages only under "
            "the batch backend: extraction is deferred to poll, which never "
            "forces an already-completed partition. To re-extract, run a poll "
            "tick with --force while no job is active."
        )
    config = _pipeline_config(args)
    # Prepare writes registries a poll tick also writes, so the lease covers it.
    resolved_root = resolve_artifact_root(args.artifact_root)
    lease = acquire_lease_with_wait(resolved_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.error(
            "Pipeline-writer lease still held after %ss; not running. The holder "
            "may be wedged — check for a stuck poll tick.",
            LEASE_WAIT_SECONDS,
        )
        return 1
    renew = renewer(lease)
    try:
        prepared = run_prepare_stages(config, renew=renew)
        artifact_root = prepared.artifact_root
        LOGGER.info(
            "Extraction deferred to the batch poller: pending items are claimed by "
            "the next `poll` run. Ensure the poll schedule is enabled, or run "
            "`cdt-orchestrator poll` manually — nothing else drives extraction."
        )
        # Publish from the mentions that already exist; poll publishes again
        # when the batch job completes.
        renew()
        run_match_and_finalize(
            artifact_root=artifact_root,
            final_database_root=args.final_database_root,
            batch_size=args.match_batch_size,
            force=args.force,
            renew=renew,
        )
    except LeaseLostError as exc:
        LOGGER.error("Aborting run: %s", exc)
        return 1
    finally:
        release_lease(lease)
    if prepared.failed_genres:
        # A partial success is a failure: exit non-zero and skip the heartbeat
        # below, so the daily-heartbeat alarm still fires.
        LOGGER.error(
            "Run finished with failed genres: %s; their partitions stay pending",
            ",".join(prepared.failed_genres),
        )
        return 1
    # The daily-heartbeat CloudWatch alarm matches this literal; keep it in
    # sync with pulumi/infra/alerts.py.
    LOGGER.info(
        "Orchestrator run complete: mode=%s artifact_root=%s",
        config.mode,
        artifact_root,
    )
    print(artifact_root)
    return 0


def run_poll(args: argparse.Namespace) -> int:
    """Advance the OpenAI batch extract job by one tick; finalize on completion.

    Runs under the pipeline-writer lease. Prints the job status, or ``locked``
    when the lease is held (exit 0). Returns 1 if the lease is lost mid-tick.
    """
    resolved_root = resolve_artifact_root(args.artifact_root)
    lease = acquire_lease(resolved_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.warning(
            "Pipeline-writer lease is held; skipping this poll tick — the next "
            "scheduled tick will pick the job up."
        )
        print("locked")
        return 0
    renew = renewer(lease)
    try:
        tick_kwargs: dict[str, object] = {
            "batch_client": OpenAIBatchClient(),
            "artifact_root": resolved_root,
            "max_attempts": args.max_attempts,
            "force": args.force,
            # Renewed at phase boundaries; raises LeaseLostError if stolen.
            "renew_lease": renew,
        }
        if args.max_requests_per_batch is not None:
            tick_kwargs["max_requests_per_batch"] = args.max_requests_per_batch
        if args.max_batch_bytes is not None:
            tick_kwargs["max_batch_bytes"] = args.max_batch_bytes
        if args.max_rows_per_job is not None:
            tick_kwargs["max_rows_per_job"] = args.max_rows_per_job
        result = advance_extract_job(**tick_kwargs)
        # The poll-liveness CloudWatch alarm matches this literal; keep it in
        # sync with pulumi/infra/alerts.py.
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
            renew()
            run_match_and_finalize(
                artifact_root=resolved_root,
                final_database_root=args.final_database_root,
                batch_size=args.match_batch_size,
                force=args.force,
                renew=renew,
            )
        print(result.status)
    except LeaseLostError as exc:
        LOGGER.error("Aborting poll tick: %s", exc)
        return 1
    finally:
        release_lease(lease)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the orchestrator."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(quiet=args.quiet)
    # Before any S3 access, so artifact reads, writes and the lease all use it.
    configure_s3_profile(args.aws_profile)
    reject_placeholder_secrets()
    start_runtime_watchdog(args.mode, args.max_runtime_hours)

    if args.mode == "poll":
        return run_poll(args)
    if args.extractor_backend == "batch":
        return run_batch_backend(args)

    # The live backend writes what a poll tick writes, so it takes the lease too.
    resolved_root = resolve_artifact_root(args.artifact_root)
    lease = acquire_lease_with_wait(resolved_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.error(
            "Pipeline-writer lease still held after %ss; not starting a live run. "
            "The holder may be wedged — check for a stuck poll tick.",
            LEASE_WAIT_SECONDS,
        )
        return 1
    try:
        result = run_pipeline(_pipeline_config(args), renew=renewer(lease))
    except LeaseLostError as exc:
        LOGGER.error("Aborting live run: %s", exc)
        return 1
    finally:
        release_lease(lease)
    print(result.artifact_root)
    if result.failed_genres:
        LOGGER.error(
            "Live run finished with failed genres: %s; their partitions stay pending",
            ",".join(result.failed_genres),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
