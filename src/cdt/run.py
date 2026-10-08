"""Whole runs: ``cdt run daily``, ``cdt run historical`` and ``cdt run poll``.

- ``daily`` and ``historical`` prepare every selected genre (ingest → segment →
  classify), then match and publish. With the ``batch`` extractor backend
  extraction is left to ``poll``; with ``live`` the run extracts synchronously.
- ``poll`` advances the OpenAI batch extract job by one tick and, when the job
  completes, matches and publishes.

Every mode runs under the ``pipeline-writer`` lease. ``daily`` and
``historical`` wait up to LEASE_WAIT_SECONDS for it and fail if it stays held;
``poll`` skips its tick and prints ``locked``. The CLI arms the runtime
watchdog and the placeholder-secret check before any of these start.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from time import monotonic, sleep

from cdt.datasets import resolve_artifact_root
from cdt.extractor import DEFAULT_MAX_ATTEMPTS, OpenAIBatchClient, advance_extract_job
from cdt.lease import (
    PIPELINE_WRITER_LEASE,
    Lease,
    LeaseLostError,
    acquire_lease,
    release_lease,
    renewer,
)
from cdt.pipeline import (
    DEFAULT_STAGE_BATCH_SIZE,
    PipelineConfig,
    PipelineRunResult,
    run_match_and_finalize,
    run_pipeline,
    run_prepare_stages,
)
from cdt.shared import get_logger
from cdt.storage.objects import ArtifactPath

LOGGER = get_logger(__name__)

RUN_MODES = ("daily", "historical", "poll")
EXTRACTOR_BACKENDS = ("batch", "live")

# How long daily/historical wait for a lease held by a poll tick before failing.
LEASE_WAIT_SECONDS = 15 * 60
LEASE_POLL_SECONDS = 30

# Per-mode wall-clock deadline for the runtime watchdog: ECS has no task-level
# timeout, so a wedged run must reap itself. Override with --max-runtime-hours.
MODE_DEADLINE_HOURS: dict[str, float] = {"poll": 2, "daily": 12, "historical": 72}
# EX_SOFTWARE, so a watchdog self-reap is distinguishable from a crash.
WATCHDOG_EXIT_CODE = 70

# The daily-heartbeat and poll-liveness CloudWatch alarms match these literals;
# keep them in sync with pulumi/infra/alerts.py.
RUN_COMPLETE_MESSAGE = "Run complete: mode=%s artifact_root=%s"
POLL_TICK_COMPLETE_MESSAGE = (
    "Poll tick complete: status=%s job=%s folded=%s submitted=%s in_flight=%s "
    "terminal=%s"
)


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
        else MODE_DEADLINE_HOURS[mode]
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


def _acquire_or_report(artifact_root: str, noun: str) -> Lease | None:
    lease = acquire_lease_with_wait(artifact_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.error(
            "Pipeline-writer lease still held after %ss; not starting %s. The "
            "holder may be wedged — check for a stuck poll tick.",
            LEASE_WAIT_SECONDS,
            noun,
        )
    return lease


def run_prepare_then_publish(
    config: PipelineConfig, *, extract_batch_size_given: bool = False
) -> int:
    """Run a batch-backend ``daily``/``historical``: prepare, then match and publish.

    Extraction is left to ``poll``. ``extract_batch_size_given`` says the caller
    passed an extract batch size, which this backend cannot use. Returns the
    exit code: 1 when the lease cannot be acquired or is lost, or a genre
    failed; else 0, after logging RUN_COMPLETE_MESSAGE and printing the
    artifact root.
    """
    if extract_batch_size_given:
        LOGGER.warning(
            "Ignoring --extract-batch-size=%s: the batch backend defers extraction "
            "to poll, which chunks by --max-requests-per-batch/--max-batch-bytes. "
            "Use --extractor-backend live to size synchronous extract batches.",
            config.extract_batch_size,
        )
    if config.force:
        LOGGER.warning(
            "--force applies to the prepare and match/publish stages only under "
            "the batch backend: extraction is deferred to poll, which never "
            "forces an already-completed partition. To re-extract, run "
            "`cdt run poll --force` while no job is active."
        )
    # Prepare writes registries a poll tick also writes, so the lease covers it.
    lease = _acquire_or_report(resolve_artifact_root(config.artifact_root), "the run")
    if lease is None:
        return 1
    renew = renewer(lease)
    try:
        prepared = run_prepare_stages(config, renew=renew)
        artifact_root = prepared.artifact_root
        LOGGER.info(
            "Extraction deferred to the batch poller: pending items are claimed by "
            "the next `cdt run poll`. Ensure the poll schedule is enabled, or run "
            "it by hand — nothing else drives batch extraction."
        )
        # Publish from the mentions that already exist; poll publishes again
        # when the batch job completes.
        renew()
        run_match_and_finalize(
            artifact_root=artifact_root,
            final_database_root=config.final_database_root,
            batch_size=config.match_batch_size,
            force=config.force,
            renew=renew,
        )
    except LeaseLostError as exc:
        LOGGER.error("Aborting run: %s", exc)
        return 1
    finally:
        release_lease(lease)
    if prepared.failed_genres:
        # A partial success is a failure: exit non-zero and skip the heartbeat,
        # so the daily-heartbeat alarm still fires.
        LOGGER.error(
            "Run finished with failed genres: %s; their partitions stay pending",
            ",".join(prepared.failed_genres),
        )
        return 1
    LOGGER.info(RUN_COMPLETE_MESSAGE, config.mode, artifact_root)
    print(artifact_root)
    return 0


def run_live(config: PipelineConfig) -> tuple[int, PipelineRunResult | None]:
    """Run a live-backend ``daily``/``historical``: the whole pipeline, synchronously.

    Returns the exit code and the run's result (None when it did not run to
    the end). Exit 1 when the lease cannot be acquired or is lost, or a genre
    failed; else 0, after logging RUN_COMPLETE_MESSAGE.
    """
    # The live backend writes what a poll tick writes, so it takes the lease too.
    lease = _acquire_or_report(
        resolve_artifact_root(config.artifact_root), "a live run"
    )
    if lease is None:
        return 1, None
    try:
        result = run_pipeline(config, renew=renewer(lease))
    except LeaseLostError as exc:
        LOGGER.error("Aborting live run: %s", exc)
        return 1, None
    finally:
        release_lease(lease)
    if result.failed_genres:
        LOGGER.error(
            "Live run finished with failed genres: %s; their partitions stay pending",
            ",".join(result.failed_genres),
        )
        return 1, result
    LOGGER.info(RUN_COMPLETE_MESSAGE, config.mode, result.artifact_root)
    return 0, result


def run_poll(
    *,
    artifact_root: ArtifactPath | None,
    final_database_root: ArtifactPath | None = None,
    force: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_requests_per_batch: int | None = None,
    max_batch_bytes: int | None = None,
    max_rows_per_job: int | None = None,
    match_batch_size: int = DEFAULT_STAGE_BATCH_SIZE,
) -> int:
    """Advance the OpenAI batch extract job by one tick; publish on completion.

    Runs under the pipeline-writer lease. Prints the job status, or ``locked``
    when the lease is held (exit 0). Returns 1 if the lease is lost mid-tick.
    ``force`` applies only when the tick creates a new job. The ``max_*`` limits
    default to the batch backend's own.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    lease = acquire_lease(resolved_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.warning(
            "Pipeline-writer lease is held; skipping this poll tick — the next "
            "scheduled tick will pick the job up."
        )
        print("locked")
        return 0
    renew = renewer(lease)
    limits = {
        name: value
        for name, value in (
            ("max_requests_per_batch", max_requests_per_batch),
            ("max_batch_bytes", max_batch_bytes),
            ("max_rows_per_job", max_rows_per_job),
        )
        if value is not None
    }
    try:
        result = advance_extract_job(
            batch_client=OpenAIBatchClient(),
            artifact_root=resolved_root,
            max_attempts=max_attempts,
            force=force,
            # Renewed at phase boundaries; raises LeaseLostError if stolen.
            renew_lease=renew,
            **limits,
        )
        LOGGER.info(
            POLL_TICK_COMPLETE_MESSAGE,
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
                final_database_root=final_database_root,
                batch_size=match_batch_size,
                force=force,
                renew=renew,
            )
        print(result.status)
    except LeaseLostError as exc:
        LOGGER.error("Aborting poll tick: %s", exc)
        return 1
    finally:
        release_lease(lease)
    return 0
