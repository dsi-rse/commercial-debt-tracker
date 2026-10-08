"""The ``cdt`` command line: one command per pipeline module, and ``cdt run``.

Stage commands, each named after the module it runs::

    cdt ingest | segment | classify [train] | extract [job show|reset] | match | publish

Whole runs, as deployed (see :mod:`cdt.run`)::

    cdt run daily | historical | poll

``--genres`` (default every genre) narrows ``ingest``, ``segment``, ``classify``
and ``run daily|historical``: ``--genres 8-K`` is the 8-K-only pipeline.
``extract``, ``match`` and ``publish`` read every genre's output.

An option's default resolves in order: the flag, the environment variable in
:data:`ENVIRONMENT_DEFAULTS`, then the built-in default.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from cdt.classifier.core import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_RECALL,
    default_model_dir,
    train_classifier_model,
)
from cdt.classifier.sixk import DEFAULT_CONCURRENCY as SIXK_DEFAULT_CONCURRENCY
from cdt.cli_support import configure_logging, parse_date, positive_int
from cdt.datasets import GENRES, dataset_root, resolve_artifact_root
from cdt.extractor import (
    DEFAULT_MAX_ATTEMPTS as DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
)
from cdt.extractor import (
    ActiveJobSummary,
    describe_active_job,
    extract_pending_items,
    extracted_tables_path,
    mentions_root,
    reset_active_job,
)
from cdt.ingest.core import (
    DEFAULT_BUCKET,
    DEFAULT_FLUSH_ROWS,
    DEFAULT_S3_PREFIX,
    IngestConfig,
    IngestRunResult,
)
from cdt.ingest.genres import ingest_genre
from cdt.lease import (
    PIPELINE_WRITER_LEASE,
    Lease,
    LeaseLostError,
    acquire_lease,
    release_lease,
    renewer,
)
from cdt.matcher import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    debt_instruments_root,
    match_pending_mentions,
    mention_cluster_edges_root,
)
from cdt.matcher.lineage_inference import apply_lineage_inference_pass
from cdt.pipeline import (
    DEFAULT_GENRES,
    PipelineConfig,
    PipelineRunResult,
    classify_genre,
    normalize_genres,
    read_cik_file,
    resolve_mode_dates,
    segment_genre,
)
from cdt.publish import publish_final_tables
from cdt.run import (
    EXTRACTOR_BACKENDS,
    MODE_DEADLINE_HOURS,
    WATCHDOG_EXIT_CODE,
    advance_batch_extract,
    reject_placeholder_secrets,
    run_live,
    run_poll,
    run_prepare_then_publish,
    start_runtime_watchdog,
)
from cdt.segmenter.eightk import POTENTIALLY_RELEVANT_ITEM_NUMBERS
from cdt.storage.objects import configure_s3_profile

#: Option → the environment variable that supplies its default.
ENVIRONMENT_DEFAULTS: dict[str, str] = {
    "--artifact-root": "ARTIFACT_ROOT",
    "--final-database-root": "FINAL_DATABASE_ROOT",
    "--bucket": "BUCKET_NAME",
    "--cik-file": "CDT_DEFAULT_CIK_FILE",
    "--genres": "GENRES",
    "--extractor-backend": "EXTRACTOR_BACKEND",
    "--backend": "EXTRACTOR_BACKEND",
}

#: Exit code for invalid arguments, as argparse uses.
USAGE_EXIT_CODE = 2

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cdt command-line interface."""
    args = build_parser().parse_args(argv)
    # Process-wide, so every S3 client any command builds uses the profile.
    configure_s3_profile(args.aws_profile)
    configure_logging(quiet=args.quiet, log_file=args.log_file)
    return int(args.func(args))


def _env_default(flag: str, fallback: object = None) -> object:
    """Return ``flag``'s environment default (ENVIRONMENT_DEFAULTS), else ``fallback``."""
    return os.environ.get(ENVIRONMENT_DEFAULTS[flag]) or fallback


def _genres(value: str) -> tuple[str, ...]:
    try:
        return normalize_genres(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _item_numbers(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        msg = "expected a comma-separated list of item numbers"
        raise argparse.ArgumentTypeError(msg)
    return parsed


# --- Option groups ------------------------------------------------------------
#
# Each is a parent parser. ``suppress`` builds one whose defaults are not set,
# for a nested command (``classify train``, ``extract job show``) whose parent
# command already defines the same option: a nested parser's defaults would
# otherwise overwrite a value given before the nested command's name.


def _common_options(*, suppress: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--artifact-root",
        default=argparse.SUPPRESS if suppress else _env_default("--artifact-root"),
        help="artifact root, a local path or s3:// URI (env ARTIFACT_ROOT; "
        "default DATA_DIR)",
    )
    parser.add_argument(
        "--aws-profile",
        default=argparse.SUPPRESS if suppress else "",
        help="AWS profile for every S3 client (default: the ambient credential chain)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="log warnings and errors only",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=argparse.SUPPRESS if suppress else None,
        help="also write the log to this file",
    )
    return parser


def _genre_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--genres",
        type=_genres,
        # A string, so argparse parses it with ``type`` and reports a bad GENRES.
        default=str(_env_default("--genres", ",".join(DEFAULT_GENRES))),
        help=f"comma-separated filing genres (env GENRES; default "
        f"{','.join(DEFAULT_GENRES)})",
    )
    return parser


def _filing_window_options(*, dates_required: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cik-file",
        default=_env_default("--cik-file"),
        help="one-CIK-per-line file, a local path or s3:// URI, or 'all' for "
        "every filer (env CDT_DEFAULT_CIK_FILE; required)",
    )
    window = "required" if dates_required else "default: the daily window"
    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=dates_required,
        help=f"first filing date, YYYY-MM-DD ({window})",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=dates_required,
        help=f"last filing date, YYYY-MM-DD ({window})",
    )
    parser.add_argument(
        "--bucket",
        default=_env_default("--bucket", DEFAULT_BUCKET),
        help=f"scraper bucket (env BUCKET_NAME; default {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help=f"scraper key prefix (default {DEFAULT_S3_PREFIX})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="store 8-K document bodies in the documents partitions",
    )
    parser.add_argument(
        "--failure-file",
        default=None,
        help="ingest failure registry (default failures/ingest/failures.json "
        "under the artifact root)",
    )
    return parser


def _force_option(help_text: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force", action="store_true", help=help_text)
    return parser


def _force_publish_option() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--force-publish",
        action="store_true",
        help="publish even when no source changed since the last publish, and "
        "even when a table would shrink below half its published rows",
    )
    return parser


def _segment_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--item-numbers",
        type=_item_numbers,
        default=POTENTIALLY_RELEVANT_ITEM_NUMBERS,
        help="8-K items to keep (default "
        f"{','.join(POTENTIALLY_RELEVANT_ITEM_NUMBERS)})",
    )
    return parser


def _classify_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="8-K classifier artifact directory (default: the committed model)",
    )
    parser.add_argument(
        "--sixk-model-dir",
        type=Path,
        default=None,
        help="6-K stage-1 artifact directory (default: the committed model)",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=SIXK_DEFAULT_CONCURRENCY,
        help="6-K filings whose stage-2 calls may be in flight at once "
        f"(default {SIXK_DEFAULT_CONCURRENCY})",
    )
    return parser


def _extract_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        default=None,
        help="extractor model (default: env EXTRACTOR_MODEL live, "
        "EXTRACTOR_BATCH_MODEL batch)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="reasoning effort (default: env EXTRACTOR_REASONING live, "
        "EXTRACTOR_BATCH_REASONING batch)",
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
        help=f"scored attempts per stage (default {DEFAULT_EXTRACTOR_MAX_ATTEMPTS})",
    )
    return parser


def _batch_tick_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-requests-per-batch",
        type=positive_int,
        default=None,
        help="requests per OpenAI batch file (default: the backend's limit)",
    )
    parser.add_argument(
        "--max-batch-bytes",
        type=positive_int,
        default=None,
        help="bytes per OpenAI batch file (default: the backend's limit)",
    )
    parser.add_argument(
        "--max-rows-per-job",
        type=positive_int,
        default=None,
        help="rows one job may claim, so its state fits the task's memory "
        "(default: the backend's limit)",
    )
    return parser


def _match_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--strong-match-threshold",
        type=float,
        default=DEFAULT_MEMBERSHIP_THRESHOLD,
        help=f"membership score (default {DEFAULT_MEMBERSHIP_THRESHOLD})",
    )
    parser.add_argument(
        "--loose-match-threshold",
        type=float,
        default=DEFAULT_RELATED_THRESHOLD,
        help=f"related-edge score (default {DEFAULT_RELATED_THRESHOLD})",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=DEFAULT_AMBIGUITY_MARGIN,
        help="score gap that separates two candidates (default "
        f"{DEFAULT_AMBIGUITY_MARGIN})",
    )
    return parser


def _final_database_root_option() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--final-database-root",
        default=_env_default("--final-database-root"),
        help="where the four latest.parquet tables are published, a local path "
        "or s3:// URI (env FINAL_DATABASE_ROOT; unset publishes nothing)",
    )
    return parser


def _runtime_option() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help=f"wall-clock deadline before the run exits {WATCHDOG_EXIT_CODE}; "
        f"defaults per mode: {MODE_DEADLINE_HOURS}",
    )
    return parser


# --- Parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="cdt",
        description="Commercial Debt Tracker: SEC 8-K and 6-K filings to debt "
        "instrument histories.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_ingest(commands)
    _add_segment(commands)
    _add_classify(commands)
    _add_extract(commands)
    _add_match(commands)
    _add_publish(commands)
    _add_run(commands)
    return parser


def _add_ingest(commands: argparse._SubParsersAction) -> None:
    ingest = commands.add_parser(
        "ingest",
        parents=[
            _common_options(),
            _genre_options(),
            _filing_window_options(dates_required=False),
            _force_option("re-acquire filings already in the documents datasets"),
        ],
        help="acquire each genre's filings for a CIK list into its documents dataset",
    )
    ingest.add_argument(
        "--flush-rows",
        type=positive_int,
        default=DEFAULT_FLUSH_ROWS,
        help="document rows buffered before each partition write; each write "
        f"rewrites the partitions its rows land in (default {DEFAULT_FLUSH_ROWS})",
    )
    ingest.set_defaults(func=run_ingest)


def _add_segment(commands: argparse._SubParsersAction) -> None:
    segment = commands.add_parser(
        "segment",
        parents=[
            _common_options(),
            _genre_options(),
            _force_option("re-segment partitions already segmented"),
            _segment_options(),
        ],
        help="cut documents into 8-K items and 6-K window spans",
    )
    segment.set_defaults(func=run_segment)


def _add_classify(commands: argparse._SubParsersAction) -> None:
    classify = commands.add_parser(
        "classify",
        parents=[
            _common_options(),
            _genre_options(),
            _force_option("re-classify partitions already classified"),
            _classify_options(),
        ],
        help="decide which 8-K items and 6-K windows reach the extractor",
    )
    classify.set_defaults(func=run_classify)
    nested = classify.add_subparsers(dest="classify_command")
    train = nested.add_parser(
        "train",
        parents=[_common_options(suppress=True)],
        help="train the 8-K item classifier from a labelled CSV",
    )
    train.add_argument("--train-csv", type=Path, required=True, help="labelled rows")
    train.add_argument(
        "--model-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="where to write the artifacts (default: the committed model's directory)",
    )
    train.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help=f"recall the threshold is chosen for (default {DEFAULT_TARGET_RECALL})",
    )
    train.add_argument(
        "--cv-splits",
        type=positive_int,
        default=DEFAULT_CV_SPLITS,
        help=f"cross-validation folds (default {DEFAULT_CV_SPLITS})",
    )
    train.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"seed (default {DEFAULT_RANDOM_SEED})",
    )
    train.set_defaults(func=run_classify_train)


def _add_extract(commands: argparse._SubParsersAction) -> None:
    extract = commands.add_parser(
        "extract",
        parents=[
            _common_options(),
            _force_option(
                "re-extract partitions already extracted (batch: when this "
                "tick starts a new job)"
            ),
            _extract_options(),
            _batch_tick_options(),
        ],
        help="extract debt-instrument mentions from classified rows; batch "
        "(default) advances the OpenAI batch job one tick",
    )
    extract.add_argument(
        "--backend",
        choices=EXTRACTOR_BACKENDS,
        default=_env_default("--backend", "batch"),
        help="'batch' (env EXTRACTOR_BACKEND; default) advances the OpenAI batch "
        "job by one tick and prints its status; 'live' extracts every pending row "
        "synchronously",
    )
    extract.set_defaults(func=run_extract)
    nested = extract.add_subparsers(dest="extract_command")
    job = nested.add_parser("job", help="inspect or clear the batch extract job")
    job_commands = job.add_subparsers(dest="job_command", required=True)
    show = job_commands.add_parser(
        "show",
        parents=[_common_options(suppress=True)],
        help="show the active batch extract job (read-only)",
    )
    show.set_defaults(func=run_extract_job_show)
    reset = job_commands.add_parser(
        "reset",
        parents=[_common_options(suppress=True)],
        help="clear the active batch extract job so the next poll tick starts "
        "fresh; abandons any batch still in flight",
    )
    reset.add_argument(
        "--yes",
        action="store_true",
        help="clear the marker; without it, only report what would be abandoned",
    )
    reset.set_defaults(func=run_extract_job_reset)


def _add_match(commands: argparse._SubParsersAction) -> None:
    match = commands.add_parser(
        "match",
        parents=[
            _common_options(),
            _force_option("re-match every shard"),
            _match_options(),
        ],
        help="group mentions into debt instruments, then infer lineage",
    )
    match.set_defaults(func=run_match)


def _add_publish(commands: argparse._SubParsersAction) -> None:
    publish = commands.add_parser(
        "publish",
        parents=[
            _common_options(),
            _final_database_root_option(),
            _force_publish_option(),
        ],
        help="write the four latest.parquet tables from the current datasets",
    )
    publish.set_defaults(func=run_publish)


def _add_run(commands: argparse._SubParsersAction) -> None:
    run = commands.add_parser("run", help="run the whole pipeline, as deployed")
    modes = run.add_subparsers(dest="run_mode", required=True)
    for mode, help_text in (
        ("daily", "prepare the daily window, then match and publish"),
        ("historical", "prepare a date range, then match and publish"),
    ):
        prepare = modes.add_parser(
            mode,
            parents=[
                _common_options(),
                _genre_options(),
                _filing_window_options(dates_required=mode == "historical"),
                _force_option("reprocess partitions already recorded complete"),
                _force_publish_option(),
                _final_database_root_option(),
                _runtime_option(),
                _segment_options(),
                _classify_options(),
                _extract_options(),
                _match_options(),
            ],
            help=help_text,
        )
        prepare.add_argument(
            "--extractor-backend",
            choices=EXTRACTOR_BACKENDS,
            default=_env_default("--extractor-backend", "batch"),
            help="'batch' (env EXTRACTOR_BACKEND; default) leaves extraction to "
            "`cdt run poll`; 'live' extracts synchronously in this run",
        )
        prepare.add_argument(
            "--ingest-flush-rows",
            type=positive_int,
            default=DEFAULT_FLUSH_ROWS,
            help="document rows ingest buffers before each partition write "
            f"(default {DEFAULT_FLUSH_ROWS})",
        )
        prepare.set_defaults(func=run_run)

    poll = modes.add_parser(
        "poll",
        parents=[
            _common_options(),
            _final_database_root_option(),
            _runtime_option(),
            _force_option(
                "when this tick starts a new job, claim partitions already extracted"
            ),
            _force_publish_option(),
            _batch_tick_options(),
        ],
        help="advance the batch extract job one tick; match and publish when it "
        "completes",
    )
    poll.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_EXTRACTOR_MAX_ATTEMPTS,
        help=f"scored attempts per stage (default {DEFAULT_EXTRACTOR_MAX_ATTEMPTS})",
    )
    poll.set_defaults(func=run_run)


# --- Shared command plumbing --------------------------------------------------


def _artifact_root(args: argparse.Namespace) -> str:
    return resolve_artifact_root(args.artifact_root)


def _with_writer_lease(
    args: argparse.Namespace, noun: str, body: Callable[[str, Lease], int]
) -> int:
    """Run ``body(artifact_root, lease)`` under the pipeline-writer lease.

    Exit 1 without running when the lease is held, 1 when ``body`` raises
    (logged), 2 when it raises ValueError (an invalid argument).
    """
    artifact_root = _artifact_root(args)
    lease = acquire_lease(artifact_root, PIPELINE_WRITER_LEASE)
    if lease is None:
        LOGGER.error(
            "Pipeline-writer lease is held (a scheduled run or poll tick is "
            "active); not starting %s. Retry when it finishes.",
            noun,
        )
        return 1
    try:
        return body(artifact_root, lease)
    except LeaseLostError as exc:
        LOGGER.error("Aborting %s: %s", noun, exc)
        return 1
    except ValueError as exc:
        LOGGER.error("Invalid %s arguments: %s", noun, exc)
        return USAGE_EXIT_CODE
    except Exception:
        LOGGER.exception("%s failed", noun)
        return 1
    finally:
        release_lease(lease)


def _per_genre(
    genres: Sequence[str], noun: str, stage: Callable[[str], pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Run ``stage`` per genre; a genre that raises is logged and the rest still run.

    ``LeaseLostError`` is not caught: a command that lost its lease stops.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for genre in genres:
        try:
            results[genre] = stage(genre)
        except LeaseLostError:
            raise
        except Exception:
            LOGGER.exception("Genre %s failed: genre=%s", noun, genre)
            failed.append(genre)
    return results, failed


def _report_genres(
    results: dict[str, pd.DataFrame],
    failed: list[str],
    describe: Callable[[str, pd.DataFrame], str],
) -> int:
    for genre, rows in results.items():
        print(describe(genre, rows))
    if failed:
        print(f"Failed genres: {','.join(failed)}; see the log.")
        return 1
    return 0


# --- Stage commands -----------------------------------------------------------


def run_ingest(args: argparse.Namespace) -> int:
    """Run ``cdt ingest``."""
    if not args.cik_file:
        LOGGER.error("--cik-file (or CDT_DEFAULT_CIK_FILE) is required")
        return USAGE_EXIT_CODE

    def body(artifact_root: str, lease: Lease) -> int:
        mode = (
            "daily"
            if args.start_date is None and args.end_date is None
            else "historical"
        )
        start_date, end_date = resolve_mode_dates(mode, args.start_date, args.end_date)
        config = IngestConfig(
            mode=mode,
            bucket=args.bucket,
            cik_file=Path(str(args.cik_file)),
            start_date=start_date,
            end_date=end_date,
            output_root=artifact_root,
            force=args.force,
            flush_rows=args.flush_rows,
            download=args.download,
            failure_file=args.failure_file,
            aws_profile=args.aws_profile,
            s3_prefix=args.s3_prefix,
        )
        ciks = read_cik_file(args.cik_file)
        results: dict[str, IngestRunResult] = {}
        failed: list[str] = []
        for genre in args.genres:
            try:
                results[genre] = ingest_genre(
                    genre, config, ciks=ciks, renew=renewer(lease)
                )[1]
            except LeaseLostError:
                raise
            except Exception:
                LOGGER.exception("Genre ingest failed: genre=%s", genre)
                failed.append(genre)
        for genre, result in results.items():
            print(
                f"{genre}: indexed {result.total_rows} document rows from "
                f"{result.start_date} through {result.end_date} into "
                f"{result.documents_root}."
            )
            if result.failures:
                print(f"  Filings that could not be acquired: {result.failures}.")
        if results:
            print(f"Failure registry: {next(iter(results.values())).failure_file}.")
        if failed:
            print(f"Failed genres: {','.join(failed)}; see the log.")
            return 1
        return 0

    return _with_writer_lease(args, "ingest", body)


def run_segment(args: argparse.Namespace) -> int:
    """Run ``cdt segment``."""

    def body(artifact_root: str, lease: Lease) -> int:
        results, failed = _per_genre(
            args.genres,
            "segment",
            lambda genre: segment_genre(
                genre,
                artifact_root=artifact_root,
                force=args.force,
                item_numbers=args.item_numbers,
                renew=renewer(lease),
            ),
        )

        def describe(genre: str, rows: pd.DataFrame) -> str:
            target = dataset_root(
                GENRES[genre].segment_dataset, artifact_root=artifact_root
            )
            return f"{genre}: segmented {len(rows)} rows into {target}."

        return _report_genres(results, failed, describe)

    return _with_writer_lease(args, "segment", body)


def run_classify(args: argparse.Namespace) -> int:
    """Run ``cdt classify``."""

    def body(artifact_root: str, lease: Lease) -> int:
        results, failed = _per_genre(
            args.genres,
            "classify",
            lambda genre: classify_genre(
                genre,
                artifact_root=artifact_root,
                force=args.force,
                model_dir=args.model_dir,
                sixk_model_dir=args.sixk_model_dir,
                sixk_concurrency=args.concurrency,
                renew=renewer(lease),
            ),
        )

        def describe(genre: str, rows: pd.DataFrame) -> str:
            relevant = int(rows["relevance"].fillna(False).sum()) if len(rows) else 0
            target = dataset_root(
                GENRES[genre].classified_dataset, artifact_root=artifact_root
            )
            return (
                f"{genre}: classified {len(rows)} rows, {relevant} relevant, "
                f"into {target}."
            )

        return _report_genres(results, failed, describe)

    return _with_writer_lease(args, "classify", body)


def run_classify_train(args: argparse.Namespace) -> int:
    """Run ``cdt classify train``."""
    model_dir = getattr(args, "model_dir", None) or default_model_dir()
    try:
        metadata = train_classifier_model(
            train_csv=args.train_csv,
            model_dir=model_dir,
            target_recall=args.target_recall,
            cv_splits=args.cv_splits,
            random_seed=args.random_seed,
        )
    except Exception:
        LOGGER.exception("Classifier training failed")
        return 1
    print(f"Trained classifier on {metadata['training_row_count']} labeled rows.")
    print(f"Wrote model artifacts to {model_dir}.")
    return 0


def run_extract(args: argparse.Namespace) -> int:
    """Run ``cdt extract``: one batch tick (default), or a live run."""

    def batch_tick(artifact_root: str, lease: Lease) -> int:
        result = advance_batch_extract(
            artifact_root=artifact_root,
            force=args.force,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
            max_requests_per_batch=args.max_requests_per_batch,
            max_batch_bytes=args.max_batch_bytes,
            max_rows_per_job=args.max_rows_per_job,
            renew=renewer(lease),
        )
        print(
            f"Batch extract job {result.job_id or '-'}: {result.status} "
            f"(submitted {result.submitted_batches}, in flight "
            f"{result.in_flight_batches}, terminal rows {result.terminal_rows})."
        )
        if result.status == "completed":
            print(
                f"Wrote mentions into {mentions_root(artifact_root)}; run `cdt match`."
            )
        return 0

    def body(artifact_root: str, lease: Lease) -> int:
        mentions = extract_pending_items(
            artifact_root=artifact_root,
            force=args.force,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
            renew=renewer(lease),
        )
        print(
            f"Extracted {len(mentions)} mention rows into {mentions_root(artifact_root)}."
        )
        print(f"Audit log under {extracted_tables_path(artifact_root)}.")
        return 0

    if args.backend == "batch":
        return _with_writer_lease(args, "extract", batch_tick)
    return _with_writer_lease(args, "extract", body)


def run_extract_job_show(args: argparse.Namespace) -> int:
    """Run ``cdt extract job show``: report the active batch job, read-only."""
    summary: ActiveJobSummary = describe_active_job(_artifact_root(args))
    if summary.status == "idle":
        print("No active extract job; the next poll tick will start one.")
        return 0
    if summary.status == "corrupt":
        print(f"Active job {summary.job_id} is unusable: {summary.detail}")
        print(
            "The next poll tick clears this automatically. To clear it now, run "
            "`cdt extract job reset --yes`."
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


def run_extract_job_reset(args: argparse.Namespace) -> int:
    """Run ``cdt extract job reset``: clear the active batch job's marker."""
    summary: ActiveJobSummary = describe_active_job(_artifact_root(args))
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

    def body(artifact_root: str, lease: Lease) -> int:
        del lease
        # Pin the clear to the job shown above: if a poll tick completed it and
        # started another in between, abort instead of abandoning the new job.
        job_id = reset_active_job(artifact_root, expected_job_id=summary.job_id)
        if job_id is None:
            print(
                "Not reset: the active job changed (or completed) since inspection. "
                "Re-run to inspect the current state."
            )
            return 1
        print(f"Cleared the active extract job marker for {job_id}.")
        print("The next poll tick starts a fresh job from unclaimed partitions.")
        return 0

    # The lease a poll tick holds, so a running tick cannot rewrite the marker
    # underneath the reset.
    return _with_writer_lease(args, "the extract job reset", body)


def run_match(args: argparse.Namespace) -> int:
    """Run ``cdt match``: match, then the lineage pass over every shard."""

    def body(artifact_root: str, lease: Lease) -> int:
        tables = match_pending_mentions(
            artifact_root=artifact_root,
            force=args.force,
            renew=renewer(lease),
            strong_match_threshold=args.strong_match_threshold,
            loose_match_threshold=args.loose_match_threshold,
            ambiguity_margin=args.ambiguity_margin,
        )
        # Lineage spans filings, so it is a post-pass over every shard.
        stats = apply_lineage_inference_pass(artifact_root, renew=renewer(lease))
        LOGGER.info(
            "Lineage inference: %s links (%s re-opened), lineage heads %s -> %s",
            stats["links"],
            stats["reopened"],
            stats["heads_before"],
            stats["heads_after"],
        )
        print(
            f"Matched {len(tables['debt_instrument_mentions'])} mention-cluster "
            f"edge rows into {mention_cluster_edges_root(artifact_root)} and "
            f"{debt_instruments_root(artifact_root)}."
        )
        return 0

    return _with_writer_lease(args, "match", body)


def run_publish(args: argparse.Namespace) -> int:
    """Run ``cdt publish``."""
    if not args.final_database_root:
        LOGGER.error("--final-database-root (or FINAL_DATABASE_ROOT) is required")
        return USAGE_EXIT_CODE

    def body(artifact_root: str, lease: Lease) -> int:
        published = publish_final_tables(
            artifact_root=artifact_root,
            final_database_root=args.final_database_root,
            force_publish=args.force_publish,
            renew=renewer(lease),
        )
        if not published:
            print("Nothing to publish: no source changed since the last publish.")
        for table, path in published.items():
            print(f"Published {table}: {path}")
        return 0

    return _with_writer_lease(args, "publish", body)


# --- cdt run ------------------------------------------------------------------


def run_run(args: argparse.Namespace) -> int:
    """Run ``cdt run daily|historical|poll``."""
    reject_placeholder_secrets()
    start_runtime_watchdog(args.run_mode, args.max_runtime_hours)
    if args.run_mode == "poll":
        return run_poll(
            artifact_root=args.artifact_root,
            final_database_root=args.final_database_root,
            force=args.force,
            force_publish=args.force_publish,
            max_attempts=args.max_attempts,
            max_requests_per_batch=args.max_requests_per_batch,
            max_batch_bytes=args.max_batch_bytes,
            max_rows_per_job=args.max_rows_per_job,
        )
    if not args.cik_file:
        LOGGER.error("--cik-file (or CDT_DEFAULT_CIK_FILE) is required")
        return USAGE_EXIT_CODE
    try:
        config = _pipeline_config(args)
    except ValueError as exc:
        LOGGER.error("Invalid run arguments: %s", exc)
        return USAGE_EXIT_CODE
    if args.extractor_backend == "batch":
        return run_prepare_then_publish(config)
    code, result = run_live(config)
    if result is not None:
        _print_run_summary(result)
    return code


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a daily/historical run's config, resolving its dates.

    Raises:
        ValueError: On an invalid date window.
    """
    start_date, end_date = resolve_mode_dates(
        args.run_mode, args.start_date, args.end_date
    )
    return PipelineConfig(
        mode=args.run_mode,
        cik_file=args.cik_file,
        bucket=args.bucket,
        start_date=start_date,
        end_date=end_date,
        artifact_root=args.artifact_root,
        final_database_root=args.final_database_root,
        force=args.force,
        force_publish=args.force_publish,
        download=args.download,
        failure_file=args.failure_file,
        aws_profile=args.aws_profile,
        s3_prefix=args.s3_prefix,
        ingest_flush_rows=args.ingest_flush_rows,
        item_numbers=args.item_numbers,
        classifier_model_dir=args.model_dir,
        sixk_model_dir=args.sixk_model_dir,
        sixk_concurrency=args.concurrency,
        extractor_model=args.model,
        extractor_reasoning_effort=args.reasoning_effort,
        extractor_max_attempts=args.max_attempts,
        strong_match_threshold=args.strong_match_threshold,
        loose_match_threshold=args.loose_match_threshold,
        ambiguity_margin=args.ambiguity_margin,
        genres=args.genres,
    )


def _print_run_summary(result: PipelineRunResult) -> None:
    print(
        f"Ran {result.mode} from {result.start_date} through {result.end_date} "
        f"over genres {','.join(result.genres)}."
    )
    for genre, genre_result in result.genre_results.items():
        print(
            f"{genre}: ingested {genre_result.ingest.total_rows} documents, "
            f"segmented {genre_result.segmented_rows}, classified "
            f"{genre_result.classified_rows}."
        )
    print(
        f"Extracted {result.extracted_rows} and matched {result.matched_rows} "
        f"mentions into {result.debt_instrument_rows} debt instruments."
    )
    print(f"Artifact root: {result.artifact_root}.")
    if result.failed_genres:
        print(f"Failed genres: {','.join(result.failed_genres)}; see the log.")


if __name__ == "__main__":
    raise SystemExit(main())
