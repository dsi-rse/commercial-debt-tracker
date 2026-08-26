# Deployment

This repository deploys CDT as a single ECS Fargate processor with Pulumi-managed infrastructure and GitHub Actions-based delivery.

## Deployed Shape

The deployed stack contains:

- an ECR repository for the runtime image
- one ECS cluster and Fargate task definition
- task execution and runtime IAM roles
- a CloudWatch log group
- two EventBridge Scheduler schedules: a daily `cdt-orchestrator daily` and an hourly `cdt-orchestrator poll`
- SSM SecureString parameters holding `OPENAI_API_KEY` and `OPENROUTER_API_KEY` under `/idi/<env>/cdt/secrets/`

The ECS task runs the `cdt-orchestrator` console script from [dockerfiles/Dockerfile.orchestrator](../dockerfiles/Dockerfile.orchestrator).

### Task role S3 scope

The scraper's source data and CDT's own artifacts can live in the same bucket (they do in
`dev`), so the task role scopes object permissions **by prefix rather than by bucket**:

| Prefix | Permission |
| --- | --- |
| `{source_prefix}/` (default `sec/`) | `GetObject` only — scraper-owned, CDT never writes here |
| `{artifact_prefix}/` (default `processors/cdt/`) | read + write + delete + multipart |
| `{final_database_prefix}/` (default `database/cdt/`) | read + write + delete + multipart |

`s3:ListBucket` remains bucket-level: it is a bucket-level action that can only be
narrowed with an `s3:prefix` condition, not a resource path, and an incomplete prefix list
would produce silent empty listings rather than an error.

`idi:source_prefix` must match `cdt.ingest.DEFAULT_S3_PREFIX` — Pulumi cannot import the
package, so the two are coupled by convention. A mismatch denies every read ingest
attempts.

## Runtime Entry Point

The task definition injects these environment variables directly:

- `AWS_REGION`
- `BUCKET_NAME`
- `ARTIFACT_ROOT`
- `FINAL_DATABASE_ROOT`
- `CDT_DEFAULT_CIK_FILE`
- `PYTHONUNBUFFERED`

It injects these secrets, by SSM parameter ARN, so the values never appear in the
task definition:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

The daily scheduler target overrides the container command to `daily`, and the hourly
poll scheduler target overrides it to `poll`. Scheduled production runs are therefore
equivalent to:

```bash
cdt-orchestrator daily   # daily schedule: ingest/itemize/classify + match/finalize, submits no extract
cdt-orchestrator poll    # hourly schedule: advances the OpenAI batch extract job one step
```

`daily` and `historical` both use the OpenAI batch extract backend by default and
defer extraction to the poller — a historical backfill's classified items are claimed
by the next poll tick. Historical runs are never scheduled automatically. Pass
`--extractor-backend live` (before the mode) for the synchronous OpenRouter pipeline
that extracts within the run itself.

## CI/CD Flow

GitHub Actions defines three operational workflows. The first two are thin callers
of the shared compositions in
[dsi-rse/idi-ftm2j-shared](https://github.com/dsi-rse/idi-ftm2j-shared), pinned to
an exact release tag; upgrading the pipeline means bumping that one `@vX.Y.Z` pin.

- `.github/workflows/checks.yml` → `pipeline-checks.yml`
  On pull requests: `Lint`, `Test`, `Security` (pip-audit + CodeQL), and
  `Pulumi Preview`. Those four job names are the required checks on the `dev` and
  `main` rulesets.
- `.github/workflows/deploy.yml` → `pipeline-docker.yml`
  On pushes to `dev` and `main`: version → build/push image to GHCR →
  `pulumi up` → sync the image to ECR. A `main` push also commits the patch
  version bump, cuts a tag and GitHub Release, and merges `main` back into `dev`.
- `.github/workflows/run-historical.yml`
  CDT-specific: starts an ECS task for a manual historical backfill.

Points worth knowing about the shared flow:

- Pulumi owns the AWS infrastructure and the ECR repository; GHCR is the build
  target and the ECR sync makes the image available to ECS, so Pulumi never builds
  an image. The ECR repository name is `{pulumi_project}-{env}-{app}-orchestrator`,
  because that is what the sync job pushes to.
- `idi:app_name` is not committed to the stack files — the pipeline sets it from
  the `app-name` caller input (`cdt`).
- A push to `main` deploys the prod stack only when the `PROD_INFRA_READY`
  repository variable is `"true"`; otherwise it still versions, releases, and
  pushes to GHCR, and skips the prod deploy and ECR sync. Dev always deploys.
- The pipeline's own version-bump and merge-back commits are authored by
  `idi-deploy-bot`, and the version job skips commits from that committer, which is
  what stops a deploy from triggering another deploy.

## Pulumi Configuration

Values live in one of three places, per the shared
[onboarding standard](https://github.com/dsi-rse/idi-ftm2j-shared/blob/dev/docs/onboarding-a-processor.md).

**Read from SSM, not configured here.** The shared stack publishes these, and
`pulumi/infra/config.py` reads them at plan time:

| Parameter | Used as |
| --- | --- |
| `/idi/<env>/shared/processor_bucket_name` | the ingest source bucket, and the default output bucket |
| `/idi/<env>/shared/dlq_name` | the scheduler dead-letter queue |

**Genuine secrets** are SSM `SecureString` parameters that Pulumi creates with a
placeholder value and never manages thereafter (`ignore_changes`). Set the real
value out-of-band, once per environment:

```bash
aws ssm put-parameter --name /idi/dev/cdt/secrets/openai_api_key \
  --type SecureString --value '<key>' --overwrite
aws ssm put-parameter --name /idi/dev/cdt/secrets/openrouter_api_key \
  --type SecureString --value '<key>' --overwrite
```

Rotation is another `put-parameter --overwrite`, picked up at the next task
launch — no deploy needed. Both keys also belong in the Core Facility Bitwarden.

**Committed `idi:` config** in `pulumi/Pulumi.dev.yaml` and `pulumi/Pulumi.prod.yaml`.
Required:

- `default_cik_key` — bucket-relative key of the default CIK universe

Optional:

- `output_bucket_name` (defaults to the shared processor bucket)
- `artifact_prefix` (default `processors/cdt`)
- `final_database_prefix` (default `database/cdt`)
- `source_prefix` (default `sec`)
- `app_name` (set by the pipeline from the caller input)
- `cpu`, `memory`
- `cron` (daily schedule; default `cron(0 8 * * ? *)`)
- `poll_cron` (hourly extract poll; default `cron(30 * * * ? *)`, offset from the daily run)
- `schedule_enabled` (gates the daily schedule; also the poll default)
- `poll_schedule_enabled` (gates the hourly poll on its own — the poller is the
  only driver of batch extraction, so it can be enabled to drain a manual
  historical run while the daily schedule stays off)
- `log_retention_days`
- `ecr_image_retention_count`
- `alerts_enabled` (gates every alarm/SNS resource; requires the deploy-role
  statements from dsi-rse/idi-ftm2j-shared#79 — enabling earlier fails the
  deploy with AccessDenied on `sns:CreateTopic`)
- `alert_email` (SNS subscription for every alarm; **required** when
  `alerts_enabled` is true — the deploy fails fast rather than creating alarms
  that notify nobody)

CDT does not publish Cloudflare R2 JSON, and no longer carries R2 config: it writes
final parquet snapshots under `final_database_prefix`, and the publisher stack in
`../commercial-debt-tracker-dashboard` reads those and updates R2.

The container's ingest source bucket is the SSM `processor_bucket_name` value. The
artifact and final-database roots are derived:

```text
s3://<output_bucket_name or processor bucket>/<artifact_prefix>
s3://<output_bucket_name or processor bucket>/<final_database_prefix>
```

and the default CIK file is `s3://<processor bucket>/<default_cik_key>`.

## Daily Operations

Normal daily processing is:

1. GitHub deploys code and infrastructure
2. the daily EventBridge Scheduler runs one ECS task that executes `cdt-orchestrator daily`:
   ingest → itemize → classify, then match + finalize on existing mentions. It submits no
   extract batch itself.
3. the hourly EventBridge Scheduler runs `cdt-orchestrator poll`, which starts an OpenAI
   batch extract job when classified work is pending and advances it one step per tick
   (extraction can span multiple hours/days per its 24h batch windows)
4. when an extract job completes, that poll tick writes new `mentions` partitions and
   re-runs match + finalize
5. outputs land under the configured artifact root in S3, and if `FINAL_DATABASE_ROOT` is
   set, finalize publishes one atomic snapshot generation: the four tables are written to
   an immutable `snapshots/snapshot=<run_id>/<table>.parquet` prefix, then a single
   `latest.json` pointer (run id, schema version, per-table row counts and paths) is
   replaced as the last step. Consumers should resolve the pointer, never the prefix —
   that guarantees a consistent generation across all four tables. A publish that would
   shrink a table below half its prior row count (or empty it) is refused unless forced.
   Only the current and prior generations are retained.
6. the dashboard publisher in `../commercial-debt-tracker-dashboard` can then read those
   snapshots and publish `generated/*` JSON to R2. Until it migrates to `latest.json`,
   the deprecated per-table `<table>/latest.parquet` objects are still refreshed after
   the pointer; they are individually atomic but not consistent as a set.

Because extraction is asynchronous, final snapshots for a given filing date can lag the
daily run by up to a few days. The daily run still refreshes match/final outputs from
whatever mentions already exist, so previously extracted instruments stay current.

Local note:

- `cdt pipeline` does not read `FINAL_DATABASE_ROOT`; pass `--final-database-root` to write final snapshots from that CLI.
- `cdt-orchestrator` reads `FINAL_DATABASE_ROOT`, and also accepts `--final-database-root` as a top-level option before `daily` or `historical`.

The scheduler state is controlled by the Pulumi `idi:schedule_enabled` setting
(`idi:poll_schedule_enabled` overrides it for the poll schedule alone).

`--force` on a batch-backend `daily`/`historical` run applies to the prepare and
match/finalize stages only; to force a re-extract, run a poll tick with
`--force` while no job is active.

## Monitoring and Response

With `idi:alerts_enabled` on, every alarm notifies the `idi:alert_email` SNS
subscription (topic ARN is the `alerts_topic_arn` stack output). What each
alarm means and what to do:

| Alarm | Meaning | First response |
|---|---|---|
| `*-poll-liveness` | No poll tick completed for 6h; extraction is stalled. | Check the poll schedule state and the latest task logs; a wedged holder shows up as repeated `locked` ticks. |
| `*-daily-heartbeat` | No `daily` run completed for 24h. | Check the daily schedule, the task-failure alerts, and the scheduler DLQ. |
| `*-task-failures` | An ECS task exited nonzero or failed to start (includes OOM kills, exit 137). | Read the task's log stream; OOM usually means a backfill outgrew `idi:memory`. |
| `*-job-stall` | The active extract job has run ~4 days of ticks without finishing; it blocks all newer filings. | `cdt show-extract-job`; if genuinely wedged, `cdt reset-extract-job --yes` (abandons in-flight batches). |
| `*-lease-theft` | A run died (or overran its TTL) still holding the writer lease. | Find the previous holder's logs; its partial work is recomputed by the next run, but check why it died. |
| `*-dlq-depth` | The shared scheduler DLQ has messages: a RunTask invocation failed after retries. | Inspect the queue; the message may belong to another processor sharing the DLQ. |

The log-literal → metric-filter couplings ("Poll tick complete",
"Orchestrator run complete: mode=daily", "Extract job stalled", "Stole lease")
are annotated at both ends; change them together.

## Historical Backfills

There are two supported historical paths:

1. GitHub Actions
   Use `.github/workflows/run-historical.yml` and provide `stack`, `start_date`, `end_date`, `cik_file`, and optional `force`.
2. Manual AWS CLI
   Run `aws ecs run-task` with a container override command beginning with `historical`.

The GitHub workflow uses Pulumi outputs to resolve the cluster, task definition, subnet, and security group before launching the task.
For manual AWS CLI runs, export `PULUMI_CONFIG_PASSPHRASE` before reading Pulumi outputs; the passphrase is stored in the Core Facility Bitwarden.
See [docs/deployment-dev.md](deployment-dev.md) for a complete command that logs into the S3 Pulumi backend, selects the stack, and starts an ECS historical task.

## Operational Guidance

- Keep the scheduler disabled on a new environment until a manual historical smoke test succeeds.
- Treat the configured default CIK file as the environment's normal run scope.
- Use a smaller CIK file and narrow date range for first backfills.
- Prefer `--force` only when intentionally recomputing existing partitions.

## Dev First-Deploy Walkthrough

For the concrete `dev` stack bootstrap flow, recommended config values, and an example manual backfill command, see [docs/deployment-dev.md](deployment-dev.md).
