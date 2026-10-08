# Deployment

This repository deploys CDT as a single ECS Fargate processor with Pulumi-managed infrastructure and GitHub Actions-based delivery.

## Deployed Shape

The deployed stack contains:

- an ECR repository for the runtime image
- one ECS cluster and Fargate task definition
- task execution and runtime IAM roles
- a CloudWatch log group
- two EventBridge Scheduler schedules: a daily `cdt run daily` and an hourly `cdt run poll`
- SSM SecureString parameters holding `OPENAI_API_KEY` and `OPENROUTER_API_KEY` under `/idi/<env>/cdt/secrets/`

The ECS task runs the `cdt` console script (the image's entrypoint) from [dockerfiles/Dockerfile.orchestrator](../dockerfiles/Dockerfile.orchestrator); its container is named `cdt`. The image and ECR repository keep the name `orchestrator`.

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

`idi:source_prefix` must match `cdt.ingest.core.DEFAULT_S3_PREFIX` — Pulumi cannot import the
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

The daily scheduler target overrides the container command to `run daily`, and the hourly
poll scheduler target overrides it to `run poll`. Scheduled production runs are therefore
equivalent to:

```bash
cdt run daily   # daily schedule: prepare both genres + match/publish, submits no extract
cdt run poll    # hourly schedule: advances the OpenAI batch extract job one step
```

Every `cdt` option defaults from its flag, then the environment variable above
(`ARTIFACT_ROOT`, `FINAL_DATABASE_ROOT`, `BUCKET_NAME`, `CDT_DEFAULT_CIK_FILE`, and
optionally `GENRES` and `EXTRACTOR_BACKEND`), then the built-in default.

`daily` and `historical` both use the OpenAI batch extract backend by default and
defer extraction to the poller — a historical backfill's classified items are claimed
by the next poll tick. Historical runs are never scheduled automatically. Pass
`--extractor-backend live` for the synchronous OpenRouter pipeline
that extracts within the run itself.

### The 6-K chain

Every `daily` and `historical` run prepares both filing genres (the hourly `poll`
run only advances the batch extract job), each as ingest → segment → classify, for
the one CIK list the run is given. For 6-K, segment writes window spans and
classify is the two-stage triage. The task definition sets none of the 6-K settings, so
the defaults below are what runs in production. Each can be set as an environment
variable on the task, or passed as the matching flag.

| Setting | Default | Effect |
|---|---|---|
| `GENRES` / `--genres` | `8-K,6-K` | Restrict a run to one genre. |
| `SIXK_TRIAGE_PROVIDER` | `openrouter` | Stage-2 triage backend: `openrouter` (uses `OPENROUTER_API_KEY`) or `openai` (uses `OPENAI_API_KEY`). |
| `SIXK_TRIAGE_MODEL` | `openai/gpt-5.6-luna` | Stage-2 model, as an OpenRouter slug. |
| `SIXK_TRIAGE_REASONING` | `none` | Stage-2 reasoning effort. |

**Cost:** unlike the 8-K prepare chain, 6-K classify is not free. The `daily` run makes one
synchronous LLM call per 6-K filing that has windows admitted by the local stage-1 model.
That call is billed to the triage provider's account, outside the OpenAI batch discount.
See [sixk-two-stage-triage.md](sixk-two-stage-triage.md) for measured cost per filing.
Set `GENRES=8-K` to turn the chain off.

## CI/CD Flow

GitHub Actions defines two operational workflows, both thin callers
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

- `cik_scope`: what a run covers when no `--cik-file` is given. Either the
  bucket-relative key of a one-CIK-per-line file, or `all` for every filer.
  It is required, so a stack can never default to `all` by omission. During
  the beta both stacks use `processors/cdt/inputs/ciks/beta-1k.txt`. At the
  production launch prod moves to `all` and dev keeps a beta list; after
  that the beta lists can be deleted from S3, because nothing in the code
  names them.

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
final parquet snapshots under `final_database_prefix`, and the website publisher
([dsi-rse/commercial-debt-tracker-website](https://github.com/dsi-rse/commercial-debt-tracker-website))
reads those and updates R2.

The container's ingest source bucket is the SSM `processor_bucket_name` value. The
artifact and final-database roots are derived:

```text
s3://<output_bucket_name or processor bucket>/<artifact_prefix>
s3://<output_bucket_name or processor bucket>/<final_database_prefix>
```

and the default CIK file (`CDT_DEFAULT_CIK_FILE` on the task) is
`s3://<processor bucket>/<cik_scope>`, or `all` when `cik_scope` is `all`. The task
role gets a read grant on that one file, and none when the scope is `all`.

## Daily Operations

Normal daily processing is:

1. GitHub deploys code and infrastructure
2. the daily EventBridge Scheduler runs one ECS task that executes `cdt run daily`:
   ingest → segment → classify for both genres over the 5 filing dates ending yesterday,
   then match + publish on existing mentions. It submits no extract batch itself.
3. the hourly EventBridge Scheduler runs `cdt run poll`, which starts an OpenAI
   batch extract job when classified work is pending and advances it one step per tick
   (extraction can span multiple hours/days per its 24h batch windows)
4. when an extract job completes, that poll tick writes new `mentions` partitions and
   re-runs match + finalize
5. outputs land under the configured artifact root in S3, and if `FINAL_DATABASE_ROOT` is
   set, finalize publishes in two layers. Consistent generations live under the
   **artifact root**: the four tables are written to an immutable
   `<artifact root>/final-snapshots/snapshot=<run_id>/<table>.parquet` prefix, then a
   single `latest.json` pointer there (run id, schema version, per-table row counts and
   paths) is replaced as the last, atomic step — resolve the pointer to read a
   consistent generation across all four tables. The **final database root stays
   parquet-only**: `<table>/latest.parquet` per table, refreshed after the pointer; each
   object is individually atomic but the set is not consistent mid-publish. A publish
   that would shrink a table below half its prior row count (or empty it) is refused
   unless forced. Only the current and prior generations are retained.
6. the website publisher ([dsi-rse/commercial-debt-tracker-website](https://github.com/dsi-rse/commercial-debt-tracker-website)) reads the final
   database root's parquet and publishes `generated/*` JSON to R2; if it needs
   cross-table consistency, it should resolve the `latest.json` pointer instead.

Because extraction is asynchronous, final snapshots for a given filing date can lag the
daily run by up to a few days. The daily run still refreshes match/final outputs from
whatever mentions already exist, so previously extracted instruments stay current.

Local note: `cdt run` and `cdt publish` read `FINAL_DATABASE_ROOT`, or take
`--final-database-root`; without either, nothing is published.

The scheduler state is controlled by the Pulumi `idi:schedule_enabled` setting
(`idi:poll_schedule_enabled` overrides it for the poll schedule alone). As
committed, the daily schedule is disabled in both stacks and the hourly poll is
enabled in `dev` only (`pulumi/Pulumi.<stack>.yaml`); check those before
assuming a scheduled run happened.

### Run-time limits and batch tuning

These flags are rarely needed; the defaults are what the schedules use.

| Flag | Applies to | Default | What it bounds |
|---|---|---|---|
| `--max-runtime-hours` | `cdt run daily\|historical\|poll` | poll 2, daily 12, historical 72 | Wall-clock deadline. Past it the runtime watchdog exits the task with code 70, without releasing the lease (its TTL recovers it). |
| `--max-rows-per-job` | `cdt run poll` | 10,000 | Rows one batch extract job may claim, so the job's state fits the poll task's memory. Pending rows beyond it wait for the next job. |
| `--max-requests-per-batch` | `cdt run poll` | 40,000 | Requests per OpenAI batch input file. |
| `--max-batch-bytes` | `cdt run poll` | 100 MiB | Bytes per OpenAI batch input file. |
| `--max-attempts` | `cdt run poll`, `cdt extract`, `cdt run … --extractor-backend live` | 3 | Scored attempts per extractor stage per row. |

`--force` reprocesses partitions the completion registries already record. On a
batch-backend `daily`/`historical` run it applies to the prepare and match stages
only; to force a re-extract, run `cdt run poll --force` while no job is active.
`--force` never lowers the publish guards. To publish when no source changed, or
past the shrinkage guard (a table falling below half its published rows), pass
`--force-publish`, on `cdt publish` or any `cdt run` mode.

## Monitoring and Response

With `idi:alerts_enabled` on, every alarm notifies the `idi:alert_email` SNS
subscription (topic ARN is the `alerts_topic_arn` stack output). What each
alarm means and what to do:

| Alarm | Meaning | First response |
|---|---|---|
| `*-poll-liveness` | No poll tick completed for 6h; extraction is stalled. | Check the poll schedule state and the latest task logs; a wedged holder shows up as repeated `locked` ticks. |
| `*-daily-heartbeat` | No `daily` run completed for 24h. A run where one genre's prepare chain failed still publishes the others but counts as not completed (it exits nonzero without the heartbeat line). | Check the daily schedule, the task-failure alerts, and the scheduler DLQ. Search the task log for `Genre prepare failed` to see whether one genre failed. |
| `*-task-failures` | An ECS task exited nonzero or failed to start (includes OOM kills, exit 137, and the runtime watchdog's exit 70: a run past its deadline of 2h poll, 12h daily, 72h historical). | Read the task's log stream; OOM usually means a backfill outgrew `idi:memory`; exit 70 logs `Runtime watchdog expired` — find what the run was stuck on, or split a long historical run. |
| `*-job-stall` | The active extract job has run ~4 days of ticks without finishing; it blocks all newer filings. | `cdt extract job show`; if genuinely wedged, `cdt extract job reset --yes` (abandons in-flight batches). |
| `*-lease-theft` | A run died (or overran its TTL) still holding the writer lease. | Find the previous holder's logs; its partial work is recomputed by the next run, but check why it died. |
| `*-dlq-depth` | The shared scheduler DLQ has messages: a RunTask invocation failed after retries. | Inspect the queue; the message may belong to another processor sharing the DLQ. |

The log-literal → metric-filter couplings ("Poll tick complete",
"Run complete: mode=daily", "Extract job stalled", "Stole lease")
are annotated at both ends; change them together.

## Historical Backfills

Historical runs are manual and admin-driven by design. The GitHub deploy role
cannot call `ecs:RunTask`, and granting it in the shared bootstrap policy was
judged not worth it for the handful of runs historical will ever see (#108) --
a `run-historical.yml` workflow existed but never worked and has been removed.

Use the launcher script with admin credentials:

```bash
export PULUMI_CONFIG_PASSPHRASE='<from the Core Facility Bitwarden>'
./scripts/run-historical.sh --stack dev \
  --start-date 2024-01-01 --end-date 2024-01-31 \
  --cik-file s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/inputs/ciks/beta-1k.txt
```

It resolves the cluster, task definition, subnet, and security group from
Pulumi stack outputs, builds the same container overrides the workflow did
(`--force` and `--extractor-backend batch|live` are optional flags), echoes a
one-line launch record (parameters plus caller ARN) for your shell history,
and prints the task ARN and a log-tail command. See
[docs/deployment-dev.md](deployment-dev.md) for the underlying `aws ecs
run-task` pattern the script wraps.

## Operational Guidance

- Keep the scheduler disabled on a new environment until a manual historical smoke test succeeds.
- Treat the configured default CIK file as the environment's normal run scope.
- Use a smaller CIK file and narrow date range for first backfills.
- Prefer `--force` only when intentionally recomputing existing partitions.
- Reserve `--force-publish` for a deliberate overwrite of the published tables,
  after checking why the shrinkage guard refused; `run-historical.sh` never
  passes it.

## Prod Launch Checklist

Prod stays dark until this list is walked in order. The gate variable
(`PROD_INFRA_READY=false`), the `prod` GitHub environment, and the committed
`pulumi/Pulumi.prod.yaml` already exist, so nothing here is urgent before
launch day — but do it in order then.

1. **Shared prerequisites** (owned outside this repo): the shared stack
   publishes `/idi/prod/shared/processor_bucket_name` and
   `/idi/prod/shared/dlq_name`; `pulumi-bootstrap` provisions the prod OIDC
   deploy-role pair; the `prod` GitHub environment gets `AWS_ROLE_ARN_DEPLOY`
   and `PULUMI_CONFIG_PASSPHRASE`.
2. **Initialize the stack**: `make infra-login PULUMI_STACK=prod` (each stack
   has its own state bucket, `idi-ftm2j-<stack>-pulumi-state`), then
   `pulumi stack init prod` (creates the stack record and the
   `encryptionsalt` — until this runs, any prod deploy dies at
   `pulumi stack select prod`). Creates no AWS resources.
3. **Set the real API keys** (SecureStrings, picked up at task launch):
   `aws ssm put-parameter --name /idi/prod/cdt/secrets/openai_api_key ...`
   and `.../openrouter_api_key`, per the secrets section above.
4. **Enable alerting**: uncomment `idi:alerts_enabled: "true"` in
   `Pulumi.prod.yaml` (requires the deploy-role statements from
   dsi-rse/idi-ftm2j-shared#79; enabling earlier fails on
   `sns:CreateTopic`). Verify `idi:alert_email` is the address that should
   be paged.
5. **Flip the gate**: `gh variable set PROD_INFRA_READY --body "true"`, then
   deploy (a `main` release, or `make infra-up PULUMI_STACK=prod`). Confirm
   the SNS subscription email after the deploy.
6. **Smoke test before any schedule**: run
   `./scripts/run-historical.sh --stack prod ...` with a small CIK file and a
   narrow date range (admin credentials; see Historical Backfills). Verify document/
   item/classification partitions appear under `processors/cdt/`, a manual
   (or scheduled) poll tick drains the extract job, `database/cdt/` holds
   exactly the four `latest.parquet` tables, and
   `processors/cdt/final-snapshots/latest.json` points at a consistent
   generation with sane row counts.
7. **Enable schedules**: set `idi:poll_schedule_enabled: "true"` first and
   watch a few ticks (the poller alone drains extraction), then
   `idi:schedule_enabled: "true"` for the daily run. Confirm the poll-liveness
   and daily-heartbeat alarms settle into OK.
8. **Leave the beta scope**: when prod should cover every filer, set
   `idi:cik_scope: all` in `Pulumi.prod.yaml` and deploy. The task then reads
   no CIK file and loses its read grant on the beta list. Size the first run
   on all filers deliberately: run a bounded `run-historical.sh` backfill
   first rather than letting the daily schedule discover the new scope.

## Dev First-Deploy Walkthrough

For the concrete `dev` stack bootstrap flow, recommended config values, and an example manual backfill command, see [docs/deployment-dev.md](deployment-dev.md).
