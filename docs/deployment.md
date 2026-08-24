# Deployment

This repository deploys CDT as a single ECS Fargate processor with Pulumi-managed infrastructure and GitHub Actions-based delivery.

## Deployed Shape

The deployed stack contains:

- an ECR repository for the runtime image
- one ECS cluster and Fargate task definition
- task execution and runtime IAM roles
- a CloudWatch log group
- two EventBridge Scheduler schedules: a daily `cdt-orchestrator daily` and an hourly `cdt-orchestrator poll`
- Secrets Manager secrets for `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `SEC_USER_AGENT`

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

It injects these secrets:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `SEC_USER_AGENT`

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

GitHub Actions currently defines three operational workflows:

- `.github/workflows/checks.yml`
  Runs Ruff, pytest, dependency and CodeQL security checks, and `pulumi preview` on pull requests and manual dispatch.
- `.github/workflows/deploy.yml`
  Runs on pushes to `main` and `dev`, plus manual dispatch.
- `.github/workflows/run-historical.yml`
  Starts an ECS task for a manual historical backfill.

`deploy.yml` performs three steps:

1. build and push the orchestrator image to GHCR as `ghcr.io/<owner>/cdt-orchestrator`
2. run `pulumi up` against the `prod` stack for `main` or the `dev` stack for `dev`
3. copy the GHCR image into the Pulumi-managed ECR repository as both `:latest` and `:${GITHUB_SHA}`

This split matters:

- Pulumi owns the AWS infrastructure and the ECR repository name
- GHCR is the primary CI build target
- the final ECR sync makes the image available to ECS without requiring Pulumi to build Docker images itself

## Pulumi Configuration

Required `idi:` Pulumi config:

- `bucket_name`
- `default_cik_file`
- `shared_dlq_name`
- `openai_api_key` as a secret
- `openrouter_api_key` as a secret
- `sec_user_agent` as a secret

Set the OpenAI key with:

```bash
pulumi config set --secret idi:openai_api_key <key>
```

Common optional config:

- `output_bucket_name`
- `artifact_prefix`
- `final_database_prefix`
- `app_name`
- `cpu`
- `memory`
- `cron` (daily schedule; default `cron(0 8 * * ? *)`)
- `poll_cron` (hourly extract poll; default `cron(30 * * * ? *)`, offset from the daily run)
- `schedule_enabled` (gates both the daily and poll schedules)
- `log_retention_days`
- `ecr_image_retention_count`

The Pulumi code still contains optional R2 passthrough config from an older deployment shape. Those values are not used by CDT for publishing; configure the dashboard publisher stack in `../commercial-debt-tracker-dashboard` when R2 JSON publishing is needed.

The ingest source bucket passed to the container is:

```text
<bucket_name>
```

The artifact root passed to the container is derived from:

```text
s3://<output_bucket_name or bucket_name>/<artifact_prefix>
```

The final database root passed to the container is derived from:

```text
s3://<output_bucket_name or bucket_name>/<final_database_prefix>
```

If `artifact_prefix` is omitted, Pulumi defaults it to `processors/cdt`.

If `final_database_prefix` is omitted, Pulumi defaults it to `database/cdt`.

If `output_bucket_name` is omitted, Pulumi reuses `bucket_name` for outputs.

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
   set, four final parquet snapshots are written: `items/latest.parquet`,
   `debt-instruments/latest.parquet`, `debt-instrument-mentions/latest.parquet`, and
   `mention-cluster-edges/latest.parquet`
6. the dashboard publisher in `../commercial-debt-tracker-dashboard` can then read those parquet snapshots and publish `generated/*` JSON to R2

Because extraction is asynchronous, final snapshots for a given filing date can lag the
daily run by up to a few days. The daily run still refreshes match/final outputs from
whatever mentions already exist, so previously extracted instruments stay current.

Local note:

- `cdt pipeline` does not read `FINAL_DATABASE_ROOT`; pass `--final-database-root` to write final snapshots from that CLI.
- `cdt-orchestrator` reads `FINAL_DATABASE_ROOT`, and also accepts `--final-database-root` as a top-level option before `daily` or `historical`.

The scheduler state is controlled by the Pulumi `idi:schedule_enabled` setting.

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
