# Deployment

This repository deploys CDT as a single ECS Fargate processor with Pulumi-managed infrastructure and GitHub Actions-based delivery.

## Deployed Shape

The deployed stack contains:

- an ECR repository for the runtime image
- one ECS cluster and Fargate task definition
- task execution and runtime IAM roles
- a CloudWatch log group
- one EventBridge Scheduler schedule for daily runs
- Secrets Manager secrets for `OPENROUTER_API_KEY` and `SEC_USER_AGENT`
- optional Secrets Manager secrets for Cloudflare R2 publishing credentials

The ECS task runs the `cdt-orchestrator` console script from [dockerfiles/Dockerfile.orchestrator](../dockerfiles/Dockerfile.orchestrator).

## Runtime Entry Point

The task definition injects these environment variables directly:

- `AWS_REGION`
- `BUCKET_NAME`
- `ARTIFACT_ROOT`
- `FINAL_DATABASE_ROOT`
- `CDT_DEFAULT_CIK_FILE`
- `PYTHONUNBUFFERED`

It injects these secrets:

- `OPENROUTER_API_KEY`
- `SEC_USER_AGENT`

When R2 publishing is configured, it also injects:

- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`
- `R2_OBJECT_PREFIX`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

The scheduler target overrides the container command to:

```text
daily
```

That means scheduled production runs are always equivalent to:

```bash
cdt-orchestrator daily
```

Historical runs are never scheduled automatically.

## CI/CD Flow

GitHub Actions currently defines three operational workflows:

- `.github/workflows/checks.yml`
  Runs Ruff, pytest, and `pulumi preview` on pull requests and manual dispatch.
- `.github/workflows/deploy.yml`
  Runs on pushes to `main` and `dev`, plus manual dispatch.
- `.github/workflows/run-historical.yml`
  Starts an ECS task for a manual historical backfill.

`deploy.yml` performs three steps:

1. build and push the orchestrator image to GHCR as `ghcr.io/<owner>/cdt-orchestrator`
2. run `pulumi up` against the `prod` stack for `main` or the `dev` stack for `dev`
3. copy the GHCR `:latest` image into the Pulumi-managed ECR repository

This split matters:

- Pulumi owns the AWS infrastructure and the ECR repository name
- GHCR is the primary CI build target
- the final ECR sync makes the image available to ECS without requiring Pulumi to build Docker images itself

## Pulumi Configuration

Required `idi:` Pulumi config:

- `bucket_name`
- `default_cik_file`
- `shared_dlq_name`
- `openrouter_api_key` as a secret
- `sec_user_agent` as a secret

Common optional config:

- `artifact_prefix`
- `final_database_prefix`
- `app_name`
- `cpu`
- `memory`
- `cron`
- `schedule_enabled`
- `log_retention_days`
- `r2_account_id`
- `r2_bucket_name`
- `r2_object_prefix`
- `r2_access_key_id` as a secret
- `r2_secret_access_key` as a secret

The artifact root passed to the container is derived from:

```text
s3://<bucket_name>/<artifact_prefix>
```

The final database root passed to the container is derived from:

```text
s3://<bucket_name>/<final_database_prefix>
```

If `artifact_prefix` is omitted, Pulumi defaults it to `processors/cdt`.

If `final_database_prefix` is omitted, Pulumi defaults it to `database/cdt`.

## Daily Operations

Normal daily processing is:

1. GitHub deploys code and infrastructure
2. EventBridge Scheduler runs one ECS task per day
3. the task executes `cdt-orchestrator daily`
4. outputs land under the configured artifact root in S3
5. if `FINAL_DATABASE_ROOT` is set, four final parquet snapshots are written:
   `items/latest.parquet`, `debt-instruments/latest.parquet`,
   `debt-instrument-mentions/latest.parquet`, and
   `mention-cluster-edges/latest.parquet`
6. if R2 credentials are present, dashboard snapshot JSON is published after matching

Local note:

- local runs do not write final snapshots unless you pass `--final-database-root` or set `FINAL_DATABASE_ROOT`

The scheduler state is controlled by the Pulumi `idi:schedule_enabled` setting.

## Historical Backfills

There are two supported historical paths:

1. GitHub Actions
   Use `.github/workflows/run-historical.yml` and provide `stack`, `start_date`, `end_date`, `cik_file`, and optional `force`.
2. Manual AWS CLI
   Run `aws ecs run-task` with a container override command beginning with `historical`.

The GitHub workflow uses Pulumi outputs to resolve the cluster, task definition, subnet, and security group before launching the task.

## Operational Guidance

- Keep the scheduler disabled on a new environment until a manual historical smoke test succeeds.
- Treat the configured default CIK file as the environment's normal run scope.
- Use a smaller CIK file and narrow date range for first backfills.
- Prefer `--force` only when intentionally recomputing existing partitions.

## Dev First-Deploy Walkthrough

For the concrete `dev` stack bootstrap flow, recommended config values, and an example manual backfill command, see [docs/deployment-dev.md](deployment-dev.md).
