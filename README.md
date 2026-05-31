# Commercial Debt Tracker

Commercial Debt Tracker (CDT) is a file-native processor for finding debt-related disclosures in SEC 8-K filings, extracting structured debt-instrument mentions, and matching those mentions into instrument histories.

CDT follows the same deployment pattern as the other IDI / FTM2J processors:

- one ECS Fargate task running a container from ECR
- daily EventBridge Scheduler trigger
- historical runs via ECS command overrides
- durable state in S3
- optional dashboard snapshot publishing to Cloudflare R2
- shared logging and failure handling via `idi-ftm2j-shared`

## Runtime model

CDT does not use SQLite or any other database. Canonical state is stored as Parquet partitions and run metadata under a single artifact root, typically:

```text
s3://<bucket>/<artifact-prefix>
```

Production and beta filing scope are controlled by a CIK file stored in S3. The deployed orchestrator reads the default path from `CDT_DEFAULT_CIK_FILE`, and any run can override it with `--cik-file`.

Recommended input layout:

```text
s3://<bucket>/<artifact-prefix>/inputs/ciks/all.txt
s3://<bucket>/<artifact-prefix>/inputs/ciks/beta-100k.txt
s3://<bucket>/<artifact-prefix>/inputs/ciks/beta-10k.txt
s3://<bucket>/<artifact-prefix>/inputs/ciks/beta-1k.txt
```

## Canonical datasets

```text
<artifact-root>/
  documents/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
  items/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
  classifications/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
  mentions/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
  mention-matches/cik_shard=NNNN/part-0000.parquet
  debt-instruments/cik_shard=NNNN/part-0000.parquet
  extractor-runs/run_id=<run_id>/full.jsonl
  runs/<stage>/run_id=<run_id>.json
  failures/<stage>/failures.json
```

Completion is determined by final partition presence, not by mutable status rows.

## CLI

Local stage commands operate against a local artifact root:

```bash
uv run cdt ingest --artifact-root ./data daily ./ciks.txt
uv run cdt itemize --artifact-root ./data
uv run cdt classifier --artifact-root ./data --model-dir ./models/classifier/tfidf-linear-svc
uv run cdt extractor --artifact-root ./data
uv run cdt matcher --artifact-root ./data
uv run cdt pipeline --artifact-root ./data daily ./ciks.txt
```

The ECS-oriented entrypoint is:

```bash
uv run cdt-orchestrator daily
uv run cdt-orchestrator historical --start-date 2024-01-01 --end-date 2024-12-31
```

The orchestrator expects:

- `ARTIFACT_ROOT`
- `BUCKET_NAME`
- `CDT_DEFAULT_CIK_FILE`
- `OPENROUTER_API_KEY`
- `SEC_USER_AGENT`

When R2 publishing is enabled, the task also receives:

- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`
- `R2_OBJECT_PREFIX`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

The publisher builds dashboard JSON under `generated/` and only writes objects whose
content changed.

## Deployment

This repo includes in-repo deployment scaffolding aligned to the other processors:

- `dockerfiles/Dockerfile.orchestrator`
- `.github/workflows/checks.yml`
- `.github/workflows/deploy.yml`
- `pulumi/`

Pulumi provisions:

- ECR repository
- ECS cluster and task definition
- task execution and runtime IAM roles
- CloudWatch log group
- EventBridge schedule
- Secrets Manager secrets for `OPENROUTER_API_KEY` and `SEC_USER_AGENT`
- optional Secrets Manager secrets for Cloudflare R2 upload credentials

Required Pulumi config:

- `idi:bucket_name`
- `idi:default_cik_file`
- `idi:shared_dlq_name`
- `idi:openrouter_api_key` as secret
- `idi:sec_user_agent` as secret

Optional Pulumi config:

- `idi:artifact_prefix`
- `idi:cpu`
- `idi:memory`
- `idi:cron`
- `idi:schedule_enabled`
- `idi:r2_account_id`
- `idi:r2_bucket_name`
- `idi:r2_object_prefix`
- `idi:r2_access_key_id` as secret
- `idi:r2_secret_access_key` as secret

For the full first-deploy flow, including local Pulumi authentication, `dev` stack values, and the manual ECS historical backfill command, see [docs/deployment-dev.md](docs/deployment-dev.md).

## Development

Install dependencies and run checks:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
```

The test suite runs entirely locally against filesystem-backed artifact roots.

## Local deployment-style run

Use the `Makefile` target to exercise the ECS-style orchestrator locally:

```bash
make local-run
```

Default behavior:

- runs `cdt-orchestrator daily`
- uses `$(DATA_DIR)/local` as the artifact root
- uses `idi-dev-processor-s3` as the source scraper bucket
- uses `idi-analysis` as the AWS profile
- uses `1000-ciks.txt` as the default CIK file

Override values as needed:

```bash
make local-run LOCAL_CIK_FILE=./10K-ciks.txt
make local-run LOCAL_MODE=historical LOCAL_RUN_ARGS="--start-date 2024-01-01 --end-date 2024-01-31"
make local-run LOCAL_ARTIFACT_ROOT=./data/smoke LOCAL_RUN_ARGS="--cik-file ./100K-ciks.txt"
make local-run LOCAL_AWS_PROFILE=other-profile
```
