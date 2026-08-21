# Commercial Debt Tracker

Commercial Debt Tracker (CDT) processes SEC 8-K filings to build a file-native history of debt instruments. It:

- ingests complete submission text files for a configured CIK universe
- itemizes the 8-K sections most likely to contain debt disclosures
- classifies those sections for debt relevance
- uses an LLM-backed extractor to produce structured debt-instrument mentions
- matches mentions into instrument-level histories
- optionally writes dashboard-facing final parquet snapshots

## Repository Map

- [docs/architecture.md](docs/architecture.md): what the pipeline does and why it is designed this way
- [docs/deployment.md](docs/deployment.md): how production deployment works, including CI/CD and manual historical runs
- [docs/deployment-dev.md](docs/deployment-dev.md): first-time `dev` deployment walkthrough
- [docs/schema.md](docs/schema.md): canonical artifact layout and dataset schemas
- [DataPolicy.md](DataPolicy.md): data handling expectations

## Runtime Model

CDT is intentionally file-native. Canonical state lives under one artifact root as Parquet partitions plus JSON and JSONL manifests, usually in S3 for deployed runs and under `data/` for local runs. The pipeline can also write optional final snapshot parquet files under a separate final database root.

This avoids a mutable database dependency and keeps reruns deterministic:

- date-partitioned stages write `documents`, `items`, `classifications`, and `mentions`
- CIK-sharded matcher outputs write `mention-cluster-edges` and `debt-instruments`
- stage manifests and extractor audit logs are written alongside those datasets

See [docs/schema.md](docs/schema.md) for the concrete layout, including the optional `latest.parquet` final snapshots.

## Deployment Summary

The deployed service is a single ECS Fargate task running `cdt-orchestrator`, with:

- a container image built from [dockerfiles/Dockerfile.orchestrator](dockerfiles/Dockerfile.orchestrator)
- infrastructure provisioned from [`pulumi/`](pulumi/)
- a daily EventBridge Scheduler trigger that runs `cdt-orchestrator daily`
- manual historical backfills via ECS task command overrides or the `run-historical` GitHub Actions workflow

The GitHub Actions deployment path is:

1. build and push the orchestrator image to GHCR
2. run `pulumi up`
3. sync the GHCR image into the Pulumi-managed ECR repository as both `:latest` and `:${GITHUB_SHA}`

Full details are in [docs/deployment.md](docs/deployment.md).

## Local Development

Install dependencies and run the checks:

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -v
```

Main local entrypoints:

```bash
uv run cdt pipeline --artifact-root ./data historical ./1000-ciks.txt --start-date 2024-01-01 --end-date 2024-01-31
uv run cdt-orchestrator --artifact-root ./data/local daily --cik-file ./1000-ciks.txt
make local-run
./scripts/local-pipeline.sh historical --start-date 2024-01-01 --end-date 2024-01-31
```

Notes:

- `cdt` is the stage-oriented CLI for local and ad hoc runs.
- `cdt extract --retry-failures` re-drives only the rows in `failures/extract/failures.json`, which is where rows that died on provider errors are recorded.
- `cdt-orchestrator` is the deployment-oriented entrypoint used by ECS.
- `cdt pipeline` writes final snapshots only when `--final-database-root` is passed.
- `cdt-orchestrator` reads `FINAL_DATABASE_ROOT` from the environment, or accepts `--final-database-root` before the mode.
- `make local-run` and `./scripts/local-pipeline.sh` exercise the orchestrator with deployment-like environment variables from `.env`.
- The shared local convention is `DATA_DIR/commercial-debt-tracker/local` for canonical artifacts and `DATA_DIR/commercial-debt-tracker/database/cdt` for dashboard-consumable `latest.parquet` outputs.

## Dashboard Handoff

This repo communicates with `../commercial-debt-tracker-dashboard` through the final snapshot parquet contract, not through a database or API.

For local development with a shared `DATA_DIR`:

- CDT writes canonical working artifacts to `DATA_DIR/commercial-debt-tracker/local`
- CDT writes dashboard-facing final snapshots to `DATA_DIR/commercial-debt-tracker/database/cdt`
- the dashboard repo reads those four `latest.parquet` files and builds its local `generated/*.json` snapshot from them

The dashboard-facing files are:

- `items/latest.parquet`
- `debt-instruments/latest.parquet`
- `debt-instrument-mentions/latest.parquet`
- `mention-cluster-edges/latest.parquet`

This mirrors the cloud contract: in deployment, CDT writes the same final snapshots to S3, and the dashboard publisher turns them into `generated/*` JSON for R2.

## Local End-to-End Test

To test the processor and dashboard together on your machine:

1. Set the same `DATA_DIR` in both repos' `.env` files.
2. Run CDT locally against your target date range and optional CIK file:

```bash
./scripts/local-pipeline.sh historical --start-date 2020-01-01 --end-date 2021-12-31
LOCAL_CIK_FILE=/abs/path/to/ciks.txt ./scripts/local-pipeline.sh historical --start-date 2020-01-01 --end-date 2021-12-31
```

3. In `../commercial-debt-tracker-dashboard`, run:

```bash
npm run local:dev
```

4. Open the local dashboard and confirm it loads data from the run you just produced.

If you rerun the processor, rerun `npm run local:dev` to rebuild the dashboard JSON from the new parquet outputs.

## Required Runtime Configuration

Deployed runs require:

- `ARTIFACT_ROOT`
- `BUCKET_NAME`
- `CDT_DEFAULT_CIK_FILE`
- `OPENROUTER_API_KEY`

Optional runtime configuration:

- `AWS_PROFILE` for local runs against AWS
- `EXTRACTOR_MODEL`
- `EXTRACTOR_REASONING`

Pulumi also provisions a `SEC_USER_AGENT` secret into the ECS task to match the shared processor deployment pattern, even though CDT itself currently reads filings from scraper-managed S3 rather than calling SEC endpoints directly.
