# Commercial Debt Tracker

Commercial Debt Tracker (CDT) processes SEC 8-K filings to build a file-native history of debt instruments. It:

- ingests complete submission text files for a configured CIK universe
- itemizes the 8-K sections most likely to contain debt disclosures
- classifies those sections for debt relevance
- uses an LLM-backed extractor to produce structured debt-instrument mentions
- matches mentions into instrument-level histories
- optionally publishes dashboard snapshot JSON to Cloudflare R2

## Repository Map

- [docs/architecture.md](docs/architecture.md): what the pipeline does and why it is designed this way
- [docs/deployment.md](docs/deployment.md): how production deployment works, including CI/CD and manual historical runs
- [docs/deployment-dev.md](docs/deployment-dev.md): first-time `dev` deployment walkthrough
- [docs/schema.md](docs/schema.md): canonical artifact layout and dataset schemas
- [DataPolicy.md](DataPolicy.md): data handling expectations

## Runtime Model

CDT is intentionally file-native. Canonical state lives under one artifact root as Parquet partitions plus JSON and JSONL manifests, usually in S3 for deployed runs and under `data/` for local runs.

This avoids a mutable database dependency and keeps reruns deterministic:

- date-partitioned stages write `documents`, `items`, `classifications`, and `mentions`
- CIK-sharded matcher outputs write `mention-matches` and `debt-instruments`
- stage manifests and extractor audit logs are written alongside those datasets

See [docs/schema.md](docs/schema.md) for the concrete layout.

## Deployment Summary

The deployed service is a single ECS Fargate task running `cdt-orchestrator`, with:

- a container image built from [dockerfiles/Dockerfile.orchestrator](dockerfiles/Dockerfile.orchestrator)
- infrastructure provisioned from [`pulumi/`](pulumi/)
- a daily EventBridge Scheduler trigger that runs `cdt-orchestrator daily`
- manual historical backfills via ECS task command overrides or the `run-historical` GitHub Actions workflow

The GitHub Actions deployment path is:

1. build and push the orchestrator image to GHCR
2. run `pulumi up`
3. sync the `:latest` image from GHCR into the Pulumi-managed ECR repository

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
uv run cdt-orchestrator daily --artifact-root ./data/local --cik-file ./1000-ciks.txt
make local-run
```

Notes:

- `cdt` is the stage-oriented CLI for local and ad hoc runs.
- `cdt-orchestrator` is the deployment-oriented entrypoint used by ECS.
- `make local-run` exercises the orchestrator with deployment-like environment variables from `.env`.

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
- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`
- `R2_OBJECT_PREFIX`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

Pulumi also provisions a `SEC_USER_AGENT` secret into the ECS task to match the shared processor deployment pattern, even though CDT itself currently reads filings from scraper-managed S3 rather than calling SEC endpoints directly.
