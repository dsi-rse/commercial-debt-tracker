# Dev Deployment Guide

This guide is the concrete first-deploy companion to [docs/deployment.md](deployment.md). Use it when standing up or refreshing the `dev` environment.

This guide documents the first `dev` deployment flow for CDT, including the initial Pulumi setup and a manual historical backfill.

## What Gets Deployed

The `dev` stack provisions:

- an ECR repository for the CDT image
- an ECS Fargate cluster and task definition
- IAM roles for ECS execution and runtime access
- a CloudWatch log group
- two EventBridge Scheduler schedules: a daily `cdt-orchestrator daily` and an hourly `cdt-orchestrator poll`
- SSM SecureString parameters holding `OPENAI_API_KEY` and `OPENROUTER_API_KEY` under `/idi/dev/cdt/secrets/`

The daily task runs `cdt-orchestrator daily`; the hourly task runs `cdt-orchestrator poll` to advance the OpenAI batch extract job. Historical runs are manual ECS task invocations with a container command override.

## Dev Values

These are the current recommended `dev` values:

```text
aws:region = us-east-2
idi:artifact_prefix = processors/cdt
idi:final_database_prefix = database/cdt
idi:default_cik_key = processors/cdt/inputs/ciks/beta-1k.txt
idi:cpu = 1024
idi:memory = 4096
idi:ecr_image_retention_count = 5
idi:cron = cron(0 7 * * ? *)
idi:schedule_enabled = false
```

These are already committed in `pulumi/Pulumi.dev.yaml`; the bucket and DLQ names
are read from `/idi/dev/shared/*` in SSM and the API keys are SecureStrings, so
neither appears above.

Notes:

- `cron(0 7 * * ? *)` runs at `2:00 AM CDT` on May 31, 2026. Because EventBridge Scheduler cron expressions are UTC-based here, that becomes `1:00 AM CST` in winter.
- `idi:schedule_enabled = false` is deliberate for the first deploy. Create the infrastructure, run a manual task, inspect outputs, and only then enable the daily schedule.

## Local Setup (direnv + make)

The quickest local setup uses [direnv](https://direnv.net) and the `make infra-*`
targets, so you never re-export AWS creds or the Pulumi passphrase by hand.

1. Copy `.envrc.example` to `.envrc` (both `.env` and `.envrc` are gitignored, so the AWS
   profile stays a local choice) and put your secrets in `.env` — at minimum
   `PULUMI_CONFIG_PASSPHRASE` (from the Core Facility Bitwarden) and `OPENAI_API_KEY`.
   Then trust the directory:

```bash
direnv allow
```

`.envrc` loads `.env` and exports `AWS_PROFILE=idi-analysis`, `AWS_REGION=us-east-2`, and
`AWS_SDK_LOAD_CONFIG=1`, so AWS resolves the SSO profile (and its cached token) directly —
no exported temporary credentials that vanish between shells.

2. Log into the S3 Pulumi backend once per session and preview/deploy:

```bash
make infra-login      # aws sso login if the session is dead, then pulumi login s3://...
make infra-preview    # preview the dev stack (PULUMI_STACK=prod to target prod)
make infra-up         # deploy
```

`make infra-outputs` prints stack outputs. The `infra-*` targets inline the AWS profile and
region, so they work even in a fresh shell without direnv.

The rest of this section documents the equivalent manual steps.

## Prerequisites

Before running Pulumi locally:

1. Authenticate to AWS with the `idi-analysis` SSO profile:

```bash
aws sso login --profile idi-analysis
```

2. Export short-lived AWS credentials into the current shell. This avoids local Pulumi issues with direct SSO profile resolution:

```bash
eval "$(AWS_PROFILE=idi-analysis aws configure export-credentials --format env)"
```

3. Export the Pulumi config passphrase used for this repo's stack secrets:

```bash
export PULUMI_CONFIG_PASSPHRASE='your-passphrase'
```

The passphrase is stored in the Core Facility Bitwarden.

4. Log Pulumi into the shared S3 backend:

```bash
cd pulumi
pulumi login s3://idi-ftm2j-dev-pulumi-state/commercial-debt-tracker
cd ..
```

## Upload the Beta CIK File

For the first `dev` deploy, use the local `data/ciks/1000-ciks.txt` file as the default run scope. CIK
universes live under `data/ciks/`, which is gitignored, so fetch or regenerate the file if it is absent:

```bash
aws s3 cp \
  data/ciks/1000-ciks.txt \
  s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/inputs/ciks/beta-1k.txt
```

The deployed daily job and any manual historical run can override the CIK file, but this path is the default `dev` value.

## Create and Configure the Pulumi Stack

`pulumi/Pulumi.dev.yaml` is committed, so a fresh checkout needs no `pulumi config
set` calls — select the stack and go:

```bash
pulumi stack select dev   # or `pulumi stack init dev` on a brand-new backend
```

The two API keys are SSM SecureStrings, not stack config. The first `pulumi up`
creates them holding a placeholder; set the real values once per environment, and
again whenever they rotate:

```bash
aws ssm put-parameter --name /idi/dev/cdt/secrets/openai_api_key \
  --type SecureString --value '<openai-api-key>' --overwrite
aws ssm put-parameter --name /idi/dev/cdt/secrets/openrouter_api_key \
  --type SecureString --value '<openrouter-api-key>' --overwrite
```

A task that starts while a parameter still holds the placeholder will fail to
authenticate against that provider; the value is re-read at each task launch, so
fixing it needs no deploy. Both keys belong in the Core Facility Bitwarden.

The OpenAI key powers the deployed batch extract poller and is required. The
optional `idi:poll_cron` (default `cron(30 * * * ? *)`) controls the hourly poll
schedule.

This processor stack does not publish Cloudflare R2 JSON. It writes final parquet snapshots under `idi:final_database_prefix`; the dashboard publisher stack in `../commercial-debt-tracker-dashboard` reads those snapshots and updates R2.

## Preview and Deploy

From `pulumi/`:

```bash
uv sync --group pulumi
uv run pulumi preview
uv run pulumi up
```

After `pulumi up`, useful outputs are:

```bash
pulumi stack output ecs_cluster_name
pulumi stack output task_definition_arn
pulumi stack output security_group_id
pulumi stack output primary_subnet_id
pulumi stack output log_group_name
```

## What the Cron Job Runs

The EventBridge schedule runs the container with:

```bash
cdt-orchestrator daily
```

It does not run `historical`.

Because `idi:schedule_enabled` should be `false` for the first deploy, nothing runs automatically until you enable the schedule and deploy again.

## Run a Historical Backfill Manually

`./scripts/run-historical.sh` wraps everything below -- prefer it for real runs.
The raw pattern is kept here to show what the script does.

Use the deployed ECS task definition with a container command override. From the repository's `pulumi/` directory, log into the Pulumi backend, select the stack, and capture the relevant Pulumi outputs:

```bash
export PULUMI_CONFIG_PASSPHRASE='your-passphrase'
pulumi login s3://idi-ftm2j-dev-pulumi-state/commercial-debt-tracker
pulumi stack select dev

CLUSTER_NAME="$(pulumi stack output ecs_cluster_name)"
TASK_DEFINITION_ARN="$(pulumi stack output task_definition_arn)"
SECURITY_GROUP_ID="$(pulumi stack output security_group_id)"
PRIMARY_SUBNET_ID="$(pulumi stack output primary_subnet_id)"
```

The Pulumi config passphrase is required to read stack secrets and is stored in the Core Facility Bitwarden.

Then start a manual historical run:

```bash
aws ecs run-task \
  --cluster "$CLUSTER_NAME" \
  --launch-type FARGATE \
  --task-definition "$TASK_DEFINITION_ARN" \
  --network-configuration "awsvpcConfiguration={subnets=[$PRIMARY_SUBNET_ID],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
  --overrides '{
    "containerOverrides": [
      {
        "name": "cdt-orchestrator",
        "command": [
          "historical",
          "--cik-file", "s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/inputs/ciks/beta-1k.txt",
          "--start-date", "2024-01-01",
          "--end-date", "2024-01-31"
        ]
      }
    ]
  }'
```

Recommended first manual run:

- use `beta-1k.txt`
- use a one-month window
- inspect the output before widening the date range or CIK set

For a larger 50K-CIK historical run from 2016 through today, use the same ECS task launch pattern with the 50K CIK file:

```bash
export PULUMI_CONFIG_PASSPHRASE='your-passphrase'
pulumi login s3://idi-ftm2j-dev-pulumi-state/commercial-debt-tracker
pulumi stack select dev

CLUSTER_NAME="$(pulumi stack output ecs_cluster_name)"
TASK_DEFINITION_ARN="$(pulumi stack output task_definition_arn)"
SECURITY_GROUP_ID="$(pulumi stack output security_group_id)"
PRIMARY_SUBNET_ID="$(pulumi stack output primary_subnet_id)"

aws ecs run-task \
  --cluster "$CLUSTER_NAME" \
  --launch-type FARGATE \
  --task-definition "$TASK_DEFINITION_ARN" \
  --network-configuration "awsvpcConfiguration={subnets=[$PRIMARY_SUBNET_ID],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
  --overrides "{
    \"containerOverrides\": [
      {
        \"name\": \"cdt-orchestrator\",
        \"command\": [
          \"historical\",
          \"--cik-file\", \"s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/inputs/ciks/beta-50k.txt\",
          \"--start-date\", \"2016-01-01\",
          \"--end-date\", \"$(date +%F)\"
        ]
      }
    ]
  }"
```

## Check Logs and Artifacts

CloudWatch logs:

```bash
pulumi stack output log_group_name
```

Expected `dev` artifacts:

```text
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/documents/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/items/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/classifications/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/mentions/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/mention-cluster-edges/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/debt-instruments/...
s3://idi-dev-ftm2j-shared-processor-storage/processors/cdt/runs/...

Final snapshots are also written to:

s3://idi-dev-ftm2j-shared-processor-storage/database/cdt/items/latest.parquet
s3://idi-dev-ftm2j-shared-processor-storage/database/cdt/debt-instruments/latest.parquet
s3://idi-dev-ftm2j-shared-processor-storage/database/cdt/debt-instrument-mentions/latest.parquet
s3://idi-dev-ftm2j-shared-processor-storage/database/cdt/mention-cluster-edges/latest.parquet
```

## Enable the Daily Schedule

After the manual run looks healthy:

```bash
pulumi config set idi:schedule_enabled true
uv run pulumi up
```

At that point the scheduler will run `daily` mode automatically using the default CIK file configured in `idi:default_cik_key`.
