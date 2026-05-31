# Dev Deployment Guide

This guide documents the first `dev` deployment flow for CDT, including the initial Pulumi setup and a manual historical backfill.

## What Gets Deployed

The `dev` stack provisions:

- an ECR repository for the CDT image
- an ECS Fargate cluster and task definition
- IAM roles for ECS execution and runtime access
- a CloudWatch log group
- an EventBridge Scheduler schedule
- Secrets Manager secrets for `OPENROUTER_API_KEY` and `SEC_USER_AGENT`
- optional Secrets Manager secrets for Cloudflare R2 upload credentials

The scheduled task runs `cdt-orchestrator daily`. Historical runs are manual ECS task invocations with a container command override.

## Dev Values

These are the current recommended `dev` values:

```text
aws:region = us-east-2
idi:bucket_name = idi-dev-processor-s3
idi:artifact_prefix = commercial-debt-tracker/dev
idi:default_cik_file = s3://idi-dev-processor-s3/commercial-debt-tracker/dev/inputs/ciks/beta-1k.txt
idi:shared_dlq_name = idi-dev-ftm2j-shared-scheduler-dlq
idi:cpu = 1024
idi:memory = 4096
idi:cron = cron(0 7 * * ? *)
idi:schedule_enabled = false
```

Notes:

- `cron(0 7 * * ? *)` runs at `2:00 AM CDT` on May 31, 2026. Because EventBridge Scheduler cron expressions are UTC-based here, that becomes `1:00 AM CST` in winter.
- `idi:schedule_enabled = false` is deliberate for the first deploy. Create the infrastructure, run a manual task, inspect outputs, and only then enable the daily schedule.

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

4. Log Pulumi into the shared S3 backend:

```bash
cd pulumi
pulumi login s3://idi-ftm2j-dev-pulumi-state/commercial-debt-tracker
cd ..
```

## Upload the Beta CIK File

For the first `dev` deploy, use the repository's `1000-ciks.txt` file as the default run scope:

```bash
aws s3 cp \
  1000-ciks.txt \
  s3://idi-dev-processor-s3/commercial-debt-tracker/dev/inputs/ciks/beta-1k.txt
```

The deployed daily job and any manual historical run can override the CIK file, but this path is the default `dev` value.

## Create and Configure the Pulumi Stack

Run these commands from the `pulumi/` directory:

```bash
pulumi stack init dev
pulumi config set aws:region us-east-2
pulumi config set idi:bucket_name idi-dev-processor-s3
pulumi config set idi:artifact_prefix commercial-debt-tracker/dev
pulumi config set idi:default_cik_file s3://idi-dev-processor-s3/commercial-debt-tracker/dev/inputs/ciks/beta-1k.txt
pulumi config set idi:shared_dlq_name idi-dev-ftm2j-shared-scheduler-dlq
pulumi config set idi:cpu 1024
pulumi config set idi:memory 4096
pulumi config set idi:cron "cron(0 7 * * ? *)"
pulumi config set idi:schedule_enabled false
pulumi config set --secret idi:openrouter_api_key <openrouter-api-key>
pulumi config set --secret idi:sec_user_agent "Trevor Spreadbury dsicorefacility_project3@uchicago.edu"
```

To enable dashboard snapshot publishing into Cloudflare R2, also set:

```bash
pulumi config set idi:r2_account_id <cloudflare-account-id>
pulumi config set idi:r2_bucket_name <r2-bucket-name>
pulumi config set idi:r2_object_prefix generated
pulumi config set --secret idi:r2_access_key_id <r2-access-key-id>
pulumi config set --secret idi:r2_secret_access_key <r2-secret-access-key>
```

The ECS task will then publish `generated/index.json`, `generated/companies/*`, and
`generated/debt-instruments/*` after a successful pipeline run. Unchanged objects are
skipped.

If the `dev` stack already exists, use `pulumi stack select dev` instead of `pulumi stack init dev`.

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

Use the deployed ECS task definition with a container command override. First, capture the relevant Pulumi outputs:

```bash
CLUSTER_NAME="$(pulumi stack output ecs_cluster_name)"
TASK_DEFINITION_ARN="$(pulumi stack output task_definition_arn)"
SECURITY_GROUP_ID="$(pulumi stack output security_group_id)"
PRIMARY_SUBNET_ID="$(pulumi stack output primary_subnet_id)"
```

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
          "--cik-file", "s3://idi-dev-processor-s3/commercial-debt-tracker/dev/inputs/ciks/beta-1k.txt",
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

## Check Logs and Artifacts

CloudWatch logs:

```bash
pulumi stack output log_group_name
```

Expected `dev` artifacts:

```text
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/documents/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/items/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/classifications/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/mentions/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/mention-matches/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/debt-instruments/...
s3://idi-dev-processor-s3/commercial-debt-tracker/dev/runs/...
```

## Enable the Daily Schedule

After the manual run looks healthy:

```bash
pulumi config set idi:schedule_enabled true
uv run pulumi up
```

At that point the scheduler will run `daily` mode automatically using the default CIK file configured in `idi:default_cik_file`.
