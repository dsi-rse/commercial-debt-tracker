"""Pulumi configuration and shared constants."""

import pulumi_aws as aws

import pulumi

config = pulumi.Config("idi")
project_name = pulumi.get_project()
stack_name = pulumi.get_stack()
app_name = config.get("app_name") or "cdt"
name_prefix = f"{project_name}-{stack_name}-{app_name}"
# Shared values are published to SSM (/idi/<stack>/shared/*) by the shared stack
# and read here, so there is one source of truth instead of a committed literal
# per repo. A committed literal is what broke every ingest entrypoint in #30: the
# name drifted to a bucket that never existed and nothing could detect it.
#
# ``bucket_name`` is the SEC scraper's bucket that ingest reads (prefix ``sec/``);
# ``output_bucket_name`` is where CDT writes its own artifacts, defaulting to the
# same shared bucket. Today they are that one bucket, separated only by prefix, so
# the task role cannot be scoped read-only by bucket — only by prefix (see #8).
bucket_name = aws.ssm.get_parameter(
    name=f"/idi/{stack_name}/shared/processor_bucket_name"
).value
output_bucket_name = config.get("output_bucket_name") or bucket_name
artifact_prefix = config.get("artifact_prefix") or "processors/cdt"
final_database_prefix = config.get("final_database_prefix") or "database/cdt"
# The scraper-owned prefix ingest reads. MUST match cdt.ingest.DEFAULT_S3_PREFIX:
# Pulumi cannot import the package, so the two are coupled by convention. If they
# drift, the task role denies every GetObject ingest attempts.
source_prefix = config.get("source_prefix") or "sec"
# Bucket-relative so the bucket name stays out of the committed stack files;
# the orchestrator wants a full s3:// URI. The key is kept separately because the
# task role grants GetObject on it explicitly — reads on the shared bucket are
# otherwise scoped to the scraper's source prefix, so the CIK file would become
# unreadable the day output_bucket_name diverges from the shared bucket.
default_cik_key = config.require("default_cik_key")
default_cik_file = f"s3://{bucket_name}/{default_cik_key}"
shared_dlq_name = aws.ssm.get_parameter(name=f"/idi/{stack_name}/shared/dlq_name").value
cpu = config.get("cpu") or "1024"
memory = config.get("memory") or "4096"
log_retention_days = int(config.get("log_retention_days") or "30")
ecr_image_retention_count = int(config.get("ecr_image_retention_count") or "5")
schedule_expression = config.get("cron") or "cron(0 8 * * ? *)"
# The extract batch poller runs on its own hourly schedule, offset from the daily
# run so the daily classify writes settle first. It shares ``schedule_enabled``.
poll_schedule_expression = config.get("poll_cron") or "cron(30 * * * ? *)"
schedule_enabled = (config.get("schedule_enabled") or "false").lower() == "true"
# The poller is the only driver of batch extraction, so it can be enabled on its
# own (e.g. to drain a manual historical run) while the daily schedule stays off.
# Defaults to schedule_enabled so flipping one knob still enables both.
_poll_enabled_raw = config.get("poll_schedule_enabled")
poll_schedule_enabled = (
    schedule_enabled
    if _poll_enabled_raw is None
    else _poll_enabled_raw.lower() == "true"
)
# Off by default: the alarm/SNS resources need bootstrap role statements that
# land with dsi-rse/idi-ftm2j-shared#79 — enabling before that bootstrap
# redeploy fails the CDT deploy with AccessDenied on sns:CreateTopic.
alerts_enabled = (config.get("alerts_enabled") or "false").lower() == "true"
caller = aws.get_caller_identity()
aws_region = pulumi.Config("aws").require("region")


def tags(extra: dict | None = None) -> dict:
    """Common resource tags."""
    payload = {
        "project": project_name,
        "environment": stack_name,
        "managed_by": "Pulumi",
        "app_name": app_name,
    }
    if extra:
        payload.update(extra)
    return payload
