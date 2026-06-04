"""Pulumi configuration and shared constants."""

import pulumi_aws as aws

import pulumi

config = pulumi.Config("idi")
project_name = pulumi.get_project()
stack_name = pulumi.get_stack()
app_name = config.get("app_name") or "cdt"
name_prefix = f"{project_name}-{stack_name}-{app_name}"
bucket_name = config.require("bucket_name")
output_bucket_name = config.get("output_bucket_name") or bucket_name
artifact_prefix = config.get("artifact_prefix") or "processors/cdt"
final_database_prefix = config.get("final_database_prefix") or "database/cdt"
default_cik_file = config.require("default_cik_file")
shared_dlq_name = config.require("shared_dlq_name")
cpu = config.get("cpu") or "1024"
memory = config.get("memory") or "4096"
log_retention_days = int(config.get("log_retention_days") or "30")
ecr_image_retention_count = int(config.get("ecr_image_retention_count") or "5")
schedule_expression = config.get("cron") or "cron(0 8 * * ? *)"
schedule_enabled = (config.get("schedule_enabled") or "false").lower() == "true"
r2_account_id = config.get("r2_account_id")
r2_bucket_name = config.get("r2_bucket_name")
r2_object_prefix = config.get("r2_object_prefix") or "generated"
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
