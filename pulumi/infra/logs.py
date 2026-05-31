"""CloudWatch Log Group for ECS task logs."""

import pulumi_aws as aws

from . import config

log_group = aws.cloudwatch.LogGroup(
    "cdt-ecs-log-group",
    name=f"/ecs/{config.name_prefix}",
    retention_in_days=config.log_retention_days,
    tags=config.tags(),
)
