"""Pulumi infrastructure for the CDT ECS processor."""

from infra import alerts, ecr, ecs, iam, logs, networking, scheduling, secrets

import pulumi

pulumi.export("ecs_cluster_name", ecs.cluster.name)
pulumi.export("task_definition_arn", ecs.task_definition.arn)
pulumi.export("ecr_repo_url", ecr.ecr_repo.repository_url)
pulumi.export("log_group_name", logs.log_group.name)
pulumi.export("task_execution_role_arn", iam.task_execution_role.arn)
pulumi.export("task_role_arn", iam.task_role.arn)
pulumi.export("schedule_arn", scheduling.schedule.arn)
pulumi.export("poll_schedule_arn", scheduling.poll_schedule.arn)
pulumi.export("security_group_id", networking.ecs_sg.id)
pulumi.export("primary_subnet_id", networking.primary_subnet_id)
pulumi.export("openrouter_param_arn", secrets.openrouter_api_key_param.arn)
pulumi.export("openai_param_arn", secrets.openai_api_key_param.arn)

if alerts.alerts_topic is not None:
    pulumi.export("alerts_topic_arn", alerts.alerts_topic.arn)
if alerts.poll_liveness_alarm is not None:
    pulumi.export("poll_liveness_alarm_arn", alerts.poll_liveness_alarm.arn)
if alerts.daily_heartbeat_alarm is not None:
    pulumi.export("daily_heartbeat_alarm_arn", alerts.daily_heartbeat_alarm.arn)
if alerts.task_failure_rule is not None:
    pulumi.export("task_failure_rule_arn", alerts.task_failure_rule.arn)
if alerts.job_stall_alarm is not None:
    pulumi.export("job_stall_alarm_arn", alerts.job_stall_alarm.arn)
if alerts.lease_theft_alarm is not None:
    pulumi.export("lease_theft_alarm_arn", alerts.lease_theft_alarm.arn)
if alerts.dlq_depth_alarm is not None:
    pulumi.export("dlq_depth_alarm_arn", alerts.dlq_depth_alarm.arn)
