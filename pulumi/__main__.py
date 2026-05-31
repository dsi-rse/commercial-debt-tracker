"""Pulumi infrastructure for the CDT ECS processor."""

from infra import ecr, ecs, iam, logs, networking, scheduling, secrets

import pulumi

pulumi.export("ecs_cluster_name", ecs.cluster.name)
pulumi.export("task_definition_arn", ecs.task_definition.arn)
pulumi.export("ecr_repo_url", ecr.ecr_repo.repository_url)
pulumi.export("log_group_name", logs.log_group.name)
pulumi.export("task_execution_role_arn", iam.task_execution_role.arn)
pulumi.export("task_role_arn", iam.task_role.arn)
pulumi.export("schedule_arn", scheduling.schedule.arn)
pulumi.export("security_group_id", networking.ecs_sg.id)
pulumi.export("primary_subnet_id", networking.primary_subnet_id)
pulumi.export("openrouter_secret_arn", secrets.openrouter_api_key_secret.arn)
pulumi.export("sec_user_agent_secret_arn", secrets.sec_user_agent_secret.arn)
