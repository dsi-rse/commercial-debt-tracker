"""EventBridge Scheduler for CDT daily ECS runs."""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecs, iam, networking

shared_dlq = aws.sqs.get_queue_output(name=config.shared_dlq_name)

scheduler_role = aws.iam.Role(
    "cdt-scheduler-role",
    name=f"{config.name_prefix}-role-scheduler",
    assume_role_policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "scheduler.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    ),
    tags=config.tags(),
)

aws.iam.RolePolicy(
    "cdt-scheduler-policy",
    role=scheduler_role.id,
    policy=pulumi.Output.all(
        task_execution_role_arn=iam.task_execution_role.arn,
        task_role_arn=iam.task_role.arn,
        dlq_arn=shared_dlq.arn,
        task_definition_arn=ecs.task_definition.arn,
    ).apply(
        lambda args: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "ecs:RunTask",
                        "Resource": args["task_definition_arn"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": [
                            args["task_execution_role_arn"],
                            args["task_role_arn"],
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": "sqs:SendMessage",
                        "Resource": args["dlq_arn"],
                    },
                ],
            }
        )
    ),
)

schedule = aws.scheduler.Schedule(
    "cdt-daily-schedule",
    name=f"{config.name_prefix}-schedule",
    description="Triggers the CDT orchestrator ECS task daily",
    schedule_expression=config.schedule_expression,
    flexible_time_window=aws.scheduler.ScheduleFlexibleTimeWindowArgs(mode="OFF"),
    state="ENABLED" if config.schedule_enabled else "DISABLED",
    target=aws.scheduler.ScheduleTargetArgs(
        arn=ecs.cluster.arn,
        role_arn=scheduler_role.arn,
        input=json.dumps(
            {"containerOverrides": [{"name": ecs.CONTAINER_NAME, "command": ["daily"]}]}
        ),
        ecs_parameters=aws.scheduler.ScheduleTargetEcsParametersArgs(
            task_definition_arn=ecs.task_definition.arn,
            launch_type="FARGATE",
            platform_version="LATEST",
            propagate_tags="TASK_DEFINITION",
            network_configuration=aws.scheduler.ScheduleTargetEcsParametersNetworkConfigurationArgs(
                assign_public_ip=True,
                subnets=[networking.primary_subnet_id],
                security_groups=[networking.ecs_sg.id],
            ),
        ),
        retry_policy=aws.scheduler.ScheduleTargetRetryPolicyArgs(
            maximum_retry_attempts=2,
            maximum_event_age_in_seconds=3600,
        ),
        dead_letter_config=aws.scheduler.ScheduleTargetDeadLetterConfigArgs(
            arn=shared_dlq.arn,
        ),
    ),
)
