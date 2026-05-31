"""ECS cluster and Fargate task definition for CDT."""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecr, iam, logs, secrets

CONTAINER_NAME = "cdt-orchestrator"
artifact_root = f"s3://{config.bucket_name}/{config.artifact_prefix}"

cluster = aws.ecs.Cluster(
    "cdt-ecs-cluster",
    name=f"{config.name_prefix}-cluster",
    settings=[aws.ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
    tags=config.tags(),
)

container_definitions = pulumi.Output.all(
    image=ecr.orchestrator_image,
    log_group_name=logs.log_group.name,
    region=config.aws_region,
    openrouter_secret_arn=secrets.openrouter_api_key_secret.arn,
    sec_user_agent_secret_arn=secrets.sec_user_agent_secret.arn,
).apply(
    lambda args: json.dumps(
        [
            {
                "name": CONTAINER_NAME,
                "image": args["image"],
                "essential": True,
                "command": ["--help"],
                "environment": [
                    {"name": "AWS_REGION", "value": args["region"]},
                    {"name": "BUCKET_NAME", "value": config.bucket_name},
                    {"name": "ARTIFACT_ROOT", "value": artifact_root},
                    {"name": "CDT_DEFAULT_CIK_FILE", "value": config.default_cik_file},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                ],
                "secrets": [
                    {
                        "name": "OPENROUTER_API_KEY",
                        "valueFrom": args["openrouter_secret_arn"],
                    },
                    {
                        "name": "SEC_USER_AGENT",
                        "valueFrom": args["sec_user_agent_secret_arn"],
                    },
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": args["log_group_name"],
                        "awslogs-region": args["region"],
                        "awslogs-stream-prefix": "orchestrator",
                    },
                },
                "stopTimeout": 30,
            }
        ]
    )
)

task_definition = aws.ecs.TaskDefinition(
    "cdt-ecs-task-definition",
    family=config.name_prefix,
    requires_compatibilities=["FARGATE"],
    network_mode="awsvpc",
    cpu=config.cpu,
    memory=config.memory,
    execution_role_arn=iam.task_execution_role.arn,
    task_role_arn=iam.task_role.arn,
    container_definitions=container_definitions,
    tags=config.tags(),
)
