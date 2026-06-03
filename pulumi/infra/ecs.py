"""ECS cluster and Fargate task definition for CDT."""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecr, iam, logs, secrets

CONTAINER_NAME = "cdt-orchestrator"
artifact_root = f"s3://{config.bucket_name}/{config.artifact_prefix}"
final_database_root = f"s3://{config.bucket_name}/{config.final_database_prefix}"

cluster = aws.ecs.Cluster(
    "cdt-ecs-cluster",
    name=f"idi-{config.stack_name}-{config.app_name}-cluster",
    settings=[aws.ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
    tags=config.tags(),
)

container_definitions = pulumi.Output.all(
    image=ecr.orchestrator_image,
    log_group_name=logs.log_group.name,
    region=config.aws_region,
    openrouter_secret_arn=secrets.openrouter_api_key_secret.arn,
    sec_user_agent_secret_arn=secrets.sec_user_agent_secret.arn,
    r2_access_key_id_secret_arn=(
        secrets.r2_access_key_id_secret.arn
        if secrets.r2_access_key_id_secret is not None
        else None
    ),
    r2_secret_access_key_secret_arn=(
        secrets.r2_secret_access_key_secret.arn
        if secrets.r2_secret_access_key_secret is not None
        else None
    ),
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
                    {"name": "FINAL_DATABASE_ROOT", "value": final_database_root},
                    {"name": "CDT_DEFAULT_CIK_FILE", "value": config.default_cik_file},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                    *(
                        [
                            {"name": "R2_ACCOUNT_ID", "value": config.r2_account_id},
                            {"name": "R2_BUCKET_NAME", "value": config.r2_bucket_name},
                            {
                                "name": "R2_OBJECT_PREFIX",
                                "value": config.r2_object_prefix,
                            },
                        ]
                        if config.r2_account_id
                        and config.r2_bucket_name
                        and args["r2_access_key_id_secret_arn"]
                        and args["r2_secret_access_key_secret_arn"]
                        else []
                    ),
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
                    *(
                        [
                            {
                                "name": "R2_ACCESS_KEY_ID",
                                "valueFrom": args["r2_access_key_id_secret_arn"],
                            },
                            {
                                "name": "R2_SECRET_ACCESS_KEY",
                                "valueFrom": args["r2_secret_access_key_secret_arn"],
                            },
                        ]
                        if args["r2_access_key_id_secret_arn"]
                        and args["r2_secret_access_key_secret_arn"]
                        else []
                    ),
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
