"""IAM roles and policies for the CDT ECS task and scheduler."""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecr, logs, secrets

task_execution_role = aws.iam.Role(
    "cdt-ecs-execution-role",
    name=f"{config.name_prefix}-role-ecs-execution",
    assume_role_policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    ),
    tags=config.tags(),
)

aws.iam.RolePolicy(
    "cdt-ecs-execution-ecr-policy",
    role=task_execution_role.id,
    policy=ecr.ecr_repo.arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "ecr:GetAuthorizationToken",
                        "Resource": "*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:BatchGetImage",
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchCheckLayerAvailability",
                        ],
                        "Resource": arn,
                    },
                ],
            }
        )
    ),
)

aws.iam.RolePolicy(
    "cdt-ecs-execution-logs-policy",
    role=task_execution_role.id,
    policy=logs.log_group.arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                        "Resource": f"{arn}:*",
                    }
                ],
            }
        )
    ),
)

aws.iam.RolePolicy(
    "cdt-ecs-execution-secrets-policy",
    role=task_execution_role.id,
    policy=pulumi.Output.all(
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
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "secretsmanager:GetSecretValue",
                            "secretsmanager:DescribeSecret",
                        ],
                        "Resource": [
                            arn
                            for arn in [
                                args["openrouter_secret_arn"],
                                args["sec_user_agent_secret_arn"],
                                args["r2_access_key_id_secret_arn"],
                                args["r2_secret_access_key_secret_arn"],
                            ]
                            if arn is not None
                        ],
                    }
                ],
            }
        )
    ),
)

task_role = aws.iam.Role(
    "cdt-ecs-task-role",
    name=f"{config.name_prefix}-role-ecs-task",
    assume_role_policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    ),
    tags=config.tags(),
)

bucket = aws.s3.get_bucket_output(bucket=config.bucket_name)

aws.iam.RolePolicy(
    "cdt-ecs-task-s3-policy",
    role=task_role.id,
    policy=bucket.arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:ListBucket"],
                        "Resource": arn,
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:DeleteObject",
                            "s3:AbortMultipartUpload",
                            "s3:CreateMultipartUpload",
                            "s3:UploadPart",
                            "s3:CompleteMultipartUpload",
                            "s3:ListMultipartUploadParts",
                        ],
                        "Resource": f"{arn}/*",
                    },
                ],
            }
        )
    ),
)
