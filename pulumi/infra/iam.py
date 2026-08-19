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

# Reading a SecureString parameter takes two permissions: ssm:GetParameters on
# the parameter, plus kms:Decrypt to unwrap it. KMS is scoped by ViaService so the
# role can only use the key through SSM.
aws.iam.RolePolicy(
    "cdt-ecs-execution-secrets-policy",
    role=task_execution_role.id,
    policy=pulumi.Output.json_dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ReadSecretParams",
                    "Effect": "Allow",
                    "Action": ["ssm:GetParameter", "ssm:GetParameters"],
                    "Resource": [
                        secrets.openrouter_api_key_param.arn,
                        secrets.openai_api_key_param.arn,
                    ],
                },
                {
                    "Sid": "DecryptViaSSM",
                    "Effect": "Allow",
                    "Action": ["kms:Decrypt"],
                    "Resource": "*",
                    "Condition": {
                        "StringEquals": {
                            "kms:ViaService": f"ssm.{config.aws_region}.amazonaws.com"
                        }
                    },
                },
            ],
        }
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

# S3 bucket ARNs are deterministic, so construct them directly rather than a
# plan-time aws.s3.get_bucket lookup, which would require read access to the
# scraper-managed source bucket just to obtain an ARN (see issue #7). The set
# collapses to one ARN when the source and output buckets are the same bucket,
# which is the dev configuration.
bucket_arns = sorted(
    {
        f"arn:aws:s3:::{config.bucket_name}",
        f"arn:aws:s3:::{config.output_bucket_name}",
    }
)

# Object permissions are scoped by prefix, not by bucket, because in dev the
# scraper's source bucket and CDT's output bucket are the same bucket (see #8).
# Splitting by bucket there would grant PutObject/DeleteObject over the scraper's
# whole archive — every form type back to 2016 — which CDT only ever reads.
source_read_arns = [f"arn:aws:s3:::{config.bucket_name}/{config.source_prefix}/*"]
# Everything CDT writes lives under one of these two prefixes: canonical
# artifacts (datasets, run manifests, completion + failure registries, extract
# job state, locks) and the final dashboard snapshots.
output_write_arns = sorted(
    {
        f"arn:aws:s3:::{config.output_bucket_name}/{config.artifact_prefix}/*",
        f"arn:aws:s3:::{config.output_bucket_name}/{config.final_database_prefix}/*",
    }
)

aws.iam.RolePolicy(
    "cdt-ecs-task-s3-policy",
    role=task_role.id,
    policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    # ListBucket is a bucket-level action: it cannot be scoped by
                    # resource path, only by an s3:prefix condition. Left
                    # unconstrained for now — tightening it means enumerating
                    # every prefix the code lists under, and a wrong list yields
                    # silent empty listings rather than an error (see #8).
                    "Sid": "ListBuckets",
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": bucket_arns,
                },
                {
                    "Sid": "ReadScraperSource",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject"],
                    "Resource": source_read_arns,
                },
                {
                    "Sid": "WriteOwnArtifacts",
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
                    "Resource": output_write_arns,
                },
            ],
        }
    ),
)
