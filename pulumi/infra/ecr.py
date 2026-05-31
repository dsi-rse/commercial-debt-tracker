"""ECR repository for the CDT orchestrator image."""

import json

import pulumi_aws as aws

import pulumi

from . import config

ecr_registry = pulumi.Output.from_input(config.caller.account_id).apply(
    lambda account_id: f"{account_id}.dkr.ecr.{config.aws_region}.amazonaws.com"
)

ecr_repo = aws.ecr.Repository(
    "cdt-ecr",
    name=config.name_prefix,
    force_delete=True,
    tags=config.tags(),
)

orchestrator_image = ecr_registry.apply(
    lambda registry: f"{registry}/{config.name_prefix}:latest"
)

ecr_lifecycle_policy = aws.ecr.LifecyclePolicy(
    "cdt-ecr-lifecycle",
    repository=ecr_repo.name,
    policy=json.dumps(
        {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Keep the last five images",
                    "selection": {
                        "tagStatus": "any",
                        "countType": "imageCountMoreThan",
                        "countNumber": 5,
                    },
                    "action": {"type": "expire"},
                }
            ]
        }
    ),
)
