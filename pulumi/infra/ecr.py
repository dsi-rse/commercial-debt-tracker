"""ECR repository for the CDT orchestrator image."""

import json

import pulumi_aws as aws

import pulumi

from . import config

ecr_registry = pulumi.Output.from_input(config.caller.account_id).apply(
    lambda account_id: f"{account_id}.dkr.ecr.{config.aws_region}.amazonaws.com"
)

# The repo name carries the image name (`-orchestrator`) so it matches what the
# shared pipeline's sync-ecr job pushes to:
# {pulumi_project}-{stack}-{app_name}-{image_name}.
ecr_repo = aws.ecr.Repository(
    "cdt-ecr",
    name=f"{config.name_prefix}-orchestrator",
    force_delete=True,
    tags=config.tags(),
)

orchestrator_image = ecr_registry.apply(
    lambda registry: f"{registry}/{config.name_prefix}-orchestrator:latest"
)

ecr_lifecycle_policy = aws.ecr.LifecyclePolicy(
    "cdt-ecr-lifecycle",
    repository=ecr_repo.name,
    policy=json.dumps(
        {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": (
                        f"Keep the last {config.ecr_image_retention_count} images"
                    ),
                    "selection": {
                        "tagStatus": "any",
                        "countType": "imageCountMoreThan",
                        "countNumber": config.ecr_image_retention_count,
                    },
                    "action": {"type": "expire"},
                }
            ]
        }
    ),
)
