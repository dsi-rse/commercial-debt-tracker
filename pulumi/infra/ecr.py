"""ECR repository for the CDT orchestrator image."""

import json

import pulumi_aws as aws

from . import config

# The repo name carries the image name (`-orchestrator`) so it matches what the
# shared pipeline's sync-ecr job pushes to:
# {pulumi_project}-{stack}-{app_name}-{image_name}.
ecr_repo = aws.ecr.Repository(
    "cdt-ecr",
    name=f"{config.name_prefix}-orchestrator",
    force_delete=True,
    tags=config.tags(),
)

# Derived from the repository resource so the name has exactly one home; a rename
# cannot leave the task definition pointing at a repo sync-ecr never pushes to.
orchestrator_image = ecr_repo.repository_url.apply(lambda url: f"{url}:latest")

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
