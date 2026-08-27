"""Genuine secrets as SSM Parameter Store SecureString parameters.

The two LLM API keys are stored as SSM ``SecureString`` parameters under
``/idi/<env>/cdt/secrets/``. The task definition injects each by ARN via
``secrets:``, so the values never touch CI logs, git, or Pulumi state. Rotation is
a ``put-parameter --overwrite``, picked up at the next task launch — no deploy.

Pulumi only creates the parameter with a placeholder; the real value is set
out-of-band:

    aws ssm put-parameter --name /idi/<env>/cdt/secrets/<key>
        --type SecureString --value '<v>' --overwrite
"""

import pulumi_aws as aws

import pulumi

from . import config

_secrets_prefix = f"/idi/{config.stack_name}/{config.app_name}/secrets"
_PLACEHOLDER = "PLACEHOLDER-set-via-aws-ssm-put-parameter"


def _secret_parameter(key: str, description: str) -> aws.ssm.Parameter:
    """Create one SecureString parameter whose value Pulumi does not manage."""
    return aws.ssm.Parameter(
        f"cdt-ssm-secret-{key.replace('_', '-')}",
        name=f"{_secrets_prefix}/{key}",
        type="SecureString",
        value=_PLACEHOLDER,
        description=description,
        tags=config.tags(),
        # The real value is set out-of-band, so Pulumi must not overwrite it with
        # the placeholder on the next `up`.
        opts=pulumi.ResourceOptions(ignore_changes=["value"]),
    )


openrouter_api_key_param = _secret_parameter(
    "openrouter_api_key", "OpenRouter API key (real value set out-of-band)."
)
openai_api_key_param = _secret_parameter(
    "openai_api_key", "OpenAI API key, used by the batch extractor (set out-of-band)."
)
