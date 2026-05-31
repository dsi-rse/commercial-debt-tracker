"""Secrets Manager resources for CDT runtime secrets."""

import pulumi_aws as aws

import pulumi

from . import config

openrouter_api_key = config.config.require_secret("openrouter_api_key")
sec_user_agent = config.config.require_secret("sec_user_agent")

openrouter_api_key_secret = aws.secretsmanager.Secret(
    "cdt-openrouter-api-key",
    name=f"{config.name_prefix}-openrouter-api-key",
    recovery_window_in_days=0,
    tags=config.tags(),
)

openrouter_api_key_secret_version = aws.secretsmanager.SecretVersion(
    "cdt-openrouter-api-key-version",
    secret_id=openrouter_api_key_secret.id,
    secret_string=openrouter_api_key,
    opts=pulumi.ResourceOptions(depends_on=[openrouter_api_key_secret]),
)

sec_user_agent_secret = aws.secretsmanager.Secret(
    "cdt-sec-user-agent",
    name=f"{config.name_prefix}-sec-user-agent",
    recovery_window_in_days=0,
    tags=config.tags(),
)

sec_user_agent_secret_version = aws.secretsmanager.SecretVersion(
    "cdt-sec-user-agent-version",
    secret_id=sec_user_agent_secret.id,
    secret_string=sec_user_agent,
    opts=pulumi.ResourceOptions(depends_on=[sec_user_agent_secret]),
)
