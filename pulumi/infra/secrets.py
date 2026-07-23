"""Secrets Manager resources for CDT runtime secrets."""

import pulumi_aws as aws

import pulumi

from . import config

openrouter_api_key = config.config.require_secret("openrouter_api_key")
openai_api_key = config.config.require_secret("openai_api_key")
sec_user_agent = config.config.require_secret("sec_user_agent")
r2_access_key_id = config.config.get_secret("r2_access_key_id")
r2_secret_access_key = config.config.get_secret("r2_secret_access_key")

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

openai_api_key_secret = aws.secretsmanager.Secret(
    "cdt-openai-api-key",
    name=f"{config.name_prefix}-openai-api-key",
    recovery_window_in_days=0,
    tags=config.tags(),
)

openai_api_key_secret_version = aws.secretsmanager.SecretVersion(
    "cdt-openai-api-key-version",
    secret_id=openai_api_key_secret.id,
    secret_string=openai_api_key,
    opts=pulumi.ResourceOptions(depends_on=[openai_api_key_secret]),
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

r2_access_key_id_secret = None
r2_access_key_id_secret_version = None
if r2_access_key_id is not None:
    r2_access_key_id_secret = aws.secretsmanager.Secret(
        "cdt-r2-access-key-id",
        name=f"{config.name_prefix}-r2-access-key-id",
        recovery_window_in_days=0,
        tags=config.tags(),
    )
    r2_access_key_id_secret_version = aws.secretsmanager.SecretVersion(
        "cdt-r2-access-key-id-version",
        secret_id=r2_access_key_id_secret.id,
        secret_string=r2_access_key_id,
        opts=pulumi.ResourceOptions(depends_on=[r2_access_key_id_secret]),
    )

r2_secret_access_key_secret = None
r2_secret_access_key_secret_version = None
if r2_secret_access_key is not None:
    r2_secret_access_key_secret = aws.secretsmanager.Secret(
        "cdt-r2-secret-access-key",
        name=f"{config.name_prefix}-r2-secret-access-key",
        recovery_window_in_days=0,
        tags=config.tags(),
    )
    r2_secret_access_key_secret_version = aws.secretsmanager.SecretVersion(
        "cdt-r2-secret-access-key-version",
        secret_id=r2_secret_access_key_secret.id,
        secret_string=r2_secret_access_key,
        opts=pulumi.ResourceOptions(depends_on=[r2_secret_access_key_secret]),
    )
