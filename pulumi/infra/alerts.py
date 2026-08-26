"""Liveness alerting for the batch extract poller.

Extraction is entirely poller-driven, so a wedged poller (crash-looping ticks,
a stuck lease, a scheduler misfire) silently stops all extraction. Every healthy
tick logs exactly one "Poll tick complete" line; a metric filter counts them and
an alarm fires after six consecutive tickless hours (the schedule is hourly, so
that is six missed ticks — missing data is treated as breaching on purpose).

Only created when the poll schedule is enabled (a deliberately disabled poller
must not page anyone) and ``idi:alerts_enabled`` is set — the required deploy-
role permissions ship separately (idi-ftm2j-shared#79). Notification goes to an SNS topic; subscribe an email by
setting ``idi:alert_email`` (or subscribe out-of-band — the topic ARN is a stack
output).
"""

import pulumi_aws as aws

from . import config, logs

poll_tick_metric_filter = None
poll_liveness_alarm = None
alerts_topic = None

if config.poll_schedule_enabled and config.alerts_enabled:
    alerts_topic = aws.sns.Topic(
        "cdt-alerts-topic",
        name=f"{config.name_prefix}-alerts",
        tags=config.tags(),
    )

    alert_email = config.config.get("alert_email")
    if alert_email:
        aws.sns.TopicSubscription(
            "cdt-alerts-email",
            topic=alerts_topic.arn,
            protocol="email",
            endpoint=alert_email,
        )

    _METRIC_NAME = f"{config.name_prefix}-poll-ticks"
    poll_tick_metric_filter = aws.cloudwatch.LogMetricFilter(
        "cdt-poll-tick-metric",
        name=_METRIC_NAME,
        log_group_name=logs.log_group.name,
        # The literal the orchestrator logs once per completed tick; a tick that
        # crashes, wedges on the lease, or never launches does not log it.
        pattern='"Poll tick complete"',
        metric_transformation=aws.cloudwatch.LogMetricFilterMetricTransformationArgs(
            name=_METRIC_NAME,
            namespace="CDT",
            value="1",
            default_value="0",
        ),
    )

    poll_liveness_alarm = aws.cloudwatch.MetricAlarm(
        "cdt-poll-liveness-alarm",
        name=f"{config.name_prefix}-poll-liveness",
        alarm_description=(
            "No batch-extract poll tick has completed for 6 hours; the poller "
            "is wedged or the schedule is not firing. Extraction is stalled."
        ),
        namespace="CDT",
        metric_name=_METRIC_NAME,
        statistic="Sum",
        period=3600,
        evaluation_periods=6,
        datapoints_to_alarm=6,
        threshold=1,
        comparison_operator="LessThanThreshold",
        treat_missing_data="breaching",
        alarm_actions=[alerts_topic.arn],
        ok_actions=[alerts_topic.arn],
        tags=config.tags(),
    )
