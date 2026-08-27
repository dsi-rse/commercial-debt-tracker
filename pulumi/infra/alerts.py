"""Alerting for the CDT pipeline (#85).

The pipeline's failure modes are mostly silent: extraction is poller-driven, so
a wedged poller stops everything while the schedule keeps "succeeding"; the
EventBridge Scheduler DLQ only captures RunTask API failures, so a task that
launches and then crashes or is OOM-killed notifies nobody; and a stolen lease
means a run died mid-write. Each alarm below covers one of those blind spots:

- **Poll liveness** — no "Poll tick complete" line for six hours.
- **Daily heartbeat** — no "Orchestrator run complete: mode=daily" line for a
  day (crashed, wedged, or never-launched daily run).
- **Task failures** — an EventBridge rule on ECS Task State Change for tasks
  that stop with a nonzero exit code or fail to start (covers OOM kills, which
  the scheduler DLQ never sees).
- **Job stall** — the poll tick logs "Extract job stalled" once the active job
  exceeds STALL_WARNING_TICKS; one job runs at a time, so a stuck job blocks
  every newly classified filing while poll liveness stays green.
- **Lease theft** — "Stole lease" means a run died still holding the writer
  lease; the work it was doing needs a look.
- **DLQ depth** — messages in the shared scheduler DLQ otherwise rot unseen.
  The queue is shared across processors, so this may also page for a sibling.

Everything is gated on ``idi:alerts_enabled``: the required deploy-role
permissions (sns, cloudwatch alarms/metric filters, events) ship separately
with dsi-rse/idi-ftm2j-shared#79, and enabling before that bootstrap redeploy
fails the CDT deploy with AccessDenied. When enabled, ``idi:alert_email`` is
required — an alarm publishing to a topic nobody subscribes to is silent by
construction, which defeats the point.
"""

import pulumi_aws as aws

import pulumi

from . import config, ecs, logs

alerts_topic = None
poll_tick_metric_filter = None
poll_liveness_alarm = None
daily_heartbeat_alarm = None
task_failure_rule = None
job_stall_alarm = None
lease_theft_alarm = None
dlq_depth_alarm = None


def _log_count_metric(resource_name: str, metric_suffix: str, pattern: str) -> str:
    """Create a metric filter counting one log literal; return the metric name.

    The patterns are exact literals the code logs — each is annotated at its
    log site with a pointer back here so the two cannot drift silently.
    """
    metric_name = f"{config.name_prefix}-{metric_suffix}"
    aws.cloudwatch.LogMetricFilter(
        resource_name,
        name=metric_name,
        log_group_name=logs.log_group.name,
        pattern=pattern,
        metric_transformation=aws.cloudwatch.LogMetricFilterMetricTransformationArgs(
            name=metric_name,
            namespace="CDT",
            value="1",
            default_value="0",
        ),
    )
    return metric_name


if config.alerts_enabled:
    _alert_email = config.config.get("alert_email")
    if not _alert_email:
        _msg = (
            "idi:alerts_enabled is true but idi:alert_email is not set. The "
            "alarms would publish to an SNS topic with no subscriber — silent "
            "by construction. Set idi:alert_email (or disable alerts)."
        )
        raise ValueError(_msg)

    alerts_topic = aws.sns.Topic(
        "cdt-alerts-topic",
        name=f"{config.name_prefix}-alerts",
        tags=config.tags(),
    )
    aws.sns.TopicSubscription(
        "cdt-alerts-email",
        topic=alerts_topic.arn,
        protocol="email",
        endpoint=_alert_email,
    )

    # --- ECS task failures -------------------------------------------------
    # The scheduler DLQ only captures failures of the RunTask API call; a task
    # that launches and then exits nonzero (or is OOM-killed, exit 137) is a
    # scheduler success. This rule catches the exit itself.
    task_failure_rule = aws.cloudwatch.EventRule(
        "cdt-task-failure-rule",
        name=f"{config.name_prefix}-task-failures",
        description=(
            "A CDT orchestrator ECS task stopped with a nonzero exit code or "
            "failed to start."
        ),
        event_pattern=pulumi.Output.json_dumps(
            {
                "source": ["aws.ecs"],
                "detail-type": ["ECS Task State Change"],
                "detail": {
                    "clusterArn": [ecs.cluster.arn],
                    "lastStatus": ["STOPPED"],
                    "$or": [
                        {"containers": {"exitCode": [{"anything-but": [0]}]}},
                        {"stopCode": ["TaskFailedToStart"]},
                    ],
                },
            }
        ),
        tags=config.tags(),
    )
    aws.cloudwatch.EventTarget(
        "cdt-task-failure-target",
        rule=task_failure_rule.name,
        arn=alerts_topic.arn,
    )
    # CloudWatch alarms may publish to a same-account topic implicitly;
    # EventBridge needs an explicit grant.
    aws.sns.TopicPolicy(
        "cdt-alerts-topic-policy",
        arn=alerts_topic.arn,
        policy=pulumi.Output.json_dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "events.amazonaws.com"},
                        "Action": "sns:Publish",
                        "Resource": alerts_topic.arn,
                        "Condition": {
                            "ArnEquals": {"aws:SourceArn": task_failure_rule.arn}
                        },
                    }
                ],
            }
        ),
    )

    # --- Lease theft --------------------------------------------------------
    # cdt.lease logs this exact literal when a lease is stolen from a holder
    # that never released — i.e. a run died (or overran its TTL) mid-write.
    _lease_theft_metric = _log_count_metric(
        "cdt-lease-theft-metric", "lease-thefts", '"Stole lease"'
    )
    lease_theft_alarm = aws.cloudwatch.MetricAlarm(
        "cdt-lease-theft-alarm",
        name=f"{config.name_prefix}-lease-theft",
        alarm_description=(
            "The pipeline-writer lease was stolen from a holder that never "
            "released it: a run died or overran its TTL mid-write. Check the "
            "previous run's logs for lost work."
        ),
        namespace="CDT",
        metric_name=_lease_theft_metric,
        statistic="Sum",
        period=3600,
        evaluation_periods=1,
        threshold=1,
        comparison_operator="GreaterThanOrEqualToThreshold",
        treat_missing_data="notBreaching",
        alarm_actions=[alerts_topic.arn],
        ok_actions=[alerts_topic.arn],
        tags=config.tags(),
    )

    # --- Extract job stall --------------------------------------------------
    # cdt.extractor.batch logs this literal every tick once the active job
    # exceeds STALL_WARNING_TICKS, so the alarm holds ALARM until the job
    # finishes or an operator resets it.
    _job_stall_metric = _log_count_metric(
        "cdt-job-stall-metric", "job-stalls", '"Extract job stalled"'
    )
    job_stall_alarm = aws.cloudwatch.MetricAlarm(
        "cdt-job-stall-alarm",
        name=f"{config.name_prefix}-job-stall",
        alarm_description=(
            "The active batch extract job has not finished after ~4 days of "
            "hourly ticks; it is blocking all newer filings. Inspect with "
            "`cdt show-extract-job`; clear with `cdt reset-extract-job --yes`."
        ),
        namespace="CDT",
        metric_name=_job_stall_metric,
        statistic="Sum",
        period=3600,
        evaluation_periods=1,
        threshold=1,
        comparison_operator="GreaterThanOrEqualToThreshold",
        treat_missing_data="notBreaching",
        alarm_actions=[alerts_topic.arn],
        ok_actions=[alerts_topic.arn],
        tags=config.tags(),
    )

    # --- Scheduler DLQ depth ------------------------------------------------
    dlq_depth_alarm = aws.cloudwatch.MetricAlarm(
        "cdt-dlq-depth-alarm",
        name=f"{config.name_prefix}-dlq-depth",
        alarm_description=(
            "The shared scheduler DLQ has messages: an EventBridge Scheduler "
            "invocation (RunTask call) failed even after retries. The queue is "
            "shared, so the failed invocation may belong to another processor."
        ),
        namespace="AWS/SQS",
        metric_name="ApproximateNumberOfMessagesVisible",
        dimensions={"QueueName": config.shared_dlq_name},
        statistic="Maximum",
        period=300,
        evaluation_periods=1,
        threshold=1,
        comparison_operator="GreaterThanOrEqualToThreshold",
        treat_missing_data="notBreaching",
        alarm_actions=[alerts_topic.arn],
        ok_actions=[alerts_topic.arn],
        tags=config.tags(),
    )

    # --- Poll liveness (only when the poller is meant to run) ---------------
    if config.poll_schedule_enabled:
        _poll_metric = _log_count_metric(
            # Every healthy tick logs exactly one of these; a tick that crashes,
            # wedges on the lease, or never launches does not.
            "cdt-poll-tick-metric",
            "poll-ticks",
            '"Poll tick complete"',
        )
        poll_liveness_alarm = aws.cloudwatch.MetricAlarm(
            "cdt-poll-liveness-alarm",
            name=f"{config.name_prefix}-poll-liveness",
            alarm_description=(
                "No batch-extract poll tick has completed for 6 hours; the "
                "poller is wedged or the schedule is not firing. Extraction is "
                "stalled."
            ),
            namespace="CDT",
            metric_name=_poll_metric,
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

    # --- Daily heartbeat (only when the daily schedule is meant to run) -----
    if config.schedule_enabled:
        _daily_metric = _log_count_metric(
            "cdt-daily-heartbeat-metric",
            "daily-completions",
            '"Orchestrator run complete: mode=daily"',
        )
        daily_heartbeat_alarm = aws.cloudwatch.MetricAlarm(
            "cdt-daily-heartbeat-alarm",
            name=f"{config.name_prefix}-daily-heartbeat",
            alarm_description=(
                "No daily orchestrator run has completed for 24 hours; the run "
                "crashed, wedged, or never launched. Ingest/itemize/classify "
                "are not advancing."
            ),
            namespace="CDT",
            metric_name=_daily_metric,
            statistic="Sum",
            # 24 hourly buckets, all empty: a fully missed day. Hour-granular
            # buckets avoid the partial-period flicker of one 86400s period,
            # and 24x3600 stays within the one-day evaluation-range limit.
            period=3600,
            evaluation_periods=24,
            datapoints_to_alarm=24,
            threshold=1,
            comparison_operator="LessThanThreshold",
            treat_missing_data="breaching",
            alarm_actions=[alerts_topic.arn],
            ok_actions=[alerts_topic.arn],
            tags=config.tags(),
        )
