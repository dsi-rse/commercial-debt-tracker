#!/usr/bin/env bash

# Start a historical backfill as an ECS task, with the same overrides the
# retired run-historical.yml workflow constructed. Historical runs are manual
# and admin-driven by design: the GitHub deploy role cannot call ecs:RunTask,
# and granting it upstream was judged not worth it for a handful of runs (#108).
#
# Requires: admin AWS credentials, pulumi, jq, and PULUMI_CONFIG_PASSPHRASE
# (Core Facility Bitwarden) for reading stack outputs.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run-historical.sh --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
         --cik-file <local-or-s3-path> [--stack dev|prod] [--force] \
         [--extractor-backend batch|live]

The task role can only read the shared bucket under sec/, processors/cdt/,
database/cdt/, and the committed default CIK key -- s3:// CIK paths elsewhere
fail with AccessDenied. `batch` defers extraction to the hourly poll schedule;
`live` extracts synchronously within the task via OpenRouter.
EOF
  exit 1
}

stack="dev"
start_date=""
end_date=""
cik_file=""
force="false"
extractor_backend="batch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack) stack="$2"; shift 2 ;;
    --start-date) start_date="$2"; shift 2 ;;
    --end-date) end_date="$2"; shift 2 ;;
    --cik-file) cik_file="$2"; shift 2 ;;
    --force) force="true"; shift ;;
    --extractor-backend) extractor_backend="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ "$stack" == "dev" || "$stack" == "prod" ]] || usage
[[ "$start_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || usage
[[ "$end_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || usage
# The shape regex admits impossible dates (2024-13-01); reject them here rather
# than paying a Fargate spin-up for the orchestrator to do it.
date -d "$start_date" >/dev/null 2>&1 || usage
date -d "$end_date" >/dev/null 2>&1 || usage
[[ -n "$cik_file" ]] || usage
[[ "$extractor_backend" == "batch" || "$extractor_backend" == "live" ]] || usage

for tool in aws pulumi jq; do
  command -v "$tool" >/dev/null || { echo "$tool is required." >&2; exit 1; }
done

if [[ -z "${PULUMI_CONFIG_PASSPHRASE:-}" ]]; then
  echo "PULUMI_CONFIG_PASSPHRASE must be set (Core Facility Bitwarden)." >&2
  exit 1
fi

state_bucket="${PULUMI_STATE_BUCKET:-idi-ftm2j-${stack}-pulumi-state/commercial-debt-tracker}"

cd "${repo_root}/pulumi"
pulumi login "s3://${state_bucket}" >/dev/null
pulumi stack select "$stack"

cluster_name="$(pulumi stack output ecs_cluster_name)"
task_definition_arn="$(pulumi stack output task_definition_arn)"
security_group_id="$(pulumi stack output security_group_id)"
primary_subnet_id="$(pulumi stack output primary_subnet_id)"
log_group_name="$(pulumi stack output log_group_name)"

# One line the operator's shell history and any log capture both keep: who
# launched what -- the record the Actions log used to provide.
echo "run-historical: stack=${stack} start_date=${start_date} end_date=${end_date}" \
  "cik_file=${cik_file} force=${force} extractor_backend=${extractor_backend}" \
  "caller=$(aws sts get-caller-identity --query Arn --output text)"

overrides_json="$(jq -n \
  --arg start_date "$start_date" \
  --arg end_date "$end_date" \
  --arg cik_file "$cik_file" \
  --arg backend "$extractor_backend" \
  --arg force "$force" \
  '{
    containerOverrides: [
      {
        name: "cdt-orchestrator",
        command: (
          ["--extractor-backend", $backend]
          + (if $force == "true" then ["--force"] else [] end)
          + ["historical", "--cik-file", $cik_file, "--start-date", $start_date, "--end-date", $end_date]
        )
      }
    ]
  }')"

run_output="$(aws ecs run-task \
  --cluster "$cluster_name" \
  --launch-type FARGATE \
  --task-definition "$task_definition_arn" \
  --network-configuration "awsvpcConfiguration={subnets=[${primary_subnet_id}],securityGroups=[${security_group_id}],assignPublicIp=ENABLED}" \
  --overrides "$overrides_json")"

failure_count="$(jq '.failures | length' <<<"$run_output")"
if [[ "$failure_count" -gt 0 ]]; then
  jq '.failures' <<<"$run_output" >&2
  exit 1
fi

task_arn="$(jq -r '.tasks[0].taskArn' <<<"$run_output")"
if [[ -z "$task_arn" || "$task_arn" == "null" ]]; then
  echo "$run_output" >&2
  exit 1
fi

echo "Task started: ${task_arn}"
echo "Logs:         aws logs tail '${log_group_name}' --follow"
