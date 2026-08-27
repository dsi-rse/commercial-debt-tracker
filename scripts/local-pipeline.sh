#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${repo_root}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
fi

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR must be set in the environment or ${repo_root}/.env." >&2
  exit 1
fi

local_mode="${LOCAL_MODE:-daily}"
local_artifact_root="${LOCAL_ARTIFACT_ROOT:-${DATA_DIR}/commercial-debt-tracker/local}"
local_final_database_root="${LOCAL_FINAL_DATABASE_ROOT:-${DATA_DIR}/commercial-debt-tracker/database/cdt}"
local_bucket_name="${LOCAL_BUCKET_NAME:-idi-dev-ftm2j-shared-processor-storage}"
local_aws_profile="${LOCAL_AWS_PROFILE:-idi-analysis}"
local_cik_file="${LOCAL_CIK_FILE:-${repo_root}/data/ciks/1000-ciks.txt}"

mode="${local_mode}"
if [[ $# -gt 0 && ( "$1" == "daily" || "$1" == "historical" ) ]]; then
  mode="$1"
  shift
fi

mkdir -p "${local_artifact_root}" "${local_final_database_root}"

cd "${repo_root}"
ARTIFACT_ROOT="${local_artifact_root}" \
FINAL_DATABASE_ROOT="${local_final_database_root}" \
BUCKET_NAME="${local_bucket_name}" \
AWS_PROFILE="${local_aws_profile}" \
CDT_DEFAULT_CIK_FILE="${local_cik_file}" \
uv run cdt-orchestrator --aws-profile "${local_aws_profile}" "${mode}" "$@"
