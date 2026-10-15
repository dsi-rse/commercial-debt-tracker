#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${DATA_DIR+set}" ]]; then
  environment_data_dir="${DATA_DIR}"
fi
if [[ -f "${repo_root}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
fi

# DATA_DIR resolves as cdt's settings.DATA_DIR does: the environment, then
# .env, else <repo>/data, with a leading ~ expanded. An empty value also means
# <repo>/data. A bare `cdt` command with no --artifact-root or ARTIFACT_ROOT
# uses it as its artifact root too.
if [[ -n "${environment_data_dir+set}" ]]; then
  DATA_DIR="${environment_data_dir}"
fi
DATA_DIR="${DATA_DIR:-${repo_root}/data}"
DATA_DIR="${DATA_DIR/#\~/${HOME}}"

local_mode="${LOCAL_MODE:-daily}"
local_artifact_root="${LOCAL_ARTIFACT_ROOT:-${DATA_DIR}}"
local_final_database_root="${LOCAL_FINAL_DATABASE_ROOT:-${DATA_DIR}/database/cdt}"
local_bucket_name="${LOCAL_BUCKET_NAME:-idi-dev-ftm2j-shared-processor-storage}"
local_aws_profile="${LOCAL_AWS_PROFILE:-idi-analysis}"
local_cik_file="${LOCAL_CIK_FILE:-${repo_root}/data/ciks/smoke-10.txt}"

mode="${local_mode}"
if [[ $# -gt 0 && ( "$1" == "daily" || "$1" == "historical" ) ]]; then
  mode="$1"
  shift
fi

# A relative DATA_DIR is relative to the repo root, as in cdt's settings.
cd "${repo_root}"
mkdir -p "${local_artifact_root}" "${local_final_database_root}"

ARTIFACT_ROOT="${local_artifact_root}" \
FINAL_DATABASE_ROOT="${local_final_database_root}" \
BUCKET_NAME="${local_bucket_name}" \
AWS_PROFILE="${local_aws_profile}" \
CDT_DEFAULT_CIK_FILE="${local_cik_file}" \
uv run cdt run "${mode}" --aws-profile "${local_aws_profile}" "$@"
