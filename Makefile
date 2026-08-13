
# general
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# pipeline constants
# PROJECT_NAME
project_name := "commercial-debt-tracker"
project_dir := "$(current_abs_path)"

# environment variables
include .env

# Check required environment variables
ifeq ($(DATA_DIR),)
	$(error DATA_DIR must be set in .env file)
endif


# local runtime defaults
LOCAL_MODE ?= daily
LOCAL_ARTIFACT_ROOT ?= $(DATA_DIR)/commercial-debt-tracker/local
LOCAL_FINAL_DATABASE_ROOT ?= $(DATA_DIR)/commercial-debt-tracker/database/cdt
LOCAL_BUCKET_NAME ?= idi-dev-processor-s3
LOCAL_AWS_PROFILE ?= idi-analysis
LOCAL_CIK_FILE ?= $(current_abs_path)data/ciks/1000-ciks.txt
LOCAL_RUN_ARGS ?=

# Pulumi / infra defaults. The AWS profile + SSO token cache are durable across
# shells, so infra targets never juggle exported temporary credentials.
PULUMI_AWS_PROFILE ?= $(or $(AWS_PROFILE),idi-analysis)
PULUMI_AWS_REGION ?= us-east-2
PULUMI_STATE_BUCKET ?= idi-ftm2j-dev-pulumi-state/commercial-debt-tracker
PULUMI_STACK ?= dev
PULUMI_AWS_ENV = AWS_PROFILE=$(PULUMI_AWS_PROFILE) AWS_REGION=$(PULUMI_AWS_REGION) AWS_DEFAULT_REGION=$(PULUMI_AWS_REGION) AWS_SDK_LOAD_CONFIG=1
PULUMI_WITH_AWS = cd pulumi && $(PULUMI_AWS_ENV) pulumi
# PULUMI_CONFIG_PASSPHRASE decrypts stack secrets; set it in .env (from Bitwarden).
export PULUMI_CONFIG_PASSPHRASE

# Build Docker image
.PHONY: build-only run-interactive run-notebook local-run local-pipeline

# Build Docker image 
build-only: 
	docker compose build

run-interactive: build-only	
	docker compose run -it --rm $(project_name) /bin/bash

run-notebooks: build-only	
	docker compose run --rm -p 8888:8888 -t $(project_name) \
	jupyter lab --port=8888 --ip='*' --NotebookApp.token='' --NotebookApp.password='' \
	--no-browser --allow-root

local-run:
	mkdir -p "$(LOCAL_ARTIFACT_ROOT)"
	mkdir -p "$(LOCAL_FINAL_DATABASE_ROOT)"
	ARTIFACT_ROOT="$(LOCAL_ARTIFACT_ROOT)" \
	FINAL_DATABASE_ROOT="$(LOCAL_FINAL_DATABASE_ROOT)" \
	BUCKET_NAME="$(LOCAL_BUCKET_NAME)" \
	AWS_PROFILE="$(LOCAL_AWS_PROFILE)" \
	CDT_DEFAULT_CIK_FILE="$(LOCAL_CIK_FILE)" \
	uv run cdt-orchestrator --aws-profile "$(LOCAL_AWS_PROFILE)" $(LOCAL_MODE) $(LOCAL_RUN_ARGS)

local-pipeline:
	bash scripts/local-pipeline.sh

# Infra / Pulumi. Typical flow: `make infra-login` once per session, then
# `make infra-preview` / `make infra-up`. Override the stack with e.g.
# `make infra-preview PULUMI_STACK=prod`.
.PHONY: infra-auth infra-login infra-stack-select infra-preview infra-up infra-refresh infra-outputs

# Refresh the SSO session only when the current one is dead.
infra-auth:
	$(PULUMI_AWS_ENV) aws sts get-caller-identity >/dev/null 2>&1 || \
		AWS_PROFILE=$(PULUMI_AWS_PROFILE) aws sso login --no-browser --use-device-code --profile $(PULUMI_AWS_PROFILE)

# Point Pulumi at this project's S3 backend (persists across shells).
infra-login: infra-auth
	$(PULUMI_WITH_AWS) login s3://$(PULUMI_STATE_BUCKET)

infra-stack-select: infra-auth
	$(PULUMI_WITH_AWS) stack select $(PULUMI_STACK)

infra-preview: infra-auth
	$(PULUMI_WITH_AWS) preview --stack $(PULUMI_STACK)

infra-up: infra-auth
	$(PULUMI_WITH_AWS) up --stack $(PULUMI_STACK)

infra-refresh: infra-auth
	$(PULUMI_WITH_AWS) refresh --stack $(PULUMI_STACK)

infra-outputs: infra-auth
	$(PULUMI_WITH_AWS) stack output --stack $(PULUMI_STACK)
