
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
LOCAL_ARTIFACT_ROOT ?= $(DATA_DIR)/local
LOCAL_BUCKET_NAME ?= idi-dev-ftm2j-shared-processor-storage
LOCAL_AWS_PROFILE ?= idi-analysis
LOCAL_CIK_FILE ?= $(current_abs_path)1000-ciks.txt
LOCAL_RUN_ARGS ?=

# Build Docker image
.PHONY: build-only run-interactive run-notebook local-run

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
	ARTIFACT_ROOT="$(LOCAL_ARTIFACT_ROOT)" \
	BUCKET_NAME="$(LOCAL_BUCKET_NAME)" \
	AWS_PROFILE="$(LOCAL_AWS_PROFILE)" \
	CDT_DEFAULT_CIK_FILE="$(LOCAL_CIK_FILE)" \
	uv run cdt-orchestrator --aws-profile "$(LOCAL_AWS_PROFILE)" $(LOCAL_MODE) $(LOCAL_RUN_ARGS)
