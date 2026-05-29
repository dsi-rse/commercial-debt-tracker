# Commercial Debt Tracker

Commercial Debt Tracker (CDT) is a multi-stage processor for finding debt-related disclosures in SEC 8-K filings, extracting structured debt-instrument mentions, and matching those mentions into filing-level instrument histories.

The ingest stage consumes `manifest.json` artifacts produced by [`idi-sec-scraper`](https://github.com/dsi-clinic/idi-sec-scraper). CDT does not scrape SEC EDGAR directly; it indexes and optionally downloads only the 8-K complete-submission documents it needs from the shared scraper bucket.

## Pipeline Overview

The current pipeline stages are:

1. **Ingest**: read CIKs, scan scraper manifests, index matching 8-K complete-submission text files, and optionally download document batches
2. **Itemize**: extract potentially relevant 8-K item sections
3. **Classifier**: score item rows as `relevant` or `irrelevant`
4. **Extractor**: run the LLM extraction workflow on relevant rows
5. **Matcher**: group extracted mention rows into logical debt instruments

Processing is resumable. SQLite stores stage progress, and ingest also maintains a permanent failure registry for upstream source artifacts that should not be retried.

## Output Layout

All generated artifacts live under `DATA_DIR`.

```text
{DATA_DIR}/
  cdt.sqlite
  documents/
    document-batch-*.parquet
  items/
    item-batch-*.parquet
  extractor_runs/
    run-*/full.jsonl
  failures/
    ingest_failures.json
  models/
    classifier/tfidf-linear-svc/
```

`cdt.sqlite` is the canonical pipeline state store. It tracks document rows, item rows, extraction status, matcher outputs, and related audit metadata.

## Quick Start

### Requirements

- Python 3.13+
- `uv`
- AWS credentials with read access to the shared SEC scraper bucket

### Installation

```bash
uv sync
```

For development dependencies:

```bash
uv sync --all-groups
```

### Environment

Set `DATA_DIR` in `.env` or the shell before running CDT:

```bash
DATA_DIR=/path/to/commercial-debt-tracker/processor
```

Ingest uses the `idi-analysis` AWS profile by default. Override it with `--aws-profile` when needed.

### Local Runtime

The repo includes a simple Docker-based local environment:

```bash
docker compose build
docker compose run --rm commercial-debt-tracker /bin/bash
```

The compose setup mounts the repo at `/project` and mounts `${DATA_DIR}` at `/data`.

## Ingest

### What Ingest Reads

Ingest expects the bucket layout written by `idi-sec-scraper`:

```text
s3://{bucket}/
  sec/
    {YYYY-MM-DD}/
      8-K/
        {cik}/
          {accession_without_dashes}/
            manifest.json
            ...
```

Each `manifest.json` is treated as the source of truth for a filing. CDT selects the `COMPLETE SUBMISSION TEXT FILE` document entry from that manifest.

### Modes

`cdt ingest` follows the same mode structure used in the standalone processors:

- `daily`: inclusive date window; if neither date is supplied, both default to yesterday
- `historical`: full backfill window by default (`1994-01-01` through today), with optional bounds for partial backfills

### Run

Daily ingest for an explicit window:

```bash
uv run cdt ingest \
  --bucket idi-dev-processor-s3 \
  --download \
  daily 100K-ciks.txt \
  --start-date 2024-01-01 \
  --end-date 2024-01-31
```

Daily ingest using the default yesterday window:

```bash
uv run cdt ingest daily 100K-ciks.txt
```

Historical ingest across the full CDT range:

```bash
uv run cdt ingest historical 100K-ciks.txt
```

Historical ingest for a bounded backfill:

```bash
uv run cdt ingest \
  --batch-size 250 \
  --download \
  --log-file ingest.log \
  historical 100K-ciks.txt \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

Without `--download`, ingest only indexes matching resources in SQLite. With `--download`, it also writes append-only Parquet document batches under `documents/`.

### CLI Reference

Common ingest options must appear before the mode:

| Flag | Default | Description |
|---|---|---|
| `--bucket` | `idi-dev-processor-s3` | Shared scraper bucket |
| `--force` | `false` | Replace existing rows for requested accessions |
| `--batch-size` | `100` | Matching documents processed per batch |
| `--download` | `false` | Download matched document bodies into Parquet batches |
| `--failure-file` | `{DATA_DIR}/failures/ingest_failures.json` | Permanent ingest failure registry path |
| `--aws-profile` | `idi-analysis` | AWS profile used to create the S3 client |
| `--s3-prefix` | `sec` | Top-level scraper prefix inside the bucket |
| `--log-file` | disabled | Optional file logging for long-running runs |
| `--quiet` | `false` | Suppress info-level progress logs |

Mode-specific arguments:

| Mode | Arguments | Behavior |
|---|---|---|
| `daily` | `cik_file`, optional `--start-date`, `--end-date` | Defaults both dates to yesterday when omitted; if either date is supplied, both are required |
| `historical` | `cik_file`, optional `--start-date`, `--end-date` | Defaults to `1994-01-01` through today |

### Failure Registry

Ingest writes permanent source failures to:

```text
{DATA_DIR}/failures/ingest_failures.json
```

These failures are distinct from SQLite pipeline state:

- **SQLite** tracks document indexing and downstream stage status
- **Failure registry** tracks source artifacts that should not be retried

Currently ingest records permanent failures for:

- unreadable manifests
- invalid manifest payloads
- manifests that do not contain a complete-submission document
- document download failures in `--download` mode

On rerun, ingest skips entries already recorded in the failure registry.

## Itemize

`cdt itemize` processes document rows that have not yet been itemized. It reads pending document rows from `cdt.sqlite`, loads the recorded S3 URI or local file path, extracts Form 8-K item sections, writes item batch Parquet files, and marks source documents as itemized.

```bash
uv run cdt itemize --batch-size 100 --log-file itemize.log
```

Itemization writes:

```text
{DATA_DIR}/items/item-batch-*.parquet
```

Use `--force` to re-itemize documents already marked itemized.

## Classifier

The classifier is binary: `relevant` or `irrelevant`.

Train it with:

```bash
uv run cdt classifier train --train-csv path/to/training.csv
```

The training CSV must contain:

```text
text
label
```

`text` is the item text to score. `label` must be one of `relevant`, `irrelevant`, `true`, `false`, `t`, `f`, `1`, or `0`.

The default model output directory is:

```text
{DATA_DIR}/models/classifier/tfidf-linear-svc
```

Run classification with:

```bash
uv run cdt classifier
```

That command reads pending item rows from `cdt.sqlite`, scores the corresponding item batch Parquet rows, and updates each item row in SQLite with classification metadata. Reruns skip rows already marked classified unless `--force` is used.

## Extractor

`cdt extractor` processes item rows already classified as relevant. It reads pending rows from `cdt.sqlite`, loads the corresponding item text from the recorded item batch Parquet file, runs the OpenRouter-backed three-stage workflow (`ner`, `instrument_ie`, `instrument_relation`), persists extracted `debt_instrument_mentions` into SQLite, and writes a per-run `full.jsonl` audit log.

Set the following environment variables before running the extractor:

```text
OPENROUTER_API_KEY=...
EXTRACTOR_MODEL=openai/gpt-5.4
EXTRACTOR_REASONING=none
```

`OPENROUTER_API_TOKEN` is also accepted as a fallback key name. The model value is passed directly to OpenRouter, so provider-prefixed model IDs such as `openai/...` or `anthropic/...` can be swapped without code changes.

Run extraction with:

```bash
uv run cdt extractor --batch-size 100 --log-file extractor.log
```

Useful flags:

- `--force`: re-extract rows already marked `extracted` or `extraction_failed`
- `--model`: override the OpenRouter model ID for this run
- `--reasoning-effort`: override the requested OpenRouter reasoning effort
- `--max-attempts`: validation retries per stage before the item is marked failed

The extractor writes run artifacts under:

```text
{DATA_DIR}/extractor_runs/run-*/full.jsonl
```

`full.jsonl` contains per-item attempt history, raw stage responses, final status, and the extracted `debt_instrument_mentions` payload used to populate SQLite.

## Matcher

`cdt matcher` groups extracted `debt_instrument_mentions` into logical `debt_instrument` rows. Matching is conservative and same-issuer only: the stage compares mentions only within the same `cik`, derives direct mention matches first, and then builds amendment and split parent links between matched debt instruments.

```bash
uv run cdt matcher --batch-size 100
```

Useful flags:

- `--force`: recompute matcher outputs for all extracted mentions

By default, matcher reruns are idempotent and skip work when no `debt_instrument_mentions` remain with a null matcher status.

Matcher writes to:

- SQLite `debt_instrument`
- SQLite view `active_debt_instruments`
- additional `debt_instrument_mentions` columns including matcher status and candidate metadata

## Development

Run tests with:

```bash
uv run pytest
```

Run linting with:

```bash
uv run ruff check .
```
