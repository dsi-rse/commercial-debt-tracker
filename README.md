# Commercial Debt Tracker

## Project Background

[Please add project background]

## Project Goals

[Please add project background]

## Usage

Set `DATA_DIR` in `.env`; CDT writes generated artifacts under that directory.

```bash
DATA_DIR=/path/to/commercial-debt-tracker/processor
```

### SEC 8-K Ingest

`cdt ingest` reads one CIK per line, scans the shared SEC scraper bucket for
matching 8-K complete submission text files, and records the resource
locations in the shared SQLite database.

```bash
uv run cdt ingest 100K-ciks.txt \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --batch-size 100 \
  --download \
  --log-file ingest.log
```

If dates are omitted, the command scans from `1994-01-01` through today. The
default bucket is `idi-dev-processor-s3`, with scraper data under the `sec/`
prefix. The command expects AWS credentials for the `idi-analysis` profile.
Without `--download`, ingest only indexes resources; with `--download`, it also
writes Parquet document batches.

Ingest writes:

```text
$DATA_DIR/cdt.sqlite
$DATA_DIR/documents/document-batch-*.parquet
```

The SQLite database tracks document accessions, resource URIs, and pipeline
status. Reruns skip documents that are already indexed or itemized.

### 8-K Itemization

`cdt itemize` processes resources that have not yet been itemized. It reads
pending document rows from `cdt.sqlite`, loads each source from its recorded
S3 URI or local file path, extracts Form 8-K item sections, writes item batch
Parquet files, and marks source documents as itemized in SQLite.

```bash
uv run cdt itemize --batch-size 100 --log-file itemize.log
```

Itemization writes:

```text
$DATA_DIR/items/item-batch-*.parquet
```

Use `--force` to re-itemize documents already marked itemized.


### 8-K Classification

Using dsi-core/11th-hour/commercial-debt-tracker/data/annotations/classification/svm-annotated-8K-20260318.csv to train with 5 CV splits and a random seed of 42, a classifier was created with recall of .9903 and precision of .8483. 

This is saved to dsi-core/11th-hour/commercial-debt-tracker/processor/models/classifier/tfidf-linear-svc.

In this repo, the classifier is binary: `relevant` or `irrelevant`.

Train it with:

```bash
uv run cdt classifier train --train-csv path/to/training.csv
```

The training CSV must contain these columns:

```text
text
label
```

`text` is the item text to score. `label` is the binary training target and must be one of `relevant`, `irrelevant`, `true`, `false`, `t`, `f`, `1`, or `0`.

The default model output directory is:

```text
$DATA_DIR/models/classifier/tfidf-linear-svc
```

Training targets `99%` recall and logs the resulting precision, recall, and PR-AUC after fitting. You can override the default recall target, CV split count, random seed, and model directory with CLI flags.

Run classification with:

```bash
uv run cdt classifier
```

That command reads pending item rows from `$DATA_DIR/cdt.sqlite`, scores the corresponding item batch Parquet rows, and updates each item row in SQLite with:

- `status = classified`
- `label = relevant` or `irrelevant`
- `relevance = 1` for relevant rows, `0` for irrelevant rows
- `classification_score`
- `classified_at`

Reruns are idempotent and skip rows already marked classified unless `--force` is used.

### LLM Extraction

`cdt extractor` processes item rows that have already been classified as relevant. It reads pending rows from `cdt.sqlite`, loads the corresponding item text from the recorded item batch Parquet file, runs an OpenRouter-backed three-stage workflow (`ner`, `instrument_ie`, `instrument_relation`), persists extracted `debt_instrument_mentions` into SQLite, and writes a per-run `full.jsonl` audit log.

Set the following environment variables in `.env` before running the extractor:

```text
OPENROUTER_API_KEY=...
EXTRACTOR_MODEL=openai/gpt-5.4
EXTRACTOR_REASONING=none
```

`OPENROUTER_API_TOKEN` is also accepted as a fallback key name, but `OPENROUTER_API_KEY` is preferred. The model value is passed directly to OpenRouter, so provider-prefixed model IDs such as `openai/...` or `anthropic/...` can be swapped without code changes. The default reasoning effort is `none`.

Run extraction with:

```bash
uv run cdt extractor --batch-size 100 --log-file extractor.log
```

Useful flags:

- `--force`: re-extract rows already marked `extracted` or `extraction_failed`
- `--model`: override the OpenRouter model ID for this run
- `--reasoning-effort`: override the requested OpenRouter reasoning effort
- `--max-attempts`: number of validation retries per stage before the item is marked failed

Only rows with:

- `status = classified`
- `relevance = 1`

are eligible by default.

On success, each processed item row is updated in SQLite with:

- `status = extracted`
- `extracted_at`
- `extractor_model`
- `extractor_reasoning`
- `extractor_run_path`

On failure, each processed item row is updated with:

- `status = extraction_failed`
- `extractor_error`
- the same extractor metadata fields for auditability

Extractor outputs are persisted to the SQLite `debt_instrument_mentions` table. These rows are mention-level outputs only. They are not canonical instruments, and they should not be interpreted as consolidated entities across items or filings. A later stage will resolve and consolidate mentions into actual instruments.

Each `debt_instrument_mentions` row stores:

- stable mention identity: `debt_instrument_mention_id`, `item_id`, `raw_id`
- scalar mention properties: `name`, `start_date`, `end_date`, `amount`
- mention lineage links: `amendment_of`, `split_of`
- JSON payloads for richer mention data and audit details:
  - `lenders_json`
  - `other_interested_parties_json`
  - `mention_corefs_json`
  - `start_date_corefs_json`
  - `end_date_corefs_json`
  - `amount_corefs_json`
  - `instrument_mention_json`

The extractor writes run artifacts under:

```text
$DATA_DIR/extractor_runs/run-*/full.jsonl
```

`full.jsonl` contains per-item attempt history, raw stage responses, final status, and the extracted `debt_instrument_mentions` payload used to populate SQLite.

### Matcher

`cdt matcher` groups extracted `debt_instrument_mentions` into logical `debt_instrument` rows. Matching is conservative and same-issuer only: the stage compares mentions only within the same `cik`, derives direct mention matches first, and then builds amendment and split parent links between matched debt instruments.

Run matching with:

```bash
uv run cdt matcher --batch-size 100
```

Useful flags:

- `--force`: recompute matcher outputs for all extracted mentions

By default, matcher reruns are idempotent and skip work when no `debt_instrument_mentions` remain with a null matcher status.

Matcher decision rules:

- Primary auto-match path:
  - exact normalized `amount`
  - exact normalized `start_date`
  - usable lender evidence on both sides
  - lender string similarity above the strong match threshold
- Fallback auto-match path, used only when lender evidence is missing or generic on at least one side:
  - exact normalized `amount`
  - exact normalized `start_date`
  - exact normalized debt-name fingerprint
  - compatible `end_date` values
    - equal when both are present
    - otherwise one side may be null
- Loose candidates are recorded, not merged, when amount and start date match but the primary lender-similarity path does not clear the strong threshold.

Lender evidence is treated as unusable when it is empty or only contains generic role labels such as `lender`, `lenders`, `purchaser`, `purchasers`, `holder`, `holders`, `investor`, `investors`, `buyer`, `buyers`, `noteholder`, `noteholders`, `trustee`, or `trustees`. These generic labels do not count as real counterparty identity for direct matching.

Matcher writes to:

- SQLite `debt_instrument`, one row per matched debt-instrument state
- SQLite view `active_debt_instruments`, which returns terminal current rows only and adds a computed `mentions_json`
- additional `debt_instrument_mentions` columns:
  - `debt_instrument_id`
  - `matcher_status = singleton | matched | ambiguous`
  - `matched_at`
  - `potential_matches_json`

`potential_matches_json` stores loose candidates that matched on normalized amount and start date but did not meet the automatic-merge rules.
