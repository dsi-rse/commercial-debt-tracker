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
