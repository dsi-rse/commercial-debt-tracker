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
