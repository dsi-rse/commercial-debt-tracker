# CDT Storage Schema

CDT writes canonical artifacts under one artifact root. That root can be a local directory or an `s3://` prefix.

## Root Layout

```text
<artifact-root>/
  documents/
  items/
  classifications/
  mentions/
  mention-matches/
  debt-instruments/
  extractor-runs/
  runs/
  failures/
```

## Partitioning Rules

Date-partitioned datasets:

- `documents`
- `items`
- `classifications`
- `mentions`

Canonical path shape:

```text
<artifact-root>/<dataset>/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
```

CIK-sharded datasets:

- `mention-matches`
- `debt-instruments`

Canonical path shape:

```text
<artifact-root>/<dataset>/cik_shard=NNNN/part-0000.parquet
```

Notes:

- date shards are derived from accession number hashes
- CIK shards are derived from CIK hashes
- both currently use 64 shards

## Dataset Schemas

### `documents`

Columns:

- `accession_number`
- `cik`
- `url`
- `text`
- `date`
- `resource_uri`

Primary key: `accession_number`

### `items`

Columns:

- `item_id`
- `item`
- `accession_number`
- `cik`
- `url`
- `text`
- `date`
- `resource_uri`
- `item_information`
- `extraction_status`
- `duplicate_resolution`
- `section_heading`
- `start_line`
- `end_line`
- `section_char_count`

Primary key: `item_id`

### `classifications`

Columns:

- all `items` columns
- `label`
- `relevance`
- `classification_score`

Primary key: `item_id`

### `mentions`

Columns:

- `debt_instrument_mention_id`
- `item_id`
- `accession_number`
- `cik`
- `date`
- `raw_id`
- `name`
- `start_date`
- `end_date`
- `amount`
- `amendment_of`
- `split_of`
- `lenders_json`
- `other_interested_parties_json`
- `name_json`
- `start_date_json`
- `end_date_json`
- `amount_json`

Primary key: `debt_instrument_mention_id`

### `mention-matches`

Columns:

- `debt_instrument_mention_id`
- `debt_instrument_id`
- `matcher_status`

### `debt-instruments`

Columns:

- `debt_instrument_id`
- `cik`
- `seed_debt_instrument_mention_id`
- `amendment_of_debt_instrument_id`
- `split_of_debt_instrument_id`
- `name`
- `start_date`
- `end_date`
- `amount`
- `direct_mentions_json`
- `lenders_json`
- `other_interested_parties_json`
- `possibly_related_json`

Primary key: `debt_instrument_id`

## Run Metadata

### Ingest manifests

Ingest writes one manifest per run with a generated timestamp-based run ID:

```text
<artifact-root>/runs/ingest/run_id=<run_id>.json
```

### Itemize, classify, and match manifests

These stages currently overwrite a `latest` manifest:

```text
<artifact-root>/runs/itemize/run_id=latest.json
<artifact-root>/runs/classify/run_id=latest.json
<artifact-root>/runs/match/run_id=latest.json
```

### Extractor manifests and audit logs

Extractor writes a per-run manifest and a matching full audit log:

```text
<artifact-root>/runs/extract/run_id=<run_id>.json
<artifact-root>/extractor-runs/run_id=<run_id>/full.jsonl
```

### Failure registries

The ingest stage maintains a permanent failure registry at:

```text
<artifact-root>/failures/ingest/failures.json
```

## Operational Semantics

- canonical truth is the partition data, not the run manifest
- stage completion is inferred from output partition presence
- `force=false` skips already-written partitions
- local runs and deployed runs use the same layout and code paths
- the default operating model is one active writer per environment
