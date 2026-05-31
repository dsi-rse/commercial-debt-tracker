# CDT Storage Schema

CDT is a file-native pipeline. Canonical state lives in Parquet datasets and JSON/JSONL run artifacts under one artifact root.

## Root layout

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

## Date/shard datasets

These datasets use deterministic `date` and `shard` partitions:

- `documents`
- `items`
- `classifications`
- `mentions`

Canonical file shape:

```text
<artifact-root>/<dataset>/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
```

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
- document columns above
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

- item columns above
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

## CIK-shard datasets

These datasets are written by matcher:

- `mention-matches`
- `debt-instruments`

Canonical file shape:

```text
<artifact-root>/<dataset>/cik_shard=NNNN/part-0000.parquet
```

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

## Run metadata

Stage manifests:

```text
<artifact-root>/runs/<stage>/run_id=<run_id>.json
```

Extractor audit logs:

```text
<artifact-root>/extractor-runs/run_id=<run_id>/full.jsonl
```

Permanent failure registries:

```text
<artifact-root>/failures/<stage>/failures.json
```

## Processing rules

- final partition presence is the completion signal
- partitions are rewritten deterministically
- run manifests are advisory
- one active writer per environment is the default operating model
