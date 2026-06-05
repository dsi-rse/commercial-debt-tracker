# CDT Storage Schema

CDT writes canonical artifacts under one artifact root. That root can be a local directory or an `s3://` prefix.

The pipeline can also write optional final snapshot parquet files under a separate final database root. Those snapshots are flattened table-wide exports intended for downstream database loading and are not the canonical working state for the pipeline.

## Root Layout

```text
<artifact-root>/
  documents/
  items/
  classifications/
  mentions/
  mention-cluster-edges/
  debt-instruments/
  extractor-runs/
  runs/
  failures/
```

Optional final snapshot layout:

```text
<final-database-root>/
  items/latest.parquet
  debt-instruments/latest.parquet
  debt-instrument-mentions/latest.parquet
  mention-cluster-edges/latest.parquet
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

- `mention-cluster-edges`
- `debt-instruments`

Canonical path shape:

```text
<artifact-root>/<dataset>/cik_shard=NNNN/part-0000.parquet
```

Notes:

- `documents` shards currently use Python's process-level `hash(accession_number)` modulo 64
- `items`, `classifications`, and `mentions` preserve their source document `date` and `shard`
- CIK shards are derived from CIK hashes
- `documents` currently use 64 date shards
- downstream date-partitioned datasets currently preserve whichever document shards they read
- CIK-sharded datasets currently use 64 shards
- Changing `documents` to stable accession hashing would require migration or forced reruns of existing document partitions.

### What "Partition", "Shard", and "Batch" Mean

- A `partition` is one physical parquet file at a canonical path such as `documents/date=2026-05-31/shard=0017/part-0000.parquet`.
- A `shard` is the hash bucket inside a dataset's partitioning scheme. `documents` currently use 64 shards, rendered as `0000` through `0063`. Downstream date-partitioned datasets preserve source document shard values. CIK-sharded datasets also use 64 shards, rendered as `0000` through `0063`.
- For `documents`, all rows for the same filing date are split across 64 shard files by Python `hash(accession_number)`.
- For `items`, `classifications`, and `mentions`, rows keep the `date` and `shard` partition of their upstream source partition.
- For CIK-sharded datasets, rows are split across 64 shard files by hashed CIK, regardless of filing date.
- A `batch` is not a second storage layer. It is just the internal chunk size one pipeline invocation uses while draining all work in scope.

### Date-Partitioned Stages

`documents`, `items`, `classifications`, and `mentions` all use the same path shape:

```text
<artifact-root>/<dataset>/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
```

How rows land there:

- `documents`: rows are grouped by filing `date`, then by Python `hash(accession_number) % 64`.
- `items`: each item row is written to the same `date` and `shard` partition as its parent document partition.
- `classifications`: each classified row is written to the same `date` and `shard` partition as its source item partition.
- `mentions`: each extracted mention row is written to the same `date` and `shard` partition as its source classification partition.

Practical implication:

- one parquet file usually represents "one filing date, one shard bucket"
- the number of rows in that file is variable and depends on how many filings hashed into that bucket
- downstream stages preserve the partition shape rather than reshuffling by a new key
- stages may mark a source partition completed without writing an output parquet when that partition produces zero downstream rows

### CIK-Sharded Stages

`mention-cluster-edges` and `debt-instruments` use this path shape:

```text
<artifact-root>/<dataset>/cik_shard=NNNN/part-0000.parquet
```

How rows land there:

- the matcher reads all `mentions`
- each mention is assigned to `shard_for_cik(cik)`
- all mentions for companies whose CIK hashes to the same shard are processed together
- the matcher writes one `mention-cluster-edges` parquet and one `debt-instruments` parquet for that `cik_shard`

Practical implication:

- matching is company-scoped, not filing-date-scoped
- rows from many filing dates can coexist in the same `cik_shard` parquet
- this lets the matcher compare debt mentions across time for the same issuer

### How `batch_size` Works

`batch_size` controls the chunk size used while draining pending work in one invocation. It does not control parquet file size.

- `itemize`: processes all pending `documents` partitions, in chunks of up to `batch_size` partitions at a time.
- `classify`: processes all pending `items` partitions, in chunks of up to `batch_size` partitions at a time.
- `extract`: processes all pending `classifications` partitions, in chunks of up to `batch_size` partitions at a time.
- `match`: processes all `cik_shard` groups present in the mentions dataset, in chunks of up to `batch_size` shard groups at a time.
- `ingest`: different from the other stages; here `batch_size` is a row buffer threshold for flushing accumulated document rows to their target partitions.

Examples:

- if `extract_batch_size=100`, one extractor run drains all pending `date/shard` parquet partitions, processing them in chunks of up to 100 partitions
- if `match_batch_size=100`, one matcher run drains all shard groups, processing them in chunks of up to 100 groups, though only 64 shards currently exist
- if `ingest_batch_size=100`, ingest flushes after accumulating roughly 100 document rows, and those rows may be written into multiple `date/shard` partition files

## Dataset Schemas

### `documents`

Columns:

- `accession_number`: SEC accession number normalized by removing dashes; stable document key.
- `cik`: SEC Central Index Key for the filing issuer.
- `company_name`: Filing issuer display name from the upstream SEC manifest.
- `url`: SEC source URL for the complete submission text file.
- `text`: Decoded filing text when the document body is stored inline in the dataset.
- `date`: Filing date in `YYYY-MM-DD` format.
- `resource_uri`: Alternate storage location for the filing text when `text` is omitted, typically a local path or `s3://` URI.

Primary key: `accession_number`

### `items`

Columns:

- `item_id`: Deterministic item identifier built from `accession_number` and the SEC item number.
- `item`: SEC 8-K item number for the extracted section, for example `1.01` or `2.03`.
- `accession_number`: Parent filing accession number.
- `cik`: Filing issuer CIK copied from the parent document.
- `company_name`: Filing issuer display name copied from the parent document.
- `url`: SEC source URL copied from the parent document.
- `text`: Extracted text for the item section only.
- `date`: Filing date copied from the parent document.
- `resource_uri`: Reserved pointer for externally stored item text; currently written as `null` by the itemizer.
- `item_information`: Free-text item label parsed from the filing, such as the descriptive name that follows an item number.
- `extraction_status`: Itemizer status describing how confidently the section boundary was extracted.
- `duplicate_resolution`: Notes how duplicate or repeated item sections were resolved.
- `section_heading`: Raw heading text associated with the extracted section.
- `start_line`: 1-based line number where the item section starts in the filing text.
- `end_line`: 1-based line number where the item section ends in the filing text.
- `section_char_count`: Character count for the extracted section text.

Primary key: `item_id`

### `classifications`

Columns:

- all `items` columns: The full item row is carried forward unchanged.
- `label`: Classifier output label, currently `relevant` or `irrelevant`.
- `relevance`: Boolean convenience flag derived from `label`.
- `classification_score`: Raw model decision score used to threshold relevance; higher means more likely relevant.

Primary key: `item_id`

### `mentions`

Columns:

- `debt_instrument_mention_id`: Deterministic identifier for one extracted debt-instrument mention.
- `item_id`: Source item section identifier.
- `accession_number`: Filing accession number for the source item.
- `cik`: Issuer CIK for the source item.
- `company_name`: Issuer display name for the source item.
- `date`: Filing date for the source item in `YYYY-MM-DD` format.
- `raw_id`: Row-local extractor identifier used inside a single item during relation extraction.
- `name`: Canonicalized debt instrument name text extracted from the item.
- `start_date`: Normalized instrument start or issuance date when present.
- `end_date`: Normalized maturity, termination, or end date when present.
- `amount`: Normalized principal or commitment amount when present.
- `amendment_of`: `debt_instrument_mention_id` of the mention this row amends, when the extractor found that relation.
- `split_of`: `debt_instrument_mention_id` of the mention this row splits from, when the extractor found that relation.
- `lenders_json`: JSON array of lender or counterparty mention clusters with evidence text.
- `other_interested_parties_json`: JSON array of additional related-party clusters with evidence text.
- `name_json`: JSON payload describing the evidence tags and surface text used to construct `name`.
- `start_date_json`: JSON payload containing normalized start-date value plus extraction evidence.
- `end_date_json`: JSON payload containing normalized end-date value plus extraction evidence.
- `amount_json`: JSON payload containing normalized amount value plus extraction evidence.

Primary key: `debt_instrument_mention_id`

### `mention-cluster-edges`

Columns:

- `debt_instrument_mention_id`: Mention-level identifier from the `mentions` dataset.
- `debt_instrument_id`: Canonical debt instrument entity the matcher assigned the mention to.
- `edge_type`: Relationship type, currently `member`, `related`, or `ambiguous_candidate`.
- `match_score`: Numeric matcher confidence score for the candidate relationship.
- `candidate_rank`: Rank of this candidate among evaluated instrument candidates for the mention.
- `match_via`: Short explanation of which feature family drove the match decision.
- `evaluated_run_id`: Matcher run identifier that evaluated the relationship.

### `debt-instruments`

Columns:

- `debt_instrument_id`: Canonical entity identifier for one consolidated debt instrument history.
- `cik`: Issuer CIK shared by the instrument's directly matched mentions.
- `company_name`: Issuer display name shared by the instrument's directly matched mentions.
- `seed_debt_instrument_mention_id`: First direct mention used as the representative seed for the instrument record.
- `amendment_of_debt_instrument_id`: Parent instrument ID when this instrument is an amendment lineage child.
- `split_of_debt_instrument_id`: Parent instrument ID when this instrument is a split lineage child.
- `name`: Matcher-selected canonical instrument name derived from the instrument's direct mentions.
- `start_date`: Matcher-selected canonical start date derived from the instrument's direct mentions.
- `end_date`: Matcher-selected canonical end date derived from the instrument's direct mentions.
- `amount`: Matcher-selected canonical amount derived from the instrument's direct mentions.
- `direct_mentions_json`: JSON array of directly assigned `debt_instrument_mention_id` values for this instrument.
- `lenders_json`: JSON aggregation of lender clusters carried forward from the instrument's direct mentions.
- `other_interested_parties_json`: JSON aggregation of other related-party clusters carried forward from the instrument's direct mentions.
- `possibly_related_json`: JSON array of advisory mention IDs that look related but were not directly matched into the instrument.

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
- final snapshot parquet files are derived convenience outputs, not the canonical working state
- `cdt pipeline` writes final snapshots only when `--final-database-root` is passed
- `cdt-orchestrator` writes final snapshots when `FINAL_DATABASE_ROOT` is set or `--final-database-root` is passed before the mode
- stage completion is inferred from output partition presence plus stage completion registries for zero-row outputs
- `force=false` skips already-written partitions
- local runs and deployed runs use the same layout and code paths
- the default operating model is one active writer per environment
