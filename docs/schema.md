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

- `accession_number`: SEC accession number normalized by removing dashes; stable document key.
- `cik`: SEC Central Index Key for the filing issuer.
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

### `mention-matches`

Columns:

- `debt_instrument_mention_id`: Mention-level identifier from the `mentions` dataset.
- `debt_instrument_id`: Canonical debt instrument entity the matcher assigned the mention to.
- `matcher_status`: Match outcome for the mention, currently `singleton`, `matched`, or `ambiguous`.

### `debt-instruments`

Columns:

- `debt_instrument_id`: Canonical entity identifier for one consolidated debt instrument history.
- `cik`: Issuer CIK shared by the instrument's directly matched mentions.
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
- stage completion is inferred from output partition presence
- `force=false` skips already-written partitions
- local runs and deployed runs use the same layout and code paths
- the default operating model is one active writer per environment
