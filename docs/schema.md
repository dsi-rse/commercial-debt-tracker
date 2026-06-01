# CDT Data Schema

This document describes the persisted data model used by the `cdt` package.

CDT stores state in two layers:

- A canonical SQLite database at `{DATA_DIR}/cdt.sqlite` used for orchestration and cross-stage state.
- Append-only Parquet batches under `{DATA_DIR}/documents`, `{DATA_DIR}/items`, and `{DATA_DIR}/extractor_runs` used as stage artifacts and audit outputs.

## Pipeline Overview

The main flow is:

1. `ingest` indexes SEC filings in `documents` and optionally writes raw filing text to document Parquet batches.
2. `itemizer` extracts relevant 8-K item sections into item Parquet batches and indexes them in `items`.
3. `classifier` adds relevance labels to `items`.
4. `extractor` writes debt-instrument mention rows into `debt_instrument_mentions`.
5. `matcher` consolidates mentions into canonical rows in `debt_instrument`.

## SQLite Database

The SQLite schema is defined in [src/cdt/database.py](/home/tspread/idi/cdt/commercial-debt-tracker/src/cdt/database.py:11).

### `documents`

One row per SEC filing accession number.

| Column | Type | Meaning |
| --- | --- | --- |
| `accession_number` | `TEXT` | Primary key. SEC accession number normalized without dashes. |
| `cik` | `TEXT` | Issuer CIK, normalized as text. |
| `url` | `TEXT` | Source SEC or scraper URL for the complete submission text file. |
| `resource_uri` | `TEXT` | Canonical storage location for the source document, typically an `s3://bucket/key` URI. |
| `date` | `TEXT` | Filing date in ISO `YYYY-MM-DD` format. |
| `batch_path` | `TEXT` | Path to the document Parquet batch when filing text was downloaded locally; `NULL` for index-only rows. |
| `status` | `TEXT` | Pipeline stage for the document. Current values are `indexed`, `downloaded`, and `itemized`. |
| `updated_at` | `TEXT` | UTC timestamp for the latest write to this row. |

Notes:

- `indexed` means CDT knows the filing exists but may not have downloaded full text.
- `downloaded` means the full document text was saved into a document Parquet batch.
- `itemized` means the document has already been processed into item rows.

Indexes:

- `idx_documents_cik`
- `idx_documents_date`
- `idx_documents_status`

### `items`

One row per extracted 8-K item section selected by the itemizer.

| Column | Type | Meaning |
| --- | --- | --- |
| `item_id` | `TEXT` | Primary key. Deterministic identifier built from accession number and item number, such as `000114036126006577-8-01`. |
| `accession_number` | `TEXT` | Parent filing accession number. |
| `item` | `TEXT` | 8-K item number, such as `1.01` or `8.01`. |
| `batch_path` | `TEXT` | Path to the item Parquet batch containing the full extracted section text and metadata. |
| `status` | `TEXT` | Current stage for the item. Current values are `itemized`, `classified`, `extracted`, and `extraction_failed`. |
| `updated_at` | `TEXT` | UTC timestamp for the latest write to this row. |
| `label` | `TEXT` | Classifier output label, currently `relevant` or `irrelevant`. |
| `relevance` | `INTEGER` | Boolean-like classifier output stored as `0` or `1`. |
| `classification_score` | `REAL` | Classifier score after thresholding logic. |
| `classified_at` | `TEXT` | UTC timestamp when the classifier last updated this row. |
| `extracted_at` | `TEXT` | UTC timestamp when extraction succeeded or failed. |
| `extractor_model` | `TEXT` | LLM model identifier used for extraction. |
| `extractor_reasoning` | `TEXT` | Extractor reasoning setting used for the LLM call. |
| `extractor_run_path` | `TEXT` | Path to the extractor run directory that contains `full.jsonl` audit records. |
| `extractor_error` | `TEXT` | Failure summary when extraction ends in `extraction_failed`. |

Notes:

- The `items` table is a stage index, not the full item payload. The actual text and section metadata live in item Parquet batches.
- Re-itemizing an `item_id` resets downstream classifier and extractor fields.

Indexes:

- `idx_items_accession_number`
- `idx_items_status`

### `debt_instrument_mentions`

One row per extracted debt-instrument mention from one item. A debt instrument mention is all the extracted properties associated with a single debt instrument in a single item text.

| Column | Type | Meaning |
| --- | --- | --- |
| `debt_instrument_mention_id` | `TEXT` | Primary key. Stable content-derived identifier for one extracted mention. |
| `item_id` | `TEXT` | Parent item row in `items`. |
| `raw_id` | `TEXT` | Per-item local identifier assigned during extraction, used in intermediate relation prompts. |
| `name` | `TEXT` | Canonicalized debt instrument name chosen from the mention cluster. |
| `start_date` | `TEXT` | Canonicalized start date, when available. |
| `end_date` | `TEXT` | Canonicalized end date, when available. |
| `amount` | `TEXT` | Canonicalized amount string, when available. |
| `amendment_of` | `TEXT` | Mention-level lineage pointer to another `debt_instrument_mention_id` if this mention amends a prior one. |
| `split_of` | `TEXT` | Mention-level lineage pointer to another `debt_instrument_mention_id` if this mention splits a prior one. |
| `lenders_json` | `TEXT` | JSON list of lender clusters extracted from the item text. |
| `other_interested_parties_json` | `TEXT` | JSON list of non-lender party clusters associated with the mention. |
| `name_json` | `TEXT` | JSON payload describing the evidence cluster for the instrument name itself. |
| `start_date_json` | `TEXT` | JSON payload describing evidence tags used for the standardized `start_date`. |
| `end_date_json` | `TEXT` | JSON payload describing evidence tags used for the standardized `end_date`. |
| `amount_json` | `TEXT` | JSON payload describing evidence tags used for the standardized `amount`. |
| `debt_instrument_id` | `TEXT` | Canonical debt instrument assigned by the matcher. |
| `matcher_status` | `TEXT` | Matcher outcome for this mention. Current values are `singleton`, `matched`, and `ambiguous`. |
| `matched_at` | `TEXT` | UTC timestamp when the matcher last processed this row. |
| `updated_at` | `TEXT` | UTC timestamp for the latest write to this row. |

Notes:

- This table is replaced item-by-item by the extractor.
- The JSON columns are part of the public persisted schema in practice; downstream consumers should expect them to be serialized JSON strings in SQLite.

Indexes:

- `idx_debt_instrument_mentions_item_id`
- `idx_debt_instrument_mentions_matcher_status`
- `idx_debt_instrument_mentions_debt_instrument_id`

### `debt_instrument`

Canonicalized debt instruments built by consolidating mention rows.

| Column | Type | Meaning |
| --- | --- | --- |
| `debt_instrument_id` | `TEXT` | Primary key. Canonical instrument identifier, currently based on the earliest direct mention in the group. |
| `cik` | `TEXT` | Issuer CIK for the canonical instrument. |
| `seed_debt_instrument_mention_id` | `TEXT` | Mention ID used as the seed for this canonical row. |
| `amendment_of_debt_instrument_id` | `TEXT` | Parent canonical instrument if this row represents an amendment lineage step. |
| `split_of_debt_instrument_id` | `TEXT` | Parent canonical instrument if this row represents a split lineage step. |
| `name` | `TEXT` | Best available canonical name chosen from cumulative mentions. |
| `start_date` | `TEXT` | Best available canonical start date chosen from cumulative mentions. |
| `end_date` | `TEXT` | Best available canonical end date chosen from cumulative mentions. |
| `amount` | `TEXT` | Best available canonical amount chosen from cumulative mentions. |
| `direct_mentions_json` | `TEXT` | JSON array of mention IDs directly assigned to this canonical instrument. |
| `lenders_json` | `TEXT` | JSON list of deduplicated lender clusters across the instrument lineage. |
| `other_interested_parties_json` | `TEXT` | JSON list of deduplicated non-lender party clusters across the instrument lineage. |
| `possibly_related_json` | `TEXT` | JSON list of mention IDs that look related under looser matching logic but were not directly merged. |
| `created_at` | `TEXT` | UTC timestamp when the current canonical row set was written. |
| `updated_at` | `TEXT` | UTC timestamp when the current canonical row was last rewritten. |

Notes:

- The matcher currently rewrites the full `debt_instrument` table on each run.
- Canonical field values are chosen from the most recent usable mention in the lineage.

Indexes:

- `idx_debt_instrument_cik`
- `idx_debt_instrument_amendment_of`
- `idx_debt_instrument_split_of`

### `active_debt_instruments` View

`active_debt_instruments` is a derived SQLite view that returns only the currently active end-state debt instruments.

An instrument is considered active when no child row points to it through `amendment_of_debt_instrument_id` or `split_of_debt_instrument_id`.

The view includes all columns from `debt_instrument` plus:

| Column | Type | Meaning |
| --- | --- | --- |
| `mentions_json` | Derived JSON text | JSON array of all mention IDs reachable through the active instrument’s lineage, including ancestor direct mentions. |

## Parquet Artifact Schemas

The SQLite index tables intentionally do not store every text field. Full stage payloads live in Parquet batches.

### Document Batches

Written under `{DATA_DIR}/documents/document-batch-*.parquet`.

Columns come from `cdt.ingest.DOCUMENT_COLUMNS`:

| Column | Meaning |
| --- | --- |
| `accession_number` | Filing accession number. |
| `cik` | Issuer CIK. |
| `url` | Source URL for the complete submission text file. |
| `text` | Full decoded SEC submission text. |
| `date` | Filing date. |

Notes:

- These batches exist only when ingest runs with `download=True`.
- The `documents` SQLite table points to the batch via `batch_path`, but does not duplicate `text`.

### Item Batches

Written under `{DATA_DIR}/items/item-batch-*.parquet`.

Columns come from `cdt.itemizer.core.ITEM_COLUMNS`:

| Column | Meaning |
| --- | --- |
| `item_id` | Deterministic item identifier. |
| `item` | 8-K item number. |
| `accession_number` | Parent filing accession number. |
| `cik` | Issuer CIK. |
| `url` | Source filing URL. |
| `text` | Extracted item section text. |
| `date` | Filing date. |
| `item_information` | Normalized `ITEM INFORMATION:` header value from the source filing. |
| `extraction_status` | Itemizer result such as `ok`, `duplicate_heading`, `missing_heading`, or `unmapped_item_information`. |
| `duplicate_resolution` | How duplicate item headings were resolved, such as `single_heading`, `benign_equivalent`, or `benign_contained`. |
| `section_heading` | Selected heading line for the extracted section. |
| `start_line` | One-based starting line number in the normalized body text. |
| `end_line` | One-based ending line number in the normalized body text. |
| `section_char_count` | Character length of `text`. |

### Extractor Audit Files

Written under `{DATA_DIR}/extractor_runs/run-*/full.jsonl`.

Each line is a JSON object representing one processed item and includes:

- Item identity: `item_id`, `accession_number`, `item`
- `stage_responses`: raw stage outputs keyed by stage name
- `debt_instrument_mentions`: extracted mention rows prior to SQLite persistence
- `state`: terminal workflow state such as `SUCCESS`, `FAILED`, or `ERROR`
- `attempts`: per-stage prompt, response, validation, and retry history

This file is the most detailed provenance record for LLM extraction behavior.

## Status Fields

### Document statuses

| Value | Meaning |
| --- | --- |
| `indexed` | Filing metadata has been indexed, but text may not have been downloaded. |
| `downloaded` | Filing text was downloaded into a document Parquet batch. |
| `itemized` | Filing was processed into item rows. |

### Item statuses

| Value | Meaning |
| --- | --- |
| `itemized` | Item section exists and is awaiting classification. |
| `classified` | Classifier output has been written. |
| `extracted` | Extractor succeeded and mention rows were written. |
| `extraction_failed` | Extractor attempted the row and persisted a failure summary. |

### Matcher statuses

| Value | Meaning |
| --- | --- |
| `singleton` | No strong enough canonical match was found; the mention stands alone. |
| `matched` | The mention was merged into an existing canonical instrument. |
| `ambiguous` | The matcher found plausible candidates but did not merge the mention into another canonical instrument. |

## Relationships

- `documents.accession_number -> items.accession_number`
- `items.item_id -> debt_instrument_mentions.item_id`
- `debt_instrument_mentions.debt_instrument_id -> debt_instrument.debt_instrument_id`
- `debt_instrument_mentions.amendment_of -> debt_instrument_mentions.debt_instrument_mention_id`
- `debt_instrument_mentions.split_of -> debt_instrument_mentions.debt_instrument_mention_id`
- `debt_instrument.amendment_of_debt_instrument_id -> debt_instrument.debt_instrument_id`
- `debt_instrument.split_of_debt_instrument_id -> debt_instrument.debt_instrument_id`

## Things To Watch

- Several important fields are stored as serialized JSON strings inside SQLite.
- `items` is not a complete item payload table; full text lives in Parquet.
- `debt_instrument` is a derived canonical table, not raw extraction output.
- The matcher currently rewrites canonical instrument state globally rather than incrementally.
