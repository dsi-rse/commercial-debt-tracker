# CDT Storage Schema

CDT writes canonical artifacts under one artifact root. That root can be a local directory or an `s3://` prefix.

The pipeline can also write optional final snapshot parquet files under a separate final database root. Those snapshots are flattened table-wide exports intended for downstream database loading and are not the canonical working state for the pipeline.

Text columns are normalized on the way into a snapshot: a cell whose whole value is a placeholder such as `nan`, `none`, `null`, `<na>`, or `n/a` is written as a null. Partitions written before a column existed, or by a stage that stringified a missing value, would otherwise publish the placeholder as if it were real text.

Two schema-wide contracts:

- `cik` columns carry SEC's canonical 10-digit zero-padded form (#153). Snapshots pad legacy unpadded partitions on the way out, and `shard_for_cik` hashes the unpadded form so existing `cik_shard` partitions stay where they are.
- Every evidence payload records `spans`: a list of `{tag_id, char_start, char_end, text}` whose offsets index the source item's `text` **exactly** (#154) — the extractor realigns model output whose whitespace drifted. Value-bearing payloads also record `derived_from`: `"stated"` when the value was parsed from cited evidence, `"name"` when it was derived from the instrument's own name (a `due 2028` maturity, a `$183.36 million term loan` principal, a coupon in the name), `"computed"` for arithmetic over cited spans, `"inherited"` for a term a synthesized row carries unchanged from the row it was minted from (#203), and null when there is no value (#128).

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
  extract-batches/
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

- `documents` shards by `crc32(accession_number) % 64`, computed by `datasets.shard_label` — the single source of the shard contract (crc32, modulo, four-digit label). Changing it strands every existing partition, so it is defined nowhere else (#61).
- `items`, `classifications`, and `mentions` preserve their source document `date` and `shard`
- CIK shards are `crc32 % 64` of the **unpadded** CIK, so the padded and unpadded spellings of one CIK always land in the same shard (#153)
- `documents` currently use 64 date shards
- downstream date-partitioned datasets currently preserve whichever document shards they read
- CIK-sharded datasets currently use 64 shards
- Shard assignment is stable across runs and processes. It was not always: before #61, `documents` used Python's builtin `hash`, which is salted per process, so one accession landed in a different shard on every run and a forced re-ingest wrote a second copy that per-partition dedup could not see. That is fixed, not pending. `ingest.repair_document_shards` sweeps any surviving pre-#61 stray into its canonical shard on `--force`, dropping the stray copy when the canonical one was just refreshed.

### What "Partition", "Shard", and "Batch" Mean

- A `partition` is one physical parquet file at a canonical path such as `documents/date=2026-05-31/shard=0017/part-0000.parquet`.
- A `shard` is the hash bucket inside a dataset's partitioning scheme. `documents` currently use 64 shards, rendered as `0000` through `0063`. Downstream date-partitioned datasets preserve source document shard values. CIK-sharded datasets also use 64 shards, rendered as `0000` through `0063`.
- For `documents`, all rows for the same filing date are split across 64 shard files by `crc32(accession_number) % 64`.
- For `items`, `classifications`, and `mentions`, rows keep the `date` and `shard` partition of their upstream source partition.
- For CIK-sharded datasets, rows are split across 64 shard files by `crc32` of the unpadded CIK, regardless of filing date.
- A `batch` is not a second storage layer. It is just the internal chunk size one pipeline invocation uses while draining all work in scope.

### Date-Partitioned Stages

`documents`, `items`, `classifications`, and `mentions` all use the same path shape:

```text
<artifact-root>/<dataset>/date=YYYY-MM-DD/shard=NNNN/part-0000.parquet
```

How rows land there:

- `documents`: rows are grouped by filing `date`, then by `crc32(accession_number) % 64`.
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
- each mention is assigned to `shard_for_cik(cik)`, which is `crc32` of the unpadded CIK modulo 64
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

### Column types

Every column is **nullable text** unless it appears below. A missing value is
null, and the readers treat the placeholder strings `nan`, `none`, `null`,
`<na>` and `n/a` as missing too, because parquet round-trips a missing value as
NaN and `str(float("nan"))` is the literal text `nan`.

| column | physical type | notes |
|---|---|---|
| `principal_amount`, `outstanding_balance` | `decimal128(38, 2)` | Exact money. Not a float: `float("372246148.11")` is not that number, and rendering it at fixed precision leaked the difference, which is what made every amount carrying cents publish as null (#119). Not text either: a text column sorts `962500000` before `2000000000` (#185). |
| `interest_rate_pct` | `decimal128(9, 4)` | Exact percentage. Four places carries basis points; the corpus uses at most three. Published canonical, so one rate has one spelling — it previously persisted the model's own text, giving 141 distinct strings for 115 distinct rates. |
| `mention_count`, `document_count` | `int64` | |
| `is_lineage_head`, `relevance`, `synthesized_only`, `outstanding_balance_as_of_is_filing_date` | `bool` | |
| `classification_score` | `double` | A model decision score, not a measured quantity. |
| `match_score` | `double` | |
| `start_line`, `end_line`, `section_char_count`, `candidate_rank` | `int64` | |

Decimal columns read back scale-padded (`Decimal("5.0000")` for a rate stored at
scale four). The pipeline's own readers canonicalise that to one spelling, since
every internal comparison — match keys, prior-amount equality — is textual.

Types are declared once, at the single parquet write path, so they hold for
every dataset and every partition — including a partition whose values are all
null, and an empty one. Inferring them per write made the physical type a
function of the data: a column with no value in one partition serialised as
parquet `null` and as `string` in the next, which is why 23 of 42
`debt-instruments` columns disagreed across partitions and `pyarrow.dataset`,
`pq.read_table`, `ParquetDataset` and `pandas.read_parquet` all failed on the
directory with "Unsupported cast from string to null" (#187).

So any standard parquet reader can be pointed at a dataset directory:

```python
import pandas as pd
pd.read_parquet("<artifact-root>/debt-instruments")
```

Partitions written before that fix keep the types they were written with, and a
dataset that mixes the two is readable only in the order that happens to put a
typed partition first. A root carried over from an earlier run therefore needs
rebuilding once (#107).

Dates are text in `YYYY-MM-DD`, not a date type: a year-only maturity normalizes
to `YYYY-12-31` carrying `derived_from: "name"`, and the distinction between a
stated and a derived date lives in the fact payload rather than the column.

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
- `item_information`: Canonical SEC caption for the item this row covers, read from an `ITEM INFORMATION:` line in the filing's `<SEC-HEADER>` block and normalized to lowercase, such as `entry into a material definitive agreement`. This column drives the table: the itemizer reads the header's captions, maps each one to the item number in `item`, then searches the document body for the matching section. A caption the itemizer does not recognize still produces a row, with an empty `item`. Captions that repeat, or that map to an item number another caption already claimed, are dropped so that `item_id` stays unique (#74).
- `extraction_status`: Whether the itemizer located this item's section in the document body, and whether the headings it found were ambiguous. Not a confidence score. One of:
  - `ok`: A body heading for `item` was found and the section boundaries resolved, either from a single heading or from several the itemizer judged benign. `duplicate_resolution` records which case applied.
  - `duplicate_heading`: Several body headings carried `item` and their sections differ materially, so the itemizer could not tell which one the header caption meant. The first is used; treat `text` as one of several possible readings of the filing.
  - `missing_heading`: The header declared the item but no body heading matches it. No section was extracted, so `text` and `section_heading` are empty and `start_line`, `end_line` are null.
  - `unmapped_item_information`: `item_information` is not a caption the itemizer maps to an item number, so no extraction was attempted. `item` is empty, as are the section fields listed above.

  No stage downstream filters on this column, so the classifier and the extractor both see the empty-text rows produced by the last two statuses.
- `duplicate_resolution`: How the itemizer chose between body headings carrying this item number. Populated for every row that reached extraction, not only for rows with more than one heading. Comparisons use a normalized form of each candidate section: casefolded, punctuation collapsed, and the item-number heading line itself dropped. One of:
  - `single_heading`: Exactly one body heading carried the item number.
  - `benign_equivalent`: Several headings, and one candidate section is equivalent to every other, meaning either that one contains the other or that their token sets overlap by at least 0.95. That candidate is kept.
  - `benign_contained`: Several headings, and one candidate section contains every other. Rare, because the equivalence check above already covers containment for non-empty sections; this value effectively marks the case where a competing heading's section normalizes to nothing.
  - `unresolved_duplicate`: Several headings whose sections differ materially. This is the value that sets `extraction_status` to `duplicate_heading`.
  - Empty string: no extraction was attempted, so the row is `missing_heading` or `unmapped_item_information`.
- `section_heading`: The body heading line the itemizer selected as the start of the section, verbatim from the filing and not normalized, such as `Item 1.01. Entry into a Material Definitive Agreement.`. Distinct from `item_information`, which carries SEC's own caption from the filing header rather than the text the filer wrote. Empty when no heading matched.
- `start_line`: 1-based inclusive line number where the section in `text` begins, which is the line holding `section_heading`. Line numbers index the normalized lines of the filing's primary 8-K document block, not `documents.text`, so they cannot be used to slice that column directly. Null when no section was extracted. Recorded for provenance and for debugging section boundaries; nothing downstream reads it.
- `end_line`: 1-based inclusive line number of the last line in `text`. The section ends at whichever comes first: the line before the next body heading carrying a different item number, the line before a `SIGNATURES` or `EXHIBIT INDEX` line, or the end of the body. Indexed and nulled the same way as `start_line`.
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

- `debt_instrument_mention_id`: Deterministic identifier for one extracted debt-instrument mention (a content hash over the extracted fields plus `item_id`).
- `item_id`: Source item section identifier.
- `accession_number`: Filing accession number for the source item.
- `cik`: Issuer CIK for the source item, zero-padded.
- `company_name`: Issuer display name for the source item.
- `date`: Filing date for the source item in `YYYY-MM-DD` format.
- `raw_id`: Row-local extractor identifier used inside a single item during relation extraction.
- `name`: Canonicalized debt instrument name text extracted from the item.
- `instrument_type`: One of `term_loan`, `revolving_credit`, `credit_line`, `note_bond`, or null when none fits or the document does not say (#156).
- `start_date`: Normalized instrument start or issuance date when present — the current `closing` fact in `dates_json`. An instrument whose status is `announced` has not started and carries none; its projected close lives in `dates_json` as a `closing` fact marked `expected` (the stage-1 `expected_closing` kind is rewritten to that shape on the way in). The publisher's expected-start leg (`dsi-rse/commercial-debt-tracker-website#15`) is measured against it.
- `maturity_date`: Normalized final maturity or expiration of the obligation — when the borrowed money must be repaid (#158). Year-only maturities normalize to `YYYY-12-31` with `derived_from: "name"`.
- `commitment_termination_date`: When the lender's obligation to lend ends — the close of a draw, availability, or revolving period — when the document states one distinct from the maturity (#158). Null for notes and bonds.
- `principal_amount`: The single commitment or principal figure, as digits with at most one decimal point. Balances, draws, repayments, and proceeds never populate this column (#140).
- `principal_currency`: ISO 4217 code for `principal_amount` when stated.
- `principal_amount_kind`: `commitment` or `principal`; null on rows replayed from the pre-#140 single-amount shape.
- `status`: What this mention says happened to the instrument, derived from its event facts in `dates_json`: the newest completed event wins (`retirement`→`repaid`, `termination`→`terminated`, `exchange`→`exchanged`, `default`→`defaulted`, `amendment`→`amended`, `closing`→`entered_into`, `announcement`→`announced`); an instrument whose only closing is `expected` is `announced`; a planned retirement — a `retirement` fact marked `expected` — decides nothing here and stays in `dates_json` for the publisher's cascade. Null when the mention states no event. Pre-stage-2 responses that carried a `status_event` replay it verbatim.
- `status_date`: The date of the event that decided `status`, when stated.
- `interest_rate_kind`: `fixed` or `floating` (#157).
- `interest_rate_pct`: The stated fixed or all-in rate as a numeric string, parser-verified against the cited evidence or the instrument's name. Null for floating rates; benchmarks and margins are not recorded.
- `amendment_of`: `debt_instrument_mention_id` of the mention this row amends, when the extractor found that relation.
- `split_of`: `debt_instrument_mention_id` of the mention this row splits from, when the extractor found that relation.
- `retired_by_json`: JSON array of `debt_instrument_mention_id`s of the mentions that retired this row's obligation. The pointer sits on the retired instrument's mention; proceeds-financed redemption counts (#142).
- `parties_json`: JSON array of every party cluster with `canonical_name` (longest span), `role` (`lender`, `borrower`, `agent`, `trustee`, `underwriter`, `guarantor`, `other`), `kind` (`named` or `collective`), and evidence `spans` (#150). The extractor returns one `parties` list with a role per cluster; nothing the model labels is discarded.
- `name_json`: Evidence payload (`spans`) for `name`.
- `start_date_json`, `maturity_date_json`, `commitment_termination_date_json`: Evidence payloads with `normalized_date` and `derived_from`.
- `amounts_json`: JSON array of kind-typed money facts (#140): `{kind, normalized_amount, currency, as_of_date, spans, derived_from, prior}` with `kind` one of `commitment`, `principal`, `outstanding_balance`, `draw`, `repayment`, `proceeds` (null on legacy replays). `as_of_date` is normally present only on balances. `prior` is true for a figure the filing states as it stood before an amendment (`from $25,000,000 to $50,000,000`); prior facts never supply `principal_amount`.
- `dates_json`: JSON array of kind-typed date facts: `{kind, normalized_date, precision, prior, expected, spans, derived_from}`. Kinds: `agreement` (the instrument's own dated-as-of date), `announcement`, `closing` (closing, issuance, funding, effective — the start), `amendment`, `repayment` (a payment that leaves the obligation outstanding), `retirement` (repaid in full, redeemed in whole, defeased, discharged), `termination`, `exchange`, `default`, `maturity`, `commitment_termination`. `precision` is `day`, `month` or `year` for how precisely the cited text states the date. `prior` marks a term stated as it stood before an amendment; `expected` marks a date the filing states as planned rather than occurred (an expected closing, a noticed redemption). An event the filing states without a date is a fact with `normalized_date` null. The flat `start_date`, `maturity_date` and `commitment_termination_date` columns are the current (non-prior, non-expected) `closing`, `maturity` and `commitment_termination` facts. Responses in the pre-dates[] shape replay with the kind implied by the old property name.
- `status_json`: `{status, status_date}` where `status_date` is a full evidence payload (#141).
- `interest_rate_json`: `{kind, rate_pct, spans, derived_from}` (#157).
- `lender_disclosure`: How completely this mention identifies who holds the debt, derived from the party clusters: `complete` when every `lender` cluster is `named`; `collective_present` when any is `collective` (`the other lenders party thereto`); `none_named` when the mention names no lender at all (a public-market series, a redemption notice, a syndicate where only the agent is named). Replaces the `lenders_known_incomplete` boolean, which was true for the second and third cases alike and so could not distinguish "something is undisclosed" from "nothing was disclosed here". Pre-stage-2 responses replay the flag the model declared, mapped onto these values.
- `synthesized_by`: The rule that minted this row, when the extractor synthesized it rather than the model returning it (#203); null on every model-emitted row. The only value today is `prior_state`: the state of an amended instrument before the amendment, built from the `prior`-marked terms on the object that describes the amendment.
- `synthesized_from_mention_id`: The `debt_instrument_mention_id` of the model-emitted row a synthesized row was minted from — always in the same item. Neither column is hashed into the mention id.

Primary key: `debt_instrument_mention_id`

Rows publish from extractor states `SUCCESS` and `PARTIAL` (#152). A `PARTIAL` row salvaged what a terminal failure left intact — individually valid entries after a final `instrument_ie` validation failure, or mentions without lineage after a terminal `instrument_relation` failure — and also carries a failure-registry entry recording what was lost.

#### Evidence payload shapes

Every `_json` column on `mentions` holds a JSON document serialized into a text
column, not a nested parquet type. The column descriptions above give each
one's meaning; this section gives their structure, which is shared.

Three container shapes:

- **One fact object** — `name_json`, `start_date_json`, `maturity_date_json`,
  `commitment_termination_date_json`, `status_json`, `interest_rate_json`.
  Always written, even when the fact carries no value: a payload whose value
  key is null while `spans` is populated is how "the filing discusses this, and
  here is where, but states no resolvable value" is recorded. That is a
  different claim from an empty payload, and both are different from the flat
  column being null.
- **An array of fact objects** — `amounts_json`, `dates_json`, `parties_json`.
  One element per fact the filing states; `[]` when it states none. Element
  order is the extractor's and carries no meaning, so consumers must key on the
  attributes rather than on position.
- **An array of identifiers** — `retired_by_json` holds
  `debt_instrument_mention_id` strings and no evidence.

##### `spans`

Fact objects carry a `spans` list, and its element shape is the same
everywhere:

```json
{"tag_id": "tag-6", "char_start": 186, "char_end": 219,
 "text": "Convertible Senior Notes due 2031"}
```

- `char_start`, `char_end`: a half-open offset pair indexing the **parent
  item's own `text`** exactly (#154) — not the filing, and not
  `documents.text`. The extractor realigns model output whose whitespace
  drifted so this holds. `items.text[char_start:char_end] == text`, which makes
  `text` redundant and recoverable; it is stored anyway so a consumer can
  render evidence without joining back to `items`.
- `tag_id`: the NER tag this span came from, stable within one item. It is how
  two facts are known to cite the same piece of text.
- An empty `spans` list means the value was not cited. That is legitimate for a
  name-derived value — a `notes due 2028` maturity has no separate evidence
  span — and those payloads carry `derived_from: "name"`.
- `status_json` is the one exception to the placement: it has no top-level
  `spans` at all. Its evidence sits one level down, in
  `status_json.status_date.spans`, and `status_date` is itself null when the
  filing states the event without a date. A consumer that walks `spans` at the
  top level of every payload silently collects nothing for status.

##### Fact attributes

| attribute | appears on | meaning |
|---|---|---|
| `kind` | `dates_json`, `amounts_json`, the three date payloads, `interest_rate_json`, `parties_json` | Which fact this is. Per-column vocabularies are listed in the column descriptions above. |
| `normalized_date` | date payloads, `dates_json` | `YYYY-MM-DD`, or null when no date resolves. |
| `normalized_amount` | `amounts_json` | Digits with at most one decimal point. |
| `rate_pct` | `interest_rate_json` | Numeric string; null for floating rates. |
| `derived_from` | every value-bearing payload | `"stated"` when parsed from cited evidence, `"name"` when read out of the instrument's own name, `"computed"` when arithmetic over cited spans produced it (a summed increase, a tenor added to a closing date), `"inherited"` when a synthesized row carries a term unchanged from the row it was minted from (#203), null when there is no value (#128). |
| `precision` | date payloads, `dates_json` | `day`, `month`, or `year` — how precisely the cited text states the date, not how precisely `normalized_date` is written. |
| `prior` | `dates_json`, `amounts_json` | True for a term stated as it stood *before* an amendment (`from $25,000,000 to $50,000,000`). Prior facts never supply a flat column on the row that states them; a synthesized prior state (below) carries them flipped, as its own current terms. |
| `expected` | date payloads, `dates_json` | True for a date the filing states as planned rather than occurred — an expected closing, a noticed redemption. |
| `as_of_date` | `amounts_json` | Normally present only on balances. |
| `currency` | `amounts_json` | ISO 4217. |
| `role`, `kind`, `canonical_name` | `parties_json` | Role in the instrument, `named` or `collective`, and the longest span's text as the cluster's key (#150). |
| `status`, `status_date`, `derived_from_kind` | `status_json` | `status_date` is itself a full evidence payload, nested one level deeper. `derived_from_kind` names the `dates_json` event kind that decided the status. |

##### Worked examples

A value-bearing single payload, `start_date_json`:

```json
{"kind": "agreement", "normalized_date": "2022-09-20", "precision": "day",
 "derived_from": "stated", "prior": false, "expected": false,
 "spans": [{"tag_id": "tag-27", "char_start": 6709, "char_end": 6727,
            "text": "September 20, 2022"}]}
```

An array payload where one fact is a pre-amendment figure, `amounts_json`:

```json
[{"kind": "commitment", "normalized_amount": "1500000000", "currency": "USD",
  "as_of_date": null, "derived_from": "stated", "prior": true,
  "spans": [{"tag_id": "tag-6", "char_start": 448, "char_end": 462,
             "text": "$1,500,000,000"}]},
 {"kind": "commitment", "normalized_amount": "2500000000", "currency": "USD",
  "as_of_date": null, "derived_from": "stated", "prior": false,
  "spans": [{"tag_id": "tag-7", "char_start": 466, "char_end": 480,
             "text": "$2,500,000,000"}]}]
```

A payload that found evidence but no resolvable value, `maturity_date_json` —
the filing says "fifth anniversary of the Initial Secured Loan Closing Date":

```json
{"kind": "maturity", "normalized_date": null, "precision": null,
 "derived_from": null, "prior": false, "expected": false,
 "spans": [{"tag_id": "tag-24", "char_start": 907, "char_end": 924,
            "text": "fifth anniversary"},
           {"tag_id": "tag-25", "char_start": 932, "char_end": 965,
            "text": "Initial Secured Loan Closing Date"}]}
```

The nested `status_date` payload inside `status_json`:

```json
{"status": "entered_into", "derived_from_kind": "agreement",
 "status_date": {"normalized_date": "2022-09-20", "derived_from": "stated",
                 "spans": [{"tag_id": "tag-27", "char_start": 6709,
                            "char_end": 6727, "text": "September 20, 2022"}]}}
```

##### Synthesized rows

An amendment 8-K extracts as **one** object — the terms as amended, plus every
old term the filing states marked `prior`. That leaves nothing for
`amendment_of` to name, so the extractor mints the instrument's prior state as
its own mention row (#203): `synthesized_by = "prior_state"`,
`synthesized_from_mention_id` naming the amended object, and the amended
object's `amendment_of` naming the minted row. Nothing chooses a predecessor;
the row is constructed from the amended object's own cited `prior` facts.

What a minted row carries, and how to read it:

- The `prior`-marked commitment/principal, agreement, maturity or
  commitment-termination facts, **flipped to `prior: false`** — on this row
  they are the current terms. Their spans are the successor's; the text they
  cite is the same before-figure.
- Every other commitment/principal, maturity or commitment-termination fact of
  a kind the filing marked no `prior` value for, copied with
  `derived_from: "inherited"`: the filing gives no evidence the term changed.
  This is the one assertion the rule makes that a filing cannot confirm — a
  filing stating a *new* value with no before-figure looks identical — and the
  marker is what lets a reader tell an inherited term from a stated one.
- The origin: a `prior`-marked `agreement` when there is one (the
  predecessor's own dated-as-of), else the current `closing` and `agreement` —
  the same agreement's dates — unless they equal an `amendment` date, in which
  case they are the restatement's own and the row has **no** `start_date`.
- No event facts (`amendment`, `repayment`, `retirement`, …): events belong to
  the filing's own moment. No balances, draws, repayments or proceeds, for the
  same reason.
- `parties_json` holds the borrower only, and `lender_disclosure` is
  `none_named`: a joinder adds and removes lenders, so the filing never states
  who lent under the earlier terms.
- `name`, `name_json`, `instrument_type`, `interest_rate_*` and the filing
  metadata are the successor's. `raw_id` is the successor's with `-prior`.

A minted row's id comes from the same hash as any other; the successor's id
does not change (`amendment_of` and the `synthesized_*` columns are not
hashed). Downstream, the matcher never lets a synthesized member supply a
cluster's canonical fields or widen its name class, may borrow the successor's
lender signature for scoring only, and publishes `synthesized_only` on an
instrument whose every member is synthesized. `cdt backfill-mentions` mints
over partitions written before #203; it is a pure function of the model-emitted
rows and a no-op when run twice.

##### Reading them safely

- **Older partitions key spans as `mentions`, not `spans`.** Partitions written
  before the evidence-shape change (#128) use the old key with the same element
  shape. `matcher.cluster_canonical_key` carries the fallback; any new consumer
  needs it too. This is schema versioning living inside an opaque string, where
  neither the column types declared above nor a parquet reader can see it.
- **A missing column does not read as a missing value.** `read_table` reindexes
  an absent column to `NaN`, and `NaN` is truthy, so the common
  `json.loads(str(value or "[]"))` idiom passes it the literal text `nan` and
  raises `JSONDecodeError` (#193). Test for `pd.isna` before the `or`.
- **Do not assume a payload's presence implies a value**, or a value implies
  evidence. The three cases — no payload, payload without a value, value
  without spans — are distinct and all occur.

##### Observed multiplicity

Measured over a 669-mention window, for anyone sizing a consumer or a rewrite:
**11.1 fact objects and 18.5 spans per mention**, 1.7 spans per fact. The
distribution is skewed: `parties_json` accounts for 8.4 spans per mention and
`name_json` for 5.1, because a defined term like `the Company` or `Notes` is
tagged at every recurrence. Serialized, the payloads are ~2.7 KB per mention
uncompressed against ~200 bytes for all the flat columns together, so the
evidence is effectively the whole row.

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
- `cik`: Issuer CIK shared by the instrument's directly matched mentions, zero-padded.
- `company_name`: Issuer display name resolved from the instrument's directly matched mentions, falling back to the newest name any mention for the same CIK carries.
- `seed_debt_instrument_mention_id`: First direct mention used as the representative seed for the instrument record.
- `amendment_of_debt_instrument_id`: Parent instrument ID when this instrument is an amendment lineage child. Every lineage pointer on this table carries a relation the **extractor** asserted and cited; the matcher resolves the `raw_id` to an instrument ID but never invents the relation. A pointer that records a matcher inference rather than an extracted fact needs a provenance column alongside it, documented here, and that column has to survive an incremental rematch — see the stage boundary in `docs/architecture.md` (#184).
- `retired_by_debt_instrument_ids`: JSON array of IDs of the instruments that retired this one, set on the retired instrument's own row (null when none).
- `split_of_debt_instrument_id`: Parent instrument ID when this instrument is a split lineage child.
- `amendment_inferred_by`: Which rule inferred `amendment_of_debt_instrument_id`, when the matcher filled it rather than the extractor: `ordinal_chain` (the amend-and-restate ordinal in the name). Null means the pointer came from an extracted, cited relation — including the pointer from an amended instrument to the prior state the extractor minted for it (#203) — so an inferred pointer is never mistaken for an extracted one (#170, #184). Rows written before #203 may carry `prior_fact`; the lineage pass re-opens every inferred pointer on each run and re-derives it, so such a row is re-linked by `ordinal_chain` or cleared (#204). Set by the lineage pass, which runs after every match — in `cdt match`, in the pipeline's match-and-finalize step, and in `PipelineOrchestrator.run` (`cdt pipeline` and the live extractor backend) alike.
- `superseded_by_debt_instrument_id`: The amendment child that replaced this state, when exactly one exists (#155). A row with this set is a superseded state, not a live obligation.
- `lineage_family_id`: One ID per connected lineage component over amendment, split, and retirement pointers — every state of one obligation history shares it (#155). Singleton instruments use their own ID.
- `is_lineage_head`: True when no amendment child supersedes this row; the browse index should show heads and collapse the rest of the family beneath them.
- `first_seen_filing_date` / `last_seen_filing_date`: Filing-date range of the instrument's direct mentions.
- `mention_count` / `document_count`: Direct mentions, and distinct filings containing them.
- `name`, `instrument_type`, `start_date`, `commitment_termination_date`: Matcher-selected canonical values — the newest non-null across direct mentions, falling back to the value already on the row when no mention carries one.
- `maturity_date`: Selected on a different rule from the fields above. The newest **stated** maturity wins; a derived one — read out of the instrument's own name, or computed from a tenor (#166) — publishes only when no mention in the cluster states any. Recency alone was not enough, because every post-closing `due 2030` mention re-introduces the synthesized year-end, which let a name-derived `2030-12-31` outrank the closing 8-K's stated `2030-07-01` (#162). Extending the same preference to the other canonical fields is still open under #162.
- `principal_amount`, `principal_currency`, `principal_amount_kind`: Canonical headline amount, taken together from the newest mention that carries one so the currency can never detach from its figure (#140).
- `outstanding_balance`, `outstanding_balance_currency`, `outstanding_balance_as_of`, `outstanding_balance_as_of_is_filing_date`: The newest balance observation, kept apart from principal so it never double-counts (#140). When the filing dates the balance no other way, `as_of` falls back to the observing mention's filing date and `outstanding_balance_as_of_is_filing_date` is true, so a substituted date is never mistaken for a stated one (#203).
- `synthesized_only`: True when every member mention was synthesized by the extractor (`mentions.synthesized_by` set) rather than returned by the model — a minted prior state that never merged with a mention describing that state on its own (#203). The row is a real, cited prior state of its successor, but no filing describes it independently; a reader summing capacity or counting obligations needs to know that.
- `interest_rate_kind`, `interest_rate_pct`: Canonical interest rate (#157).
- `*_source_mention_id` (name, instrument_type, start_date, maturity, commitment_termination, principal, outstanding_balance, interest_rate): The mention each canonical value actually came from (#151), so evidence attribution never has to be guessed.
- `parties_json`: JSON aggregation of party clusters from direct mentions, deduped by role plus normalized canonical name (#150).
- `lender_disclosure`: The instrument's worst-case answer across its direct mentions. `collective_present` wins outright — one filing showing a collective lender phrase means holders are hidden however many others name some — and `complete` beats `none_named`, so a filing that named every lender is not erased by one that named none.

Primary key: `debt_instrument_id`

**No lifecycle `status` is published here, by design (#196).** Answering "is
this borrowing still alive" needs a notion of *now*. A wall clock would make the
output non-reproducible, and the only other value this repository can reach is
one read off the data — the newest filing date in the run — which makes a
published status a function of the run's **scope** rather than of the filings.
Re-deriving one 542-instrument corpus against a reference date two years earlier
moved 218 statuses and changed nothing else: same edges, same clusters, every
other column identical. Nothing read the answer back either, here or in the
publisher.

The cascade therefore lives in the dashboard publisher, which derives it at
publish time against an explicit `asOf` recorded alongside `generatedAt`
(`dsi-rse/commercial-debt-tracker-website#15`). Every input it needs is already
published: the lineage pointers, `start_date`, `maturity_date` and
`commitment_termination_date`, and the member mentions' own `status`,
`status_date` and `dates_json`.

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
<artifact-root>/runs/infer-lineage/run_id=latest.json
```

`infer-lineage` is the amendment-lineage post-pass. It runs after every match
and rewrites every `debt-instruments` partition the match manifest just listed,
so it records its own — otherwise the last manifest of that dataset describes a
state something else changed afterwards (#211). It carries the pass's counters
(`links`, `reopened`, `heads_before`, `heads_after`) alongside the partitions it
rewrote.

The match manifest and the `final-snapshots/latest.json` pointer carry
`schema_version` (`MATCHER_SCHEMA_VERSION`, currently 7). It is bumped whenever a
`debt-instruments` or `mention-cluster-edges` column is added or removed — 5 → 6
for the four `status_*` columns #196 removed, 6 → 7 for `synthesized_only` and
`outstanding_balance_as_of_is_filing_date` (#203) — so a reader of an older root
knows its columns differ from the current contract.

`match_pending_mentions` reads the recorded value back and **promotes a plain
run to a full rematch when it is lower than the running version**. It has to:
mention ids are content hashes, so a schema change that alters the hashed
payload changes every id, and the clusters carried over from the older root are
then keyed on ids the mentions dataset no longer contains. It also silently
degraded #203 — minting an amended instrument's prior state adds mentions, so
the carried-over clusters hold slots the mints would take on a clean build, and
a cluster can end up with two members naming two different amendment parents,
which `derive_parent_links` correctly refuses. Measured on a 542-instrument
root recorded at version 4: `cdt backfill-mentions` plus one plain `cdt match`
published 19 amendment pointers and 539 lineage heads against a forced match's
22 and 536, and further plain matches never recovered it. Promoting rather than
refusing keeps the scheduled run self-healing; a rematch is deterministic local
compute, and the manifest it writes records the current version, so the next run
is an ordinary incremental match again (#208).

### Extractor manifests and audit logs

Extractor writes a per-run manifest and a matching full audit log:

```text
<artifact-root>/runs/extract/run_id=<run_id>.json
<artifact-root>/extractor-runs/run_id=<run_id>/full.jsonl
```

### Extract batch job state

The OpenAI batch extract backend keeps its resumable, file-native job state under
`extract-batches/`. The hourly `poll` run is the only writer, apart from the
`cdt reset-extract-job` admin command, which rewrites `active.json` under the same lease.

```text
<artifact-root>/extract-batches/
  active.json                        # {"job_id": ...}; job_id is null when idle
  job_id=<run_id>/manifest.json      # static job config + claimed classification partitions
  job_id=<run_id>/state.jsonl        # one line per item: source partition + pending request
                                     # marker + expiry resubmission counter + resumable row state
  job_id=<run_id>/batches.json       # in-flight OpenAI batches, seen batch ids, tick counter
  job_id=<run_id>/ticks/tick=<n>.json  # per-tick audit counts
```

The orchestrator also keeps advisory locks directly under the artifact root:

```text
locks/pipeline-writer.json           # single-writer lease: {holder, acquired_at, expires_at}
```

When a job finishes, its mentions are written to the canonical `mentions` partitions and its
audit log to `extractor-runs/run_id=<run_id>/full.jsonl`, exactly like the synchronous
backend. `state.jsonl` and `batches.json` are working state, not canonical outputs.

A `job_id=<run_id>/` directory is never deleted, including when a job is abandoned as
corrupt, so a wedged poller leaves its evidence behind. Only `active.json` decides which
job a tick advances: it is set to `{"job_id": null}` on completion or reset, which
`_read_active_job` treats the same as an absent marker (a write, not a delete, so no
delete permission on the artifact bucket is needed).

### Failure registries

The ingest stage maintains a permanent failure registry at:

```text
<artifact-root>/failures/ingest/failures.json
```

The extract stage maintains an equivalent row-level registry, written by both the `live`
and `batch` backends:

```text
<artifact-root>/failures/extract/failures.json
```

```json
{
  "stage": "extract",
  "failure_count": 1,
  "failures": {
    "<item_id>": {
      "item_id": "...",
      "accession_number": "...",
      "cik": "...",
      "date": "2024-01-02",
      "shard": "0001",
      "state": "ERROR",
      "stage": "ner",
      "run_id": "20260814T134326943342Z",
      "backend": "batch",
      "error": "..."
    }
  }
}
```

This registry exists because a finished extract run marks all of its claimed
classification partitions completed regardless of individual row outcomes, so rows that
produced no mentions are never revisited. It is **diagnostic, not control flow**: nothing
reads it to decide what to process, and writing it does not change which partitions are
skipped. What it provides is a durable, queryable work-list of dropped rows — previously
recoverable only by parsing every `extractor-runs/run_id=*/full.jsonl` audit file.

Entries are keyed by `item_id` and accumulate across runs. A row that fully succeeds in a
later run (typically a `--force` re-extract) has its entry removed, so the registry always
reflects the latest known outcome per row rather than a growing history. A `PARTIAL` row
(#152) appears here too — its mentions published, but the entry records what salvage
dropped. Retrying the listed rows is still manual, and still partition-granular via
`--force`.

## Operational Semantics

- canonical truth is the partition data, not the run manifest
- final snapshot parquet files are derived convenience outputs, not the canonical working state
- `cdt pipeline` writes final snapshots only when `--final-database-root` is passed
- `cdt-orchestrator` writes final snapshots when `FINAL_DATABASE_ROOT` is set or `--final-database-root` is passed before the mode
- stage completion is inferred from output partition presence plus stage completion registries for zero-row outputs
- `force=false` skips already-written partitions
- local runs and deployed runs use the same layout and code paths
- the default operating model is one active writer per environment
