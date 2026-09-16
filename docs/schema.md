# CDT Storage Schema

CDT writes canonical artifacts under one artifact root. That root can be a local directory or an `s3://` prefix.

The pipeline can also write optional final snapshot parquet files under a separate final database root. Those snapshots are flattened table-wide exports intended for downstream database loading and are not the canonical working state for the pipeline.

Text columns are normalized on the way into a snapshot: a cell whose whole value is a placeholder such as `nan`, `none`, `null`, `<na>`, or `n/a` is written as a null. Partitions written before a column existed, or by a stage that stringified a missing value, would otherwise publish the placeholder as if it were real text.

Two schema-wide contracts:

- `cik` columns carry SEC's canonical 10-digit zero-padded form (#153). Snapshots pad legacy unpadded partitions on the way out, and `shard_for_cik` hashes the unpadded form so existing `cik_shard` partitions stay where they are.
- Every evidence payload records `spans`: a list of `{tag_id, char_start, char_end, text}` whose offsets index the source item's `text` **exactly** (#154) — the extractor realigns model output whose whitespace drifted. Value-bearing payloads also record `derived_from`: `"stated"` when the value was parsed from cited evidence, `"name"` when it was derived from the instrument's own name (a `due 2028` maturity, a `$183.36 million term loan` principal, a coupon in the name), and null when there is no value (#128).

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
| `is_lineage_head`, `relevance` | `bool` | |
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

- `debt_instrument_mention_id`: Deterministic identifier for one extracted debt-instrument mention (a content hash over the extracted fields plus `item_id`).
- `item_id`: Source item section identifier.
- `accession_number`: Filing accession number for the source item.
- `cik`: Issuer CIK for the source item, zero-padded.
- `company_name`: Issuer display name for the source item.
- `date`: Filing date for the source item in `YYYY-MM-DD` format.
- `raw_id`: Row-local extractor identifier used inside a single item during relation extraction.
- `name`: Canonicalized debt instrument name text extracted from the item.
- `instrument_type`: One of `term_loan`, `revolving_credit`, `credit_line`, `note_bond`, or null when none fits or the document does not say (#156).
- `start_date`: Normalized instrument start or issuance date when present — the current `closing` fact in `dates_json`. An instrument whose status is `announced` has not started and carries none; its projected close lives in `dates_json` as a `closing` fact marked `expected` (the stage-1 `expected_closing` kind is rewritten to that shape on the way in). The instrument-level `expected_active` leg is measured against it.
- `maturity_date`: Normalized final maturity or expiration of the obligation — when the borrowed money must be repaid (#158). Year-only maturities normalize to `YYYY-12-31` with `derived_from: "name"`.
- `commitment_termination_date`: When the lender's obligation to lend ends — the close of a draw, availability, or revolving period — when the document states one distinct from the maturity (#158). Null for notes and bonds.
- `principal_amount`: The single commitment or principal figure, as digits with at most one decimal point. Balances, draws, repayments, and proceeds never populate this column (#140).
- `principal_currency`: ISO 4217 code for `principal_amount` when stated.
- `principal_amount_kind`: `commitment` or `principal`; null on rows replayed from the pre-#140 single-amount shape.
- `status`: What this mention says happened to the instrument, derived from its event facts in `dates_json`: the newest completed event wins (`retirement`→`repaid`, `termination`→`terminated`, `exchange`→`exchanged`, `default`→`defaulted`, `amendment`→`amended`, `closing`→`entered_into`, `announcement`→`announced`); an instrument whose only closing is `expected` is `announced`; a planned retirement decides nothing here and is read by the matcher as pending. Null when the mention states no event. `matured` is never extracted; the matcher derives it. Pre-stage-2 responses that carried a `status_event` replay it verbatim.
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

Primary key: `debt_instrument_mention_id`

Rows publish from extractor states `SUCCESS` and `PARTIAL` (#152). A `PARTIAL` row salvaged what a terminal failure left intact — individually valid entries after a final `instrument_ie` validation failure, or mentions without lineage after a terminal `instrument_relation` failure — and also carries a failure-registry entry recording what was lost.

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
- `amendment_inferred_by`: Which rule inferred `amendment_of_debt_instrument_id`, when the matcher filled it rather than the extractor: `prior_fact` (a `prior`-marked amount equal to the predecessor's principal) or `ordinal_chain` (the amend-and-restate ordinal in the name). Null means the pointer came from an extracted, cited relation — so an inferred pointer is never mistaken for an extracted one (#170, #184). Only set behind `cdt match --infer-lineage`, and carried forward across incremental rematches for as long as the pointer it describes is unchanged.
- `superseded_by_debt_instrument_id`: The amendment child that replaced this state, when exactly one exists (#155). A row with this set is a superseded state, not a live obligation.
- `lineage_family_id`: One ID per connected lineage component over amendment, split, and retirement pointers — every state of one obligation history shares it (#155). Singleton instruments use their own ID.
- `is_lineage_head`: True when no amendment child supersedes this row; the browse index should show heads and collapse the rest of the family beneath them.
- `status`: Derived lifecycle answer (#155, vocabulary per #183). One of five values, which separate what the filings confirm from what they only imply:

  | value | meaning |
  |---|---|
  | `announced` | The newest decisive event is the announcement, and either no planned start is recorded or that planned start is still ahead of the reference date. |
  | `active` | The reference date is past an explicit `start_date`, and not yet past every end date the instrument records. |
  | `expected_active` | No filing confirms it started, but the reference date is past its planned start. |
  | `closed` | An explicit terminal status. Always carries a `status_subtype`. |
  | `expected_closed` | The reference date is past every end date the instrument records — scheduled or merely planned — with no terminal event recorded. |

  The legs are tried in order: an extracted terminal event; then `closed`/`superseded` when any amendment child replaced this state; then `closed`/`repaid` when only the retirement lineage says the obligation ended; then the date legs. An explicit `start_date` the reference date has passed outranks a later announcement, so re-announcing an instrument that already closed does not revert it (#169). An instrument with no dates and no events stays `active`: the filing describes an obligation it treats as outstanding, and there is no evidence against that.

  The reference date is what the date legs treat as "now": the newest filing date among **every** mention in the run, resolved once before the matcher shards its work and passed into the rollup, so a rerun over the same inputs reproduces the same statuses. It is deliberately not the wall clock, which would make every rerun differ. It is also not per `cik_shard` — a shard is a hash bucket, so deriving it there made a published status depend on which unrelated issuers happened to share a bucket, and left a shard of quiet filers judged against a date months behind the corpus (#188). End dates are `maturity_date`, `commitment_termination_date`, and any planned retirement the mentions record (an `expected` terminal date fact, or a terminal status dated after the filing that carries it — a redemption notice). The *latest* of them governs: a lapsed commitment does not close a facility whose principal is still owed to a later maturity. A planned retirement stated with no date can never be shown to have come due, so it blocks `expected_closed` indefinitely.
- `status_subtype`: The cause of a `closed` status, null for every other status: `repaid`, `terminated`, `exchanged` or `defaulted` from the extracted terminal event, or `superseded` when a later amendment replaced this state. The display form joins the two with a dash — `closed - repaid` — so a reader sees the state and its cause together.
- `status_date`: The date that decided `status`: the winning event's date, the `start_date` behind `active`, the planned start behind `expected_active`, or the end date behind `expected_closed`. Null for a `closed` status derived from lineage rather than an event. An `announced` status is dated no later than the filing that announced it.
- `status_source_mention_id`: The mention whose extracted event decided `status`, when one did. Null on the inferred statuses, because no mention states them.

Presenting `status` (UI contract): a `closed` or `expected_closed` row should name the instrument that ended it when one is known — `superseded by <name>` linking to `superseded_by_debt_instrument_id`, or `retired by <name>` linking to the entries in `retired_by_debt_instrument_ids`. Both pointers are published on the row, so the link needs no extra lookup beyond resolving the target's `name`.
- `first_seen_filing_date` / `last_seen_filing_date`: Filing-date range of the instrument's direct mentions.
- `mention_count` / `document_count`: Direct mentions, and distinct filings containing them.
- `name`, `instrument_type`, `start_date`, `maturity_date`, `commitment_termination_date`: Matcher-selected canonical values (newest non-null across direct mentions).
- `principal_amount`, `principal_currency`, `principal_amount_kind`: Canonical headline amount, taken together from the newest mention that carries one so the currency can never detach from its figure (#140).
- `outstanding_balance`, `outstanding_balance_currency`, `outstanding_balance_as_of`: The newest balance observation, kept apart from principal so it never double-counts (#140). `as_of` falls back to the observing mention's filing date.
- `interest_rate_kind`, `interest_rate_pct`: Canonical interest rate (#157).
- `*_source_mention_id` (name, instrument_type, start_date, maturity, commitment_termination, principal, outstanding_balance, interest_rate): The mention each canonical value actually came from (#151), so evidence attribution never has to be guessed.
- `parties_json`: JSON aggregation of party clusters from direct mentions, deduped by role plus normalized canonical name (#150).
- `lender_disclosure`: The instrument's worst-case answer across its direct mentions. `collective_present` wins outright — one filing showing a collective lender phrase means holders are hidden however many others name some — and `complete` beats `none_named`, so a filing that named every lender is not erased by one that named none.

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
