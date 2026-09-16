# Wiring the 6-K path into the pipeline

`cdt.sixk` landed on `dev` as a library with tests and no callers. Its own doc
closes with the gap this plan fills:

> **What is not in this change:** orchestrator wiring. This adds the stage as a
> library with its own tests; making it a pipeline stage alongside ingest →
> itemize → classify → extract needs decisions about partitioning and dataset
> registration that are better taken separately.

Those decisions are taken below. Branch: `sixk-pipeline-wiring`, based on `dev`
(`15598dd`).

## What is already true

Verified against the code on `dev` and against the live dev bucket, not assumed:

| fact | where |
|---|---|
| `cdt.sixk` is complete and unreferenced outside its tests | `src/cdt/sixk/`, `tests/test_sixk_triage.py` |
| Stage-1 artifact is committed and loads under the pinned sklearn | `data/models/sixk/stage1-tfidf-linear-svc/` |
| `ingest.iter_filings` already takes `form_types: str \| list[str]` and normalizes `6-K/A` → `6-K_A` | `src/cdt/ingest.py:613`, `:822` |
| …but no production caller uses it; `iter_document_candidates_for_date_range` hardcodes `CDT_FORM_TYPE = "8-K"` | `src/cdt/ingest.py:55`, `:565` |
| The extractor reads exactly one input dataset, by constant | `extractor/core.py:1040` (`CLASSIFICATION_DATASET_NAME`) |
| `read_table(path, columns)` falls back to a full read + reindex when a column is absent, so adding a column to an existing dataset is backward compatible | `storage.py:469-474` |
| `PARTITION_PATTERN`'s `dataset` capture group is matched but never consumed — scans join a dataset root and read only `date`/`shard` | `datasets.py:33`, `:491-498` |

And two facts about the upstream bucket that shape the fallback:

- **There is no 6-K in the scraper bucket at all.** `s3://…/sec/<date>/` holds
  `8-K`, `8-K_A`, `10-K`, `10-K_A`, `10-Q`, `10-Q_A`, `13F-HR`, `13F-HR_A`. The
  scraper-wide index `sec/manifest.parquet` (1,693,077 document rows,
  2016-01-04 → 2026-09-08) contains **zero** `6-K` or `6-K/A` rows.
- **The scraper's document shape is not uniform across forms.** All 749,077 8-K
  rows are a single `Complete submission text file` per filing — which is what
  `ingest._is_cdt_document` matches (`ingest.py:730`) and what the whole
  one-`DocumentCandidate`-per-filing model assumes. The 20-F rows are the
  opposite: per-document entries (`20-F`, `EX-8.1`, …) and **no** complete
  submission entry. Which of the two shapes 6-K arrives in is therefore an
  upstream question, not a given (see [Source A](#source-a-the-scraper-bucket)).

## Target flow

Two genre paths that converge at `extract`:

```
                                     ┌─ documents ──── itemize ─── items ─── classify ─── classifications ─┐
  scraper S3 (8-K) ──── ingest ──────┤                                                                      │
                                     │                                                                      ├── extract ── mentions ── match ── debt-instruments
  scraper S3 (6-K) ──── ingest ──────┴─ documents-sixk ─── sixk ──────────────── sixk-snippets ──────────────┘
                                                            (window → stage 1 → stage 2)
```

Everything downstream of `extract` is untouched: mentions carry `item_id`, the
matcher shards by CIK, and a 6-K mention consolidates against an 8-K mention for
the same issuer for free.

## Design decisions

### 1. 6-K documents get their own dataset, `documents-sixk`

The obvious alternative — one `documents` dataset with a `form_type` column — is
cheaper to write and worse to run. `ingest._write_document_partitions`
(`ingest.py:968`) *merges* new rows into the existing `date=/shard=` partition,
so adding 6-K rows changes that partition's fingerprint, and itemize's
fingerprint-keyed selection (`itemizer/core.py`, #62) then makes it pending
again. A historical 6-K backfill would re-itemize essentially the whole 8-K
corpus — the code puts that at ~2.5h of itemize (#111) plus one S3 GET per 8-K
document re-read.

A separate dataset also keeps the production 8-K path byte-identical: itemize
needs no genre filter and cannot regress. The cost is threading a dataset name
through `IngestConfig` instead of hardcoding `DOCUMENT_DATASET_NAME`.

Name it `documents-sixk` for consistency with the `cdt.sixk` package. An
earlier draft of this plan claimed `documents-6k` would break the partition
contract, because `PARTITION_PATTERN` matches the dataset segment as
`[a-z\-]+`. That is wrong twice over: `search` still matches such a path (it
just captures a truncated `dataset` group), and nothing reads that group —
every scan is rooted at a joined `dataset_root(name)` and only consumes `date`
and `shard`. The name carries no behaviour; pick it for readability.

Both datasets keep the same columns plus two new ones:

- `form_type` — `6-K` or `6-K/A`. Also added to `documents` (backward compatible
  per `read_table`'s reindex fallback; null reads as 8-K).
- `source` — `s3-manifest` or `edgar`, so a row's provenance survives the
  cutover between the two acquisition sources.

### 2. The 6-K stage writes `sixk-snippets`, conforming to the classified-item contract

One new stage, `cdt.sixk.stage` (`sixk_pending_documents`), reads
`documents-sixk` partitions and writes `sixk-snippets` at the same
`date=/shard=` coordinates. Per source partition it:

1. resolves each document's text (inline `text`, else `resource_uri`, exactly as
   `itemizer.core._document_text_for_record` does),
2. splits and flattens prose documents (see [§3](#3-document-flattening-moves-into-cdtsixk)),
3. windows and gates via `sixk.prepare_filing`,
4. scores with `sixk.stage1_admit` at the artifact's threshold,
5. groups a filing's admitted windows and prunes them with `sixk.triage_filing`,
6. writes **every admitted window** as a row, with `relevance = kept by stage 2`.

Writing dropped windows with `relevance=False` mirrors what classify already
does with irrelevant items, and it is the only way to audit stage 2 after the
fact — it is a non-deterministic LLM (94.9% run-to-run agreement) whose drops
are the mechanism, not a side effect.

Rows use `CLASSIFIED_ITEM_COLUMNS` verbatim so the extractor needs no per-source
schema knowledge:

| column | 6-K value |
|---|---|
| `item_id` | `{accession}-6K-{doc_index}-{window_index}` — disjoint from 8-K ids by construction |
| `item` | `6-K:{doc_index}:{window_index}` (the snippet id) |
| `text` | the window text — what `ExtractionRowState.text` reads |
| `accession_number`, `cik`, `company_name`, `date`, `url` | from the document row |
| `label`, `relevance` | stage-2 verdict |
| `classification_score` | the stage-1 score |
| `section_heading` | the document's `<TYPE>` (`6-K`, `EX-99.1`, …) |
| `start_line` / `end_line` / `section_char_count` | window char offsets and length |
| `item_information`, `extraction_status`, `duplicate_resolution` | stage-2 drop reason where one applies, else null |

Why not write into `classifications` directly (which would need zero extractor
change)? Because classify already owns
`classifications/date=D/shard=S/part-0000.parquet` and
`write_partition_table` overwrites it wholesale. Two stages writing one
partition means merge-on-write with genre-scoped row replacement and an
ordering hazard between two independent completion registries. One writer per
dataset is the invariant the whole file-native design leans on; keep it.

### 3. Document flattening moves into `cdt.sixk`

The eval harness
(`commercial-debt-tracker-models/genwindow_eval/harness/sample_and_fetch_6k.py`)
does something the runtime library does not: split `<DOCUMENT>` blocks, keep
only prose types, and flatten with the itemizer's HTML handling. Promote it to
`src/cdt/sixk/documents.py`:

```python
KEEP_TYPE_RE = re.compile(r"^(6-K(/A)?|EX-99(\.\d+)?|EX-1(\.\d+)?|EX-4(\.\d+)?|EX-10(\.\d+)?)$", re.I)

def prose_documents(submission: str) -> list[SixkDocument]:
    """Return (type, flattened text) for each prose document in a submission."""
```

reusing `itemizer.extract.DOCUMENT_RE`, `TYPE_RE` and `normalize_body_lines`.
This must be the *same* code the eval scored, or the wiring's inputs differ from
the measured ones. Graphics, XBRL and cover-only artifacts carry no prose and are
dropped here; the inline-XBRL prologue strip stays inside `prepare_filing`,
where the documented ordering already puts it.

### 4. The extractor gains a source list, not a second implementation

```python
# extractor/core.py
CLASSIFICATION_SOURCES: tuple[str, ...] = (CLASSIFICATION_DATASET_NAME, SIXK_SNIPPET_DATASET_NAME)
```

`pending_extract_partitions` loops over the sources and unions the results.
Registry keys are full paths, so they stay unique per source; `collect_pending_extract_items`,
`extract_pending_items` and `finalize_extract_outputs` already read
`CLASSIFIED_ITEM_COLUMNS` from `pending.classification_path` and need no change
beyond the loop.

**One trap to get right.** `pending_extract_partitions` has a backfill
heuristic: when a partition has no registry entry but `(date, shard)` already
exists in `mentions`, it is adopted as complete (`extractor/core.py:1063-1071`).
Both sources write into the same `mentions` `(date, shard)` space, so an
unprocessed 6-K partition whose date/shard already holds 8-K mentions would be
silently marked complete and never extracted — no error, no rows. The heuristic
exists only to adopt partitions that predate the registry, and no 6-K partition
can predate it, so scope it to the 8-K source. This is the single highest-risk
line in the change; it gets a dedicated regression test.

Mentions merging already handles two sources per partition:
`_merge_mentions_partition` replaces by `item_id` and keeps everything else
(`extractor/core.py:1102`), and 6-K item ids are disjoint from 8-K ones.

### 5. Stage 2 runs live, inside the stage — not through the batch machinery

Stage 2 is one call per filing: 550 prompt + 222 output tokens, **$0.25 per
1,000 filings**, ~0.9% of what extraction costs. Routing it through the OpenAI
Batch state machine would buy ~$0.12 per 1,000 filings and cost a second job
type in `extractor/batch.py`. Run it synchronously in the stage, where
`daily`/`historical` prepare already runs under the pipeline-writer lease.

Failure is already designed for: `triage_filing` returns a `FilingVerdict` with
`error` set and every admitted window kept. That degrades to stage-1 output —
~2x extraction cost for that filing, no recall loss — which is the right
direction to fail in. The stage logs and counts degraded filings in its run
manifest rather than aborting.

The client: `extractor.core.OpenRouterChatClient` already matches
`sixk.SupportsChatCompletion` (both `async def complete(*, messages, model,
reasoning_effort)`) and is the default. Note that the eval harness had to run
stage 2 against OpenAI directly because the shared OpenRouter account is out of
credit and OpenRouter reserves estimated max cost per in-flight request. Promote
the harness's `OpenAIChatClient` alongside it and select with
`SIXK_TRIAGE_PROVIDER` (`openrouter` | `openai`, default `openrouter`), so a
credit-limited account is a config change and not a blocked run.

`settings.SIXK_TRIAGE_MODEL` and `SIXK_TRIAGE_REASONING` already exist and are
already resolved at call time inside `triage_filing`.

## Source A: the scraper bucket

The steady state, and it is mostly a rename of existing generality:

1. `IngestConfig` gains `form_types: tuple[str, ...] = ("8-K",)` and
   `dataset_name: str = DOCUMENT_DATASET_NAME`.
2. `iter_document_candidates_for_date_range` takes `form_types` instead of
   closing over `CDT_FORM_TYPE`; the prefix normalization it needs
   (`6-K/A` → `6-K_A`) is already in `_normalize_form_types` and already tested.
3. `DocumentCandidate` gains `form_type` and `source`, populated from the
   manifest.
4. `cdt ingest` / `cdt-orchestrator` gain `--form-types`, defaulting to `8-K`.

**The upstream ask, stated precisely.** For the one-`DocumentCandidate`-per-filing
model to hold, 6-K manifests must carry a complete-submission entry the way 8-K
manifests do — `type` or `description` equal to `COMPLETE SUBMISSION TEXT FILE`
(`ingest._is_cdt_document`). The 20-F precedent shows the scraper does not always
write one. If 6-K arrives 20-F-shaped instead, the contingency is bounded and
worth pricing now: `DocumentCandidate` grows a `resource_uris: tuple[str, ...]`,
`_candidate_from_filing` selects the prose set with the same `KEEP_TYPE_RE` from
§3, and the 6-K stage concatenates them instead of splitting one submission.
That is strictly *less* work than the split path — but it is a different
`documents-sixk` row shape, so it should be settled before the stage is written,
not after.

Ask the scraper team for: `sec/<date>/6-K/` and `sec/<date>/6-K_A/` prefixes,
per-filing `manifest.json` in the existing shape, a complete-submission text
file document entry, and `6-K` rows in `sec/manifest.parquet`.

### What arrived (2026-09-16)

The prefixes and manifests arrived; the complete-submission entry did not. 6-K
landed **20-F-shaped**, as the precedent warned: every one of the 2,676
filing-date partitions from 2016-01-04 onward now carries `6-K/` (and `6-K_A/`
where amendments exist) at 50-130 filers a day, and a filing is stored as one
gzipped object *per document* with no whole-submission object and no
complete-submission manifest entry (0 `.txt` objects under any 6-K prefix).

Neither contingency above was needed, because of what the per-document objects
turned out to be: each is the document's **dissemination-format `<DOCUMENT>`
block**, header lines (`<TYPE>`, `<SEQUENCE>`, `<FILENAME>`, `<DESCRIPTION>`)
included — the scraper splits EDGAR's submission without rewriting it. So
concatenating a filing's objects in `seq` order *reconstructs* the submission,
differing from EDGAR's only in the `<SEC-HEADER>` preamble that
`prose_documents` discards anyway.

`cdt.sixk.scraper` therefore assembles the submission at ingest and mirrors it
under the `raw-documents/sixk/` path the EDGAR source had written. That keeps
the row shape (one `resource_uri`), the stage (one submission per row, split
into prose documents whose index is part of a snippet's identity) and the
measured triage behaviour all unchanged — verified byte-for-byte on 23 real
filings spanning 2016 to 2026, including a 6-K/A: every flattened prose
document came out identical to EDGAR's.

Two things this path does *not* inherit from the EDGAR one (which it replaced
outright — see [Source B](#source-b-the-edgar-fallback-built-then-removed)):

- **No dissemination-feed problem (#90).** The scraper lists by filing date, so
  a range means filing dates and late-listed filings land in their own
  partition by construction rather than by the EDGAR path's write-where-it-
  belongs rule.
- **A deposit lag instead.** The scraper writes a filing's manifest 1-2 days
  after the filing date for 8-K (measured over 75 filings across five dates),
  inside `pipeline.DAILY_LOOKBACK_DAYS = 5`. The 6-K lags measured 7-14 days,
  but every 6-K sampled was scraped on 2026-09-15/16 — that is the backfill
  timestamp, not a cadence. Worth re-measuring once the 6-K job has run daily
  for a week: if its steady-state lag exceeds the lookback, the window needs
  widening for both genres.

## Source B: the EDGAR fallback (built, then removed)

While the bucket held no 6-K, this path fetched complete submission text files
from sec.gov and mirrored them under `raw-documents/sixk/`, with `--source
edgar` selecting it. It shipped in this branch's commits 2-3 and was **removed
in commit 7**, once Source A covered the whole corpus: a second way to acquire
one genre is a second failure taxonomy, a second throttling policy and a second
thing to keep true, and the 8-K path has exactly one source. `SEC_USER_AGENT`
and the fair-access handling went with it — nothing in the pipeline talks to
sec.gov now.

Two lessons worth keeping if it is ever reinstated, neither visible from the
research harness (which only ever read a quarterly index from disk):

- The quarterly full-index spells a filing date `2026-04-29`; the daily index
  spells it `20260908`. A regex accepting only the first matches *nothing* in a
  daily index — 147 of 147 6-K rows dropped, with no error.
- A daily index is a dissemination feed, not a filing-date bucket: the
  2026-09-08 index lists filings dated 2026-09-04. Filtering its rows by the
  run's date range drops those permanently (#90). Neither problem exists on
  Source A, which lists by filing date.

The one thing the EDGAR path still buys, and the reason to remember it exists:
it could serve a filing the scraper has not scraped. Source A's mitigation for
that is the 8-K path's — `DAILY_LOOKBACK_DAYS`, a re-runnable range, and the
failure registry — not a second source.

### Cutover

Done as of 2026-09-16, and it needed no migration step. Both sources wrote the
same mirror path and ingest dedups on accession (`_existing_accessions`), so
filings EDGAR had already acquired were neither re-fetched nor re-ingested; the
mirror stays the resource for those rows. With Source B gone, `cdt ingest-sixk`
takes the same `--bucket` / `--aws-profile` / `--s3-prefix` flags `cdt ingest`
does and has no source to choose.

Residue worth naming: the `source` column on `documents-sixk` now holds one
value (`s3-manifest`) on every row it is set on. Kept rather than dropped — it
is already in the published schema, older partitions legitimately have it null,
and it is the field that would distinguish a future source — but it no longer
discriminates anything.

## Change list

New:

| file | contents |
|---|---|
| `src/cdt/sixk/documents.py` | `prose_documents`, `KEEP_TYPE_RE` — promoted from the eval harness |
| `src/cdt/sixk/stage.py` | `sixk_pending_documents`, `sixk_snippets_root`, `SIXK_SNIPPET_COLUMNS`, `snippet_id_for` |
| `src/cdt/sixk/scraper.py` | manifest scan, submission assembly, mirror write, `acquire_scraped_sixk_documents` |
| `src/cdt/sixk/mirror.py` | the mirror path contract |
| `tests/test_sixk_stage.py` | the stage, end to end, with a fake chat client |
| `tests/test_sixk_scraper.py` | assembly order, mirror/resume, malformed and missing documents |

Modified:

| file | change |
|---|---|
| `ingest.py` | `form_types` + `dataset_name` on `IngestConfig`; `form_type`/`source` on `DocumentCandidate` and `DOCUMENT_COLUMNS`; drop the `CDT_FORM_TYPE` hardcode |
| `extractor/core.py` | `CLASSIFICATION_SOURCES` loop in `pending_extract_partitions`; scope the mentions-backfill heuristic to the 8-K source |
| `pipeline.py` | `sixk_enabled` / `sixk_form_types` / `sixk_cik_file` on `PipelineConfig`; a `_sixk` phase in `_ingest_itemize_classify` (renamed `_prepare`), with a lease renew at its boundary; counts on `PipelineRunResult`; `FINAL_OUTPUT_TABLES["items"]` becomes a union of `items` and `sixk-snippets` |
| `cli.py` | `cdt sixk` stage command; `cdt ingest-sixk` with the scraper flags `cdt ingest` takes; `--form-types` and `--sixk-cik-file` on `cdt ingest` |
| `orchestrator.py` | `--sixk` / `SIXK_ENABLED` and `--sixk-cik-file` / `SIXK_CIK_FILE` (defaulting to `CDT_DEFAULT_CIK_FILE`), threaded into `PipelineConfig` |
| `settings.py` | `SIXK_TRIAGE_PROVIDER`, `SIXK_CIK_FILE` |
| `sixk/__init__.py` | re-export the new modules |
| `docs/architecture.md` | the second genre path; the 6-K stage between ingest and extract |
| `docs/schema.md` | `documents-sixk`, `sixk-snippets`, `raw-documents/sixk/`; `form_type`/`source` on `documents` |
| `docs/sixk-two-stage-triage.md` | replace "What is not in this change" with the wiring |
| `.env.example`, `README.md` | running the 6-K path |

## Tests

Alongside the existing stage tests in `tests/test_file_native_stages.py`, and
following their fake-client pattern:

- window → stage 1 → stage 2 over a synthetic partition; kept and dropped
  windows both persisted, `relevance` set from the verdict.
- a stage-2 error keeps every admitted window and records the degradation.
- a filing whose windows span two documents is judged in **one** stage-2 call
  (the measured design; per-window calls would change the numbers).
- filing-scoped grouping holds within a partition — documents shard by
  accession (`ingest._document_shard`), so all of a filing's windows are
  co-partitioned. Worth an explicit test, because the whole grouping design
  rests on it.
- **the backfill-heuristic regression:** a `sixk-snippets` partition at a
  `(date, shard)` that already has 8-K mentions is still pending.
- extract claims from both sources in one job; mentions merge, neither genre's
  rows lost.
- completion/fingerprint semantics for the new stage: re-ingest merges a row →
  partition pending again; interrupted pass → `complete=False`.
- ingest with `form_types=("6-K", "6-K/A")` writes `documents-sixk` and leaves
  `documents` untouched.
- Scraper 6-K: documents assembled in `seq` order, a document missing its
  `<DOCUMENT>` wrapper failing its filing permanently, the mirror as resume
  ledger, and the assembled prose matching EDGAR's byte for byte.

## Phases

1. **Ingest generality.** `form_types`, `dataset_name`, the two new columns.
   Ships alone; the 8-K path is unchanged and provable by the existing suite.
2. **Acquisition.** `sixk/scraper.py` + mirror (originally `sixk/edgar.py`;
   see Source B). Verifiable without any LLM call: run one day, count rows in
   `documents-sixk`. The `cdt ingest-sixk` command moved forward from phase 5
   into this phase, because "run one day" is this phase's acceptance check and
   it needs an entry point.
3. **The stage.** `sixk/documents.py` + `sixk/stage.py` + tests. Still no
   pipeline change; drive it with `cdt sixk`.
4. **Extractor source list.** Including the backfill-heuristic fix and its test.
   This is the phase to review hardest — it touches the production 8-K path.
5. **Pipeline and orchestrator.** `--sixk` off by default, so the deployed daily
   run is unchanged until it is turned on deliberately. Also the `items`
   snapshot union, and the issue on `commercial-debt-tracker-dashboard` for the
   `item`-column change it implies.
6. **Docs.**

### Phase-3 checkpoint result (2026-09-09)

One fresh day, 2026-09-08, all 146 6-K/6-K/A filings EDGAR listed for it — not
the generalization eval's window, which is a scoring set.

| measure | this day | `docs/sixk-two-stage-triage.md` |
|---|---|---|
| filings passing the debt-vocabulary gate | 13.7% (20/146) | 13.4% |
| windows stage 1 admitted | 129 of 876 gated | — |
| snippets stage 2 pruned | 61% (79 of 129) | 47.5% |
| filings degraded to stage-1 output | 0 | — |

The gate reproduces. Stage 2 pruned harder than measured, on 13 filings, so
read that as a wide interval rather than a shift.

**One filing-level recall miss, worth carrying into the eval.** Stage 2 dropped
all 17 admitted windows of Gilat Satellite Networks' filing as `no_details`.
The first of them opens `EXHIBIT 99.2 Unofficial Translation from Hebrew TRUST
DEED FOR NOTES (SERIES 1) Made and entered into on August 30, 2026` — a real
note issuance, and no window survived to extract it from. That is exactly the
metric the triage doc calls the one that matters, where it reports 53 of 53;
the doc also says that bound "is bounded, not proven", and this is a
counterexample on the first unseen day. Two other filings lost their only
admitted window each, both plausible stage-1 false positives.

Three caveats before this is treated as a rate: one day, one annotator (me,
by eye), and stage 2 is non-deterministic at ~95% run-to-run agreement. It is
a finding for the generalization eval to size, not a wiring defect — the
wiring's part worked, in that persisting dropped windows with their reasons is
what made the miss findable at all.

A related quality note, not a miss: several *kept* windows are cover-page
boilerplate ("FORM 6-K REPORT OF FOREIGN PRIVATE ISSUER..."). The header
leakage `prose_documents` preserves for fidelity is plausibly what admits them.
The measured numbers were taken with the same leakage, so changing it is an
eval-scoped decision too.

Checkpoint after 4: run the whole 6-K path locally over a **fresh** small
sample and eyeball the mentions, with `--sixk-cik-file` pointed at that sample's
CIKs — the deployed list contains no 6-K filers, so the default produces nothing
to look at. Do not use `data/genwindow-6k/` for this — that
window is a scoring set for the generalization eval
(`commercial-debt-tracker-models/genwindow_eval/PLAN.md`) and iterating against
it burns it.

### Phase-4 checkpoint result (2026-09-09)

Ran end to end on real filings from 2026-09-08: EDGAR → mirror → windows →
stage 1 → stage 2 → `sixk-snippets` → extract → `mentions` → match →
`debt-instruments`. Extraction was scoped to two partitions to keep spend to
pennies.

- Ecopetrol's bondholders'-meeting filing produced 2 mentions from one window,
  which the matcher consolidated into one instrument. The whole path works with
  no genre-specific code below the triage stage.
- Canaan's kept window (an earnings release) extracted to nothing, terminating
  SUCCESS. Correct: a stage-1/stage-2 false positive costs one extraction and
  produces no rows, which is the trade the imprecise stage 1 is chosen for.

Two quality observations, neither in this phase's scope. The Ecopetrol
instrument came out named `bond issuances made in 2010 and 2013` with no
amount, from a filing that announces a *meeting* of bondholders rather than an
issuance — and the same instrument was emitted twice from one window. Both are
`dev`'s extractor prompts, which `pre-beta-schema-update` reworks; they are not
6-K-specific.

## Decisions taken

1. **CIK universe: the deployed `CDT_DEFAULT_CIK_FILE`, with a `--sixk-cik-file`
   override.** Decided: reuse the deployed file. Carry this measurement with it,
   because it bounds what the path can produce. The deployed list is
   `processors/cdt/inputs/ciks/beta-1k.txt` (1,000 CIKs,
   `pulumi/Pulumi.{dev,prod}.yaml`). Of the 175 distinct 6-K filers in a seeded
   2026-QTR2 sample of 200 filings (`data/genwindow-6k/documents.jsonl`),
   **0 appear in that list** — 6-K filers are foreign private issuers and the
   beta list is domestic. Reusing the deployed file therefore yields a 6-K path
   that is correct and empty until the list is widened.

   So the option is implemented as chosen — the deployed file is the default and
   no new required config appears — but `--sixk-cik-file` / `SIXK_CIK_FILE` is
   added as an override, because without it the path cannot be exercised at all:
   not in a local smoke run, not in the phase-4 checkpoint, not in dev. Widening
   the deployed list to include FPIs is the alternative and makes the override
   unnecessary. Either way, the decision to *cover* 6-K filers is separate from
   the decision to *wire* the 6-K path, and this plan only does the latter.

2. **`items/latest.parquet` publishes the union of items and 6-K snippets.**
   Decided. The dashboard resolves a mention's source text through the published
   `items` table, so a 6-K mention has no provenance otherwise.
   `FINAL_OUTPUT_TABLES` maps a table name to a root callable, so the change is
   a union function in place of `items_root`; the shrinkage guard and the
   snapshot pointer both keep working unchanged. The `item` column will hold
   snippet ids (`6-K:0:3`) rather than item numbers (`1.01`) for those rows, and
   the dashboard repo renders that column — so this needs an issue on
   `commercial-debt-tracker-dashboard` before it reaches prod, filed in phase 5.

## Still open

1. **Prod gating.** The generalization eval is scoring 6-K quality now. Wiring
   should not wait on it, but flipping `--sixk` on in the deployed daily run
   should. Defaulting the flag off makes that automatic.
2. **The 20-F-shaped-manifest contingency** in [Source A](#source-a-the-scraper-bucket)
   — settle the expected 6-K manifest shape with the scraper team before phase 3.

## Not in scope

- Cross-row dedup of 6-K *mentions* after extraction. Stage 2's window dedup is
  deliberately narrow ("drop only if every attribute it states already appears
  in a snippet you are keeping") precisely because prose is the wrong place to
  resolve duplicate instruments; that belongs in the matcher, comparing
  extracted values. Still open, as `docs/sixk-two-stage-triage.md` says.
- Retraining or recalibrating stage 1. The threshold belongs to the fitted
  artifact; a new fit needs a new measured threshold.
- Any other form type. 20-F/40-F would reuse this second-genre skeleton, but
  their document shape and window size are unmeasured.
