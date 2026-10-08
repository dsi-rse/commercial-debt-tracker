# Form 6-K triage

A 6-K has no item structure, so the 8-K path's item classifier has nothing to
classify. This path windows the document and runs two stages over the windows.
Developed and evaluated in
[`uchicago-dsi/commercial-debt-tracker-models`](https://github.com/uchicago-dsi/commercial-debt-tracker-models);
this is the runtime implementation.

## The sequence

1. **Strip the inline-XBRL prologue.** Many 6-K exhibits open with hundreds of
   lines of XBRL context facts. They score highly because their tag names are
   made of debt vocabulary. The rule needs a namespaced-tag share as the
   discriminator, not just bare scalars, or it deletes numeric borrowings tables.
2. **Gate on debt vocabulary.** Nine keyword lemmas, plurals allowed. Adding the
   plural suffix moved the pass rate from 10.9% to 13.4% — those were filings
   being missed on `debentures` versus `debenture`.
3. **Window at 400 tokens.** Not the 2,000 the 8-K path uses. Only 1.2% of
   positive windows proved context-dependent at that size, and the smaller crop
   cut extraction tokens to 0.34x while still matching 146 of 154 known mentions.
4. **Stage 1: TF-IDF linear SVM, threshold 0.332.** Tuned for recall, not
   precision.
5. **Expand each admitted window backwards.** 200 tokens of context at least,
   400 at most, stopping early at a section header. A 400-token crop can keep
   an instrument's amounts and dates while cutting away the noun that names it.
6. **Stage 2: LLM over a whole filing's expanded windows at once.**

Steps 1-3 are the segment stage and steps 4-6 the classify stage (see "Segment
and classify" below). Steps 1-2 are `cdt.segmenter.sixk.gated_body` and step 3
`split_into_windows`; `prepare_filing` composes all three for one text, so the
order is code rather than prose. It matters in both directions: the gate applies to the whole
document, so applying it per window would change the 13.4% pass rate above,
and stripping has to come first because a prologue's tag names are themselves
made of debt vocabulary.

## Why stage 1 is deliberately imprecise

At 0.332 stage 1 admits 5.8% of windows, holding 95.4% of relevant ones at 35.4%
precision. Tightening it is a bad trade: the frontier is steep at the high-recall
end, and recall lost here is invisible downstream, where precision loss is merely
expensive.

| threshold | windows admitted | precision | recall |
|---|---|---|---|
| 0.332 | 5.8% | 35.4% | 95.4% |
| 0.348 | 4.5% | 52.2% | 90.2% |
| 0.379 | 3.3% | 62.0% | 85.0% |
| 0.420 | 2.7% | 77.1% | 73.8% |

Precision was also tried at source. Hand-built features encoding the actual
false-positive families — aggregates, blank templates, mechanics — moved
precision by +1.2pp with a 90% interval of [-2.7, +3.3], i.e. not at all. The
distinctions are semantic and a bag of n-grams cannot represent them, which is
what stage 2 is for.

## Why expansion comes after stage 1, not before

A window cut at a fixed 400-token boundary can hold every number belonging to an
instrument and none of the words naming it. Measured on the generalization
window, a fifth of admitted 6-K snippets carried money, a rate or a date with no
instrument noun anywhere in the text. The extractor then has nothing to anchor
on and answers erratically — not because the model is wrong, but because the
input no longer determines an answer. It also made the 6-K stratum unreviewable:
an empty result could not be told from a miss. 6-K was the least reproducible
stratum on a same-arm rerun, 4 of 10 sampled units changing object count against
15% overall.

The fix has to run *after* admission. `WINDOW_TOKENS` is what the shipped stage-1
model was fitted on and what its threshold in `metadata.json` was calibrated
against, so widening the text stage 1 scores invalidates both. Widening the text
only stage 2 and extraction see costs nothing upstream, and is paid on the 5.8%
of windows stage 1 admits rather than on all of them.

`expand_admitted_windows` walks backwards from an admitted window and stops at
the first of: a section header, a blank line once 200 tokens are taken, or the
line boundary where the 200 tokens were met. Nothing crosses 400 tokens.

Two details are load-bearing, and both were found by replaying the
generalization window's 392 admitted windows:

- **A capitalised short line is usually a table cell, not a header.** Extracted
  6-K exhibits put one cell per line, so `Currency`, `Book Value` and `I` all
  read like headings, as do the page-break artefacts inside a table — `GRUPO
  SUPERVIELLE S.A.`, `NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS`. Treating
  any of them as a header stops the walk inside the table it was meant to climb
  out of; on the Grupo Supervielle table the walk halted 32 tokens up, at a
  class letter. So a casing-based header must also stand alone in its own
  paragraph. Only an explicitly numbered heading (`Item 5.02`, `Note 12`) is
  taken on wording alone.
- **Adjacent admitted windows merge.** An expansion reaching into the window
  before it would otherwise send the same text twice. Merging also cuts the
  snippet count, 392 admitted windows becoming 219. A run of them stops merging
  once its estimate reaches 2,000 tokens, the largest snippet the 8-K path
  already sends the same extractor; the generalization window has a run of 21
  that would otherwise reach 8,191. The estimate leaves out the context each
  later member's expansion pulls in (up to 400 tokens per merge), so a merged
  window can be larger than 2,000 tokens: a synthetic run of alternate admitted
  windows produced 3,665.

### What it fixes, and what it costs

Replaying the 27 filings of the generalization window's 6-K set:

| | before | after |
|---|---|---|
| kept snippets carrying numbers with no instrument noun | 7 | 0 |
| snippets sent to stage 2 | 392 | 219 |
| stage-2 input tokens | 147,654 | 193,304 (1.31x) |
| median context added per snippet | — | 206 tokens |

The 200-token minimum is where the last of those snippets recovers a noun; 100
recovers 4 of 7, and 150 recovers 6. Grupo Supervielle
(`000151739926000013-6K-0-147`), the case reviewers could not read at all, now
opens with `Global Program for the issuance of simple Negotiable Debt
securities` and the `Date of ISSUE / Currency / Class No. / Amount` header row.

Two figures in this document were measured on unexpanded windows and are
**not** re-measured here, because no model was refitted but both stages' inputs
changed:

- **Stage 2's precision (35.4% → 70.9%).** It now judges expanded windows, and
  the labelled 500-window set describes the unexpanded ones.
- **Extraction cost per 6-K filing.** Stage 2 decides what reaches extraction,
  and its verdicts on expanded text are what would have to be re-run. The
  stage-2 input figure above, 1.31x, is the part measurable without an LLM run.

## Why stage 2 sees a whole filing

Two reasons. Whether a window merely repeats a sibling cannot be judged from the
window alone. And 75% of filings with a relevant window have more than one
(median 3, max 20), so the question comes up constantly.

Grouping also makes it cheap: prompt overhead is paid per filing rather than per
window. Measured 550 prompt and 222 output tokens per call.

The dedup rule is deliberately narrow:

> Drop a snippet as a duplicate **only if** every attribute it states already
> appears in a snippet you are keeping.

The obvious phrasing — one window per instrument — destroys data. One Golden Sun
$5,000,000 note spread across six windows carrying, separately, the principal and
issuance date, an 18% default rate, the security and share pledge, the conversion
price, the registration-rights parties, and the signature page. Keeping one loses
five-sixths of the instrument.

## Measured results

88 filings, scored against 500 hand-labelled windows.

Every row is the same one configuration per stage: stage 2 always sees all of a
filing's admitted windows at once. There is no variant here where the LLM is
shown a single window in isolation. The last two rows are one run reported at two
granularities, not two different runs.

| configuration | granularity | precision | recall |
|---|---|---|---|
| stage 1 alone | window | 35.4% | 95.4% |
| **stage 1 + stage 2** | window | **70.9%** | 74.6% |
| **stage 1 + stage 2** | filing | — | **100.0%** |

Window recall falls **by design** — consolidating siblings is the job. The metric
that matters is whether a filing still has a window an instrument can be
extracted from, and 53 of 53 do. That bound rests on 53 filings, so the rule of
three puts the miss rate below about 5.7%; it is bounded, not proven.

Of 14 relevant windows stage 2 dropped: 11 as duplicates, and inspecting every
one, it kept the richer window and dropped the thinner in each case (two
Greenbriar windows were byte-identical from overlapping crops). Two may lose a
secondary attribute — a $1m break-up fee, a pair of covenant thresholds. The
other 3 were relevance calls on windows whose label is itself low-confidence.

## Cost

`openai/gpt-5.6-luna` at $0.20/$1.20 per Mtok: **$0.25 per 1,000 6-K filings**,
which is 0.9% of what extraction costs per window. It removes 47.5% of the
windows stage 1 admits.

| | straight to extraction | with stage 2 | saving |
|---|---|---|---|
| per 1,000 filings | $41.55 | $22.08 | **47%** |

So stage 2 returns roughly 76x its cost. The quality effect is probably worth
more than the money: half as many junk mentions reach the database.

## Configuration

Following the two patterns already in the repo rather than inventing a third:

| | where | default | override |
|---|---|---|---|
| stage-1 artifact **path** | `cdt.classifier.triage.default_model_dir()` | the committed `data/models/sixk/stage1-tfidf-linear-svc` | `cdt classify --sixk-model-dir`, or pass `model_dir` |
| stage-1 **threshold** | the artifact's `metadata.json` | 0.332 | retrain and recalibrate |
| stage-2 **model id** | `settings.SIXK_TRIAGE_MODEL` | `openai/gpt-5.6-luna` | `SIXK_TRIAGE_MODEL` env |
| stage-2 **reasoning effort** | `settings.SIXK_TRIAGE_REASONING` | `none` | `SIXK_TRIAGE_REASONING` env |
| **expansion** minimum / cap / merge budget | `cdt.segmenter.sixk` constants | 200 / 400 / 2,000 tokens | pass `min_tokens`, `max_tokens`, `max_merged_tokens` |

Paths follow the 8-K classifier: both default to the committed artifacts under
`settings.MODELS_DIR` (the repo's `data/models`), not to `DATA_DIR`, because the
models are part of the code and `DATA_DIR` is where a developer's data lives. Model ids and reasoning effort follow the
extractor, which does keep
them in `settings.py` with an env override. Both are read inside
`triage_filing` rather than bound as default arguments, so an override applied
after import is honoured. The triage id is separate from
`EXTRACTOR_MODEL` because the two jobs want opposite trade-offs: triage reads a
lot of text and returns a list of ids, so it is priced for volume; extraction
returns structured records and is priced for accuracy.

The threshold lives in the artifact rather than in code, and
`load_stage1_model` returns it alongside the model. It is a property of one
fitted pipeline: refitting moves the score scale, so a constant in code would
quietly stop meaning what it was calibrated to mean.

### The artifact

Committed at `data/models/sixk/stage1-tfidf-linear-svc/`, the same way the 8-K
classifier's is, and written by the same `classifier.core.save_training_artifacts`
so both have one on-disk contract. There is no fetch step.

It was **refitted in this repo under the pinned scikit-learn**, not copied from
the research repo, which would have shipped a 1.8.0 pickle that warns on load
here. Refitting reproduces the evaluated numbers exactly: the same 255 of 500
windows admitted at 35.4% precision and 95.4% recall. Training used 5,727
labelled windows (231 positive) with the 500 evaluation windows held out.

## Caveats worth carrying forward

- **The labels are one annotator's.** A second annotator agreed on 90% of a
  50-window blind sample, and disagreements traced to two rulings rather than to
  taste. Treat any single precision figure as ±10pp.
- **Stage 2 is not deterministic.** Two identical runs at temperature 0 agreed on
  94.9% of keep/drop calls, so single-run precision carries about ±1.5pp.
- **Retraining means recalibrating.** The threshold in `metadata.json` belongs to
  the fitted pipeline beside it; a new fit needs a new threshold measured against
  a labelled sample, not the old number carried over.
- **Expansion is not measured against labels.** Its acceptance check counts
  snippets that carry numbers with no instrument noun, which is a property of
  the text rather than a judgement of relevance. It says the input now
  determines an answer; it does not say the answer improved. 6-K rerun
  stability, the other half of the check in issue #172, needs a pipeline run.

## Segment and classify: where the windows are stored

The 6-K chain is split at the window boundary, as the 8-K chain is at the item
boundary. `cdt segment --genres 6-K` (`cdt.segmenter.sixk.segment_pending_sixk_documents`)
runs the strip, the gate and the windowing and writes `sixk-windows`.
`cdt classify --genres 6-K` (`cdt.classifier.sixk.triage_pending_windows`) runs
stage 1, expansion and stage 2 into `sixk-snippets`. Retuning or retraining stage
1 then reruns only classify; the windowing and its tokenizing, most of the 6-K
CPU time, do not rerun.

**Windows are stored as spans, not text.** Measured on 264 real 6-Ks (filed
2026-09-15 to 09-17): 33 pass the gate and yield 3,567 windows, about 108 per
passing filing, of which stage 1 admits about 6%. As parquet, a span row costs
about 20 bytes and a row carrying its text about 730: roughly 0.23 MB against
9.7 MB per 1,000 filings, or about 160 MB against 7 GB over a 20-year backfill
of an estimated 700k filings.

Storing the text would save classify almost nothing, because expansion reads up
to 400 tokens *before* each admitted window, so classify needs the source text
anyway for every filing that reaches stage 2, which is nearly every filing that
has windows. Classify therefore reads each windows partition together with the
`documents-sixk` partition of the same date and shard, re-derives the gated
body of each document from the mirrored submission (`prose_documents`, then
`gated_body`), and rebuilds every window as `body[start:end]`. That costs one
extra read of the mirror per gate-passing filing.

The coupling is guarded rather than assumed. Each span row carries
`source_sha256`, the digest of the body it indexes. If the rebuilt body's digest
differs — the flattening code changed between the two stages, or the document
is missing — classify judges none of that filing's windows, writes nothing for
its partition and records no completion for it, so the partition stays
pending. After processing every other partition it raises
`StaleSegmentationError`, naming the held partitions and telling the operator
to rerun `cdt segment --genres 6-K --force`.

## Where expansion runs

Inside `cdt.classifier.sixk`, between `stage1_admit` and `triage_filing` — the only
place it can run, since stage 1's threshold was calibrated on the crop and the
extractor needs the expanded text. Admitted windows are grouped by document
first: offsets mean nothing outside the text they index into, so expanding
across two documents would splice unrelated prose together.

One consequence for `sixk-snippets`: **a row is a snippet stage 2 judged, not a
window stage 1 admitted.** Merging makes those differ, and the alternative —
one row per member — would carry the merged text on each of them, so the
extractor would read rows and pay for the same text twice, which is the cost
merging exists to avoid. `sixk_member_windows` holds the comma-separated window
indices a row answers for, so every admission remains auditable: with the row's
accession and document index (both in `item`), it names each window stage 1
admitted and the verdict that window's text received.

Replaying the generalization window's 6-K set through the wired stage
reproduces the acceptance numbers measured before it was wired: 27 filings,
392 admitted windows, 219 snippets sent, 147,654 → 193,304 stage-2 input
tokens (1.31x), 88 of the 219 being merged groups.

## Design notes behind the code

### `cdt.segmenter.sixk.strip_inline_xbrl_prologue`

The prologue is stripped rather than the document dropped, because these
documents carry real prose after it. Leaving it in costs twice: the NER stage
must echo its input verbatim, so every prologue token is paid for at output
prices and adds a chance of failing the identity check; and TF-IDF windows of
tag soup dilute the stage-1 signal. `MIN_XBRL_TAG_SHARE` exists because a
borrowings schedule is also mostly bare numbers, and stripping one would delete
table bodies the annotation codebook rules relevant.

`prepare_filing` strips idempotently: stripped text begins at prose, so a second
pass finds no prologue. `cdt.segmenter.sixk.prose_documents` therefore leaves it
alone, and the research harness, which stripped in both places, is still
reproduced.

### `MIN_EXPANSION_TOKENS`, `MAX_EXPANSION_TOKENS`, `MAX_MERGED_TOKENS`

- **Minimum, 200.** Recovers a noun in 7 of 7 of the noun-less kept snippets
  (above); 150 gets most of the way, and 200 is what also reaches the table
  header above a page break, the shape reviewers could not read at all.
- **Cap, 400.** Expansion is paid only on the 5.8% of windows stage 1 admits,
  but the walk needs a stop when no header or blank line appears, or a document
  of unbroken table rows would prepend itself to every window. Past the minimum
  the cap only buys a tidier boundary, so it is one window wide.
- **Merge ceiling, 2,000.** Without one, the generalization window's run of 21
  adjacent windows merges to 8,191 tokens. When the ceiling cuts a run, the
  window opening the next span still expands, so up to 400 tokens are sent
  twice. That is the cheaper mistake: a window starting cold at the cut is the
  noun-less failure again, and the run of 21 is exactly that case, its table
  header sitting above a cut.

Header detection (`_is_section_header`) errs towards "not a header": a missed
header lets the walk continue to its minimum and pass over it, while a false
one stops the walk early and can leave the instrument noun outside the window.

### `cdt.classifier.triage`: what stage 2 is asked

Stage 2 answers two questions that fail differently. Does the window state a
concrete attribute of a specific instrument? Rejecting one that does not costs
nothing. Is every attribute it states already in a kept window? Dropping a true
duplicate is a precision gain, but dropping one that added an attribute is
silent data loss, so the rule is conservative and every duplicate ruling must
name the kept snippet covering it (`DROP_REASONS`, `validate_verdict`).

The snippet fence carries a per-request nonce because snippet text is written
by the filer. A bare `--- snippet N ---` delimiter is forgeable: a 6-K holding
that string splits into blocks the model reads as separate snippets, a verdict
on that numbering still partitions the ids, and a real disclosure is dropped at
the filer's discretion.

A filing with no admitted window makes no call: it is the common case at a 5.8%
admission rate, and an empty user message is a 400 from several providers.

### `cdt.classifier.sixk`: dataset and row shape

`sixk-snippets` is its own dataset rather than rows in `classifications`.
Classify owns `classifications/date=D/shard=S/part-0000.parquet` and rewrites it
whole, so a second writer would need genre-scoped merge-on-write and would race
a second completion registry. One writer per dataset is the invariant the
file-native design relies on.

Dropped snippets are persisted with their verdict because stage 2 is a
non-deterministic LLM whose decisions must be auditable. Windows stage 1
rejected are not: at a 5.8% admission rate, storing them would grow the dataset
about 17x to record that nothing happened. A row's `classification_score` is
the highest score among its merged members, since the weakest would understate
why the text was sent.

`item_id` names a snippet's character span, not its first member window. A
merged row named after its first member would keep that id when regrouping
changed its text, and since the extractor skips ids it has finished, the new
text would never be extracted while the absorbed members' old mentions had
nothing to prune them. Two groupings covering one span carry the same text, so
sharing an id there is correct.

`sixk_member_windows` exists because a row is a stage-2 snippet rather than a
window: with the accession and document index in `item`, it names every window
stage 1 admitted.

Both 6-K stages recompute a whole partition when their source partition
changed: segment when ingest merged new rows into a documents partition, and
classify when that rewrote the windows partition, as the 8-K segment and
classify stages do. Unlike them, classify costs an LLM call per filing with
admitted windows; at about $0.25 per 1,000 filings that is accepted.

The OpenAI provider (`SIXK_TRIAGE_PROVIDER=openai`) exists because OpenRouter
reserves an estimated maximum cost per in-flight request, so it is the first to
refuse under this stage's shape: many concurrent long-prompt calls.

### `cdt.ingest.sixk`: assembly fidelity

Assembling a submission from the scraper's per-document objects was checked
against EDGAR on 23 real filings from 2016 to 2026, including a 6-K/A: every
flattened prose document was byte-identical, which is what carries the triage
stage's measured behaviour over from the corpus it was scored on.
`DOCUMENT_MARKER` is checked rather than assumed because a source that stopped
keeping the `<DOCUMENT>` wrapper would change the prose the window stage reads,
with nothing downstream able to tell. The mirror is written as UTF-8 because
that is what `decode_document_bytes` produced; any other encoding would change
the text the extractor quotes as evidence.

Recorded CIKs keep the manifest reader's 10-digit padded form, as 8-K rows do,
so one issuer's CIK reads the same in both genres. `shard_for_cik` hashes the
unpadded form, so sharding would survive either spelling.
