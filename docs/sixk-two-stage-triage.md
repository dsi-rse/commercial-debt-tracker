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

Steps 1-3 are composed by `cdt.sixk.prepare_filing`, so the order is code
rather than prose. It matters in both directions: the gate applies to the whole
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
  at roughly 2,000 tokens, the largest snippet the 8-K path already sends the
  same extractor; the generalization window has a run of 21 that would otherwise
  reach 8,191.

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
| stage-1 artifact **path** | `cdt.sixk.default_model_dir()` | `DATA_DIR/models/sixk/stage1-tfidf-linear-svc` | `DATA_DIR`, or pass `model_dir` |
| stage-1 **threshold** | the artifact's `metadata.json` | 0.332 | retrain and recalibrate |
| stage-2 **model id** | `settings.SIXK_TRIAGE_MODEL` | `openai/gpt-5.6-luna` | `SIXK_TRIAGE_MODEL` env |
| stage-2 **reasoning effort** | `settings.SIXK_TRIAGE_REASONING` | `none` | `SIXK_TRIAGE_REASONING` env |
| **expansion** minimum / cap / merge budget | `cdt.sixk.windows` constants | 200 / 400 / 2,000 tokens | pass `min_tokens`, `max_tokens`, `max_merged_tokens` |

Paths follow the 8-K classifier, which derives from `DATA_DIR` via
`classifier.core.default_model_dir` rather than taking a settings entry, so one
variable moves both classifiers. Model ids and reasoning effort follow the
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

## Where expansion runs

Inside `cdt.sixk.stage`, between `stage1_admit` and `triage_filing` — the only
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

## What is not in this change

Cross-row deduplication after extraction — that is where duplicate *mentions*
should be resolved, by comparing extracted values rather than inferring from
prose. Publishing 6-K snippets into `items/latest.parquet` is no longer open:
the table is a union over both genres, stamped with a `form_type` column, so a
6-K mention joins to its own snippet row rather than to nothing. What the
website makes of a 6-K row's `item` -- a snippet id where an 8-K row carries a
dotted item number -- is still a dashboard-side question.

The scheduled pipeline does now run this stage: `pipeline.py` prepares both
genres by default and the orchestrator takes `--genres` / `GENRES` to narrow a
run. Worth knowing before enabling it on a wide CIK list: unlike the 8-K
prepare chain, this stage costs an LLM call per filing with admitted windows.
