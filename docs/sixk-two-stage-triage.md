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
5. **Stage 2: LLM over a whole filing's admitted windows at once.**

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

| | precision | recall |
|---|---|---|
| stage 1 alone | 35.4% | 95.4% (window) |
| **stage 1 + stage 2** | **70.9%** | 74.6% (window) |
| | | **100.0% (filing)** |

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
| stage-2 **model id** | `settings.SIXK_TRIAGE_MODEL` | `openai/gpt-5.6-luna` | `SIXK_TRIAGE_MODEL` env |

Paths follow the 8-K classifier, which derives from `DATA_DIR` via
`classifier.core.default_model_dir` rather than taking a settings entry, so one
variable moves both classifiers. Model ids follow the extractor, which does keep
them in `settings.py` with an env override. The triage id is separate from
`EXTRACTOR_MODEL` because the two jobs want opposite trade-offs: triage reads a
lot of text and returns a list of ids, so it is priced for volume; extraction
returns structured records and is priced for accuracy.

**The artifact is not in this repo.** Place `model.pkl` and `metadata.json` from
`model-c400-v5` in the directory above before running the stage.

## Caveats worth carrying forward

- **The labels are one annotator's.** A second annotator agreed on 90% of a
  50-window blind sample, and disagreements traced to two rulings rather than to
  taste. Treat any single precision figure as ±10pp.
- **Stage 2 is not deterministic.** Two identical runs at temperature 0 agreed on
  94.9% of keep/drop calls, so single-run precision carries about ±1.5pp.
- **The threshold belongs to one model.** Retraining stage 1 requires
  recalibrating 0.332 against a labelled sample; the score scale moves.
- **The shipped artifact was pickled under scikit-learn 1.8.0** and this repo
  resolves 1.9.0, which warns on unpickle. Regenerate it under the pinned version
  before relying on it.

## What is not in this change

Orchestrator wiring. This adds the stage as a library with its own tests; making
it a pipeline stage alongside ingest → itemize → classify → extract needs
decisions about partitioning and dataset registration that are better taken
separately. Cross-row deduplication after extraction is also still open — that is
where duplicate *mentions* should be resolved, by comparing extracted values
rather than inferring from prose.
