# PR Review Findings: Lifecycle and schema cleanup (#197)

PR [#197](https://github.com/dsi-rse/commercial-debt-tracker/pull/197). Reviewed 2026-09-18 against `dev` at `ba3c589`, head at `9faf63a`.

Reviewable diff: 199 added / 1125 deleted lines across 7 files from 5 commits. Two-dot and three-dot diffs agree (`HEAD~5 == origin/dev`), so nothing in the diff belongs to another PR. No generated or vendored files.

| file | +/− | commit group |
|---|---|---|
| `src/cdt/matcher/core.py` | +13/−410 | removal (1,2), schema bump (3) |
| `tests/test_file_native_stages.py` | +4/−697 | removal (1,2) |
| `src/cdt/matcher/lineage_inference.py` | +69 | borrower guard (5) |
| `tests/test_matcher_lineage_inference.py` | +83 | borrower guard (5) |
| `docs/schema.md` | +25/−17 | removal (1), schema/maturity (3) |
| `src/cdt/extractor/core.py` | +2/−2 | comment (1) |
| `docs/architecture.md` | +1/−1 | docs (4) |

User-supplied context: **the website is pre-beta and it is acceptable for this merge to break it.** Findings whose only consequence is downstream dashboard breakage were excluded from scope by instruction.

## Verification runs

All run on the PR branch with `source .venv/bin/activate && PYTHONPATH=src`.

- `python -m pytest -q` → **499 passed, 1 xfailed in 8.56s**. Matches the PR's claim.
- `ruff check .` → **All checks passed!** (ruff 0.7.2, the repo's pinned version)
- `ruff format --check .` → **56 files already formatted**
- `gh pr checks 197` → all 5 green (CodeQL, Lint, Pulumi Preview, Security, Test)
- `python -c "import cdt.matcher, cdt.cli, cdt.pipeline, cdt.orchestrator"` → clean

### A/B against the real corpus (the PR's central claim)

Two throwaway worktrees (`ba3c589` = dev, `c3b451b` = after commit 2), each run with `PYTHONPATH` pointed at its own `src` so the worktree resolves its own branch, over the 542-instrument corpus at `data/genwindow-run-branch` (669 mentions):

```
dev  schema_version: 5 cols: 42 heads: 535 links: 3
c2   schema_version: 5 cols: 38 heads: 535 links: 3
columns removed: ['status', 'status_subtype', 'status_date', 'status_source_mention_id']
columns added:   []
shared columns: 38
row ids identical: True ( 542 rows )
every shared column byte-identical: True
```

**The removal is verified behaviour-preserving on real data.** The correctness reviewer reproduced this independently with a full `force=True` rematch and additionally found `mention-cluster-edges` byte-identical (672 rows both sides).

### The reference-date argument (the justification for removing it)

Re-derived on dev, same corpus, only the reference date changed. **Every figure in the PR's table reproduces exactly**, and the "changed nothing else" claim holds:

| reference date | status differs | non-status cell diffs |
|---|---|---|
| 2026-06-30 | 49 / 542 | 0 |
| 2026-03-31 | 184 / 542 | 0 |
| 2025-09-08 | 198 / 542 | 0 |
| 2024-09-08 | 218 / 542 | 0 |

### The borrower guard's measured effect

Guard fires 4 times on the corpus; 2 of the 4 are the EQT/EQM pair the commit targets.

```
commit2 -> HEAD: heads 535->535  links 3->3
columns changed: superseded_by ×2, amendment_of ×2, lineage_family_id ×3, amendment_inferred_by ×2
```

Lineage families before/after:

```
before: 2017-08-03 A&R Revolving -> 2017-11-14 Second A&R -> 2022-06-28 Third A&R
        + EQM's 2024-07-22 Third A&R welded in (ordinal_chain)
        2025-07-01 Fourth A&R orphaned in its own family (refused: two rank-3 parents)
after:  EQM's Third A&R detached into its own family with EQM's Prior Revolving
        2025-07-01 Fourth A&R resolves into EQT's chain (ordinal_chain)
```

Performance: 24 `_borrowers()` calls on the whole corpus, 4 ms total for the inference pass. Not a concern.

---

## Factor: correctness (reviewer scope: full diff)

### Findings

- **[correctness] A defined-term borrower name is read as positive evidence of a different party** — `src/cdt/matcher/lineage_inference.py:113-126` [proposed: Important] [confidence: verified] The extractor sets `canonical_name` to the longest span in the cluster, so a filing that only calls the obligor by its defined term records the borrower as literally `"Issuer"`. `_borrowers_disagree` then refuses the link. Real row pair, CIK 0001089113: `'HSBC Holdings plc'` → `('hsbc','holdings')` vs `'Issuer'` → `('issuer',)`, no prefix relation, refused. This contradicts the function's own docstring ("this only ever fires on positive evidence of a different borrower") — a defined-term alias is the same silence the docstring says must not refuse. Corpus measurement: **8 distinct fully-generic borrower keys** (`issuer`, `issuers`, `borrowers`, `buyer`, `buyer parent`, `parent`, `loan parties`, `other borrowers party thereto`) on **17 rows, 10 of which have no other borrower key**; my independent count found **120 of 836** refusable same-CIK pairs driven by a placeholder on one side. It cuts both ways: two different obligors that both reduce to `('issuer',)` are read as *agreeing*. Fix: filter generic party words in `_borrowers` so a generic-only row falls back to silence; `GENERIC_LENDER_TERMS` (`matcher/core.py:53-70`) already exists and already contains `buyer`/`buyers`.

- **[correctness] Prefix matching re-admits the parent-vs-subsidiary confusion the guard exists to block** — `src/cdt/matcher/lineage_inference.py:124-126` [proposed: Important] [confidence: verified, latent] `a[:len(b)] == b or b[:len(a)] == a` accepts `<Parent>` ≡ `<Parent> <Qualifier>`, the commonest finance-subsidiary naming pattern. The stated justification ("`EQT` and `EQT Corporation` are one borrower") does not need it: `BORROWER_SUFFIXES` already strips `corporation`, so both sides are `('eqt',)` and equality suffices. Verified against the real functions: `'EQT Corporation'` vs `'EQT Midstream Partners, LP'` → agree; `'Ford Motor Company'` vs `'Ford Motor Credit Company LLC'` → agree; `'Apple Inc.'` vs `'Apple Operations International'` → agree. **The motivating case survives only because `eqt` and `eqm` differ in their first token — and EQM Midstream Partners, LP was itself renamed from *EQT Midstream Partners, LP*, the name under which its 2018-10-31 agreement was signed.** Corpus scan: 15 distinct same-CIK key pairs the prefix rule collapses, including `('eqt',)`/`('eqt','production')`, `('caterpillar',)`/`('caterpillar','financial','services')`, `('andx',)`/`('andx','finance')`, `('parent',)`/`('parent','merger','sub')`; my independent count found 90 exposed same-CIK pairs. Confirmed **latent**: switching to set intersection changes no link on this corpus (both give the same 3 links and refuse the same pairs). Fix: `child_keys & parent_keys`, or restrict the extra tokens to `BORROWER_SUFFIXES`.

- **[correctness] `docs/schema.md:270` still tells the publisher the matcher reads planned retirements "as pending"** — `docs/schema.md:270` [proposed: Important] [confidence: verified] Commit 3 removed the other half of this sentence but left "a planned retirement decides nothing here and is read by the matcher as pending". `grep -rn "pending" src/cdt/matcher/*.py` returns only `match_pending_mentions` — there is no pending concept left. This matters more than a normal doc nit: the new paragraph at `docs/schema.md:476-481` points the publisher's author at exactly this bullet as the contract for mention `status`, and this clause tells them a downstream stage already handles planned retirements.

- **[correctness] `_borrowers` raises on JSON-valid non-list `parties_json` where its siblings return empty** — `src/cdt/matcher/lineage_inference.py:101-109` [proposed: Personal preference] [confidence: crash verified, reachability not demonstrated] Only `json.JSONDecodeError` is caught. Probed: `nan`/`None`/`'{not json'`/`'"x"'` → `set()`, but `'null'` → `TypeError: 'NoneType' object is not iterable` and `'3'` → `TypeError: 'int' object is not iterable`, which aborts the whole lineage pass. Sibling `parse_cluster_list` (`core.py:1622`) has `if not isinstance(payload, list): return []` and `_prior_amounts` (`lineage_inference.py:156`) catches `(TypeError, ValueError)`. No live path produces these — every writer goes through `json.dumps(dedupe_party_clusters(...))` — so this is defence-in-depth, not a live bug.

- **[correctness] The cluster's `parties_json` is a union, so the guard weakens as clusters absorb mentions** — `src/cdt/matcher/lineage_inference.py:99-126` [proposed: Opinion] [confidence: verified] `build_debt_instrument_rows` (`core.py:1298-1308`) unions the existing row's parties with every member mention's, so a row can carry several borrower keys — **46 of 542 rows carry ≥2** (27×2, 11×3, 6×4, 2×5). Because the comparison is `any()` over the cross product, one agreeing pair anywhere passes. Worst rows: CIK 0001552000 carries `andeavor logistics`, `andx`, `andx finance`, `mplx`, `tesoro logistics finance` — five obligors, so the guard cannot refuse it against almost anything under that CIK. My count: **630 of 1867** agreeing same-CIK pairs involve a multi-borrower row. This is inherent to the surface, and the union direction is also what makes "silence is not disagreement" work, so no semantic change is recommended — but it deserves a sentence in the docstring so a later reader does not assume the guard is tight. MPLX/Andeavor is structurally the same acquired-subsidiary shape as EQT/EQM.

- **[correctness] Commit 5's net effect on published `superseded_by` is zero, not a gain** — informational [proposed: Opinion] [confidence: verified] Unblocking Fourth→Third gives the 2022 Third A&R two children (the newly inferred Fourth A&R and an *extracted* pointer from a second cluster of the same 2024-07-22 agreement), so the ambiguity rule nulls its `superseded_by`. Measured: the 2017 Second A&R gains one, the 2022 Third A&R loses one, corpus `superseded_by` count unchanged at 6, heads unchanged at 535. Root cause is the duplicate clustering the PR already lists as out of scope, and "null on genuine ambiguity" is the documented policy — but the PR describes only the gain.

### Verified clean

- **The removal is a leaf.** All 20 removed symbols grepped across `src/ tests/ scripts/ notebooks/ pulumi/ .github/ docs/ Makefile README.md` over `*.py *.md *.toml *.yml *.yaml *.ipynb *.ts *.json` — zero live references. Two exceptions, both in tests and reported below.
- **No remaining code path reads `status` off an instrument row.** The only instrument-adjacent hits are `prepare_mention` (`core.py:1684`, reads the *mention* row) and `ClusterProfile.add_member` (`core.py:217`). `build_cluster_profiles` rebuilds `retired` from mentions, never from the instrument row.
- **`ClusterProfile.retired` / `TERMINAL_STATUS_EVENTS` intact as described** — both mention-level, both retained; the byte-identical edges in the A/B is the behavioural proof.
- **`reference_date` plumbing fully unwound.** Zero hits anywhere. `pipeline.py:312`/`:409` and `cli.py:883` call `match_pending_mentions` with the surviving keywords only; a stale caller would raise `TypeError`, not silently default.
- **The new schema.md paragraph's "every input is already published" claim is accurate.** Mention `status`, `status_date` and `dates_json` are in `DEBT_INSTRUMENT_MENTION_COLUMNS` and populated (669/669 rows carry `dates_json`); `mention-cluster-edges` is a `FINAL_OUTPUT_TABLES` member so the cluster→mention mapping is published; `retired_by_debt_instrument_ids`, `is_lineage_head` and `superseded_by_debt_instrument_id` are all still on the row; the cross-row "retiring instrument not started" check is reconstructible.
- **`offer()` guard ordering is right** — the borrower check runs before `candidates[child_id].setdefault`, so a refused offer cannot shadow a later valid one, and the guard applies to both `prior_fact` and `ordinal_chain`.
- **Empty-key handling is safe** — `_borrower_key("The Company")` and `_borrower_key("Co.")` return `()`, filtered by `if key`, so they become silence rather than a universal prefix match.
- **`MATCHER_SCHEMA_VERSION = 6` reaches both consumers** (`core.py:379` run manifest, `pipeline.py:571` snapshot pointer). No doc pins the old value.

### Highlights

- Placing the guard in `offer()` rather than in the resolution step means it *reduces* ambiguity instead of only removing links — measurably: it killed the wrong EQM→EQT link **and** rescued EQT Fourth→Third from a two-parent tie. A post-hoc filter would only have done the first half.
- The guard lands on the correct side of the documented stage boundary (`docs/architecture.md:220`: "A matcher heuristic may **refuse** a match. It may not **assert** a fact"). It is a `return` inside `offer()` reasoning only over extractor-bound, cited `parties_json`.
- The `docs/schema.md` removal paragraph states the measurement rather than asserting the conclusion. Documenting *absence*, with the evidence, is rare and is what stops the cascade being re-added.

---

## Factor: test coverage (reviewer scope: full diff)

### Findings

- **[tests] A `strict=True` xfail now xfails on `ImportError`, not for the reason it states** — `tests/test_file_native_stages.py:6175-6233` [proposed: Important] [confidence: verified] The `reason=` documents a bug in `event_status_for_instrument` — a function **this PR deletes**. Reproduced:

  ```
  pytest ...::test_a_planned_retirement_survives_a_newer_amendment_mention -q --runxfail
  E   ImportError: cannot import name 'event_status_for_instrument' from 'cdt.matcher.core'
  ```

  This 59-line test is now the only reference to that symbol anywhere in the repo, kept green because `strict=True` only cares *that* it failed, not why. `strict=True` is meant to be a tripwire that goes red when a bug is fixed; it now pins nothing, and its reason string documents a defect in a function that no longer exists and a rollup this PR removed, so it can never be "fixed". The suite's "1 xfailed" also misrepresents the repo as carrying one known unfixed bug. Both the test and coherence reviewers flagged this independently. Fix: delete the test and its decorator alongside the function.

- **[tests] The legal-form-suffix test passes two different ways, so it pins neither** — `tests/test_matcher_lineage_inference.py:404-422` [proposed: Important] [confidence: mutation-proved] `test_a_legal_form_suffix_is_not_a_different_borrower` compares `"EQT"` against `"EQT Corporation"`. Two independent mechanisms each make it pass alone: `BORROWER_SUFFIXES` strips `corporation` (equality suffices), **and** the prefix rule matches `('eqt','corporation')` against `('eqt',)` (the suffix list is unnecessary). Mutation proof: dropping the prefix rule alone **survived**, dropping `BORROWER_SUFFIXES` alone **survived**, doing both was killed. So a contributor can delete the entire 19-line constant, or replace the prefix comparison with `==`, and the suite stays at 499 passed. The one test whose name claims to cover suffix handling covers neither mechanism. Fix: split into two tests that isolate one mechanism each — e.g. `"EQT Corporation"` vs `"EQT Company"` for the suffix list, `"Acme Holdings"` vs `"Acme Holdings Finance"` for the prefix rule.

- **[tests] The guard's effect on the `prior_fact` rule is untested** — `src/cdt/matcher/lineage_inference.py:213-214`, offers at `:246` [proposed: Important] [confidence: verified] The guard sits in `offer()` so it gates both rules, but all three new tests use `ordinal_chain`. Verified the behaviour is real by adding disagreeing borrowers to the existing `prior_fact` fixture: result went from `{"i2": ("i1","prior_fact")}` to `{}`. `prior_fact` is the rule with the *stronger* evidence (an exact `prior`-marked amount match), so whether a borrower mismatch should override it is a genuine design decision currently made silently.

- **[tests] The `JSONDecodeError` guard is load-bearing for the parquet NaN path and has no test** — `src/cdt/matcher/lineage_inference.py:101-104` [proposed: Personal preference] [confidence: verified] Deleting the `try/except` leaves 499 passed. But `apply_lineage_inference_pass` feeds `instruments.to_dict("records")` straight in, so a missing `parties_json` arrives as float `nan`, which is truthy: `str(nan or "[]")` → `"nan"` → `json.loads("nan")` raises (Python's json accepts `NaN`, not lowercase `nan`). Remove the guard and that path crashes the whole stage, with nothing in the suite to say so.

- **[tests] Multi-borrower semantics and the empty-key filter are unasserted** — `src/cdt/matcher/lineage_inference.py:105-110` [proposed: Personal preference] [confidence: mutation-proved] Every new test uses a single borrower, so the deliberate `any()` semantics are unpinned. Dropping the `{key for key in keys if key}` filter **survived** mutation: it is behaviour-preserving for single-borrower rows (the early exit covers them) and only diverges on a mixed row like `{(), ("eqm","midstream")}`, where the empty tuple prefix-matches everything and silently disables the guard — precisely the multi-borrower case nothing covers.

- **[tests] `isinstance(party, dict)` and `NAME_NOISE` filtering in `_borrower_key` have no test** — `src/cdt/matcher/lineage_inference.py:95, 108` [proposed: Opinion] [confidence: mutation-proved] Both mutations survived. The `isinstance` guard is real (`'["oops", {...}]'` returns `{('eqt',)}` rather than raising), but nothing asserts it. `NAME_NOISE` (`the/a/an/that/certain`) in a *borrower* name has no exercised case at all and may simply be copied from `_name_rank_and_stem`.

- **[tests] No end-to-end test that a borrower on a mention reaches the guard as an instrument row** — `tests/test_file_native_stages.py:2477` [proposed: Opinion] [confidence: medium] All three new tests call `infer_amendment_parents` directly with hand-built dicts; the only e2e test of `apply_lineage_inference_pass` builds mentions with no parties. The file's own `instrument()` helper doesn't include a `parties_json` key, so the unit tests can't catch a rename or a dtype surprise. Fix: add one borrower to the two mentions in the existing e2e test.

### Verified clean

- **No test was weakened to make the suite pass.** `test_lifecycle_rollup_marks_heads_families_and_status` → `test_lifecycle_rollup_marks_heads_and_families` dropped exactly three assertions, all pure status; every surviving assertion was kept verbatim. In the e2e test at `:2477`, `assert after["m-1"]["status"] == "closed"` was **replaced** by `assert after["m-1"]["is_lineage_head"] is False` — a mutation-killing substitution, not a deletion.
- **All 19 deleted test functions tested only removed behaviour.** Every symbol they imported was deleted in the same commit. `test_two_amendment_children_leave_no_unreachable_parent`'s surviving half is fully duplicated by `test_two_amendment_children_publish_no_superseded_pointer` (`:6121`), which also adds `is_lineage_head is False`. `test_status_does_not_depend_on_which_shard_an_issuer_hashes_into`'s incidental multi-shard coverage is still held by `test_match_pending_mentions_drains_all_shards`. `_expected_closing_dates_json` has no remaining consumers.
- **`apply_lifecycle_rollup`'s surviving logic is genuinely pinned — 7/7 mutants killed**: `superseded_by` ambiguity, `lineage_family_id` (both the `min` choice and the split/retired edges), `is_lineage_head`, `first_seen`/`last_seen`, `mention_count`, `document_count`.
- **Schema-removal tripwires intact**: `MATCHER_SCHEMA_VERSION == 6` (`:5936`), the exact `DEBT_INSTRUMENT_COLUMNS` list (`:5747`), and `list(tables["debt_instrument"].columns) == DEBT_INSTRUMENT_COLUMNS` (`:5957`) — so the four removed columns cannot silently return, nor a fifth silently vanish.
- **No coverage theater in the new tests** — no assert-free tests, no snapshot-everything, no mocks, no sleeps, no ordering dependence. Suite is ~6s and deterministic across ~20 mutation runs.

### Mutation results on the new guard — 5/11 killed (45%)

| mutation | result |
|---|---|
| `_borrowers_disagree` → always `False` | KILLED |
| `_borrowers_disagree` → always `True` | KILLED (6 tests) |
| drop the `if not child_keys or not parent_keys` early exit | KILLED (4 tests) |
| prefix compare → plain `a == b` | **SURVIVED** |
| drop `BORROWER_SUFFIXES` filtering | **SURVIVED** |
| drop `NAME_NOISE` filtering | **SURVIVED** |
| `role == "borrower"` → `"lender"` | KILLED |
| remove the `except json.JSONDecodeError` | **SURVIVED** |
| remove `isinstance(party, dict)` | **SURVIVED** |
| remove the empty-key filter | **SURVIVED** |
| delete the `_borrowers_disagree` call site in `offer()` | KILLED |
| prefix→`==` **and** drop `BORROWER_SUFFIXES` together | KILLED |

### Highlights

- `test_a_different_borrower_refuses_the_ordinal_link` is the model for this kind of test: it reconstructs the real EQT/EQM filing and asserts **both** legs — the good link survives and the bad one dies. That single test kills three separate mutants including deletion of the entire call site.
- `test_a_missing_borrower_does_not_refuse_the_link` pins the subtlest and highest-blast-radius part of the guard: dropping that early exit would have disabled *every* inferred link in the corpus, and this test catches it.

---

## Factor: repo coherence (reviewer scope: full diff + all non-Python files)

### Findings

- **[repo-coherence] `apply_lineage_inference_pass`'s docstring still promises it keeps `status` consistent** — `src/cdt/matcher/core.py:2099-2101` [proposed: Important] [confidence: verified] "…re-derives the rollup columns from the updated pointers, so `superseded_by`, `lineage_family_id`, `is_lineage_head` and **`status`** stay consistent." This is the **only surviving mention of the removed instrument-level column in `src/` or `docs/`**, and it sits 1,500 lines below the new `apply_lifecycle_rollup` docstring that says the exact opposite. It is the docstring for the public entry point behind `cdt match --infer-lineage`, so the first thing a reader of the lineage pass learns is that the removed column still exists. Every other docstring in the removal was updated carefully; this one was missed.

- **[repo-coherence] `PreparedMention.status_date` is now dead, and commit 2 claims otherwise** — `src/cdt/matcher/core.py:183` (declaration), `:1685` (only write) [proposed: Important] [confidence: verified] Commit `c3b451b` argues `PreparedMention` "is a working set, where every field exists because some code path reads it, and after the previous commit these three were the only exceptions." That is false — `status_date` is a fourth exception created by the same commit. At `origin/dev` it was read at `core.py:882`, `:885`, `:952`, `:954`, all inside `planned_retirement_date` / `event_status_for_instrument`, both removed by commit 1. I confirmed: in the matcher, `status_date` appears only at `:183` and `:1685`, and no dynamic `getattr` path can reach it. Note `mention.status` (`:182`) **is** still read at `:217`, so only `status_date` is dead and the sibling must stay. Fix: drop it, or correct the commit-message claim.

- **[repo-coherence] `docs/schema.md` references `expected_active` after deleting its only definition** — `docs/schema.md:264` [proposed: Important] [confidence: verified] The PR changed "The instrument-level `expected_active` leg" → "The publisher's `expected_active` leg", but the five-value table that defined `expected_active` was deleted in the same commit. `grep -rn 'expected_active' docs/ src/` returns this line and nothing else, so the term is now undefined anywhere. This is the dangling reference left by the lifecycle-table removal.

- **[repo-coherence] `parties_json` is missing from both stated lists of lineage-inference inputs** — `src/cdt/matcher/lineage_inference.py:28-29` and `docs/architecture.md:230` [proposed: Important] [confidence: verified] The module docstring says "**Every** input here is extractor output: the `prior` marks in `amounts_json`, the canonical `name`, `principal_amount`, and `start_date`." `architecture.md:230` says the two rules reason over `amounts_json` and `name`, "**both** of which the extractor bound to an object and cited." Commit 5 adds a fourth input — `parties_json` → borrower `canonical_name`, read at `:106`. The stage-boundary docstring is load-bearing: it is what a future contributor reads to decide whether a new input is legal here. The new input *is* on the right side of the line, so adding it strengthens the argument.

- **[repo-coherence] `_borrower_key` re-implements `normalize_party_text`, the repo's existing organization-name normalizer** — `src/cdt/matcher/lineage_inference.py:69-96` vs `src/cdt/matcher/core.py:1855-1864` [proposed: Important] [confidence: high on the duplication, medium on the refactor shape] Same job, same nine-of-thirteen suffix tokens, same lowercase-and-strip-punctuation shape — and `normalize_party_text` is **already applied to borrower names in this very pipeline**: `normalize_party_text` → `cluster_canonical_key` (`:1633`) → `dedupe_party_clusters` (`:1604`), which keys all party roles including `borrower` when `build_debt_instrument_rows` writes the `parties_json` that `_borrowers()` reads back. So the same borrower string is normalized two different ways, ten lines apart in the data flow, with divergent vocabularies: `normalize_party_text` knows `national association`; `BORROWER_SUFFIXES` knows `incorporated`, `lp`, `llp`, `limited`. A future addition to either list silently does not reach the other. Candidates ruled out first: `company_names_by_cik` (display names, no normalization), `canonical_value` (longest raw span), `lender_signature`/`lender_keys` (wrappers over `cluster_canonical_key`), `normalize_name_fingerprint`/`NAME_STOPWORDS` (instrument names). **Confirmed import cycle:** `core.py:35` imports `infer_amendment_parents` from `lineage_inference`, so the shared helper would have to move to `src/cdt/shared.py` or be imported lazily. That makes this the largest change in the set and defensible as a follow-up, provided the divergence is noted meanwhile.

- **[repo-coherence] The PR's "where the cascade went" pointer resolves, through the repo's own glossary, to an archived repo** — `docs/schema.md:476-478` [proposed: Personal preference] [confidence: verified] The new paragraph says the cascade "lives in the dashboard publisher (`dsi-rse/commercial-debt-tracker-website#15`)". I confirmed that pointer is correct: `-website` is live (`archived: false`, pushed 2026-06-05) and issue #15 is OPEN, titled "Derive instrument lifecycle status at publish time, against an explicit asOf date". But every other reference in the repo names `dsi-rse/commercial-debt-tracker-dashboard`, which is **archived** (last push 2025-01-22): `README.md:87`, `:116`, `docs/architecture.md:241`, `docs/deployment.md:157`, `:193`, `docs/deployment-dev.md:149`, `docs/design-sixk-pipeline-wiring.md:387`, `:485`. The staleness is pre-existing, but this PR makes it load-bearing — schema.md says "**the** dashboard publisher", and the only definition of that term is `architecture.md:241`. Minimum fix: write "the website publisher (`dsi-rse/commercial-debt-tracker-website`)" so the prose matches the link.

- **[repo-coherence] `apply_lifecycle_rollup` is no longer a lifecycle rollup** — `src/cdt/matcher/core.py:546` [proposed: Personal preference] [confidence: verified] It now fills lineage topology plus observation counters. Its own docstring has to spend a paragraph saying there is "Deliberately no lifecycle `status`", `docs/architecture.md:204` had to be edited to say status events "decide no lifecycle status here", and `lineage_inference.py:6-7` still calls it "the #155 lifecycle rollup". A name that needs a disclaimer in three places is the wrong name. Counter-argument weighed: it has 2 call sites plus 5 test call sites, and renaming enlarges a diff whose selling point is byte-identical output — fair reason to defer, but then `lineage_inference.py:6-7` should at least stop calling it "the lifecycle rollup" unqualified.

- **[repo-coherence] Schema-regression test fixture still builds the removed columns** — `tests/test_file_native_stages.py:5833-5846` [proposed: Personal preference] [confidence: verified] The test writes `"status": "closed", "status_subtype": subtype` into a `dict.fromkeys(DEBT_INSTRUMENT_COLUMNS)` fixture. Those keys are no longer in `DEBT_INSTRUMENT_COLUMNS`, so a test whose entire point is "the consumer promise: point any parquet reader at the directory" for the *published* schema now invents two columns to prove it. It still passes and still catches its regression, but `status_subtype` is the column carrying the null-in-first-partition case the docstring says is essential. Also at `:5809`: "23 of 42 `debt-instruments` columns" — the table is now 38. Fix: swap for a real nullable column, e.g. `superseded_by_debt_instrument_id`.

- **[repo-coherence] `TERMINAL_STATUS_EVENTS` is now 326 lines from its only consumer** — `src/cdt/matcher/core.py:539-543` [proposed: Personal preference] [confidence: verified] At dev it sat among four users; three are gone and the only reader left is `ClusterProfile.add_member` at `:217`. It now sits immediately above `apply_lifecycle_rollup`, which does not use it, and its good new comment explains `ClusterProfile.retired`, a class defined 320 lines earlier. Right module, wrong place in it.

- **[repo-coherence] The new borrower block splits `NAME_NOISE`/`MIN_CHAIN_MEMBERS` from their consumer** — `src/cdt/matcher/lineage_inference.py:69-126` [proposed: Personal preference] [confidence: verified] The four borrower definitions are inserted between those constants (`:66-68`) and `_name_rank_and_stem` (`:129`), which is what uses them, interrupting the ordinal-chain story. Moving them below `_canonical_date` keeps each constant next to its user.

- **[repo-coherence] Two small imprecisions in the rewritten `debt-instruments` bullets** — `docs/schema.md:455-456` [proposed: Opinion] [confidence: high on (a)] Reading `canonical_maturity_fields` (`core.py:1451-1489`) line by line, the core rule in the new bullet is **exactly right** and the `#162`/`#166` framing matches. Two residual gaps: (a) the split moved the existing-row fallback clause onto the *other* fields' bullet, but `canonical_maturity_fields:1479-1489` has the same fallback, so the maturity bullet reads as if it has none; (b) the bullet now lists `maturity_date` after `commitment_termination_date`, whereas `DEBT_INSTRUMENT_COLUMNS` has it before.

- **[repo-coherence] Commit 1 calls `reference_date` a parameter of `match_pending_mentions`** — commit message of `a1f616e` [proposed: Opinion] [confidence: verified] At dev it was a keyword parameter on `match_tables` and `apply_lifecycle_rollup` but a **local variable** in `match_pending_mentions`, computed from `mention_rows["date"].max()`. Minor, but the point of that sentence is to let a reviewer check the removal surface.

### Verified clean

Exhaustive sweep for the instrument-level `status`/`status_subtype`/`status_source_mention_id`, the five lifecycle values, `closed - repaid`, `reference_date`, `corpus_reference_date`, "lifecycle rollup", `expected_closing`, `expected_active`, and all twelve removed symbols:

- **`src/cdt/extractor/prompts/*.md` (LLM templates)** — clean. `instrument_relation.md:4` mentions `status` and `expected_retirement="true"`: both mention-level, both still produced. `instrument_ie.md:49` `announcement` is a `dates_json` kind, not a lifecycle value. No prompt ever knew about the cascade.
- **`docs/design-completion-semantics.md`, `docs/sixk-two-stage-triage.md`, `README.md`, `DataPolicy.md`, `Makefile`** — zero occurrences of "status". `design-sixk-pipeline-wiring.md` has only `extraction_status`; `deployment*.md` only "active job"; `scripts/` only `extraction_status`; `.github/`, `pulumi/`, `conftest.py`, `pyproject.toml`, `docker-compose.yaml`, `Dockerfile`, `.devcontainer/`, `.pre-commit-config.yaml` — all clean.
- **`src/cdt/storage.py`** — `DECLARED_COLUMN_TYPES` (`:438-455`) never declared a status column, so the four removals need no type-registry edit.
- **Mention-level vs instrument-level distinction correctly maintained throughout.** The mention `status`/`status_date`/`status_json` columns and `PreparedMention.status` survive, are still documented, still produced, still read. **The PR removed the right one.**
- **`docs/schema.md` column-list completeness** — all 38 `DEBT_INSTRUMENT_COLUMNS` entries are covered by a bullet, none extra, no orphaned bullet. No doc states a column count or a schema version number.
- **The "Presenting `status` (UI contract)" removal** leaves the `superseded_by` and `retired_by` bullets standing on their own; the only dangling term is `expected_active`.
- **Commit hygiene** — `git show --stat` on all five: each stat matches its message, each touches a coherent slice, and the ordering is genuinely reviewable (removal → dead-field follow-up → review response → doc fix → independent lineage guard).

### Highlights

- The `#196` rationale paragraph (`docs/schema.md:466-481`) is the strongest thing in the diff: it states the removal, the reason, the measurement, the negative evidence, and where the capability went with a working issue link.
- The `MATCHER_SCHEMA_VERSION` comment records the 5→6 reason inline rather than only in the commit, and cites in-repo precedent.
- The `maturity_date` bullet split is a real doc-accuracy fix **not required by #196** — the old text described the pre-#162 behaviour, i.e. the bug rather than the fix. It now matches `canonical_maturity_fields` closely and honestly flags the unfinished generalization.

---

---

## Factor: data safety and migration (reviewer scope: parquet artifacts written by this repo)

### Findings

- **[data-safety] The borrower guard cannot repair lineage a previous run already published** — `src/cdt/matcher/lineage_inference.py:191-195`, `src/cdt/matcher/core.py:1176-1180` [proposed: Important, arguably High] [confidence: verified, reproduced twice independently] `infer_amendment_parents` only considers rows whose pointer is null (`open_children = [... if not row.get("amendment_of_debt_instrument_id")]`), and `derive_parent_links` re-seeds the candidate set from the pointer already on disk. So a bad link written by a pre-guard run is never re-evaluated *and* is actively re-published by the next ordinary `cdt match`. My reproduction on the real corpus:

  ```
  fresh derivation, guard on : EQM link present? False
  dev (pre-guard) published EQM link: True -> ('dim::3fe2333cac34643d0da44aa7', 'ordinal_chain')
  after re-running the GUARDED pass over that published state:
     EQM still points at        : dim::3fe2333cac34643d0da44aa7
     guarded pass reconsidered? : False (it is not an open child)
     provenance still recorded  : ordinal_chain
  ```

  The reviewer's end-to-end run agrees and quantifies it: carry-forward vs `--force` differ on **14 of 542 amendment pointers** (all 14 are pointers the carry-forward has and the clean rebuild does not; zero in the other direction), plus 5 `superseded_by`, 10 `lineage_family_id`, 8 `is_lineage_head`. Critically, the real predecessor's `superseded_by` — the exact column the commit says the two-child ambiguity was nulling — **stays null** through the guarded pass and through a guarded `cdt match`; only `--force` restores it.

  **Honest attribution:** the carry-forward itself is pre-existing #184 design, not introduced here — confirmed by re-running with the guard stubbed off, which gives the same 14. What is new is that **this PR's commit-5 fix is unreachable on already-published data**, and that the PR's "output is a pure function of the inputs" claim holds only for a `--force` derivation; on an existing artifact root, 14/542 amendment pointers are a function of run history.

  Two fixes, either sufficient. (a) State in the PR body and `docs/deployment.md` that landing commit 5 needs one `cdt match --force --infer-lineage` over the artifact root. `--force` is verified safe on this data — the `debt_instrument_id` set is byte-identical before and after (542 == 542, id sets equal); it is only destructive when mention IDs change, which is not the case. (b) Better: let `apply_lineage_inference_pass` re-open children whose `amendment_inferred_by` is non-null. I verified that provenance column **survives** a plain match (carried forward at `core.py:1232/1239`, and still reading `ordinal_chain` on the stale row above), so it already distinguishes an inferred pointer from an extracted one — re-deriving exactly those is safe and makes the pass self-healing.

- **[data-safety] `MATCHER_SCHEMA_VERSION` is write-only; nothing compares it** — `src/cdt/matcher/core.py:48-51`, `:379`, `src/cdt/pipeline.py:571` [proposed: Personal preference] [confidence: verified] Two write sites, one test pin, zero comparisons anywhere in `src/` or `tests/`. No caller reads `runs/match/run_id=latest.json` back, and `docs/schema.md` never documents `schema_version` at all. Confirmed live: a plain non-force match against a root whose recorded version was 4 silently wrote 6 — no refusal, no warning, no implicit force. The bump's real audience is the external publisher, so this is not the overclaim it first looks like; the useful point is narrower and follows from the finding above — **the manifest can say 6 while the rows still carry pre-guard lineage from a v4/v5 run.** If the self-healing fix is not taken, a version comparison in `match_pending_mentions` (refuse and point at `--force`, or implicitly force) is exactly where that protection would live.

### Verified clean

- **v5 data on disk survives read -> mutate -> rewrite.** The real `apply_lineage_inference_pass` against a 63-partition, 542-row v5 dataset: 63 partitions before and after, one distinct schema after (38 columns, nothing extra, nothing missing), 542 rows. The four columns are silently projected away by `pd.DataFrame(rows, columns=DEBT_INSTRUMENT_COLUMNS)`; no rows lost, no partition stranded. Re-sharding cannot strand a copy because padded and unpadded CIKs hash identically by design.
- **A mixed v5/v6 partition set does not raise, in either fragment order.** Built explicitly: first-partition-v6 and first-partition-v5 both read fine through `pyarrow.dataset`, `pd.read_parquet` and the repo's `read_dataset`. Column *removal* is not the same hazard class as the null/string *type* change the test at `:5823` guards. Nothing in the matcher reads instrument-level `status` any more, so a lingering column is inert.
- **A mixed state is barely reachable anyway.** `match_pending_mentions` reads the whole mentions dataset with no date filter and rewrites each `cik_shard` partition whole, so one run converts everything: 63/63 partitions at 38 columns after a single non-force `cdt match`. Only an aborted run can leave a mixed root, and the next run cleans it. There is no publish-only CLI subcommand, and both `write_final_output_tables` callers run match first.
- **Rollback works and costs one match run.** Old code (`ba3c589`, `PYTHONPATH` pointed at its own worktree) on new data restored all 42 columns and re-derived every status value with no error and no row loss.
- **Determinism - the PR's central claim holds mechanically.** No wall clock anywhere in the matcher (only `perf_counter` for a log line). Full match + lineage pass at `PYTHONHASHSEED` 0, 1 and 12345 produced an identical digest. Every set-to-scalar conversion is guarded by a `len == 1` check or a `sorted()`/`min()`. `shard_for_cik` uses `zlib.crc32`, not Python's salted `hash`. Idempotent: cycle 1 makes 2 links, cycles 2 and 3 make none, digest unchanged.
- **The v5->v6 rewrite is a net improvement for on-disk readability.** The untouched `data/lineage-probe` root today has 26 columns with mixed physical types and `pyarrow.dataset` over it raises `ArrowNotImplementedError` - pre-existing, from pre-#186 partitions. A single branch match heals it completely.

### Highlights

- "Silence is not disagreement" is backed by the numbers: 418 of 542 instruments carry a `role: "borrower"` cluster and 124 do not, so refusing on a missing party would have dropped links for 23% of rows. The published `parties_json` really does carry `role` and `canonical_name` on every one of its 1,418 clusters, so the guard has the evidence it claims to read.
- Rollback safety was preserved without anyone having to design for it, because `read_table`'s projection fallback and `read_dataset`'s per-file concat already tolerate column drift in both directions.


## Dropped during the false-positive filter

- **"The website/dashboard will break"** — excluded by the user's explicit instruction that the site is pre-beta.
- **`_borrowers` crashing on a bare JSON scalar** — downgraded from Important to Personal preference: the crash is real but I could not construct a live path, since every writer goes through `json.dumps(dedupe_party_clusters(...))` which always yields a list.
- **The dead xfail as a High** — downgraded to Important. It is a test-hygiene defect with no production consequence; HIPPO places test problems at Important.
- **Guard performance** — hypothesized O(N²) JSON re-parsing on large filers; measured 24 `_borrowers()` calls and 4 ms on the whole corpus. Not a finding.
- **Prefix matching as an active bug** — confirmed latent. Two independent runs (mine and the correctness reviewer's) agree that switching to set intersection changes no link on this corpus.
