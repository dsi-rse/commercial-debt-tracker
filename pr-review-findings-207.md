# PR Review Findings: Mint each amended instrument's prior state in the extractor (#207)

PR [#207](https://github.com/dsi-rse/commercial-debt-tracker/pull/207). Reviewed 2026-09-18 against `dev` at `974ab16`, head at `53f0993`.

Reviewable diff: 2,398 added / 338 deleted across 13 files from 13 commits. Two-dot and three-dot diffs agree, so nothing in the diff belongs to another PR. No generated or vendored files (`uv.lock` untouched).

Review factors, chosen by the author: correctness; determinism and idempotency (does it give the same answer every time, and is it safe to re-run?); tests; repo coherence; security. Backwards compatibility with existing on-disk data was ruled **out of scope** by the author — the project is pre-beta, the website is being revamped, and prior-schema data in S3 will be wiped. Severity below reflects that.

| file | +/− | area |
|---|---|---|
| `src/cdt/extractor/core.py` | +426 | the mint, the publish seam, the backfill, #206 |
| `src/cdt/matcher/core.py` | +260 | schema v7, synthesized-row identity rules, lineage pass #204 |
| `src/cdt/matcher/lineage_inference.py` | +153/−… | `prior_fact` retired, borrower guard #205 |
| `src/cdt/cli.py` | +71 | `backfill-mentions`, `--infer-lineage` removed |
| `src/cdt/pipeline.py` | +15 | pass wired into `run_match_and_finalize` |
| `src/cdt/storage.py` | +22 | decimal-to-text pre-pass |
| `docs/schema.md`, `docs/architecture.md` | +866 | synthesized rows, stage boundary, live publisher |
| 5 test files | +1,714 | mint, matcher, lineage, pipeline, CLI |

## Verification runs

All run on this branch at `53f0993`.

- `uv run pytest -q` → **529 passed** in ~6-10s. Re-run clean at the end of the review.
- `uv run ruff check .` → **All checks passed!**
- `uv run ruff format --check .` → **56 files already formatted**
- `gh pr checks 207` → all five green (CodeQL, Lint, Pulumi Preview, Security, Test)

### Live corpus exercised

The author approved copying a data root to a scratch directory and rewriting it. I used `data/lineage-verify` (542 instruments, 236 mention partitions, 20 MB). The originals were never written to — confirmed by `diff -r` after the dry run.

- `cdt backfill-mentions --artifact-root <scratch> --dry-run` → `{'partitions': 236, 'partitions_rewritten': 0, 'minted': 15, 'minted_no_origin': 2, 'skipped_no_origin': 5, 'skipped_ambiguous_prior': 1}`. Matches the PR's stated yield. Same counters on `data/genwindow-run-branch`, the root the PR cites.
- `diff -r data/lineage-verify/mentions <scratch>/mentions` after the dry run → identical. The dry run writes nothing.
- `cdt backfill-mentions` (real) → 236 partitions rewritten, same mint counters.
- `cdt match --force` → 557 instruments, 536 heads, 477 families, 22 amendment pointers, 2 `ordinal_chain`, 14 `synthesized_only`, **0 synthesized rows that are lineage heads**. Every number in the PR's comparison table reproduced exactly.
- `cdt match` (plain) after the forced match → `debt-instruments` **byte-identical** across all 557 rows and all columns.
- Longest amendment chain walk → **EQT's revolver walks six states**, ending in the $1.5B synthesized prior state. The PR's showcase claim reproduces.
- Plain `cdt match` on the untouched stale root (no backfill) → 15 stale inferred pointers from two *retired* rules (`dated_reference` ×13, `prior_fact` ×1, `ordinal_chain` ×1) were re-opened and re-derived to the correct 2 `ordinal_chain` links, with no `--force`. **The #204 self-healing fix works as advertised.**
- The PR's own documented deploy sequence (backfill, then one plain `cdt match`) → **19 pointers, 539 heads**, against the measured 22 and 536. Three further plain matches did not recover. See finding H2.
- One minted row inspected end to end: `prior` flipped to `false`, successor's spans retained, borrower-only parties, `lender_disclosure = none_named`, `raw_id` suffixed `-prior`, successor's `amendment_of` pointing back at it. Correct.
- 5 of 15 mints carry a `derived_from: "inherited"` term, so the inheritance path genuinely fires.

Environment note: everything was verified by execution. Nothing in this review is "read-only speculation" except where a finding explicitly says so.

---

## Factor: correctness (reviewer scope: full diff)

### Findings

- **[correctness] `cdt pipeline` and the live orchestrator backend still publish with no lineage pass** — `src/cdt/pipeline.py`:312-357 [proposed: **High**] [confidence: verified] — `run_match_and_finalize` (line 429) gained `apply_lineage_inference_pass`, but `PipelineOrchestrator.run` did not: it calls `match_pending_mentions` at 312 and `write_final_output_tables` at 357 with nothing between. `grep apply_lineage_inference_pass src/` returns only `pipeline.py:429` and `cli.py:926`. Affected entry points: `cdt pipeline` (via `cli.py:591` → `run_pipeline`) and `cdt-orchestrator daily|historical --extractor-backend live` (via `orchestrator.py:435`). The default backend is `batch`, which routes through `run_batch_backend` → `run_match_and_finalize` and *is* covered, so the current scheduled production path is fine. Contradicts `docs/schema.md`:135 ("runs after every match — in `cdt match` and in the pipeline's match-and-finalize step alike") and the new comment at `cli.py`:920-922 ("Always, as the pipeline does"). Fix: call the pass between the two stages in `run()`, guarded on `not matched["debt_instrument"].empty` as `run_match_and_finalize` does.

- **[correctness] Following the PR's own deploy instructions loses three amendment pointers, permanently** — `src/cdt/matcher/core.py`:1212-1213, 1253-1254 [proposed: **High**] [confidence: verified end to end on the reference corpus] — `derive_parent_links` seeds `amendment_parents` with the existing row's pointer (1212-1213) regardless of whether the lineage pass inferred it, then clears the set when it holds two (1253-1254). So a stale inferred pointer and the new extracted #203 pointer annihilate each other, and the pass re-infers its guess on the next run. Reproduced exactly: on `data/lineage-verify`, `cdt backfill-mentions` then one plain `cdt match` (the documented sequence) yields **19 pointers / 539 heads** against the clean rebuild's **22 / 536**; three further plain matches leave it at 19. `dim::3fe2333…` (EQT's Second A&R) carried a stale `prior_fact` pointer and **loses its extracted pointer to the minted prior state** — the PR's headline feature. `dim::491cf40…` is knock-on damage: with the first row unlinked, `ordinal_chain` sees a same-rank tie and refuses. Only `cdt match --force` produces the measured result, but the deploy notes say `--force` is needed only for a non-zero #206 count. Fix: don't carry an inferred pointer into the collision — skip the seed when `amendment_inferred_by` is set, or discard it from the set before the `len > 1` clear.

- **[correctness] A `prior` term whose value did not parse makes the mint inherit the post-amendment value, uncounted** — `src/cdt/extractor/core.py`:1767-1781, 1847-1848 [proposed: **Important**] [confidence: verified, zero incidence today] — `prior_amounts`/`prior_dates` filter on `normalized_amount is not None`/`normalized_date is not None` (1767-1780), and `prior_amount_kinds`/`prior_date_kinds` (1847-1848) are derived from those *filtered* lists. A `prior: true` term the parser could not resolve is therefore invisible to the "did this kind change?" test, so the current value is copied onto the predecessor marked `derived_from: "inherited"` — asserting the amended figure as the prior state's own term, which is the one thing the rule is supposed never to do. Demonstrated: a row with a parsed prior `agreement` plus an unparsed prior `commitment` and an unparsed prior `maturity` mints `principal_amount: 250000000` and `maturity_date: 2029-05-01`, both the successor's post-amendment values, both marked `inherited`, with counters `{'minted': 1}` and no refusal recorded. **Two symptoms, one root cause.** The second symptom: when the unparsed prior is the object's *only* prior term, the object is dropped at 1781 with no counter at all. On `data/genwindow-run-branch` that silently swallows one object (`dim::5542bb4c…`, a `Loan and Security Agreement` whose prior commitment has `normalized_amount: null`) — which is why the PR's "22 objects carry a prior term" doesn't match its own counters, which sum to 21. The docstring promises "every refusal increments a named counter so the rate is measurable before anyone loosens the rule", and the deploy notes call the counters "the pre-registered yield". Measured incidence of the harmful variant (a parsed *and* an unparsed prior on one object) on the reference corpus: **0**. Fix: derive the kind sets before the value filter, and add a counter for the unparseable-prior drop.

- **[correctness] `apply_lineage_inference_pass` infers against a `first_seen_filing_date` it then overwrites** — `src/cdt/matcher/core.py`:2304-2316 [proposed: **Important**] [confidence: verified behaviour, likely reachability] — `infer_amendment_parents` reads `first_seen_filing_date` as both the predecessor-ordering guard and the chain sort key (`lineage_inference.py`:259-261, 279); `apply_lifecycle_rollup` at 2314 then recomputes and rewrites that column from the member edges. When the recomputed value differs from what was on disk, pass N+1 infers against a different corpus than pass N. Demonstrated on an instrument row with no live member edges: pass 1 yields 1 link and nulls `first_seen_filing_date`; pass 2 yields 2 links, adding a pointer pass 1 refused. That row shape arises naturally — mention ids are content hashes, so re-extracting an item mints a new id, the old member edge is never deleted, and the old instrument survives with `mention_count=0`. Converges after two passes, unlike H2. Fix: run the rollup's observation-column recomputation before the inference as well as after.

- **[correctness] `mint_prior_state_rows` mutates the caller's row dicts** — `src/cdt/extractor/core.py`:1950 [proposed: **Personal preference**] [confidence: verified] — writes `amendment_of` into the dicts it was handed, while the docstring calls it "a pure function of one item's rows". Both current callers are safe (`published_mention_rows` copies at 1974; `backfill_mentions` owns its `records`), and no nested payload is aliased. The hazard is a future caller passing `row_state.debt_instrument_mentions` directly, which would persist a pointer into `state.jsonl`. Fix: copy at the top of the function, or soften the docstring.

- **[correctness] On the `minted_shared` path, two unhashed fields are decided by input order** — `src/cdt/extractor/core.py`:1910, 1943, 1951-1954 [proposed: **Opinion**] [confidence: verified order-sensitivity, unreachable as real nondeterminism] — when two successors mint a byte-identical predecessor, the first seen supplies `raw_id` and `synthesized_from_mention_id`. No observable nondeterminism could be constructed (write-time and backfill row order agree), but `synthesized_from_mention_id` is actually read later — `borrowed_lender_signature` reads it — so if the two successors named different lenders, which one wins could change cluster scoring.

- **[correctness] `skipped_sibling_is_predecessor` publishes no lineage at all** — `src/cdt/extractor/core.py`:1837-1841 [proposed: **Opinion**] [confidence: verified] — leaves `amendment_of` null rather than pointing at the sibling the comment identifies as P. Defensible, but the comment reads as though a link results.

- **[correctness] `minted_no_origin` and `minted` both fire for the same row** — `src/cdt/extractor/core.py` [proposed: **Personal preference**] [confidence: verified] — the counters don't partition, so they can't be summed.

- **[correctness] The mutual-pair drop in `infer_amendment_parents` may now be dead code** — `src/cdt/matcher/lineage_inference.py` [proposed: **Opinion**] [confidence: uncertain] — with only `ordinal_chain` left, a mutual pair may be unreachable. It is the only lineage guard with no test. Either add a direct unit test or delete the block.

### Verified clean

- **Input mutation**: all callers of `mint_prior_state_rows` pass copies; `to_state_dict` still serializes the raw model rows; no third caller exists.
- **Write-time vs backfill byte identity**: verified two ways — a real parquet round-trip across five row shapes (plain, cents, null rate, null `name_json`, null `cik`) and a hand-built read-back shape with `nan`/`Decimal`/`numpy.str_` substituted. Identical ids, no field diffs.
- **#206 string comparison** cannot miss on formatting: both sides route through `canonical_numeric_text`, the single canonical spelling. No `1e8`/`.0` variants possible.
- **`backfill_mentions` partition handling**: `PARTITION_PATTERN` pins one file per partition, so the rewrite overwrites rather than duplicating; an item can never span partitions; the successor is always in its mint's partition.
- **The `replaced` step-aside** only removes same-rank rows another same-rank row already supersedes; a mutual pair empties `parents`, which is a refusal, never a wrong link.
- **`mention_sort_key`** 5-tuple: all four consumers treat it as opaque; nothing unpacks it. Synthesized-first does achieve the documented effect.
- **`canonical_member_ids` vs `present_member_ids`**: same set, different order; `all()` is order-independent. The `or ordered_member_ids` fallback is correct.
- **`coerce_optional_bool`** over `None/bool/np.bool_/pd.NA/nan/NaT/"true"/"0"/""/"nan"/0/1` — every case correct, including numpy 2's non-`bool`-subclass `np.bool_`.
- **`storage.apply_declared_column_types`**: `coerce_dataset_text(Decimal)` is lossless (`.normalize()` + `:f`, no scientific notation); `None`/`NaN`/`pd.NA` land as nulls. The mixed Decimal/str column it fixes is real, and `decimal_column_values` raises rather than silently nulling bad text.
- **`borrowed_lender_signature`**: `mention_index` is shard-wide and a mint always shares its successor's item, so the successor is present whenever the mint is.
- **`GENERIC_BORROWER_PHRASES`**: all 18 phrases survive `_borrower_key` unchanged; `"the Borrowers"` → `borrowers`, `"Co-Borrower"` → `borrower`. No dead entries.
- **`_borrowers_disagree`** exact-intersection change, `party_dedupe_key`, `name_class_sizes` (`by_cik.get` correctly defends the new skip), `outstanding_balance_as_of_is_filing_date`, the `.fillna("")` shard fix, and the re-open loop — all behave as documented.

## Factor: determinism / idempotency (reviewer scope: full diff)

Six claims were tested. **Four hold, one holds in the narrow form the PR states, one is falsified.**

- **Claim 1 — the mint is a pure function of one item's rows: HOLDS**, with the two caveats above. Verified under `PYTHONHASHSEED` ∈ {0, 1, 7, 12345, 99999} with byte-identical output; every set in the function is membership/`len` only.
- **Claim 2 — a prior state built as data is first written matches one rebuilt later by the backfill, identical down to the hashed id: HOLDS.** Strongest form also passes: backfilling a partition the writer already minted is a byte no-op.
- **Claim 3 — running `backfill-mentions` a second time changes nothing: HOLDS.** SHA-256 of every parquet file identical across runs 1→2→3, counters identical, on-disk row order stable, including over a pre-#203 partition lacking the new columns. Note it still *rewrites* every partition (236 of 236 on the real corpus) — the bytes are identical but it is not a read-only no-op.
- **Claim 4 — every model-emitted mention id is byte-identical: HOLDS.** Recomputed on every real row before and after. The hash payload contains none of `amendment_of`, `synthesized_by`, `synthesized_from_mention_id`.
- **Claim 5 — the lineage pass is a pure function of its inputs: FALSIFIED** by two independent mechanisms (the two Important/High findings above). The narrower sub-claim the PR actually states — "a plain match-and-finalize after a forced match is byte-identical" — **does hold**, and I confirmed it on the real corpus. Rollups are genuinely recomputed, not carried, and the re-open loop cannot oscillate on the pointer axis.
- **Claim 6 — the pass's shard assignment matches `match_pending_mentions`: HOLDS.** Character-identical expressions. `fillna("")` covers `None`/`nan`/`numpy.nan`; the literal spellings `"None"`/`"nan"`/`"   "` are not covered but cannot diverge, because such a mention is dropped before an instrument row exists. Zero-padding is not a divergence either.

## Factor: tests (reviewer scope: full diff)

Judged by mutation testing: deliberately break one piece of production code, then check that some test goes red. **54 breakages were applied to `src/cdt/`, each followed by the full suite. 44 were caught, 10 were not.** All were reverted; `git status` is clean and 529 tests pass at the end. I re-ran the highest-value one myself to confirm.

### Findings

- **[tests] None of the four `published_mention_rows` publish paths is pinned** — `tests/test_file_native_stages.py`:1002-1024; call sites `src/cdt/extractor/core.py`:687, 2229, 2397, 2580 [proposed: **Important**] [confidence: verified independently] — `test_published_mention_rows_is_the_single_publish_seam` calls the helper directly on a `row_state` whose mention has no `prior` facts, so the mint is a no-op and the seam is indistinguishable from the raw list. Replacing each call site with `row_state.debt_instrument_mentions` leaves the suite green — all four. I re-ran two of them myself (2229 and 2580) and got **529 passed**. No test anywhere asserts a synthesized row reaching a written partition or a `full.jsonl` record through a real extractor entry point; the only end-to-end minting coverage is `backfill_mentions`. The seam is the linchpin of the feature and every one of its uses is deletable with a green suite. Fix: give the existing `test_extract_pending_items_writes_mentions_and_audit` fixture a `prior`-marked commitment and assert a `synthesized_by == "prior_state"` row in the written partition and in the audit record.

- **[tests] `cdt match` running the lineage pass is untested** — `src/cdt/cli.py`:926 [proposed: **Important**] [confidence: verified] — replacing the call with canned zero stats leaves the suite green. `tests/test_pipeline.py`:340 covers the `run_match_and_finalize` half well; the CLI half, named in commit `26db036`'s subject, has no test. Nothing pins that `--infer-lineage` is gone from the parser either.

- **[tests] `backfill-mentions` lease acquisition is unpinned** — `tests/test_cli.py`:755-786; `src/cdt/cli.py`:881 [proposed: **Important**] [confidence: verified] — deleting the whole lease block leaves the suite green. The test covers the `--dry-run` skip well but the real leg only asserts exit 0. The repo already has the convention at `tests/test_cli.py`:462.

- **[tests] `borrowed_lender_signature`'s wiring into `match_tables` is not pinned** — `src/cdt/matcher/core.py`:497 [proposed: **Important**] [confidence: verified] — deleting the `lender_signature=borrowed_lender_signature(...)` argument leaves the suite green. Both halves are tested in isolation; their connection is not. The one end-to-end test uses `parties_json="[]"`, so the lender path never runs there.

- **[tests] Half of `PRIOR_TERM_DATE_KINDS` / `INHERITED_DATE_KINDS` and the `expected` guard are unpinned** — `src/cdt/extractor/core.py`:1668, 1673 [proposed: **Important**] [confidence: verified] — three mutations uncaught: dropping `commitment_termination` from either frozenset, and deleting the `and not entry.get("expected")` guard. Every mint test uses `maturity`/`agreement` only and no fixture carries `expected: True`.

- **[tests] The "works on copies" test passes for the wrong reason** — `tests/test_file_native_stages.py`:1017-1022 [proposed: **Important**] [confidence: verified] — `mint_prior_state_rows` already returns a fresh list, so the assertion holds even without the dict-copy. Removing the copy from `published_mention_rows` leaves the suite green, though it makes `row["amendment_of"] = minted_id` write onto the dicts persisted to `state.jsonl`.

- **[tests] The new as-of-flag test never covers its own headline case** — `tests/test_file_native_stages.py`:6903-6982 [proposed: **Personal preference**] [confidence: verified] — hardcoding the flag to `False` leaves the new test green in isolation; only a pre-existing test catches it. The new test asserts `False` for a stated as-of and `True` for a carried-forward value, never `True` for a substituted one.

- **[tests] `synthesized_only` carry-forward and `coerce_optional_bool`'s text path are unpinned** — [proposed: **Personal preference**] [confidence: verified] — both mutations uncaught. Incremental-rematch-only paths.

- **[tests] `name_class_sizes`' all-synthesized-CIK path is unpinned and guards a reachable crash** — `src/cdt/matcher/core.py`:2221 [proposed: **Personal preference**] [confidence: verified] — reverting `by_cik.get(cik, [])` to `by_cik[cik]` leaves the suite green, but the new synthesized skip means a CIK whose mentions are all synthesized raises `KeyError` inside `match_pending_mentions`. The fix is right; only the test is missing.

- **[tests] Comment claims three cases, two are asserted** — `tests/test_matcher.py`:180-184 [proposed: **Personal preference**] [confidence: verified] — removing `or mention.lender_signature` from the guard is uncaught. The missing case is the one that could over-borrow.

- **[tests] `assert renewals` in the pipeline test is vacuous** — `tests/test_pipeline.py`:382 [proposed: **Opinion**] [confidence: verified] — `renew` is called by `match_pending_mentions` and again after the pass block, so the list is non-empty either way. Deleting the `renew()` before the pass is uncaught.

- **[tests] The "skip already-synthesized rows" filter is unpinned** — `src/cdt/extractor/core.py`:1761 [proposed: **Opinion**] [confidence: verified] — `test_mint_is_id_stable_and_idempotent` holds for a second reason (a mint's `prior` flags are already flipped), so replacing the filter with `list(rows)` is uncaught.

### Verified clean

- **All seven refusal counters are individually pinned** with exact-dict assertions, not "count > 0", and the row count is asserted unchanged on each refusal.
- **Backfill idempotency is genuinely pinned** with `assert_frame_equal` across two runs; stopping the `amendment_of` clearing turns it red.
- **Write-time vs backfill id identity is tested** at `tests/test_file_native_stages.py`:1466-1475, and the decimal canonicalization is load-bearing for it.
- **Assertion strength is high overall**: no assert-free tests, no snapshot-everything, no sleeps, no order dependence, no shared mutable fixtures. Canonical-field tests assert the specific winning value *and* its `*_source_mention_id`.
- **Fixtures are hardened**: `mention_row` seeds from `DEBT_INSTRUMENT_MENTION_COLUMNS` and asserts no override names a non-published column, rejecting a typo'd kwarg that would silently no-op.
- **The three claimed #197 test fixes in `4d75acb` are all real and complete** — verified by counterfactual for each. The type-drift test now catches an undeclared bool the old all-`None` fixture passed; the xfail's target function no longer exists anywhere in `src/`; the suffix test's fixture moved to `EQT Corporation` vs `EQT Company` and now fails when `BORROWER_SUFFIXES` is disabled.
- **No test was weakened.** All 203 removed lines map to production code this PR deliberately removed: the four `prior_fact` tests, the `--infer-lineage` flag test (replaced by a strictly better assertion), the dead xfail, and schema-version/fixture updates that track real changes.
- **Conventions followed**: all filesystem tests take `tmp_path` and rely on the autouse `_isolated_data_dir`; parametrize style matches the repo.

### Highlights

`test_a_prior_state_is_placed_before_the_object_it_was_minted_from` (`tests/test_matcher.py`:261) is the standout: it reconstructs the real EQT November-2017/April-2021 chain, runs it through `match_tables` end to end, and asserts cluster membership, both pointers, the head count, and `synthesized_only` on both rows. It independently catches three separate mutations. The #204 and #205 tests are likewise built from named real-corpus failures with measured counts in their docstrings.

## Factor: repo coherence (reviewer scope: full diff)

### Findings

- **[repo-coherence] `infer_amendment_parents` keeps two required params it immediately `del`s — the shape a test forbids** — `src/cdt/matcher/lineage_inference.py`:208-223 [proposed: **Important**] [confidence: verified] — `tests/test_matcher_lineage_inference.py`:104-110 asserts by `inspect.signature` that `item_texts` is absent from this function and that `item_texts`/`infer_lineage` are absent from `match_tables`/`match_pending_mentions`. That test exists because the #177 review flagged "accepts a parameter and never reads it" as `Important [repo-coherence]` **in this same module**. The repo turned it into an executable convention, and this PR reintroduces the pattern one function below it. Also, the docstring at :214 still says "the **two rules** support" when one rule remains. Fix: drop both parameters and the `del`; stop forwarding them at `matcher/core.py`:2304-2308. `member_groups` and `mention_index` are still needed by `apply_lifecycle_rollup`, so nothing else moves.

- **[repo-coherence] The lineage pass rewrites every `debt-instruments` shard on every run and appears in no manifest** — `src/cdt/pipeline.py`:420-431, `src/cdt/cli.py`:919-929, `src/cdt/matcher/core.py`:2319-2334 [proposed: **Personal preference**, arguably Important] [confidence: verified] — every writing stage in this repo writes a run manifest (`ingest.py`:457, `itemizer/core.py`:274, `classifier/core.py`:303, `matcher/core.py`:386, `extractor/core.py`:2311 and 2485, `sixk/stage.py`:604), and `docs/architecture.md`:38 names stage manifests as a design property. The match manifest is written with `partitions_written`, and *then* the pass rewrites every one of those partitions and writes nothing. Pre-existing, but it was behind `--infer-lineage`; this PR makes it the unconditional default, so the manifest now systematically describes a dataset something else changed afterwards. The pass already returns the counters needed.

- **[repo-coherence] `_json_list`/`_json_dict` duplicate `parse_cluster_list`** — `src/cdt/extractor/core.py`:1676-1700 [proposed: **Personal preference**] [confidence: verified] — same try/`JSONDecodeError`/`isinstance(list)`/filter-to-dicts body as `matcher/core.py`:1707-1715, plus a `coerce_dataset_text` front end. They do genuinely consolidate three ad-hoc inline parses in the extractor, but onto a private copy of a public matcher helper. `shared.py` is the wrong home (it is an `idi_ftm2j_shared` compatibility shim); `storage.py` is right — both modules already import it and `coerce_dataset_text` lives there. Note the direction constraint: `matcher/core.py`:28 imports `cdt.extractor.core`, so the extractor cannot import the matcher.

- **[repo-coherence] Two role-word vocabularies in two modules, with overlapping entries and two normalisations** — `src/cdt/matcher/lineage_inference.py`:106-127 vs `src/cdt/matcher/core.py`:55-71 [proposed: **Personal preference**] [confidence: verified] — `GENERIC_BORROWER_PHRASES` and `GENERIC_LENDER_TERMS` both encode "this string names a role, not a company"; `buyer`/`buyers` appear in both, under different normalisations, so a future addition must be made twice. Separately, `_borrowers` compares the literal `"borrower"` while `extractor/core.py`:84 defines `BORROWER_PARTY_ROLE` and this PR's own mint code uses it — though `lineage_inference.py` imports nothing from `cdt`, which is why. **The PR's explicit deferral of unifying `_borrower_key` with `normalize_party_text` is reasonable and the PR did not make it worse**: `party_dedupe_key` composes the two existing helpers rather than adding a fourth normaliser, and its docstring records the measurement that justifies it.

- **[repo-coherence] ~370 lines of a new minting rule land in a 5,262-line `core.py`** — `src/cdt/extractor/core.py`:1662-2032 [proposed: **Personal preference**] [confidence: verified] — the next largest module is `matcher/core.py` at 2,349. The closest in-repo precedent is exact: the matcher's inference rules live in `matcher/lineage_inference.py` (372 lines), not in `matcher/core.py`. An `extractor/prior_state.py` would sit in the same relation. For `backfill_mentions` specifically, the admin-command precedent is `describe_active_job`/`reset_active_job` in `extractor/batch.py`.

- **[repo-coherence] `backfill_mentions` rewrites the canonical mentions dataset with no manifest** — `src/cdt/extractor/core.py`:1978-2032 [proposed: **Personal preference**] [confidence: verified] — the counters only reach stdout; nothing on disk records that `mentions` was rewritten.

- **[repo-coherence] The "Synthesized rows" doc omits `status` and overstates `interest_rate_*`** — `docs/schema.md`:403-445 [proposed: **Personal preference**] [confidence: verified, and confirmed on real data] — the section says "No event facts", but the code sets `status_payload = derived_status_payload(minted_dates)` at `extractor/core.py`:1887, and a dated `agreement` yields `status = "entered_into"`. I confirmed this on a real mint: `status: entered_into`. A reader of that bullet would expect null. Separately, `interest_rate_*` is described as "the successor's" at :415, but :1894-1895 rewrites its `derived_from` to `"inherited"`. Given the PR's own note that a prior review found two `schema.md` sentences false, these are worth one line each.

- **[repo-coherence] `README.md` still names the archived dashboard repo as the live consumer** — `README.md`:87, 116 [proposed: **Personal preference**] [confidence: verified] — this PR's `docs/architecture.md`:241 now asserts that repo is archived, but the README tells a new contributor to `cd` into it and run a command. Same in `docs/deployment.md`:157, 193 and `docs/deployment-dev.md`:149. The PR hedges deliberately ("older documents that name it mean this one"), which is fine for design docs; the README is the front door.

- **[repo-coherence] `published_mention_rows`' `counters` parameter is passed by nobody** — `src/cdt/extractor/core.py`:1960-1975 [proposed: **Opinion**] [confidence: verified] — all four production call sites pass one argument; tests thread counters into `mint_prior_state_rows` directly. Same class as the `del` finding. **The seam itself is genuine and should stay** — it owns the copy that keeps the mutated pointer off `state.jsonl`.

- **[repo-coherence] `cdt backfill-mentions` is absent from the README's command notes** — `README.md`:81-82 [proposed: **Opinion**] [confidence: verified] — its two sibling admin subcommands are both listed there.

- **[repo-coherence] `SYNTHESIZED_PRIOR_STATE` breaks the `<COLUMN>_<VALUE>` convention, and "synthesized" now means two things** — `src/cdt/extractor/core.py`:1664 [proposed: **Opinion**] [confidence: verified] — every other enumerated-value constant is `<column>_<value>` (`DERIVED_FROM_STATED`, `LENDER_DISCLOSURE_NONE_NAMED`); `SYNTHESIZED_BY_PRIOR_STATE` would match. The word now carries both the pre-existing sense (a value derived from a name) and the new one (a whole minted row), within one file.

- **[repo-coherence] `coerce_optional_bool` is placed right but not shaped like its family** — `src/cdt/matcher/core.py`:1852-1871 [proposed: **Opinion**] [confidence: verified] — both siblings funnel through `coerce_dataset_text`, which already handles `None`, `Decimal`, `pd.isna` and the `MISSING_TEXT_VALUES` placeholders; this re-implements the `pd.isna` try/except instead.

### Verified clean

- **No clock anywhere in the processor.** Grepped every added line for `datetime`, `.now(`, `today()`, `time()`: zero hits.
- **The matcher creates no objects and asserts no facts.** Everything new reads `synthesized_by` and reacts to it — all views.
- **`outstanding_balance_as_of_is_filing_date` is a view, not a fact**, and the docs claim is honest: it records whether the *extractor's* payload dated the balance. It survives an incremental rematch via the `coerce_optional_bool(existing_row...)` fallback, as `docs/architecture.md`:228 requires. Same for `synthesized_only`.
- **Both new published columns and both new mention columns are documented**, and both bools are in `DECLARED_COLUMN_TYPES` and the `schema.md` type table.
- **`derived_from: "inherited"` is documented in both required places**, and the same edit back-filled `"computed"`, which existed in code but was undocumented.
- **Removal hygiene is clean.** `prior_fact`, `--infer-lineage`, `_prior_amounts` and `PreparedMention.status_date` have no live references anywhere in `src/`, `docs/`, `scripts/`, `notebooks/`, `Makefile`, `README.md` or `pulumi/` — only deliberate historical mentions.
- **CLI shape is coherent**: `run_backfill_mentions` follows `run_matcher` and `run_reset_extract_job` exactly, and `--dry-run` skipping the lease matches `run_show_extract_job`, the repo's other read-only command.

### Highlights

The docstrings match the repo's discursive house style precisely — they cite issues, explain the failure the rule exists for, and record measurements. `party_dedupe_key`'s in particular names the two concrete regressions that justify its choice. `docs/architecture.md`:230 does not just delete `prior_fact` from the "right side of the line" example; it keeps it as the *counter*-example and uses it to sharpen the rule, which is the correct way to edit a principles doc.

## Factor: security (reviewer scope: `src/`)

Low-surface change, and it holds up. No injection, no auth, no new network I/O, no secrets, no path traversal.

### Findings

- **[security] `cdt backfill-mentions` rewrites every partition on a lease it never renews** — `src/cdt/cli.py`:874-894, `src/cdt/extractor/core.py`:1978-2032 [proposed: **Important**] [confidence: verified mechanism, unquantified trigger] — `run_backfill_mentions` takes `PIPELINE_WRITER_LEASE` once and never renews; `DEFAULT_LEASE_TTL_SECONDS` is 2h (`lease.py`:44). Once the TTL elapses the next orchestrator tick legitimately steals the lease (`lease.py`:129-133) and starts extract/match while the backfill is still overwriting `part-0000.parquet` per partition. Both writers full-overwrite the same object, so one writer's mention rows are silently lost, and `release_lease` no-ops after a steal. This is the #89 class, and the tell is the asymmetry: **this PR itself adds `renew` to `apply_lineage_inference_pass` for exactly this reason**, then omits it on the new command. **Scale caveat I measured:** 236 partitions took ~10 seconds locally, so 2h is far away at reference-corpus size; the risk is real only if the production corpus is orders of magnitude larger or S3 round trips dominate. Fix: thread `renew` into `backfill_mentions` and call it per partition.

- **[security] `cdt match` now runs an unconditional whole-corpus rewrite without renewing** — `src/cdt/cli.py`:898-931 [proposed: **Personal preference**] [confidence: verified] — same class. `pipeline.run_match_and_finalize` correctly passes `renew=renew`; `run_matcher` passes nothing, though `lease` is in scope. The un-renewed hold predates this PR (`match_pending_mentions` is also called without `renew` there), but the PR unconditionally extends the window and adds the very parameter that fixes it without wiring it up on this path.

- **[security] An unhashable value in a `dates_json` entry aborts the whole mint pass** — `src/cdt/extractor/core.py`:1795-1799 [proposed: **Personal preference**] [confidence: verified crash, **no attack path found**] — `{"kind":"amendment","normalized_date":["2020-01-01"]}` raises `TypeError: cannot use 'list' as a set element`, killing the entire extract or backfill run rather than one item. This is the identical failure class the PR just paid to fix in `_borrowers`. **No reachable attack path**: `standardized_date_payload` overwrites `normalized_date` with this repo's own parser output (always `str | None`), so no LLM output can put a non-string there. Reaching it needs a tampered parquet, at which point the attacker already has write access. Reported as hardening. Fix: one `isinstance(..., str)` clause.

### Verified clean

- **17 adversarial payloads** through `mint_prior_state_rows`: JSON columns as objects, bare ints, bare strings, `null`, non-JSON junk, lists of scalars, `normalized_amount` as a dict or list, `None`/`nan` everywhere, a 5 MB span string. All returned normally. `_json_list`'s and `_json_dict`'s isinstance gates are the right shape.
- **The `_borrowers` fix is complete**: `{"a":1}` and `7` both now return `set()` where they previously raised.
- **Path traversal is structurally impossible**: `PARTITION_PATTERN` constrains `date` to `\d{4}-\d{2}-\d{2}` and `shard` to `\d{4}`, so only digits reach `f"{key}={value}"`. `shard_for_cik` is a crc32 modulo. `item_id` never touches a path.
- **Dry run writes nothing — verified by hashing** `(mtime, size, path)` for 245 partitions before and after: byte-identical. So skipping the lease there is safe.
- **Lease release on every exit path**: `finally: release_lease`, holder-checked.
- **Mid-backfill failure is not a corruption risk**: each partition is one atomic write (`NamedTemporaryFile` + `Path.replace` locally, single `put_object` on S3), and a half-finished backfill cannot strand pointers because `match_pending_mentions` re-reads the whole dataset each run.
- **The sibling scan is not a DoS**: `O(n²)` where n is the mentions of *one filing item*, not the corpus. Timed at n=800 → 0.125 s; realistic n is single digits.
- **`backfill_mentions` does not load the whole dataset** — strictly one partition at a time, which is the right shape.
- **Logging and `print` are clean**: only hard-coded counter names and ints; no data-controlled string reaches the terminal.

### Highlights

The hash-payload exclusion of `synthesized_by`/`synthesized_from_mention_id` is deliberate and verified — the backfill adds columns without re-keying a single existing row. `by_cik.get(mention.cik, [])` is a necessary `KeyError` fix, not cosmetic. `renewer()` raising `LeaseLostError` rather than returning a bool is the right call.

---

## Dropped during the false-positive filter

- **"Missing `renew` on `backfill_mentions` risks a lost write in practice"** — downgraded from the reviewer's framing. I measured the TTL (2h) against the actual runtime (~10s for 236 partitions). The wiring gap is real and worth fixing; the imminent-data-loss framing is not supported at reference-corpus scale.
- **"The four publish paths gap is High"** — downgraded to Important. Per the HIPPO definitions, missing tests for new behavior is Important; it is at the top of that tier, but no defect ships as a result.
- **A claimed working-tree edit to `extractor/core.py`** reported mid-review by one reviewer was another reviewer's mutation test. Confirmed reverted; `git status` shows no modified tracked files and the suite passes 529.
- One reviewer reported suite totals of 519 rather than 529 (it worked in a separate worktree). Its targeted experiments were independently evidenced and are retained; its suite counts are not relied on.
