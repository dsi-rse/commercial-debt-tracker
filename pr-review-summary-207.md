# PR Review: Mint each amended instrument's prior state in the extractor; run lineage inference in the pipeline (#203, #204, #205, #206)

## Verdict: REQUEST CHANGES

Two High findings, both verified by running the code on the real corpus. Everything else is either fixable in a follow-up or optional.

## Overview

This pull request fixes a real and well-diagnosed problem. An amendment 8-K comes out of the model as one object: the terms as amended, plus the old terms marked `prior`. That leaves nothing for `amendment_of` to point at, so the pipeline published almost no amendment history — 537 of 542 instruments were lineage heads. Rather than ask the model to name the predecessor (which picked the wrong instrument in three of five spot checks), the extractor now *builds* the predecessor from the object's own cited `prior` facts. Nothing chooses a predecessor, so that whole class of error cannot happen. The design reasoning is sound, the docstrings are the best I have read in this repo, and the docs were updated in the same commits rather than afterwards.

I reproduced the author's measurements exactly, by running the code against a scratch copy of the real corpus rather than reading it. On `data/lineage-verify`, the dry run reports 15 minted, 5 skipped for no origin, 1 ambiguous. After the backfill and a forced match: 557 instrument rows, 536 lineage heads, 477 families, 22 amendment pointers, 2 `ordinal_chain` links, and no synthesized row published as a head. EQT's revolver really does walk six states, ending in the $1.5 billion prior state the extractor minted. A plain match after a forced one produces identical files, down to the byte, across all 557 rows. The #204 self-healing fix is the standout result: I pointed a plain `cdt match` at a stale data root carrying 15 pointers from two rules that no longer exist, and it re-opened all 15 and settled on the two correct links with no `--force`. The mint's purity claims hold too. A prior state built as the data is first written, and the same prior state rebuilt later by the backfill reading from disk, come out identical — same fields, same id — and running the backfill a second time changes nothing at all.

The two blocking problems are both in the wiring rather than the logic. First, `cdt pipeline` and the live orchestrator backend never got the lineage pass; only `run_match_and_finalize` and `cdt match` did. Those paths still publish the un-inferred lineage this PR exists to fix, which contradicts both `docs/schema.md` and the PR's own code comment. The default production backend is `batch`, which is covered, so the scheduled run is not currently broken.

Second, and more serious: **following the PR's own deploy instructions produces a worse result than the one it measured.** The documented sequence is "backfill, then one plain `cdt match`". Run that on the reference corpus and you get 19 amendment pointers and 539 heads instead of 22 and 536. Three more plain matches never recover it. One of the lost links is EQT's Second Amended and Restated agreement pointing at its minted prior state — the showcase result. The cause is that a leftover guessed pointer and the new extracted pointer land in the same set in `derive_parent_links`, which throws the set away when it holds two, so both are lost. The pass then re-applies its guess on the next run, and the extracted link never comes back. `cdt match --force` gives the right answer, but the deploy notes say `--force` is only needed for an unrelated reason. In practice, an operator who follows the instructions would publish a history that is missing links, see a plausible-looking log line, and have no signal that anything went wrong.

The test work is genuinely strong. To judge it I used mutation testing — deliberately breaking a piece of production code and checking that some test goes red. Of 54 such breakages, 44 were caught. The refusal counters are pinned with exact values rather than "greater than zero", and the end-to-end EQT matcher test catches three separate breakages on its own. The gaps are all at the seams rather than in the logic. `published_mention_rows` is the single seam the whole feature depends on, and I confirmed myself that two of its four call sites can be swapped back to the raw list while all 529 tests still pass. That is worth closing before merge, even though nothing ships broken because of it.

## Severity summary

Severity follows the HIPPO scale: **High** blocks the merge, **Important** should be fixed here or in an immediate follow-up, **Personal preference** is a defensible alternative you may decline, and **Opinion** asks for no change.

| Severity | Count |
|---|---|
| High | 2 |
| Important | 10 |
| Personal preference | 13 |
| Opinion | 9 |

## Inline notes

### src/cdt/pipeline.py

- **L312–L357** `HIGH` `[correctness]` **`cdt pipeline` publishes with no lineage pass, so #170 survives on that path** — `run_match_and_finalize` got the pass at L429, but `PipelineOrchestrator.run` calls `match_pending_mentions` here and `write_final_output_tables` at L357 with nothing between. `grep apply_lineage_inference_pass src/` returns only `pipeline.py:429` and `cli.py:926`. That leaves `cdt pipeline` and `cdt-orchestrator daily|historical --extractor-backend live` publishing un-inferred lineage. The default backend is `batch`, which routes through `run_match_and_finalize`, so the scheduled run is fine today — but `docs/schema.md:135` says the pass runs "in `cdt match` and in the pipeline's match-and-finalize step alike", and your new comment at `cli.py:920` says "Always, as the pipeline does". Could you add the call here, guarded on `not matched["debt_instrument"].empty` the way `run_match_and_finalize` guards it?

- **L425** `PREF` `[correctness]` **The empty guard makes the two paths behave differently** — `cdt match` runs the pass unconditionally; this path skips it when nothing new matched. On a quiet pipeline run over a root holding stale pointers, the self-healing would not fire. I could not make a real corpus produce zero instruments, so this may be unreachable in practice — worth a sentence either way.

### src/cdt/matcher/core.py

- **L1212–L1213, L1253–L1254** `HIGH` `[correctness]` **A leftover guessed pointer cancels the new extracted one, and your documented deploy sequence hits it** — `derive_parent_links` seeds `amendment_parents` with the existing row's pointer whether or not the pass inferred it, then clears the set when it holds two. So a leftover inferred pointer and the new #203 extracted pointer cancel each other, and the pass re-infers its guess on the next run. I ran your deploy sequence on `data/lineage-verify`: `cdt backfill-mentions`, then one plain `cdt match`, gives **19 pointers and 539 heads** against the clean rebuild's **22 and 536**, and three further plain matches leave it at 19. `dim::3fe2333…` (EQT's Second Amended and Restated agreement) carried a stale `prior_fact` pointer and loses its extracted link to the minted prior state; `dim::491cf40…` is knock-on, because with the first row unlinked `ordinal_chain` sees a same-rank tie and refuses. `--force` gives the right answer, but the deploy notes say `--force` is only for a non-zero #206 count. The cheapest fix is to let the inference yield: skip the seed, or discard it from the set, when `amendment_inferred_by` is set. Note that dropping the carry-forward entirely breaks `test_lineage_inference_pass_writes_pointers_and_rederives_the_rollup`, which deliberately pins it — so the conditional version is the one to write.

- **L2304–L2316** `IMPORTANT` `[correctness]` **The pass infers against a `first_seen_filing_date` it then overwrites, so pass 1 can differ from pass 2** — `infer_amendment_parents` uses that column as both the ordering guard and the chain sort key, and `apply_lifecycle_rollup` at L2314 recomputes and rewrites it from the member edges. On a row whose members are gone, pass 1 yields one link and nulls the column; pass 2 then yields two, adding a link pass 1 refused. That row shape arises naturally, since mention ids are content hashes and re-extracting an item mints a new id while the old edge is never deleted. It does converge after two passes. Running the observation-column recomputation before the inference as well as after would close it.

- **L497** `IMPORTANT` `[tests]` **Deleting the borrowed-signature argument leaves the suite green** — `borrowed_lender_signature` and `score_candidates_for_mention` are each tested alone, but nothing pins that they are connected; removing `lender_signature=borrowed_lender_signature(mention, mention_index)` here passes all 529. The one end-to-end test that goes through `match_tables` uses `parties_json="[]"`, so the lender path never runs in it. Giving that fixture a lender payload on the model rows would close it.

- **L2221** `PREF` `[tests]` **A reachable `KeyError` guard with no test** — reverting `by_cik.get(cik, [])` to `by_cik[cik]` is uncaught, but the new synthesized skip above means a CIK whose mentions are all synthesized leaves no entry and would crash `match_pending_mentions`. The fix is right; one assertion would pin it.

- **L1852–L1871** `OPINION` `[repo-coherence]` **`coerce_optional_bool` re-implements what `coerce_dataset_text` already does** — its two siblings both funnel through `coerce_dataset_text`, which already handles `None`, `Decimal`, `pd.isna` and the placeholder set. Routing through it would match the family and inherit the placeholder handling.

### src/cdt/matcher/lineage_inference.py

- **L208–L223** `IMPORTANT` `[repo-coherence]` **Two required parameters that the function immediately `del`s — the shape a test in this module forbids** — `tests/test_matcher_lineage_inference.py:104` asserts by `inspect.signature` that unread parameters are absent from this function and from `match_tables`/`match_pending_mentions`. That test exists because the #177 review flagged exactly this pattern here as `Important`. Dropping both parameters and the `del` is easy: `member_groups` and `mention_index` are still built for `apply_lifecycle_rollup`, so only the call at `matcher/core.py:2304` changes. While you are there, L214 still says "the **two rules** support" and there is one rule now.

- **L271 onward** `OPINION` `[correctness]` **Is the mutual-pair drop still reachable?** — with only `ordinal_chain` left, a mutual pair may be impossible to construct. It is now the only lineage guard with no test. Either pin it or delete it.

- **L106–L127** `PREF` `[repo-coherence]` **A second generic-role vocabulary, overlapping the first** — `GENERIC_BORROWER_PHRASES` and `GENERIC_LENDER_TERMS` (`matcher/core.py:55`) both mean "this names a role, not a company", they share `buyer`/`buyers`, and they are applied under different normalisations, so an addition has to be made twice. Your deferral of the `_borrower_key`/`normalize_party_text` unification is reasonable and `party_dedupe_key` did not make it worse — this list is the one thing genuinely added to the pile.

### src/cdt/extractor/core.py

- **L1767–L1781, L1847–L1848** `IMPORTANT` `[correctness]` **A `prior` term that failed to parse makes the mint inherit the post-amendment value** — the kind sets at L1847–L1848 are built from lists already filtered on a non-null value, so a `prior: true` term the parser could not resolve is invisible to the "did this kind change?" test and the current value gets copied onto the predecessor marked `inherited`. I verified it: a row with a parsed prior `agreement` plus an unparsed prior commitment and maturity mints `principal_amount: 250000000` and `maturity_date: 2029-05-01` — both the successor's post-amendment values — with counters `{'minted': 1}` and no refusal recorded. The same root cause explains the other half: when the unparsed prior is the object's only one, it is dropped at L1781 with no counter at all. That is why "22 objects carry a prior term" doesn't match your counters, which sum to 21 — on `genwindow-run-branch` the missing object is `dim::5542bb4c…`, a `Loan and Security Agreement` whose prior commitment has a null amount. Your docstring promises "every refusal increments a named counter so the rate is measurable", and the deploy notes call the counters the pre-registered yield. Incidence of the harmful variant on the corpus today is **0**, so nothing is wrong in the published data. Deriving the kind sets before the value filter, plus one counter, fixes both.

- **L687, L2229, L2397, L2580** `IMPORTANT` `[tests]` **All four publish paths through the seam can be deleted with a green suite** — `test_published_mention_rows_is_the_single_publish_seam` calls the helper directly on a row state with no `prior` facts, so the mint is a no-op there and the seam is indistinguishable from the raw list. I replaced two of these call sites with `row_state.debt_instrument_mentions` myself and got 529 passed. No test asserts a synthesized row reaching a written partition or a `full.jsonl` record through a real extractor entry point — the only end-to-end minting coverage is the backfill. Giving `test_extract_pending_items_writes_mentions_and_audit` a `prior`-marked commitment and asserting a `synthesized_by == "prior_state"` row in the partition and the audit record would cover the important ones.

- **L1668, L1673** `IMPORTANT` `[tests]` **Half of each date-kind frozenset, and the `expected` guard, are unpinned** — dropping `commitment_termination` from either set, or deleting `and not entry.get("expected")` from the inherit loop, all pass. Every mint test uses `maturity`/`agreement` only and no fixture carries `expected: True`. The `expected` guard is a deliberate rule with no coverage at all.

- **L1978–L2032** `IMPORTANT` `[security]` **The backfill rewrites every partition on a lease it never renews** — the TTL is two hours (`lease.py:44`), and once it lapses the next orchestrator tick legitimately steals the lease and starts writing the same `part-0000.parquet` files. Both writers full-overwrite, so rows are lost silently. The tell is the asymmetry: this PR adds `renew` to `apply_lineage_inference_pass` for exactly this reason and then omits it here. **Scale caveat:** I measured 236 partitions at about ten seconds, so two hours is a long way off at reference-corpus size — this matters only if production is far larger or S3 round trips dominate. Threading `renew` through and calling it per partition mirrors `match_pending_mentions`.

- **L1950** `PREF` `[correctness]` **The function mutates the dicts it was handed** — it writes `amendment_of` into the caller's rows while the docstring calls it "a pure function of one item's rows". Both current callers are safe, and no nested payload is aliased, so nothing is broken. The trap is for the next caller who passes `row_state.debt_instrument_mentions` straight in and persists a pointer into `state.jsonl`. Either copy at the top or say "rewrites `amendment_of` in place" in the docstring.

- **L1662–L2032** `PREF` `[repo-coherence]` **370 lines of a new rule in a 5,262-line module** — the closest precedent in the repo is exact: the matcher's inference rules live in `matcher/lineage_inference.py` rather than in `matcher/core.py`, which is less than half this file's size. An `extractor/prior_state.py` would sit in the same relation. Not urgent, but this file is the one that most invites it.

- **L1676–L1700** `PREF` `[repo-coherence]` **`_json_list`/`_json_dict` are a private copy of `parse_cluster_list`** — same body as `matcher/core.py:1707` plus a `coerce_dataset_text` front end. They do consolidate three ad-hoc inline parses, which is a real improvement. `storage.py` is the coherent home (both modules import it, `coerce_dataset_text` already lives there) — note `shared.py` is not, since it is an `idi_ftm2j_shared` shim, and the extractor cannot import the matcher.

- **L1795–L1799** `PREF` `[security]` **A non-string date value would abort the whole run, not one item** — `{"kind":"amendment","normalized_date":["2020-01-01"]}` raises `TypeError: cannot use 'list' as a set element` out of the mint and kills the entire extract or backfill. **I could not find an attack path** — `standardized_date_payload` overwrites `normalized_date` with this repo's own parser output, so no model output reaches here as a non-string; it would take a tampered parquet. Flagging only because it is the same class you just paid to fix in `_borrowers`. One `isinstance(..., str)` clause closes it.

- **L1837–L1841** `OPINION` `[correctness]` **`skipped_sibling_is_predecessor` publishes no link at all** — the comment says "the sibling *is* P", but `amendment_of` stays null, so that case yields no lineage. Defensible; the comment just reads as though a link results.

- **L1761** `OPINION` `[tests]` **The already-synthesized filter is unpinned** — `test_mint_is_id_stable_and_idempotent` holds for a second reason (a mint's `prior` flags are already flipped), so replacing this with `list(rows)` passes. A row carrying both `synthesized_by` and a `prior: True` amount would pin it directly.

- **L1910, L1943, L1951–L1954** `OPINION` `[determinism]` **On the shared-mint path, two unhashed fields depend on input order** — whichever successor is seen first supplies `raw_id` and `synthesized_from_mention_id`. I could not make this produce a different result on any path I tried, and the ids match either way. It is worth a note because `synthesized_from_mention_id` is actually read later: `borrowed_lender_signature` reads it, so if two successors named different lenders the winner could change scoring.

- **L1960–L1975** `OPINION` `[repo-coherence]` **The `counters` parameter is passed by nobody** — all four production sites call this with one argument and the tests thread counters into `mint_prior_state_rows` directly. The seam itself is genuine and should stay: it owns the copy that keeps the mutated pointer off `state.jsonl`.

- **L1664** `OPINION` `[repo-coherence]` **`SYNTHESIZED_PRIOR_STATE` breaks the `<COLUMN>_<VALUE>` naming convention** — every other enumerated-value constant here is `<column>_<value>` (`DERIVED_FROM_STATED`, `LENDER_DISCLOSURE_NONE_NAMED`); `SYNTHESIZED_BY_PRIOR_STATE` would match. Related: "synthesized" now means both "a value derived from a name" and "a whole minted row" within this one file. Mostly a note for future naming, since the column is on disk.

### src/cdt/cli.py

- **L926** `IMPORTANT` `[tests]` **`cdt match` running the pass has no test** — replacing the call with canned zero stats passes the suite. `tests/test_pipeline.py:340` covers the `run_match_and_finalize` half well; this half, named in commit `26db036`'s subject line, is unguarded. Nothing pins that `--infer-lineage` is gone from the parser either. Mirroring the pipeline test would take a few lines.

- **L881** `IMPORTANT` `[tests]` **Deleting the whole lease block from `backfill-mentions` passes the suite** — the test covers the `--dry-run` skip well (and does catch `--dry-run` being ignored), but the real leg only asserts exit 0. This command rewrites every mentions partition, so the lease is the #88 guard. `tests/test_cli.py:462` already has the pattern to copy.

- **L898–L931** `PREF` `[security]` **`run_matcher` doesn't forward `renew` to the pass** — the pipeline path does; this one doesn't, though `lease` is in scope. The un-renewed hold predates this PR, but the PR makes the whole-corpus rewrite unconditional and adds the parameter that fixes it.

### docs/schema.md

- **L403–L445** `PREF` `[repo-coherence]` **The "Synthesized rows" section omits `status` and overstates `interest_rate_*`** — the bullet says "No event facts", but `extractor/core.py:1887` sets `status_payload = derived_status_payload(minted_dates)`, and a dated `agreement` yields `entered_into`. I confirmed it on a real mint: `status: entered_into`. A reader of that bullet would expect null. Separately, L415 says the rate is "the successor's", but L1894 rewrites its `derived_from` to `"inherited"`. Given your note that a prior review found two sentences here false, these are worth one line each. The rest of the section I checked line by line against the code and it is accurate.

### src/cdt/matcher/core.py and src/cdt/cli.py (lineage pass)

- **L2319–L2334 / L919–L929** `PREF` `[repo-coherence]` **The pass rewrites every shard and writes no manifest** — every other writing stage in this repo writes one, and `docs/architecture.md:38` names stage manifests as a design property. The match manifest records `partitions_written`, and then the pass rewrites all of them. This predates the PR, but it was behind `--infer-lineage`; making it the unconditional default means the manifest now systematically describes a dataset something else changed afterwards. The pass already returns the counters you would want in it.

### tests/test_file_native_stages.py

- **L1017–L1022** `IMPORTANT` `[tests]` **The "works on copies" assertion passes for the wrong reason** — `mint_prior_state_rows` already returns a fresh list, so this holds even without the dict-copy in `published_mention_rows`. Removing that copy passes the suite, though it makes the pointer write land on the dicts persisted to `state.jsonl`, which the docstring says must never happen. Using a row state whose mention carries a `prior` amount and asserting `row_state.debt_instrument_mentions[0]["amendment_of"] is None` afterwards would pin the real invariant.

- **L6903–L6982** `PREF` `[tests]` **The as-of-flag test doesn't cover its own headline case** — hardcoding the flag to `False` leaves this test green in isolation; only a pre-existing test catches it. It asserts `False` for a stated as-of and `True` for a carried-forward value, but never `True` for a substituted one. A third case with `balance_mention("m-substituted", None)` would do it.

### tests/test_matcher.py

- **L180–L184** `PREF` `[tests]` **The comment claims three cases and two are asserted** — the missing one is a mint that names lenders itself, which is the case that could over-borrow; removing `or mention.lender_signature` from the guard is uncaught. One line.

### tests/test_pipeline.py

- **L382** `OPINION` `[tests]` **`assert renewals` cannot fail** — `renew` is also called by `match_pending_mentions` and again after the pass block, so the list is non-empty either way; deleting the `renew()` before the pass is uncaught. Either snapshot the count around the pass or drop the comment. The neighbouring `callable(calls[0]["renew"])` assertion is real.

### README.md

- **L87, L116** `PREF` `[repo-coherence]` **Still points a new contributor into the archived dashboard repo** — `docs/architecture.md:241` now says that repo is archived, but the README tells you to `cd` into it and run a command. `docs/deployment.md:157,193` and `docs/deployment-dev.md:149` do the same. Hedging in the design docs is fine; the README is the front door.

- **L81–L82** `OPINION` `[repo-coherence]` **`cdt backfill-mentions` isn't listed with its sibling admin commands** — `show-extract-job` and `reset-extract-job` are both noted here.

## What I would do before merging

1. Wire the pass into `PipelineOrchestrator.run` (High, `pipeline.py:312–357`).
2. Stop an inferred pointer from colliding with an extracted one in `derive_parent_links` (High, `matcher/core.py:1253`), then re-run the deploy sequence and confirm it lands on 22 pointers and 536 heads. Until then, the deploy notes should say `--force` is required on any root that already carries inferred pointers.
3. Pin at least one real publish path end to end (`extractor/core.py:2229`), and the `cdt match` pass call.
4. Derive the prior-kind sets before the value filter and add the missing refusal counter (`extractor/core.py:1847`).

Nothing in the design needs to change. The stage-boundary argument is right, the mint genuinely gives the same answer every time and is safe to re-run, and #204's self-healing is the best-evidenced part of the change — it did exactly what it claims on a stale root carrying pointers from two retired rules.
