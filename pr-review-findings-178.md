# PR Review Findings: Pre-beta schema: kind-typed amounts and dates, parties, status events, provenance, lifecycle rollup

PR [#178](https://github.com/dsi-rse/commercial-debt-tracker/pull/178). Reviewed 2026-09-11 against `dev` at `ac1b07a`, head at `a71bb2a`.

Reviewable diff: 2,982 added / 656 deleted source lines across 12 files, from 29 in-scope commits.

## Diff scope note — resolved by a rebase after the review

The review was conducted while the branch still carried #149's four original commits, because
#149 was squash-merged into `dev`. The three-dot diff (`origin/dev...HEAD`) therefore
over-reported by 68 source lines and attributed #149's work to this PR, so every finding below
was scoped against the two-dot diff (`git diff origin/dev HEAD`) instead.

**That has since been fixed.** The branch was rebased with
`git rebase --onto origin/dev 9a6eed7`, dropping #149's four duplicate commits and replaying
this PR's 29 onto `dev`. The rebase was clean, and `git diff a71bb2a 6b04446` is empty — the
published tree is byte-identical, so nothing in this review is invalidated. The three-dot and
two-dot diffs now agree at 5,401 insertions / 656 deletions, and GitHub's `CONFLICTING` flag
should clear on the next push.

Line numbers below refer to the file contents, which the rebase did not change.

## Verification runs

| command | result |
|---|---|
| `pytest -q` (PYTHONPATH pinned to the branch worktree) | **394 passed**, 5.70s, exit 0 |
| `ruff check src tests` | All checks passed |
| `ruff format --check src tests` | 35 files already formatted |
| `gh pr checks 178` | **no checks have ever run** — `checks.yml` fires only on PRs targeting `main`/`dev`, and this PR was opened against `issue-142-retired-of-semantics` |
| `git diff origin/dev origin/issue-142-retired-of-semantics` | empty (confirms the conflict is cosmetic) |

Because CI has never run, the suite/lint results above are the only automated signal on this
branch, and they come from one machine rather than the pinned CI image.

## Independent replay of the eval's judge-free claims

Recomputed from the local run artifacts (`data/genwindow-run-dev` 676 mentions,
`data/genwindow-run-branch` 669 mentions), with item text joined from
`data/genwindow-eval*/items` (8,190 texts). Read-only.

| metric | claimed dev | recomputed | claimed branch | recomputed |
|---|---|---|---|---|
| `amount_without_currency` | 30 | **30** | 4 | 6 |
| `no_named_lenders_and_not_flagged_incomplete` | 189 | 183 | 29 | **29** |
| `with_maturity` | 350 | **350** | 384 | **384** |
| `evidence_spans` | 5,880 | **5,880** | 10,792 | 12,363 |
| `evidence_spans_not_matching_text` | 3 | **3** | 0 | **0** |

Seven of ten cells reproduce exactly. The three that differ are definitional, not directional,
and both differences move the branch's own number in the unfavourable direction: my
`amount_without_currency` counts every kind-typed entry rather than only principal-supplying
ones, and my `evidence_spans` sweeps every `*_json` column including `dates_json`.

**The #154 span contract holds under independent check: 11,656 branch evidence spans
re-sliced from the original item text, zero mismatches, against 3 in dev.**

Also replayed and confirmed:

- Extractor row states, branch: 362 SUCCESS, 1 FAILED, 1 PARTIAL. Salvage (#152) fired **once** in 364 units.
- Failure registry: 2 entries covering both the FAILED and the PARTIAL row, so the "publishes its mentions and keeps a registry entry" contract holds.
- `is_lineage_head` 537 True / 5 False, and `status == "announced"` on 54 of 542 — matching the PR body's own #170 and #169 figures exactly. The known gaps are reported honestly.
- `superseded_by_debt_instrument_id` set on 5 rows, `amendment_of_debt_instrument_id` on 5. Consistent.
- No `_member_ids` or other underscore-prefixed temporary leaked into the published dataset.
- Published date-fact kinds across 1,173 facts: no `expected_closing`; 40 `closing` facts carry `expected: true`.
- Multi-span instrument names are the norm: **476 of 669 mentions (71%)** carry more than one `name` span.

## Corrections to my own change map

Three claims in the change map I gave the reviewers were wrong, and reviewers caught all three:

1. I wrote that `build_debt_instrument_rows` lost a retired-parent `end_date` backfill loop. That loop was removed by **#149**, not this PR — I derived the bullet from the three-dot diff before re-scoping.
2. I gave the column-list sizes as 24 and 38. They are **32** (`DEBT_INSTRUMENT_MENTION_COLUMNS`) and **40** (`DEBT_INSTRUMENT_COLUMNS`); I had counted diff-visible entries rather than the full lists.
3. I flagged `standardized_amounts_payloads`' truthiness check as possibly swallowing a zero amount, and `select_date_payload`'s partial default dict as possibly harming a consumer. Both refuted: `normalize_numeric_string` renders zero as `"0"` (truthy), and the only consumer of the date payload reads `derived_from`. Dropped.

## Findings

Severity is my call, not the reviewers'. I downgraded 11 findings the reviewers proposed as
Important — naming, dead constants, comment placement, file size and speculative generality
are defensible-alternative territory. Where a reviewer's fixture was wrong I rebuilt it and
say so.

---

### Factor: correctness — matcher (reviewer scope: `src/cdt/matcher/core.py`)

#### Findings

- **[correctness] A non-force match over pre-#178 partitions silently publishes dangling pointers and resets lifecycle columns** — `src/cdt/matcher/core.py:40`, `597-649`, `1387-1394` [proposed: **High**] [confidence: verified]

  One root cause with three observable symptoms. `debt_instrument_mention_id_for`'s hash
  payload changed, so **every mention ID changes**, while the `dim::` prefix stays — old and
  new IDs are indistinguishable by shape. `MATCHER_SCHEMA_VERSION` went 3→4, but I traced
  every reader: `matcher/core.py:373` and `pipeline.py:571` both *write* it into a manifest
  and **nothing reads it back**. It is a label, not a guard. `match_pending_mentions` only
  discards existing partitions when the operator passes `force=True`.

  So a match run without `--force` against a stale root produces, with no error and no warning:

  1. **Dangling pointers.** `mention-cluster-edges` rows and `seed_debt_instrument_mention_id` point at mention IDs absent from the `mentions` dataset.
  2. **Reverted statuses.** `apply_lifecycle_rollup` filters members to `member_id in mention_index`; a cluster whose members are all absent is recomputed from an empty list. Verified end to end — run 1 published `terminated` / `2024-06-02` / source `m1` / 1 mention; run 2 republished the same instrument as **`active` / None / None / 0 mentions**.
  3. **Zeroed observations.** `mention_count`, `document_count`, `first_seen_filing_date`, `last_seen_filing_date` all reset, so the browse index #155 exists to feed cannot distinguish "never observed" from "not re-read this run".

  This is not #107 ("the stale artifact root needs a rebuild"), which is a known gap. It is
  that a *partial* rebuild produces silently wrong rows instead of failing. The matcher
  reviewer's original fixture dated a terminal event after its filing, which routes into the
  pending path and yields `active` for an unrelated reason; I rebuilt it with the event dated
  on its filing date and the reversal still reproduces.

  Fix: read the prior `runs/match/latest.json` `schema_version` and either refuse or implicitly
  force when it is below `MATCHER_SCHEMA_VERSION`. The value is already written. Separately,
  when the filtered member list is empty, carry the existing row's status and observation
  columns forward instead of recomputing them.

- **[correctness] Legacy instrument rows publish an unpadded CIK, blank their `company_name`, and drop their lenders** — `src/cdt/matcher/core.py:1392`, `1398`, `1404`, consequence at `1429` [proposed: Important] [confidence: verified]

  This PR converted three CIK *read* sites to `coerce_optional_cik` but left the one site that
  *writes* the published column using `coerce_optional_text`, and left `build_debt_instrument_rows`
  reading only `parties_json` where `build_cluster_profiles:815` correctly falls back to
  `lenders_json`. All three limbs reproduce in one run against a dev-shaped `existing_instruments`:

  ```
  debt_instrument_id        cik company_name            name principal_amount maturity_date parties_json
            dim::NEW 0000707605    Acme Corp Brand New Notes        500000000          None           []
            dim::OLD     707605         None    Old Facility        250000000    2030-06-30           []
  ```

  Two spellings of one CIK in one partition (contradicting `docs/schema.md:249`); `company_name`
  blank even though "Acme Corp" is known for that CIK from a sibling mention — the #47 fallback
  at line 1429 looks up the unpadded key in a dict keyed by the padded form; and the legacy
  lender "Bank of Nowhere" erased, while `amount`→`principal_amount` and `end_date`→`maturity_date`
  both round-trip correctly. `dedupe_party_clusters` and `cluster_canonical_key` already handle
  role-less legacy clusters, so the fallback would just work.

  Fix: `coerce_optional_cik` at 1392/1398; add `or existing_row.get("lenders_json")` at 1404.

- **[correctness] `name_class_sizes` ignores legacy unpadded CIKs, disarming the `NAME_CLASS_GATE`** — `src/cdt/matcher/core.py:2147` [proposed: Important] [confidence: verified]

  `by_cik` is keyed by the padded `PreparedMention.cik`; this site reads `coerce_optional_text`.
  A legacy row's unpadded CIK misses the key and its name is not counted. Verified:

  ```
  padded existing row   -> {'m1': 3, 'm2': 3}   # 3 > NAME_CLASS_GATE(2) -> relaxed key rule OFF
  unpadded existing row -> {'m1': 2, 'm2': 2}   # 2 <= gate              -> relaxed key rule ON
  ```

  The gate exists because "FHLB Dallas files 67 `Consolidated Obligation Bonds` with no dates
  and repeated round amounts" (comment at line 2065). On a pre-#178 root it is silently
  disarmed, producing wrong merges rather than a failure.

- **[correctness] A planned retirement is forgotten past a newer `amended` mention, so the matcher asserts `repaid`** — `src/cdt/matcher/core.py:717-748` (the `break` at 747), effect at `681-692` [proposed: Important] [confidence: verified]

  `event_status_for_instrument` accumulates `pending` while scanning, but `break`s at the newest
  `entered_into`/`amended` status, so a strictly-older mention's `expected_retirement` is never
  collected. The `break` is right for choosing *which* event decides the status; it should not
  also gate `pending`.

  The reviewer's illustrative dates were **inverted** — it described the `amended` mention as
  older, but the bug requires it to be newer. With the `amended` mention newer:

  ```
  [p1] alone -> (None, True)      # p1 = planned redemption
  [p1, p2]   -> (None, False)     # p2 = later "amended" mention
  ```

  And downstream, `derive_instrument_status` on a row carrying a `retired_by` pointer:
  `pending=True` → `active`; `pending=False` → **`repaid`**. So adding an amendment mention makes
  the matcher assert a retirement the filings say is only planned — the exact outcome the
  pending flag was added to prevent.

  Fix: compute `pending` in its own full pass over `member_ids`, independent of the status scan.

- **[correctness] An ambiguous amendment inverse leaves a row that is neither a head nor superseded** — `src/cdt/matcher/core.py:590-596`, `679-680` [proposed: Important] [confidence: verified]

  `superseded_by_debt_instrument_id` is nulled when there is not exactly one child, but
  `is_lineage_head` reads the raw set and `derive_instrument_status` decides `superseded` off
  the nulled pointer. With two amendment children:

  ```
  P   head=False  superseded_by=None  status=active   family=C1
  C1  head=True   superseded_by=None  status=active
  C2  head=True   superseded_by=None  status=active
  ```

  `P` is excluded from any head view *and* has no forward pointer, so it is unreachable — and
  it reads `active` though the rollup knows two amendments replaced it. The single-child case
  is coherent (`head=False`, pointer set, `status=superseded`), so this is an inconsistency
  rather than a design choice. Fix: derive the status from the set, not the pointer.

- **[correctness] `lineage_family_id` changes identity whenever the family grows** — `src/cdt/matcher/core.py:569-582`, `595` [proposed: Important] [confidence: verified by reviewer, not re-run by me]

  `family_by_id` is rebuilt from scratch each run and never seeded from existing rows, so the
  key is whichever member's content hash sorts lowest — arbitrary, and changed by every new
  amendment. Anything that persists or links a family key breaks on the next amendment. The
  walk itself is deterministic within a run. Fix: seed from any existing row's
  `lineage_family_id` in the component, falling back to `min(component)`.

- **[correctness] `reference_date` is this run's newest filing date, so `matured` flips backwards on a backfill** — `src/cdt/matcher/core.py:584-587`, `693-700` [proposed: Personal preference] [confidence: verified by reviewer]

  Not monotonic across runs: a backfill of an older filing can move an instrument from
  `matured` back to `active`. The max is also taken over a whole `cik_shard`, so one issuer's
  recent filing decides whether a co-sharded issuer's older maturity reads `matured`. The
  docstring's determinism claim ("a rerun over the same inputs reproduces the same rows") is
  true as written; it is reproducibility across *different* partitions that the incremental
  design needs. Lower than Important because the value is derived and self-corrects on a full run.

- **[correctness] `build_cluster_profiles` never sets `retired`, so the name-tie "prefer live over retired" rule is inert across runs** — `src/cdt/matcher/core.py:773-823`, consumed at `1101` [proposed: Personal preference] [confidence: verified by reviewer]

  `ClusterProfile.retired` is set only in `add_member`, which runs only for mentions present in
  this run, so a pre-existing cluster whose instrument row says `status="repaid"` still reports
  `retired=False`. The tie-break degrades to "largest cluster" in exactly the incremental case
  it was written for. Kept below Important because the fallback ordering is defensible and no
  wrong value is published. One-line fix in the existing-instrument branch.

- **[correctness] The `name_only_tie` branch emits no `related` edges** — `src/cdt/matcher/core.py:1090-1133` [proposed: Opinion] [confidence: verified]

  Unlike the `not close_competitors` branch, the new branch returns early having emitted only
  `member` and `ambiguous_candidate`. Nothing in-repo reads `related` (grep finds only
  `pipeline.py:329` counting `member`), so nothing observable is lost today — it is an
  asymmetry a future consumer will trip on.

- **[correctness] `interest_rate_fields` lets a kind-only mention shadow an older complete one** — `src/cdt/matcher/core.py:1678-1687` [proposed: Opinion] [confidence: verified by reviewer]

  Selects on `kind is not None or pct is not None`, so a newest mention with `kind="fixed"` and
  a null pct beats an older mention carrying both, with no fallback slot — unlike
  `canonical_maturity_fields`, which keeps one for exactly this shape. `kind="floating"` with a
  null pct is legitimately complete, so the predicate cannot simply require the pct.

- **[correctness] `derive_instrument_status`'s recompute guard is unreachable and weaker than the primary path** — `src/cdt/matcher/core.py:669-676` [proposed: Opinion] [confidence: verified]

  `apply_lifecycle_rollup` always passes `announced_instrument_ids`, and there is no other
  caller, so the branch is dead. If reached it would skip the announced-retirer check and
  return `repaid` unconditionally for any `retired_by` pointer. The defaults also make "there
  is genuinely no event" inexpressible.

- **[correctness] Nothing recovers a same-item add-on merge** — `src/cdt/matcher/core.py:944-955` [proposed: Opinion] [confidence: verified]

  The deleted `is_same_item_sibling` fired only when start dates agreed and amounts differed;
  the removed test used *differing* start dates, so the old guard never fired there and the
  add-on merged by name fingerprint. The new unconditional skip returns no candidate at all,
  and the add-on publishes as a second instrument. Lineage can describe the relationship if the
  relation stage emits `amendment_of`/`split_of`, but that recovers the grouping, never the
  merged `principal_amount`. Recorded as a note because the merge it removes was actively wrong
  (it published Gray Media's parent series at the add-on's $70M size), it also fixes Longevity
  Health's #131 twins and Kestra's four tranches, and it matches the extractor's stated
  one-object-per-instrument invariant. The right trade, correctly scoped — but the add-on case
  is now permanently split, and that is not in the PR's "known gaps" list.

#### Verified clean

- `_member_ids` never leaks: popped for every row, and every writer reindexes to `DEBT_INSTRUMENT_COLUMNS`. Confirmed against the real run — no underscore-prefixed column in the published dataset.
- The two-pass rollup ordering is necessary: `announced_ids` needs every row's event status before any row's `repaid` leg can ask whether its retirer has closed. A single pass cannot compute it.
- The same-item skip cannot block a cross-filing merge: `item_id` is `{accession}-{item}`, globally unique per filing-item, and `member_item_ids` is populated only for this run's mentions.
- `cluster_size`/`cluster_retired` are not stale at resolution time — `CandidateScore` is rebuilt per mention, score → resolve → `add_member` in sequence. The only staleness is the cross-run `retired` gap above.
- The `name_only_tie` sort key is fully deterministic, and every `name_fingerprint` candidate scores exactly `round(strong_match_threshold, 4)`, so `tied` is either all-`name_fingerprint` or falls through to the seed branch.
- `lender_keys`' `role == "lender"` filter applies on both paths, including the all-roles `parties_json` routed through `lender_signature`.
- `dedupe_party_clusters`' `role::canonical` key neither drops nor duplicates legacy role-less clusters (verified: legacy + new payloads for one bank yield 3 clusters, not 4).
- `normalize_cik`/`coerce_optional_cik` cannot split a cluster or a shard in the primary join: `707605`, `0000707605`, `707605` as int, `np.int64`, and whitespace-padded all normalize identically, and `shard_for_cik` lstrips so padded and unpadded share a shard. Every CIK failure found is in a secondary lookup.
- `end_dates_are_compatible`'s length-based resolution: year/month/day cases all correct, including the genuine December 31 the rewrite exists to fix.
- Value/source-pointer pairing cannot disagree — every fallback reads value and pointer from the same `existing_row`, and `canonical_scalar_fields`' `existing_keys` parameter is never passed by any caller.
- The suspected `normalized_end_date_for_matching` regression is **refuted**: a dev-era mention row yields `None` (the column reindexes to NaN), and `end_dates_are_compatible(None, x)` returns True, so the gate is skipped rather than falsely conflicting. The state "has `maturity_date` but no `derived_from` marker" cannot exist on disk, since #128 and #158 land together.
- `build_debt_instrument_rows` did **not** lose a retired-parent `end_date` backfill — that was #149.

---

### Factor: correctness — extractor (reviewer scope: `src/cdt/extractor/core.py`)

#### Findings

- **[correctness] Two current `closing` dates pass a check the prompt marks "(validated)"** — `src/cdt/extractor/core.py:2814`, message at `2888-2895`; prompt `instrument_ie.md:68` [proposed: Important] [confidence: verified]

  `validate_dates_property` counts a kind toward the single-current cap only when
  `kind not in EVENT_DATE_KINDS`. `closing` **is** an event kind, so it is never counted:

  ```
  closing                  -> ACCEPTED
  maturity                 -> REJECTED
  agreement                -> REJECTED
  commitment_termination   -> REJECTED
  ```

  This contradicts three things at once: the function's own docstring, the prompt's
  "At most one current `agreement`, `closing`, `maturity`, or `commitment_termination` entry
  per object **(validated)**", and `build_retry_message`'s "one current closing date". The
  conflict is then resolved by response order — `select_date_payload` returns the first match,
  so `start_date` publishes one arbitrary date and drops the other with no retry and no
  failure record. The prompt's "(validated)" annotations are the model's only signal that a
  rule is hard; eight of nine are accurate and this is the one that has drifted.

- **[correctness] The `amounts` list lost the multiple-distinct-values guard its predecessor has** — `src/cdt/extractor/core.py:2899-2982` [proposed: Important] [confidence: verified]

  `validate_dates_property` calls `validate_standardized_single_value_cardinality` per entry,
  and the legacy single `amount` property still routes through it. The validator for the list
  that *replaced* `amount` in #140 never calls it. On one two-span citation
  (`$2.5 billion` + `$270.5 million`):

  ```
  amounts[0] -> ACCEPTED
  dates[0]   -> "contains multiple distinct normalized values. Split this into separate..."
  ```

  The accepted entry then publishes `normalized_amount: null`, because `canonical_amount_value`
  picks the shorter span and `amounts_agree` fails — so the amount is lost with no validation
  failure and no retry, exactly what the guard was written to prevent. There is also no
  per-kind cardinality check, so two current `commitment` entries are accepted and
  `select_principal_amount` silently takes the first. Fix: one call, mirroring the dates loop.

- **[correctness] The documented name-derived principal fallback is dead code** — `src/cdt/extractor/core.py:3914-3924`, cause at `3830-3832` [proposed: Important] [confidence: verified]

  `standardized_amount_payload(None, ...)` has `model_amount = None`, and the parsed value is
  gated behind `amounts_agree(model_amount, parsed_amount)`, which is False for a non-`str`
  model amount. So the synthesized payload's `normalized_amount` is always `None` and the block
  can never append:

  ```
  standardized_amount_payload(None, td, name_text="$183.36 million term loan")
      -> {'normalized_amount': None, 'currency': None, 'derived_from': None}
  standardized_amounts_payloads({"name": [...]}, td, name_text="$183.36 million term loan")
      -> []      # docstring promises a synthesized principal entry here
  ```

  The docstring asserts live behaviour ("one name-derived principal entry is synthesized,
  preserving #129"). The twin block for dates *does* work, because
  `standardized_end_date_payload` applies its name fallback after the agreement gate rather
  than through it. #129 is not regressed — it still works whenever the model emits any
  `amounts` entry — but the no-`amounts`-at-all path publishes a null principal. Note the gate
  should also be `if not select_principal_amount(payloads)`, since a `prior` or unrelated
  `repayment` figure currently suppresses it too.

- **[correctness] `lenders_known_incomplete` publishes `false` when an entry omits both `parties` and `dates`** — `src/cdt/extractor/core.py:4396-4399`, `4430-4431` [proposed: Important] [confidence: verified]

  `stage2_shape` is inferred rather than detected. A current-schema entry that legitimately
  omits both keys — the prompt says "omit a property the document says nothing about" —
  matches neither disjunct and falls through to the legacy return:

  | entry | parties | incomplete |
  |---|---|---|
  | `{name, amounts}` (no dates, no parties) | 0 | **False** |
  | `{name, dates: []}` | 0 | True |
  | `{name, parties: []}` | 0 | True |

  So an instrument mentioned only as "our $500 million revolving credit facility" publishes
  "holders fully disclosed" beside an empty `parties_json` — affirmatively wrong, not null,
  and the opposite of what the same entry reports if the model happens to include `dates: []`.
  Fix: detect the legacy shape positively, by the presence of legacy keys.

- **[correctness] Unguarded `date()` construction can abort the whole extract run, because a helper was copied without its guard** — `src/cdt/extractor/core.py:3596-3602`, `3719`, `3724-3725` [proposed: Important] [confidence: verified for the exception, uncertain on frequency]

  Two findings with one cause. `normalized_month_year_from_text` reimplements
  `iso_month_end_from_parts` (70 lines away, both new in this PR) — `MONTH_MAP` lookup →
  `calendar.monthrange` → build ISO string — but the copy drops the original's
  `is_valid_iso_date` guard. Every other date-from-parts site in this module goes through a
  named helper. Result:

  ```
  normalized_month_year_from_text("notes due January 0000") -> ValueError: year must be in 1..9999
  normalized_maturity_from_text  ("notes due January 0000") -> None          # guarded, for contrast
  date_plus_tenor("9999-01-01", (999, "year"))              -> ValueError
  date_plus_tenor("9999-12-31", (999, "day"))               -> OverflowError
  ```

  Both are reached from `InstrumentIEStage.postprocess`, and `extract_pending_items` catches
  only `InfrastructureError` — so an escaping `ValueError` unwinds past the registry save, the
  mentions write and the audit write, and the run dies having persisted nothing. Triggers are
  improbable; the blast radius is the entire job and the fix is two lines plus one call to the
  existing helper.

- **[correctness] `validate_interest_rate` rejects the shape the prompt demonstrates** — `src/cdt/extractor/core.py:3072-3077` [proposed: Important] [confidence: verified]

  ```
  {"kind":"fixed","rate_pct":"3.875"}               -> ["'interest_rate.evidence' must be a list of tag IDs."]
  {"kind":"fixed","rate_pct":"3.875","evidence":[]} -> ACCEPTED
  ```

  The prompt's `interest_rate` section ends with a rendered object carrying no `evidence` key.
  The post-processor handles the missing key perfectly (the name-derived leg verifies the rate
  off `name_text`, publishing `rate_pct: '3.875'`, `derived_from: 'name'`), and an empty list is
  already accepted — so the strictness buys nothing and costs a full retry cycle on the one
  shape the prompt shows verbatim. Every other optional sub-object in the file tolerates an
  absent key. Fix: `value.get("evidence", [])`.

- **[correctness] Salvage bypasses `validate_no_legacy_properties`, and `status_event` then overrides `dates`** — `src/cdt/extractor/core.py:2280` vs `1107-1108` [proposed: Opinion] [confidence: verified]

  When the only failures were legacy-property rejections, every entry is individually valid, so
  `kept == data`, `dropped == 0`, and the row publishes PARTIAL with a note saying it "dropped
  0 invalid ones". Worse, `postprocess`'s condition is `"dates" in obj and "status_event" not
  in obj`, so a mixed-shape entry resolves status the wrong way — a legacy `status_event:
  announced` beats a `retirement` date fact that would derive `repaid`. Publishing something
  rather than nothing is defensible for salvage; the precedence is the part that yields wrong
  data rather than merely old-shaped data.

- **[correctness] `retirement` and `termination` ties resolve by response order** — `src/cdt/extractor/core.py:172-180`, `4204-4211` [proposed: Personal preference] [confidence: verified]

  Both carry precedence 4, so for a filing stating both on one date the `max` returns whichever
  the model listed first (`terminated` vs `repaid`). The prompt explicitly anticipates the
  co-occurrence, so the pair is not hypothetical; both are terminal, so the harm is limited to
  which terminal label shows.

- **[correctness] `rate_tokens` misreads a hyphenated vulgar fraction: `6-1/2%` → `2`** — `src/cdt/extractor/core.py:3100-3102` [proposed: Personal preference] [confidence: verified]

  `FRACTION_RATE_PATTERN` requires whitespace between whole number and numerator, so a common
  older-indenture spelling misses the fraction branch and `RATE_PCT_PATTERN` reads `2%` off the
  tail. Harm is bounded — parser verification means the model's `6.5` fails to match and
  `rate_pct` publishes null, so it is a miss, not a wrong value.

- **[correctness] `rate_tokens_in_rate_span`'s bare-number fallback would accept a basis-points cell as a percentage** — `src/cdt/extractor/core.py:3117-3130` [proposed: Opinion] [confidence: uncertain]

  A span the tagger typed `interest_rate` whose text is just `150` (a `SPREAD (BPS)` column
  rather than the `COUPON PCT` column the docstring cites) returns `['150']`, and a model
  echoing `rate_pct: "150"` would publish a 150% rate. Only fires where the model is also wrong.

#### Verified clean

- **`realign_tag_details`** — ran it against a round-trip text differing by a collapsed double space and a newline→space: all three spans landed on byte-exact original offsets. No off-by-one on `char_end` (`nonws_map[last] + 1` is right because `last` is the final non-whitespace index). Both fallback paths sound, and the `last < start` branch is provably unreachable because `NERStage.validate` rejects whitespace-only tags. Independently corroborated by the real-run replay: **11,656 spans, zero mismatches**.
- **The salvage path and every terminal outcome** — all traced. Every terminal state is reached exactly once; NER failures, unparseable JSON, a non-list response, and no-valid-entry all correctly yield `FAILED`. PARTIAL publishes and registers exactly once at both persistent call sites, and `succeeded_item_ids` correctly excludes PARTIAL so `_merge_row_failures` does not clear the entry it just wrote. `extract_tables` has no failure registry but also no caller.
- **`debt_instrument_mention_id_for`'s hash payload** — audited against all 32 mention columns. Every omitted column is a deterministic function of a hashed one. No two genuinely different mentions can collide, and the ID is stable across re-extraction of identical content.
- **`date_plus_tenor` arithmetic** — month-end clamping, negative months across a year boundary, leap years all correct (`2024-02-29 +1y → 2025-02-28`, `2026-01-31 +1mo → 2026-02-28`, `2026-05-15 +364d → 2027-05-14`).
- **`normalized_maturity_from_text`'s month-year coordination** — 12 cases. All coordinated forms correctly return `None`; `due April 2033` → `2033-04-30`, `notes due 2028` → `2028-12-31`, and the redundant `due April 2033 and 2033` → `2033-04-30`.
- **`tenor_from_text`**, including `eighteen-month` backtracking past the `eight` alternative, and `2027-year` correctly not matching.
- **`computed_sum_amount` and `computed_maturity_date`** — both refuse more than they accept: ≥2 addends, no single-span "agreement", no rate-like spans, exact equality with the model's own figure.
- **Legacy replay of response shapes** — concrete pre-#140 and pre-`dates[]` responses replay to the same `closing`/`maturity`/principal values, `end_date` correctly suppressed when `maturity_date` is also present.
- **`mark_post_filing_events_expected`** — robust to both shapes `item_row["date"]` takes across backends; same-day events correctly stay unexpected.
- **`derived_status_payload`'s remaining legs** — `repayment` exclusion yields `status: None` as #163 intends; the expected-closing and agreement legs behave as documented.
- `iso_month_end_from_parts` is itself safe against a `0000` year — which is precisely why the duplication finding above matters.

---

### Factor: correctness — prompt/validator contract (reviewer scope: `src/cdt/extractor/prompts/`)

All seven of the prompt's worked examples were reconstructed with matching tag IDs and
**all validate and publish correctly**, including the `prior` maturity, the expected-closing →
`announced` case, the month-resolution `March 2056` maturity at `precision: month`, the computed
`2031-06-24` maturity, the summed `250000000`, and the `commitment_termination_date` case.

#### Findings

- **[correctness] The validator enforces an amount-kind × instrument-type rule the prompt never states** — `src/cdt/extractor/core.py:948-952`, `1024-1031` [proposed: Important] [confidence: verified]

  `commitment`+`note_bond`, `principal`+`revolving_credit` and `principal`+`credit_line` are
  rejected. Exhaustive grep of the prompt finds no statement of this — the kind table *implies*
  the split but never forbids the pairing, and nothing tells the model which of the two fields
  to change. The error catalog documents this as an observed recurring model error (§3.6, "HBT,
  GoPro 'up to $50M'"), so these retries will actually be spent, and an item whose only amount
  carries the wrong kind loses the amount entirely after salvage.

- **[correctness] `amounts[].prior` is restricted by the validator and by nothing in the prompt** — prompt `instrument_ie.md:99`; validator `src/cdt/extractor/core.py:993-1003` [proposed: Important] [confidence: verified]

  The prompt says only "`prior: true` marks a figure stated as it stood before a change" — no
  kind restriction. The validator rejects `prior` on any kind outside commitment/principal.
  Verified: an `outstanding_balance` 60M current + 80M `prior`, from "borrowings fell from $80
  million to $60 million", fails. That is a common before/after balance sentence, so it burns
  the full retry budget then salvages or drops. The `dates` section *does* draw this line for
  its own flags, so the omission looks accidental. One sentence fixes it.

- **[correctness] A worked mini-example produces a validation failure when followed** — prompt `instrument_ie.md:106`; validator `src/cdt/extractor/core.py:1010-1023` [proposed: Important] [confidence: verified]

  The `dates` side states the repayment pairing; the `amounts` side never states the reverse, and
  its mini-example models exactly the failing shape — "will repay $68 million in outstanding
  amounts under the credit facility → `repayment` 68000000 on the facility, not its principal" —
  with no instruction to add a `repayment` date entry. An object following it that also carries
  the facility's `agreement` date (which the prompt separately tells the model to record) fails
  with "a `repayment` amount is an event; add a `repayment` entry to 'dates'". The rule only
  escapes when `dates` is empty, which for a named facility it rarely is. The error catalog
  notes examples steer harder than rules here, so this is the highest-leverage line in the set.

- **[correctness] The canonical instrument name picks the agreement name over the facility name, against `ner.md` rule 11** — `src/cdt/extractor/core.py:3297-3305` (`canonical_value`) [proposed: Important] [confidence: verified]

  `name_text` is the single **longest** `name` span, while `ner.md` rule 11 now says the
  descriptive phrase "must never be dropped in favour of the agreement name" and
  `instrument_ie.md` says to put every span in `name`. Multi-span names are the norm — **476 of
  669 mentions (71%)** — and I measured **9 real cases** where the longest-span rule does exactly
  what rule 11 forbids:

  ```
  chose 'Amended and Restated Credit Agreement'         over 'revolving credit facility'
  chose 'Second Amended and Restated Credit Agreement'  over 'term loan B facility'
  chose 'Super-Priority Senior Secured Priming DIP...'  over 'DIP Facility'
  ... 6 more
  ```

  It matters beyond display: the published `name` feeds `normalize_name_fingerprint`, and
  "Amended and Restated Credit Agreement" is precisely the template name the matcher's own
  generic-name guard and `NAME_CLASS_GATE` exist to defend against — so the picker manufactures
  the generic names the matcher then works around.

- **[correctness] A `floating` rate can publish a margin as `interest_rate_pct`, with no enforcement** — `src/cdt/extractor/core.py:3046-3095` [proposed: Personal preference] [confidence: verified mechanism, measured zero occurrences]

  The prompt is unambiguous: `rate_pct` is "`null` for a floating rate: benchmarks and margins
  are not recorded". `validate_interest_rate` checks `kind` and checks `rate_pct` is numeric,
  but never checks them against each other. Verified end to end —
  `{"kind":"floating","rate_pct":"0.875"}` citing a `0.875% per annum` span passes and publishes
  `interest_rate_pct=0.875`. That is **issue #29's exact value** reappearing in the column #157
  added, and the PR's stated criterion is "#157 fixes #29 at the source".

  Downgraded from the reviewer's Important because I measured it: **0 of 70 floating mentions**
  in the 364-unit run published a `rate_pct`. The model obeyed the prompt every time, so #29 is
  not currently regressed — but the fix rests entirely on model obedience, with no guard behind
  it, and three lines would close it.

- **[correctness] `expected` is also set from the calendar, which the prompt's definition excludes** — prompt `instrument_ie.md:62`; `src/cdt/extractor/core.py:4120-4139` [proposed: Opinion] [confidence: verified]

  Well-motivated and well-commented, but it means model output and persisted fact differ with
  nothing telling the model so. One clause on the prompt line states the invariant the code
  already enforces.

- **[correctness] `expected_closing` is accepted from live responses although documented replay-only** — `src/cdt/extractor/core.py:133` [proposed: Opinion] [confidence: verified]

  The validator accepts a kind the prompt does not teach, and it behaves oddly: `expected: true`
  on it is rejected while `prior: true` is accepted, and the post-processor rewrites it
  regardless. No observed cost.

- **[correctness] `expected_retirement` fires on `default` and not on a planned `repayment`, unlike the relation prompt's wording** — `instrument_relation.md:4`; `src/cdt/extractor/core.py:155` [proposed: Opinion] [confidence: verified]

  The prompt says the marker appears when the filing plans to "redeem, **repay**, exchange or
  terminate"; the manifest keys on `TERMINAL_DATE_KINDS`, so an expected `default` sets it and
  an expected partial `repayment` does not (correctly — but the wording misleads).

#### Verified clean

- **Every enumeration matches the prompt exactly, in both directions**: `AMOUNT_KINDS` (6), `DATE_KINDS` (11 + replay-only `expected_closing`), `PARTY_ROLES` (7), `PARTY_KINDS`, `INSTRUMENT_TYPES` (4), `INTEREST_RATE_KINDS`, `INSTRUMENT_RELATION_TYPES`. `STATUS_FOR_DATE_KIND`'s range is exactly `STATUS_EVENT_VALUES`.
- **Evidence tag types for all five properties** match the prompt's table as an exact transcription, including "agreement spans are context only: no property may cite them" — no constant set anywhere contains `agreement`.
- **`validate_no_legacy_properties` vs the rewritten prompt**: exhaustive grep for all nine names finds none as an output property anywhere — not in the shape block, not in a rule, not in any worked example, not in `build_retry_message`. The only `amount` occurrences are the tag-type name, which is correct.
- **#165, #166, #163, #150 and #158 are all carried correctly into the prompt**, each verified by constructing the response and publishing it. #150 matters most given the error catalog's warning that borrowerless examples halved borrower coverage — **all eight** examples that return `parties` include a `borrower` cluster.
- **NER's `interest_rate` is wired everywhere it should be and nowhere it shouldn't**: in `allowed_tags` and `INTEREST_RATE_EVIDENCE_TAG_TYPES`, correctly excluded from the amount, date and party sets — so "interest rates are never amounts of any kind (validated)" is enforced twice over.
- **No downstream consumer assumes one `debt_instrument` tag per facility** — `NERStage.early_stop`, `iter_instrument_entries`, `relation_prompt_xml` and the evidence validators all handle multi-span names. The only span-count sensitivity is `canonical_value`.
- **`instrument_relation.md`'s manifest vocabulary matches the manifest**, including the deliberate retention of `amount` as the attribute name for `principal_amount`.

---

### Factor: correctness/migration — schema seams (reviewer scope: `datasets.py`, `pipeline.py`, `ingest.py`, `docs/schema.md`)

#### Findings

- **[correctness] `docs/schema.md` misstates the instrument `status` precedence in two ways** — `docs/schema.md:259` [proposed: Important] [confidence: verified]

  (a) "else `superseded` when an amendment child exists" — the code checks the pointer, which is
  set only when *exactly one* child exists. Line 256 of the same document correctly says "when
  exactly one exists", so the two bullets disagree. (b) "the newest terminal extracted event
  wins" — a newer `entered_into`/`amended` `break`s the scan and suppresses an older terminal
  event. Verified: mentions `2026-03-01 repaid` + `2026-05-01 amended` publish
  `status=active`. The behaviour is deliberate and arguably right; the docs just don't say it.
  Also unstated: a mention whose only date fact is a non-prior `agreement` publishes
  `entered_into`, but the mention-level bullet lists only the seven event kinds and then says
  "Null when the mention states no event".

- **[correctness] `docs/schema.md` documents the maturity-selection bug this PR fixed** — `docs/schema.md:264` [proposed: Important] [confidence: verified]

  The bullet lumps `maturity_date` in with "newest non-null across direct mentions".
  `canonical_maturity_fields` explicitly does not do that — it prefers the newest *stated*
  maturity over a newer derived one, which is the whole point of #162 ("recency-only selection
  let a name-derived `2030-12-31` outrank the closing 8-K's stated `2030-07-01`"). The doc
  describes the bug, not the fix, and would mislead anyone reasoning about maturity provenance.

- **[correctness] `docs/schema.md` says a projected close appears in `dates_json` as `expected_closing`; it never does** — `docs/schema.md:207`; `src/cdt/extractor/core.py:4080` [proposed: Important] [confidence: verified against production output]

  `standardized_dates_payloads` rewrites the kind before publishing. Measured across the real
  run: **0 of 1,173 published date facts carry `kind="expected_closing"`**, while 40 `closing`
  facts carry `expected: true`. The document's own `dates_json` bullet gets it right and omits
  `expected_closing` from the kind list, so the file contradicts itself — and a consumer
  filtering on that kind finds nothing. `docs/schema.md` is the contract the pending publisher
  work will be written against, which is why three doc findings are Important rather than
  preference.

- **[correctness] `lenders_known_incomplete` now conflates "nothing to disclose" with "something undisclosed"** — `src/cdt/extractor/core.py:4428`; `docs/schema.md:271` [proposed: Important] [confidence: verified with measured distribution]

  The derivation is complete and internally consistent across extractor, matcher and docs — the
  *change* was done properly. The issue is what the flag now means. Measured on the real window:

  | | dev | branch |
  |---|---|---|
  | mentions with flag true | 377/676 (55.8%) | 517/669 (77.3%) |
  | instruments with flag true | 323/573 (56.4%) | 396/542 (73.1%) |

  Broken down by what `parties_json` actually says: all-named → 123 false / 0 true; a collective
  cluster → 0 false / 122 true; **no lender cluster at all → 29 false / 395 true**. So 395 of the
  517 true values (76%) are true *only* because no lender was named. Dev's docs called that state
  "nothing to disclose rather than something undisclosed"; the site now reads it as "this
  instrument has counterparties the document never named". The two states remain recoverable from
  `parties_json`, but the boolean — the thing a facet or badge binds to — no longer distinguishes
  them, and the matcher's `any(...)` plus the existing-row carry-forward makes it monotonic:
  once true, true forever. Fix: keep the boolean as "a collective cluster was present", or make
  it three-valued.

- **[migration] "Stored responses replay through postprocess" describes a code path that does not exist** — `src/cdt/extractor/core.py:955-957` (comment), `1108`; `docs/schema.md:213, 222, 223, 227` [proposed: Important] [confidence: verified; the in-flight-batch trigger is reasoned, not reproduced]

  The comment above `LEGACY_INSTRUMENT_PROPERTIES` asserts a distinction the code does not make.
  There are exactly two `postprocess` call sites, both inside `handle_response`, and
  `handle_response` always runs `stage.validate` — including `validate_no_legacy_properties`.
  Folded batch results go through the same function. There is no stored-response bypass.
  Verified on a dev-shaped response:

  ```
  validate failures: 8, incl. "'status_event' is not a property of this schema"
  -> state = PARTIAL
  -> salvage_notes = ["... dropped 0 invalid ones after 1 failed attempts"]
  -> mention: principal_amount=3500000000, maturity_date=2028-10-11, status=terminated
  ```

  So the *semantics* replay correctly — every "replay" claim in the docs holds — but only after
  burning `max_attempts` model calls, and the row publishes as PARTIAL with a registry entry
  saying it dropped nothing. `docs/schema.md:230` tells the reader a PARTIAL entry "records what
  was lost". The realistic trigger is a rolling deploy with a batch job in flight:
  `ExtractionRowState.retry` appends to the persisted conversation, so retries still carry the
  old system prompt and cannot produce new-shape output — all attempts fail deterministically.

- **[correctness] The failure registry names the wrong stage and invents an error for salvaged rows** — `src/cdt/extractor/core.py:4564` (`_failure_record`), `4532` (`summarize_failure`) [proposed: Important] [confidence: verified against real run output]

  *Found by me while replaying the run; closely related to the finding above.* A row salvaged at
  `instrument_ie` that then completes normally finishes `SUCCESS`, which `finish` coerces to
  `PARTIAL`. `_failure_record` never reads `salvage_notes`: it reports `stage` as the last stage
  *run* and `error` from `summarize_failure`, which — finding no validation errors on that last
  attempt but a response present — returns the literal `"Unexpected response at stage <stage>"`.

  The one PARTIAL row in the 364-unit run published:

  ```
  "stage": "instrument_relation",
  "error": "Unexpected response at stage instrument_relation",
  "state": "PARTIAL"
  ```

  while its audit record shows that stage's single attempt had **no** validation errors and a
  valid response, and the real loss lived only in `salvage_notes`: *"instrument_ie kept the valid
  entries and dropped 1 invalid ones after 3 failed attempts"*. An operator triaging the registry
  would re-run the wrong stage, and the one thing the entry exists for is absent. All three
  `_failure_record` callers are affected.

- **[correctness] CIK padding is applied only at the two edges, so no partition satisfies the documented contract** — `src/cdt/ingest.py:773`; `src/cdt/pipeline.py:606-611`; `src/cdt/extractor/core.py:1178`; `docs/schema.md:11, 200, 249` [proposed: Personal preference] [confidence: verified]

  Padding happens in exactly two places: the ingest manifest reader and the snapshot writer. No
  stage in between pads — the itemizer, the extractor and `_failure_record` all copy the incoming
  string. Confirmed across real roots:

  ```
  data/genwindow-run-branch | mentions: padded=True  '0001552000' | debt-instruments: padded=True
  data/schema-smoke         | mentions: padded=False '816761'     | debt-instruments: padded=True
  data/genwindow-batch-smoke| mentions: padded=False '816761'     | debt-instruments: padded=True
  ```

  One artifact root, both spellings, because only the matcher and the snapshot normalize. The
  global bullet is honest about the snapshot mechanism, but the per-dataset bullets stating the
  *partition* column is padded are false. Nothing downstream breaks today, which is why this is
  preference rather than Important — it is the documented contract that is wrong. Fix: one
  `normalize_cik` call at `extractor/core.py:1178`, or reword the three bullets.

- **[correctness] `shard_for_cik` and `normalize_cik` disagree about whitespace** — `src/cdt/datasets.py:618-637` [proposed: Personal preference] [confidence: verified behaviour, no real occurrences]

  ```
  ' 707605 ' -> normalize_cik '0000707605'  shard_for_cik 0011
  '707605'   -> normalize_cik '0000707605'  shard_for_cik 0026
  ```

  The matcher shards on the raw column but compares on the normalized value, so a
  whitespace-bearing CIK would split into two never-meeting histories. Not a regression (dev
  split them too) but a hole in the invariant the new docstring asserts. The reviewer checked 402
  distinct CIK spellings across every local partition and found zero whitespace or non-digit
  values, so live exposure looks nil.

- **[correctness] Latent `NaN` handling in two defaults** — `src/cdt/matcher/core.py:1778`, `1787`; `src/cdt/pipeline.py:606-611` [proposed: Opinion] [confidence: verified]

  `str(row.get("amounts_json") or "[]")` never fires for the case it was written for: a
  parquet-missing column arrives as `NaN`, which is truthy, so `PreparedMention` is constructed
  with the literal string `'nan'`. Harmless today because every consumer routes through
  `parse_cluster_list`, which swallows the decode error — but the field is annotated `str` and
  documented as JSON. Separately, `normalize_snapshot_text` pads only `str` cells; a numeric
  `cik` would bypass padding, and `normalize_cik(707605.0)` yields `'707605.0'`, which both fails
  to join and shards differently. Nothing writes a numeric CIK today.

#### Verified clean

- **The `#153` `shard_for_cik` docstring claim — "existing `cik_shard` partitions do not move" — is correct.** Every historical writer was enumerated via `git log -S`; the call has always hashed the value from `_filing_from_manifest`, which has used `.lstrip("0")` since the initial scaffold. Verified: `707605`, `0000707605`, `0`, `0000000000` and an 11-digit value all shard consistently. The one divergence from dev is the empty string, which is harmless because `match_tables` skips null-CIK mentions before any row is produced.
- **Column-list vs writer agreement is exact, both ways.** Instrumented and diffed: `DEBT_INSTRUMENT_MENTION_COLUMNS` (32) every one written, no extra keys; `DEBT_INSTRUMENT_COLUMNS` (40) `keys not in list: []` and `columns never written: []`, including the ten the rollup fills in place; `MENTION_CLUSTER_EDGE_COLUMNS` unchanged so edge partitions stay readable. Every column is documented and every bullet names a real column.
- **Legacy replay at the matcher seam works.** A dev-shaped mentions partition written, read back through `read_table` with the new column list, and run through `match_tables`: no crash, core fields survive, dropped columns publish as clean nulls. `read_dataset` is called **without** `columns=` for the instrument and edge frames, so legacy keys arrive genuinely absent and the `get(new) or get(old)` idiom works as intended — a subtle and correct detail.
- **Per-bullet `docs/schema.md` verdicts**: `principal_amount_kind` "null on pre-#140 replays" **accurate** (reproduced); `as_of_date` "normally present only on balances" **accurate as hedged** (unenforced, but "normally" carries the claim honestly); `outstanding_balance_as_of` filing-date fallback **accurate** (reproduced); the `spans` "index exactly" contract **accurate** and the degradation path provably unreachable for validated output; `lineage_family_id` "singletons use their own ID" **accurate** (reproduced for a singleton and a 3-node family). `commitment_termination_date` "Null for notes and bonds" is an **unenforced aspiration** — no validator, no prompt rule, and the analogous amount-kind rule *is* enforced, so a reader reasonably infers this one is too.
- `ingest.py`'s `normalize_cik("")` behaves identically to dev's `"".lstrip("0")`. `normalize_snapshot_text` is the only snapshot padding site and runs for all four `FINAL_OUTPUT_TABLES`.

---

### Factor: repo coherence (reviewer scope: full source diff)

Severity note: the reviewer proposed nine of these as Important. I kept two and moved the rest
down. Dead constants, comment placement, naming and file size are defensible-alternative
territory; none changes behaviour, and ruff is clean.

#### Findings

- **[repo-coherence] The matcher re-declares extractor vocabulary while already importing from it** — `src/cdt/matcher/core.py:526`, `1813`, `2002`, `2011` [proposed: Important] [confidence: verified]

  Four pieces of the extractor's published vocabulary are respelled as literals, and the reviewer
  *evaluated* the equalities rather than eyeballing them: `EXPECTED_RETIREMENT_KINDS` ==
  `TERMINAL_DATE_KINDS`; `TERMINAL_STATUS_EVENTS` == `{STATUS_FOR_DATE_KIND[k] for k in
  TERMINAL_DATE_KINDS}`; `DERIVED_MATURITY_KINDS` == `{DERIVED_FROM_NAME, DERIVED_FROM_COMPUTED}`;
  and the bare `"-12-31"` that the extractor names `YEAR_ONLY_MATURITY_SUFFIX`.

  The convention is set by this very file — `matcher/core.py:24-27` already imports
  `DEBT_INSTRUMENT_MENTION_COLUMNS` and `MENTIONS_DATASET_NAME` from `cdt.extractor.core`.
  Sharper still, the extractor's own comment says "Downstream consumers key on this — the matcher
  treats a name-synthesized YYYY-12-31 maturity as year-resolution only (#128)", naming the
  consumer that then hardcodes the value. Kept at Important because three vocabularies must stay
  in lockstep across a stage boundary with no test pinning them: adding a terminal kind upstream
  would silently fail to reach the matcher.

- **[repo-coherence] `docs/schema.md`'s `expected_closing` bullet** — see the schema-seams section above; both reviewers found it independently. [proposed: Important]

- **[repo-coherence] Four new constant tables are unused, three of them shadowing hardcoded copies** — `src/cdt/extractor/core.py:198-202`, `210`, `947`, `4020-4022` [proposed: Personal preference] [confidence: verified]

  Verified at exactly one usage each (the definition): `DATE_COLUMN_KINDS` (vs three hardcoded
  `select_date_payload(..., "closing"/"maturity"/"commitment_termination")` calls),
  `DATE_PRECISIONS` (vs bare `"day"`/`"month"`/`"year"` returns), `FACILITY_INSTRUMENT_TYPES`
  (respelled inside `AMOUNT_KIND_TYPE_CONFLICTS`), and `LEGACY_STATUS_DATE_KINDS` (no use at
  all). Dev's version of this file has one such constant; this PR adds four. ruff cannot see
  unused module constants, so nothing else will catch them.

- **[repo-coherence] `expected_retirement_in_payloads` has no production caller while its predicate is inlined 265 lines later** — `src/cdt/extractor/core.py:4222-4227`, `4487-4492` [proposed: Personal preference] [confidence: verified]

  One src usage (the definition) and two test usages. `relation_instrument_manifest` writes the
  identical predicate inline, differing only by an `isinstance` guard the JSON-loaded input
  needs. Same rule, two places, one with no caller to keep it honest.

- **[repo-coherence] `canonical_scalar_fields(existing_keys=...)` is never passed, and the two functions it was built for inline the fallback** — `src/cdt/matcher/core.py:1525`, `1538` vs `1577`, `1611` [proposed: Personal preference] [confidence: verified]

  Speculative generality that failed to prevent the duplication it was added for — and the two
  special-cased functions cannot use it anyway, since they return multi-column dicts.

- **[repo-coherence] `maturity_source_mention_id` drops the `_date` its sibling keeps** — `src/cdt/matcher/core.py:94`, `96` [proposed: Personal preference] [confidence: verified]

  `start_date` → `start_date_source_mention_id`, but `maturity_date` → `maturity_source_mention_id`
  and `commitment_termination_date` → `commitment_termination_source_mention_id`, two lines
  apart. The reviewer correctly *cleared* the other five: `principal_`, `outstanding_balance_` and
  `interest_rate_` are group prefixes covering multiple columns, which is deliberate. Worth
  noting that `MATCHER_SCHEMA_VERSION` already bumps here and the publisher does not yet read
  this schema, so this is the cheap moment; after beta it is a migration.

- **[repo-coherence] The `#141` comment sits above `INTEREST_RATE_KINDS` instead of `STATUS_EVENT_VALUES`** — `src/cdt/extractor/core.py:212-214` [proposed: Personal preference] [confidence: verified]

  An insertion landed between a comment and its subject, so the comment about `matured` now
  documents the interest-rate enum, and the most important new enum in the schema is bare.

- **[repo-coherence] `extractor/core.py` is 4,622 lines with a verified clean seam** — [proposed: Personal preference] [confidence: verified seam, judgement recommendation]

  Every other module in `src/cdt/` is under 1,400 lines, and the repo's practice is to split a
  stage package once a concern separates (`extractor/{core,batch}`, `itemizer/{core,extract}`,
  `sixk/{triage,windows}`). The reviewer verified a seam mechanically: **lines 2560-4430 (~1,870
  lines) contain zero references to `ExtractionRowState`, `StageSpec`, `row_state`, pandas, the
  storage helpers, or `settings`**, and `extractor/batch.py`'s twelve imported names are all in
  the stage-machine half, so no cycle is possible. Suggested split: `extractor/facts.py` (tables,
  parsers, payload builders) and `extractor/validators.py`. Recorded as preference and explicitly
  **not** something to do in this PR — it would bury 2,982 reviewed lines under a file move. A
  follow-up issue is the right vehicle.

- **[repo-coherence] Two new validators are positional where all five pre-existing ones are keyword-only** — `src/cdt/extractor/core.py:823-827`, `971` [proposed: Personal preference] [confidence: verified]

  Six of the eight new validators follow the convention; `validate_instrument_entry` and
  `validate_no_legacy_properties` don't. The cost shows at the salvage call site, where `0` is a
  meaningless placeholder index that `index=0` would have made visible. The reviewer also notes
  this file's own convention for running a check family is a table plus a loop
  (`INSTRUMENT_SINGLE_VALUE_PROPERTIES`, `PARTY_PROPERTY_ANNOTATIONS`, `EXTRACTOR_STAGES`), which
  would settle the signature question and collapse the call sequence.

- **[repo-coherence] The rate marker `(?:%|percent\b)` is spelled three times instead of composed** — `src/cdt/extractor/core.py:218`, `303`, `3100` [proposed: Personal preference] [confidence: verified]

  This file's convention is to name a regex fragment and interpolate it
  (`MATURITY_COORDINATED_YEARS`, `TENOR_WORD_NUMBERS`, `QUALIFIED_DOLLAR_CODES`). This PR
  demonstrates the hazard: it added `percent` to two of the three spellings. The reviewer checked
  and **rejected** the larger claim that `rate_tokens` duplicates `is_rate_like_amount_text`.

- **[repo-coherence] `_member_ids` is passed in-band on output rows beside three out-of-band side maps** — `src/cdt/matcher/core.py:617`, `625`, `637` [proposed: Personal preference] [confidence: verified]

  `apply_lifecycle_rollup` already builds `superseded_by`, `family_by_id` and `event_status` as
  maps keyed by instrument ID for exactly this purpose; a fourth would have cost one line and
  removed the write/read/pop dance. No behavioural risk — confirmed no leak in the real run.

- **[repo-coherence] `normalized_end_date_for_matching` is new code in the vocabulary this PR renamed** — `src/cdt/matcher/core.py:1982` [proposed: Personal preference] [confidence: verified]

  Every column it touches is `maturity_*`. In fairness, `normalized_end_date`,
  `normalized_end_dates` and `end_dates_are_compatible` are all inherited from dev unchanged, so
  the new function is locally consistent with its neighbours — but the PR grew the old-vocabulary
  island rather than shrinking it.

- **[repo-coherence] Kind-grouping sets lack the filing-and-issue grounding their neighbours carry** — `src/cdt/extractor/core.py:145`, `155`, `181`, `211`, `227` [proposed: Opinion] [confidence: verified]

  `TERMINAL_DATE_KINDS` is the one worth a line: it drives three separate behaviours and its
  membership is genuinely non-obvious — `default` is in the set, yet a default does not retire an
  obligation, and nothing says why an event of default counts as terminal for these purposes.

- **[repo-coherence] `resolve_candidates` gains a fourth near-identical edge-construction block; hash payload no longer sorted; four constants defined after first use** — `src/cdt/matcher/core.py:1106-1132`; `src/cdt/extractor/core.py:3253`, `4264-4268` [proposed: Opinion] [confidence: verified]

  The reviewer nearly dropped the first, correctly noting the existing code set the pattern with
  three blocks so the fourth is locally consistent. The hash-payload ordering is cosmetic
  (`sort_keys=True` makes it irrelevant to the digest) but sorted order is what lets a reviewer
  diff the payload against the column list.

#### Verified clean

- **All eleven new constant tables are mutually consistent** — evaluated, not eyeballed: `DATE_KINDS` is a superset of `EVENT_DATE_KINDS`, `TERMINAL_DATE_KINDS`, `DATE_KINDS_REQUIRING_EVIDENCE` and the key sets of `STATUS_FOR_DATE_KIND` and `EVENT_KIND_PRECEDENCE`; `STATUS_FOR_DATE_KIND`'s values equal `STATUS_EVENT_VALUES` exactly; `DATE_COLUMN_KINDS` and `LEGACY_DATE_PROPERTY_KINDS` values are all in `DATE_KINDS`.
- **`repayment`'s absence from `STATUS_FOR_DATE_KIND` and `EVENT_KIND_PRECEDENCE` is deliberate** (#163), and `rank`'s default is unreachable for it.
- **`expected_closing`'s asymmetry is handled consistently at every site**, and the one theoretical wrinkle requires a shape the stage-1 prompt never produced.
- **`payload_tag_ids` is not a duplicate of `single_value_evidence_tag_ids`** — different representations, real call sites at both layers. **`coerce_optional_cik` is composition, not duplication.** **`normalized_date_from_text` vs `normalized_maturity_from_text` overlap is justified.**
- **All eight validators share one return contract**; the five matcher field-group builders share one signature and fallback shape; `computed_sum_amount` and `computed_maturity_date` are deliberately parallel.
- **`relation_instrument_manifest` keeping `amount` as the XML attribute is principled** and justified on the line.
- `apply_lifecycle_rollup`'s in-place mutation is signalled by its `apply_` prefix and matches `mark_post_filing_events_expected`. `extractor/batch.py` is untouched and the live/batch split survives intact. The `datasets.py`/`pipeline.py`/`ingest.py` changes are minimal, correctly placed and fully in-style. `normalize_date`'s inline `month_map` duplicating `MONTH_MAP` is inherited from dev, not this PR's to answer for.

---

### Factor: test coverage (reviewer scope: all four test files)

**Method.** 118 source mutations applied one at a time to a private worktree, full suite
re-run per mutation, file reverted after each. Baseline 394 passed.
**71 caught / 44 uncaught / 3 patch errors — a 60% kill rate.** Every "uncaught" result was
additionally confirmed non-equivalent by executing the real code path to show the guard does
change behaviour. I re-ran three of the most severe mutations myself and reproduced all three
(394 passed in each case), and confirmed both worktrees were left clean.

Severity note: the reviewer proposed five of these as High. The HIPPO definitions place
"missing tests for new behaviour" at Important, so I moved all five down and consolidated 17
reported findings into 8. What justifies the verdict is not any single one but that they
**cluster on one risk**: the PR's central deliverable — the published schema — has no
assertion anywhere.

#### Findings

- **[tests] The published schema contract is entirely unpinned** — no test references `DEBT_INSTRUMENT_COLUMNS`, `DEBT_INSTRUMENT_MENTION_COLUMNS` or `MATCHER_SCHEMA_VERSION` [proposed: Important] [confidence: verified, three mutations re-run by me]

  `match_tables` publishes via `pd.DataFrame(rows, columns=DEBT_INSTRUMENT_COLUMNS)`, so deleting
  a column from the list silently drops the data. Confirmed by me:

  | mutation | result |
  |---|---|
  | drop `status`, `status_date`, `status_source_mention_id` | 394 passed |
  | drop `lineage_family_id`, `is_lineage_head` | 394 passed |
  | drop `interest_rate_pct` | 394 passed |
  | drop mention-level `dates_json` | 394 passed |
  | revert `MATCHER_SCHEMA_VERSION` 4 → 3 | 394 passed |
  | strip `char_start`/`char_end` from every `cluster_payload` span | 394 passed |
  | `outstanding_balance_fields` → `return {}` | 394 passed |
  | `interest_rate_fields` → `return {}` | 394 passed |

  Two of these deserve emphasis. **#154's span contract is never asserted in any payload**:
  every published-payload assertion in the suite projects to `tag_id` or `text` only, and
  `char_start` appears 34 times in the test file but always as *fixture input*. The three
  `realign_tag_details` tests check offsets on the helper in isolation, never that offsets
  survive into `*_json`. And **seven published columns from #140/#157's instrument-level
  rollup have no test at all** — `test_amounts_are_kind_typed_and_the_balance_never_becomes_principal`
  covers the extractor side only; nothing carries a balance or a rate through to an
  instrument row.

  Fix: one test asserting `list(match_tables(fixture)["debt_instrument"].columns) ==
  DEBT_INSTRUMENT_COLUMNS` against a literal expected list, the same for mentions,
  `assert MATCHER_SCHEMA_VERSION == 4`, one assertion of
  `item_text[span["char_start"]:span["char_end"]] == span["text"]` on a published payload, and
  one `build_debt_instrument_rows` test carrying an undated balance and a rate through.

- **[tests] Matcher decision logic is tested at the wrong altitude, hiding four unpinned branches** — `tests/test_file_native_stages.py:4311-4428`, `4654-4712` [proposed: Important] [confidence: verified]

  Every new lifecycle test drives `apply_lifecycle_rollup` on hand-built row dicts; none goes
  through `match_tables`. `derive_instrument_status` and `event_status_for_instrument` are never
  called directly by any test. That is precisely why these pass:

  - `lineage_family_id` `min(component)` → `max` — the test asserts only that two rows *share* a family id, never which one.
  - `retired_by` edges, and separately `split_of` edges, excluded from the component walk, fragmenting those families.
  - `superseded_by` publishing on ambiguity — no fixture has two amendment children.
  - `first_seen`/`last_seen_filing_date` swapped — both fixture mentions share `2024-01-02`, and `last_seen` is never asserted.
  - The `repaid` leg evaluated before `superseded` — no fixture has both a `superseded_by` child and a `retired_by` pointer.
  - The `announced_instrument_ids` guard removed. Notably, in `test_lifecycle_status_treats_future_dated_retirement_as_pending` the row is *already* `retirement_pending`, so `retired_by and not retirement_pending` short-circuits and the announced-retirer guard is never reached — even though the test's own comment claims "the retiring notes have not closed".
  - `and not retirement_pending` removed from the `matured` leg.

  A test whose comment asserts a guarantee it does not exercise is worse than no test, because
  it stops anyone else writing one. One `match_tables` test covering a lineage chain end to end
  would close most of this at once.

- **[tests] PARTIAL is never exercised past the unit boundary** — `tests/test_file_native_stages.py:3785-3890` [proposed: Important] [confidence: verified]

  `PARTIAL` appears in exactly three assertions in the whole suite, all on a `row_state` object.
  Treating PARTIAL as a success in the live driver, doing the same in the batch finalizer (which
  removes its failure-registry entry), and removing `"PARTIAL"` from `PUBLISHABLE_ROW_STATES`
  entirely so salvaged mentions never reach a partition — **all three pass**. #152's stated
  contract is "mentions publish, and the failure registry records the loss"; only the
  `row_state.state == "PARTIAL"` half is tested, so the robustness feature could be completely
  disconnected from the pipeline with CI green. This is the same area as my own failure-registry
  finding, and one fixture would cover both.

- **[tests] `validate_parties_property` has zero effective coverage** — `tests/test_file_native_stages.py:989-1010` [proposed: Important] [confidence: verified, re-run by me]

  `return []` on the whole function passes (I reproduced this), as does deleting just its
  bad-role branch. The test whose name implies coverage feeds the **legacy** `lenders` /
  `other_interested_parties` keys, so its `"'role' must be one of"` assertion is satisfied by the
  pre-existing `validate_party_property`, not the new one; the sibling test asserts `== []`,
  which a no-op also satisfies. All eight rejection branches are untested. This is downstream of
  the three superseded party tests being replaced by tests that also drive the legacy shape.

- **[tests] Four scoring guards are untested or mutually masked** — `src/cdt/matcher/core.py:993-1003`, `1096-1103`; `tests/test_file_native_stages.py:4549-4557`; `tests/test_matcher.py:339-368` [proposed: Important] [confidence: verified]

  - **The generic-cluster vs identifying-name guard** — the fix the source comment says stopped the cascade that "shattered GEO's note histories" — has no test; deleting it passes. The reviewer confirmed it is live: a mention named `5.25% senior notes due 2028` scored against a cluster whose only name is `senior notes` yields 0 candidates with the guard and would attach without it.
  - **`name_only_tie`'s sort key: only the first of four components is tested.** Reversing `cluster_retired`, `-cluster_size`, and the `debt_instrument_id` tiebreak all pass, because the single test supplies `exact_name=True/False` with `cluster_retired` default-False on both candidates — so the first component decides everything. The id tiebreak is what makes resolution deterministic across runs, which matters given #171's reproducibility complaint.
  - **`computed_sum_amount`'s two guards mask each other**: removing "at least two addends" *or* "no single span equals the value" each passes alone, because the only test case cites one span whose value equals the model's, so either guard alone rejects it. A two-span citation where one span already equals the value is untested.
  - **`relaxed_keys_support_membership` and the `NAME_CLASS_GATE`** likewise mask each other. `test_a_generic_issuer_name_turns_off_the_relaxed_key_rule` is actually decided by an *earlier* early-return, not the inner gate — and raising `NAME_CLASS_GATE` from 2 to 8 passes, since the test straddles with 2 and 9.

  A test that passes under either of two guards proves neither.

- **[tests] Four compat and provenance paths this PR added are untested** [proposed: Important] [confidence: verified]

  - **`lender_keys`' new `role == "lender"` filter**: removing it passes. Both `lender_signature` tests feed pre-#128 `mentions`-shaped payloads with no `role` key, so they exercise only the legacy default. Without the filter a borrower's name silently joins the lender signature and drives `lender_similarity_score`.
  - **#153's two new call sites**: disabling the snapshot writer's `cik` padding, and reverting the ingest manifest reader to `.lstrip("0")`, both pass. `normalize_cik`/`shard_for_cik` themselves are well tested; the sites this PR actually added are not, and no `normalize_snapshot_text` test constructs a `cik` column.
  - **`realign_tag_details`' documented degradation path**: turning the "cannot be aligned → return unchanged" into a `break`, so the function proceeds with a partial map and emits realigned-but-wrong offsets, passes. So does removing the leading-whitespace trim on a span's start.
  - **Whole-response legacy replay**: a stored dev response carries `name`, `start_date`, `end_date`, `amount`, `lenders` *and* `other_interested_parties` in one object — the combination that simultaneously decides `instrument_date_entries`' fallback, `standardized_amounts_payloads`' `elif`, and `party_payloads_and_incompleteness`' `stage2_shape` selector. No test replays that object; the suite tests one legacy property at a time. The reviewer ran it manually and it replays correctly today, so this is a missing regression test rather than a live bug — but it is the thing the PR's compat claim rests on.

- **[tests] Hand-built mention fixtures omit 12 of the 32 mention columns** — `tests/test_file_native_stages.py:189-225`, `tests/test_matcher.py:29-53` [proposed: Important] [confidence: verified]

  `build_mention_row` supplies 20 columns and `mention_row` 16, against a 32-column schema.
  Missing from both: `instrument_type`, `commitment_termination_date`, `principal_currency`,
  `principal_amount_kind`, `interest_rate_kind`, `interest_rate_pct`, `status`, `status_date`,
  `status_json`, `interest_rate_json`, `commitment_termination_date_json`, `dates_json`. Every
  one is read by `prepare_mention` via `row.get(...)`, so an absent column silently becomes
  `None` — which is why dropping `dates_json` from the extractor's column list goes unnoticed.
  Tests that need these fields bolt them on with `| {"status": ...}`, which is the fixture
  telling you it is out of step with the schema it claims to represent. Fix:
  `{col: None for col in COLUMNS} | defaults | overrides`, so a new or renamed column becomes a
  visible change.

- **[tests] Coverage theater and small uncovered branches** — `tests/test_file_native_stages.py:1055`, `1073`, `1091` [proposed: Personal preference] [confidence: verified]

  `assert len(json.loads(str(mention["parties_json"]))) == 1` (twice) asserts a length only — any
  implementation producing one party of any role, kind or name passes, and both tests already
  read the payload, so asserting the `(role, kind)` tuple as the sibling test does costs nothing.
  `assert all(party["canonical_name"] for party in parties)` asserts truthiness only, and is the
  only assertion on `canonical_name` anywhere, though the "longest span" rule is documented
  behaviour. Also uncovered: `date_precision`'s computed → `"day"` leg (the computed-maturity
  test asserts `derived_from` but never `precision`); `mark_post_filing_events_expected`'s
  announcement exemption (the test dates the announcement exactly *on* the filing date, so the
  comparison is already False and the exemption is never the reason); `dedupe_party_clusters`
  never actually deduplicating in any test; and unknown party roles falling back to `"other"`.

#### The five removed tests

The reviewer's assessment here corrects my initial suspicion, and I accept it.

- **`test_end_dates_treat_december_31_as_year_resolution` → `test_end_dates_treat_only_name_derived_values_as_year_resolution`: fully replaced, nothing unpinned.** All four original assertions map onto the new semantics, the case the PR deliberately reversed is now pinned *positively* as `assert not end_dates_are_compatible("2030-12-31", "2030-04-15")`, and all seven resolution mutations were caught. This is the best-covered area of the PR.
- **`test_same_item_add_on_with_its_own_start_date_still_attaches` → `test_same_item_mentions_never_merge`: the reversal is pinned, and so is its cost.** The new test retains a cross-item control so the reversal is scoped rather than blanket, `test_match_tables_still_attaches_an_add_on_to_its_series` keeps the cross-filing add-on merging end to end, and `test_match_tables_keeps_same_day_siblings_apart` pins that non-merging siblings publish as distinct rows each keeping its own `principal_amount`. The residual gap is narrow: that fixture's mentions share a start date, whereas the motivating Gray Media shape had *differing* start dates with an identical identifying name — for that shape only the scoring-level `== []` is pinned, not the published rows. Personal preference.
- The other three are legitimately superseded, by replacements with strictly stronger assertions (full `(role, kind, spans)` tuples rather than presence checks) — though all three drive the *legacy* input shape, which is why the new `parties` validator ended up uncovered.

#### The real matcher-coverage picture

My change map's "3 new matcher tests against 835 changed lines" framing was wrong, and the
reviewer corrected it: matcher-exercising tests number **54**, not 31 — 31 in
`test_matcher.py` plus 23 in `test_file_native_stages.py` (`match_tables` ×16,
`apply_lifecycle_rollup` ×4, `build_debt_instrument_rows` ×2, `resolve_candidates` ×1), and 7
of the ~10 net-new matcher tests live in the larger file. The problem is not the count but the
altitude, as above.

#### Verified clean

Behaviours the reviewer tried to break and the suite stopped:

- **`_salvage_or_fail` / `salvage_instrument_ie_entries` at the row-state level** — all four mutations caught (keep-invalid, succeed-with-nothing-valid, SUCCESS-instead-of-PARTIAL, drop the relation salvage note), each by a differently named test. Only the pipeline wiring is missing.
- **`realign_tag_details`' core alignment** — the `+1` off-by-one on `char_end` is caught by two tests, and the identity short-circuit by a third asserting object identity (`is details`), which is a genuinely strong assertion.
- **`end_dates_are_compatible` / `normalized_end_date_for_matching`** — all seven resolution mutations caught across five tests.
- **`computed_maturity_date` in both directions** — removing either the `start+tenor` or the `start-tenor` leg is caught by a distinct, well-named test. `tenor_from_text` and `date_plus_tenor` including month-end clamping are covered table-style.
- **`standardized_interest_rate_payload`'s parser verification**, the stated-vs-name `derived_from` split, the bare table-cell coupon path, and fractional-rate arithmetic — all caught, by three different tests.
- **`select_principal_amount`'s prior and kind filters** — both caught by purpose-built tests.
- **`score_candidates_for_mention`'s same-item skip** — caught by four tests across both files, including an end-to-end `match_tables` check.
- **9 of the 10 new validators die when no-op'd** (only `validate_parties_property` survives), and finer branch mutations inside `validate_cross_field_semantics`, `validate_dates_property` and `validate_amount_is_not_rate` are caught too. The near-universal habit of asserting on specific validator message substrings rather than "some failure occurred" is why.
- **`normalize_cik` and `shard_for_cik` themselves**, including the `lstrip("0")` partition-stability guarantee.
- `date_precision`'s month and year legs, `mark_post_filing_events_expected`'s core comparison, `select_date_payload`'s `prior`/`expected` filters, `instrument_entries_from_response`, and `canonical_maturity_fields`' stated-over-derived preference.

#### Highlights

- **`test_published_retirers_are_sorted_not_set_order`** (`tests/test_matcher.py:588`) is exemplary. Its docstring explains why the fixture uses eight reversed IDs rather than a realistic two: "a realistic two-retirer fixture therefore catches that only about half the time, and which half depends on `PYTHONHASHSEED`". A test author reasoning about their own test's statistical power is rare and worth imitating.
- **`test_same_item_mentions_never_merge`** keeps a cross-item control in the same test, which is the right way to replace a test whose behaviour you are inverting.
- **`test_computed_sum_amount_accepts_only_the_exact_sum_of_cited_spans`** and **`test_table_cells_publish_coupon_and_document_currency`** both test rejection paths alongside the happy path, which is why their kill rate is high.
- Two tests' house-style docstrings — naming the real filing and the real failure mode — are what let the reviewer notice that the test does not reach the guard its comment describes. Good comments make coverage gaps findable.

---

## Findings dropped in the false-positive filter

| dropped claim | why |
|---|---|
| `normalized_end_date_for_matching` loses year-resolution leniency for legacy rows (my own suspicion) | Refuted by running it: a dev-era row yields `None` because the column reindexes to NaN, and `end_dates_are_compatible(None, x)` is True, so the gate is skipped rather than falsely conflicting. The "`maturity_date` with no marker" state cannot exist on disk. |
| `standardized_amounts_payloads` truthiness check swallows a zero amount (my own suspicion) | Refuted: `normalize_numeric_string` renders zero as `"0"`, which is truthy. |
| `select_date_payload`'s partial default dict harms a consumer (my own suspicion) | Refuted: the only consumer of `*_date_json` is `maturity_derivation`, which reads `derived_from` only. Nothing reads `precision`. |
| `build_debt_instrument_rows` lost a retired-parent `end_date` backfill (my change map) | Removed by #149, not this PR. |
| `prepare_mention`'s `tuple(json.loads(...))` on `retired_by_json` can crash on NaN | #149's line, unchanged context in this diff. Out of scope. |
| `possibly_related_json` bullet deleted from the docs | Correct cleanup — the column does not exist in dev's `matcher/core.py` either. |
| A floating rate publishes a margin (severity, not the finding) | Mechanism verified but measured **0 of 70** real occurrences, so Important → Personal preference. |
| A second name span kills the name-derived maturity backstop (severity, not the finding) | Mechanism verified but measured **0** real occurrences of "name span says `due <date>` while `maturity_date` is null", so Important → folded into the canonical-name finding as an Opinion-level limb. |

## Duplicates merged

Four defects were found independently by two reviewers each and are credited once above: two
current `closing` entries passing validation; `lenders_known_incomplete` false when an entry
omits both keys; salvage bypassing `validate_no_legacy_properties`; and the `expected_closing`
documentation error. One cause/effect pair was merged: the unguarded `date()` crash exists
*because* `normalized_month_year_from_text` duplicates `iso_month_end_from_parts` and dropped
its `is_valid_iso_date` guard in the copy.
