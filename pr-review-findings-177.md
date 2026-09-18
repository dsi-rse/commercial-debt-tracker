# PR Review Findings: Matcher: infer amendment lineage behind `--infer-lineage` (partial #170); split the status vocabulary (#183)

PR [#177](https://github.com/dsi-rse/commercial-debt-tracker/pull/177). Reviewed 2026-09-13 against `dev` at `67a4db4`, head at `876bc60`.

Reviewable diff: 722 added / 87 deleted source lines, plus 642 added / 37 deleted test lines, across 6 files from 2 commits. Two-dot and three-dot diffs agree, so the stack below this PR (#149, #178) has landed and nothing in the diff belongs to another PR.

| file | +/− | half |
|---|---|---|
| `src/cdt/matcher/lineage_inference.py` | +297 | lineage (#170) |
| `src/cdt/matcher/core.py` | +388/−83 | both |
| `src/cdt/cli.py` | +18 | lineage (#170) |
| `docs/schema.md` | +19/−4 | status (#183) |
| `tests/test_matcher_lineage_inference.py` | +240 | lineage (#170) |
| `tests/test_file_native_stages.py` | +402/−37 | status (#183) |

## Verification runs

All run with `.venv/bin/python`, after confirming it resolves this checkout rather than another (`cdt` resolved to `/home/tspread/idi/cdt/commercial-debt-tracker/src/cdt/__init__.py`). This repo's venv is known to resolve the main checkout when a worktree is in play, so the check was made explicitly before any result was trusted.

- `gh pr checks 177` → all five pass (CodeQL, Lint, Pulumi Preview, Security, Test).
- `.venv/bin/python -m pytest -q` → **449 passed, 1 xfailed** in 7.49s. The PR body claims 440; the extra nine are not explained by the diff, but the suite is green either way.
- `.venv/bin/python -m ruff check .` → All checks passed.
- `.venv/bin/python -m ruff format --check .` → 49 files already formatted.
- `.venv/bin/python -m ruff --version` → 0.7.2, matching the pin in `pyproject.toml` and `.pre-commit-config.yaml`.

Because the linter and formatter are clean at the repo's own pinned version, no finding below is in linter territory.

## Diff scope note

The bundled triage script was run as `map_diff.py dev` and reported 9,453 lines across 17 files, suggesting the tier from that. That number is wrong: the local `dev` ref is stale at `15598dd`, three commits behind `origin/dev` at `67a4db4`, so the script's diff included all of #178. Every finding below is scoped against `origin/dev...HEAD`, which is 1,364 insertions / 124 deletions across 6 files.

---

## Factor: pre-review findings by the coordinating reviewer (scope: full diff)

These were confirmed before fan-out and handed to the focused reviewers so they would not spend effort re-deriving them.

### Findings

- **[data-safety] `read_item_texts` silently reads nothing from an `s3://` artifact root** — `src/cdt/matcher/core.py`:2417 [proposed: High] [confidence: verified by execution]

  `read_item_texts` resolves its input with `root = Path(str(artifact_root)) / "classifications"` and then `root.glob("date=*/shard=*/*.parquet")`. `cdt`'s `--artifact-root` explicitly accepts an S3 URI — `cli.py:79` reads "Artifact root as a local path or s3:// URI" — and `src/cdt/storage.py:108` has an `is_s3` helper precisely because artifact roots are routinely remote.

  `pathlib` collapses the double slash in a URI, so `Path("s3://bucket/root") / "classifications"` is the local relative-looking path `s3:/bucket/root/classifications`, and `.glob()` over a path that does not exist yields nothing without raising. Verified directly:

  ```
  Path() of an s3 URI -> 's3:/my-bucket/cdt-artifacts/classifications'
  glob yields: []
  read_item_texts('s3://my-bucket/cdt-artifacts') -> {}
  ```

  `infer_amendment_parents` guards rule 3 with `if item_texts:`, so an empty dict disables the `dated_reference` rule entirely — the rule the PR body measures as producing **13 of the 15 links**. There is no warning: the `except Exception` inside the loop never fires because the loop never iterates, and `apply_lineage_inference_pass` does not log the item-text count. The pass reports "15 links" locally and would report near zero against S3 with an identical, successful-looking log line.

  Every sibling read in the same function goes through `read_dataset`, which handles S3 via `iter_partition_paths`. Fix: read through `read_dataset(classifications_root(artifact_root, data_dir=data_dir), columns=["item_id", "text"])`, reusing the existing `classifications_root` helper and `CLASSIFICATION_DATASET_NAME` constant from `src/cdt/classifier/core.py:44,77`. That also removes the hardcoded partition layout.

- **[repo-coherence] `match_tables` accepts `infer_lineage` and `item_texts` and never reads either** — `src/cdt/matcher/core.py`:419-425 [proposed: Important] [confidence: verified by execution]

  Both parameters appear only in the signature and the docstring. Verified by scanning the function body: the names occur at source-relative lines 10, 11 (signature) and 15, 17 (docstring), and `infer_amendment_parents` does not occur at all.

  The docstring actively asserts behavior that does not exist: "``infer_lineage`` fills amendment pointers the item-scoped relation stage cannot express" and "``item_texts`` maps item_id to filing text and enables the dated-reference rule." A caller who reads that docstring and passes `infer_lineage=True` gets zero links and no error.

  That is exactly the failure the PR body describes discovering the hard way — "My first wiring returned **zero links** for exactly this reason, with no error." The dead parameters preserve the trap for the next person instead of removing it. `match_tables` is exported in `src/cdt/matcher/__init__.py:11,24`, so this is public API.

  Fix: delete both parameters and the docstring paragraph. The post-pass is the only supported entry point.

- **[repo-coherence] `match_pending_mentions(infer_lineage=True)` loads the whole corpus's item text and discards it** — `src/cdt/matcher/core.py`:271,283-287 [proposed: Important] [confidence: verified by execution]

  The flag causes `item_texts = read_item_texts(resolved_root)` and an informative log line, and then the value is never passed to the `match_tables` call at `core.py:332` nor used anywhere else in the function. It is read into memory and dropped.

  The CLI does not currently take this path — `cli.py:709` calls `match_pending_mentions` without the flag and then calls `apply_lineage_inference_pass` separately at `cli.py:718` — so no shipped code path is affected today. But `match_pending_mentions` is public API exported at `src/cdt/matcher/__init__.py:10,23` and is the function `src/cdt/pipeline.py:312` and `:409` call, so the parameter is reachable and its only effect is to waste memory while appearing to enable a feature.

- **[repo-coherence] `PreparedMention` has no `dates_json`, so two `lineage_inference` code paths are dead and silent** — `src/cdt/matcher/lineage_inference.py`:123-133, 147-154 [proposed: Important] [confidence: verified by execution]

  `dataclasses.fields(PreparedMention)` confirms `amounts_json` is present and `dates_json` is absent. Both readers use `getattr(mention, column, None)`, whose `None` default makes the absence indistinguishable from an empty field, so:

  - `_prior_values` iterates `("amounts_json", "normalized_amount")` and `("dates_json", "normalized_date")`; the second tuple can never contribute. `prior_fact` therefore sees prior **amounts** only, yet still compares them against each candidate's `maturity_date` (line 205-209) — a comparison that can only ever match by accident.
  - `_identity_dates` falls back to `mention.start_date` alone; its `dates_json` branch selecting `agreement` and `closing` kinds — the part that implements the #167 identity idea the module docstring describes — never runs.

  The PR body discloses the `dates_json` gap honestly as a known limit, which is to its credit. The finding is not the gap but that the code is written as though the field existed and fails silently when it does not, and the module docstring describes behavior (`a prior-marked amount **or date**`, resolution against "the agreement and closing dates of the issuer's other clusters") that the shipped code cannot perform.

  Fix: either add `dates_json` to `PreparedMention` — the PR body says this "would likely lift both rules" — or drop the dead branches and correct the docstring. If the branches are kept as forward-compatibility, replace `getattr(..., None)` with a direct attribute access so the absence is a loud `AttributeError` rather than a silent empty set.

- **[repo-coherence] The new `amendment_inferred_by` column is published but undocumented** — `src/cdt/matcher/core.py`:114, `docs/schema.md`:256-275 [proposed: Important] [confidence: verified]

  `amendment_inferred_by` is appended to `DEBT_INSTRUMENT_COLUMNS`, so it is part of the published `debt-instruments` schema. The `docs/schema.md` diff documents the sibling new column `status_subtype` in detail but never mentions `amendment_inferred_by`.

  This matters more than a typical doc gap because the column exists specifically so that "an inferred pointer is never mistaken for an extracted one" (PR body). A consumer who cannot find it documented has no way to know that `amendment_of_debt_instrument_id` now mixes extracted and inferred provenance, which is the whole safeguard.

### Verified clean

- **The new `closed`/`superseded` leg reading `is_lineage_head` is sound.** `derive_instrument_status` at `core.py:779-781` returns `closed`/`superseded` when `superseded_by_debt_instrument_id` is set **or** `is_lineage_head` is present and false. I traced whether `is_lineage_head` can be false for a reason other than having an amendment child, which would misclassify (for example) split children as superseded. It cannot: `apply_lifecycle_rollup` sets `row["is_lineage_head"] = not children` at `core.py:660`, where `children = superseded_by.get(row_id, set())` and `superseded_by` is built at `core.py:614-618` exclusively from `amendment_of_debt_instrument_id`. `lineage_family_id` is the value that spans split and retirement pointers (`core.py:620-636`), and the head flag does not read it. The leg is therefore exactly "has at least one amendment child", which is what the two-children fix requires.

  I also checked the dtype risk: `row.get("is_lineage_head")` could in principle be a numpy bool or NaN when rows come from parquet via `to_dict("records")` in `apply_lineage_inference_pass`. It cannot be stale here, because `apply_lifecycle_rollup` reassigns the flag as a Python bool at `core.py:660` in a loop that completes before the status loop at `core.py:683-706` reads it.

- **`apply_lineage_inference_pass` writes the same filename as the normal path, so the rewrite overwrites rather than accumulating.** Both the normal matcher write (`core.py:352-356`) and the rewrite (`core.py:2483-2489`) call `write_partition_table` without a `filename`, and `storage.write_partition_table` defaults to `part-0000.parquet` per partition. One file per `cik_shard`, replaced in place. (Handed to the performance reviewer to confirm against the real on-disk layout, since concatenating reads would double-count if the assumption is ever violated.)

- **Diff hygiene.** Nothing in the diff is outside the two stated concerns: no drive-by refactors, no vendored or generated files, no lockfile churn, no stray debug output. Every changed file is named in the PR description except `docs/schema.md`'s omission noted above.

### Highlights

- The PR body is unusually honest about its own limits, and the honesty is specific rather than hedging. It states plainly that the change does **not** close #170 (EQT 68 → 66 heads against a reviewed real count of 44), names the two implementation limits with their causes, and flags that reading item text in the matcher is architecturally wrong and points at #167 as the right home. The two "findings worth knowing before anyone else tries this" — that lineage inference cannot live in `match_tables`, and that the clause regex silently refused every clause containing a decimal — are the kind of negative result that normally goes unrecorded.
- The comment density in both new code paths matches this repo's established style, where a branch names the real filing or issue that motivated it. `RetirementExpectation.unfulfilled_at`'s docstring explaining why an undated plan blocks `expected_closed` indefinitely is a good example: it records a deliberate asymmetry that would otherwise read as a bug.
- The guard set on the inference rules is the right set, and is argued rather than assumed: fill-only-null, refuse-on-ambiguity, reject-a-later-parent, drop-cycles, record-provenance. "A wrong pointer silently rewrites a published history, which is worse than the status quo" is the correct instinct for derived lineage.

---

## Factor: correctness (reviewer scope: full diff, both commits)

### Findings

- **[correctness] `dated_reference` links two sibling tranches of one new agreement as parent and child** — `src/cdt/matcher/lineage_inference.py`:65, 230-252 [proposed: High] [confidence: verified by execution, independently reproduced by the coordinating reviewer]

  The trigger alternation at line 65 includes `amend\w+\s+and\s+restat\w+`, which matches the **child's own name**. In the standard Item 1.01 phrasing "the Amended and Restated Credit Agreement, dated as of July 7, 2026", the date the regex captures is the child's own dated-as-of date, not a predecessor's. `referenced` therefore holds the child's own identity date, and every other cluster in the same CIK sharing that date is offered as a parent. `offer()` blocks `child == parent` but not a sibling.

  Reproduced twice, independently. One 8-K item describing two tranches of one brand-new agreement — a Term Loan A and a Revolver, both `start_date=2026-07-07`, no predecessor in existence anywhere:

  ```
  regex captures: ['July 7, 2026'] -> 2026-07-07
  inferred link: {'i-tl': ('i-rv', 'dated_reference')}
  ```

  Neither instrument is an amendment of the other. Each offered the other, so the per-child ambiguity guard saw exactly one offer each and refused nothing; the cycle guard then dropped the reciprocal edge, leaving one arbitrary wrong link.

  The published consequence, confirmed directly:

  ```
  live revolver, matures 2031-07-07 -> ('closed', 'superseded', None, None)
  ```

  A revolving facility that matures in 2031 publishes as `closed - superseded` and disappears from every `is_lineage_head` view — unreachable and dead, when it is live and undrawn. That is the same class of failure #183 set out to fix, reintroduced from the other direction.

  Two things make this worse than an ordinary rule bug. First, `dated_reference` produced **13 of the 15** measured links, so this mechanism is inside the headline number and some fraction of those 13 may be spurious. Second, a false sibling link **lowers the head count**, which is the metric the PR uses to demonstrate success — so this failure mode flatters 537 → 528 rather than showing up as a regression. The measured improvement and this bug are not separable without re-auditing the 13 links by hand.

  Fix: subtract the child's own identity dates before offering — `referenced -= identity.get(child_id, set())` — or require the referenced date to be strictly earlier than every one of the child's own identity dates. `test_dated_reference_resolves_a_named_predecessor` stays green under either (child identity `{2026-04-13}`, referenced `{2022-07-07}`).

- **[correctness] The next ordinary `cdt match` wipes `amendment_inferred_by` but keeps the pointer it explains** — `src/cdt/matcher/core.py`:1598-1661, 2474 [proposed: High] [confidence: verified by execution]

  `derive_parent_links` deliberately carries an existing `amendment_of_debt_instrument_id` forward (`core.py:1471-1476`), so an inferred pointer survives an incremental rematch. `build_debt_instrument_rows` then rebuilds the row **without** `amendment_inferred_by`, and the reindex nulls it. Confirmed by inspection — the string occurs zero times in `build_debt_instrument_rows`, against two occurrences of `amendment_of_debt_instrument_id` — and by grep: the only producer anywhere in `src/` is the single write inside the inference pass at `core.py:2474`.

  So after `cdt match --infer-lineage` followed by any ordinary `cdt match` (no `--force`), the row keeps `amendment_of_debt_instrument_id`, keeps `is_lineage_head = False`, keeps `closed - superseded` — and reports `amendment_inferred_by = null`, which is indistinguishable from a pointer the relation stage extracted. That sequence is ordinary daily operation, not an edge case.

  The column exists for exactly one reason, stated in the PR body: "so an inferred pointer is never mistaken for an extracted one." After one ordinary match pass, it is. It also hollows out the "only fills a null pointer, never overwrites an extracted one" guard, because no later pass can tell which pointers it authored.

  Fix: carry it forward in the row dict alongside the pointer, cleared when the pointer changes.

- **[correctness] `--infer-lineage` silently re-derives `status` corpus-wide, moving rows with zero inferred links** — `src/cdt/matcher/core.py`:2476 with :648-651 [proposed: Important] [confidence: verified by execution]

  `apply_lifecycle_rollup` sets `reference_date = max(mention.date)` over whatever `mention_index` it is given. In the normal path that index is built per `cik_shard` batch (`match_pending_mentions` groups by `cik_shard` at `core.py:310` and `match_tables` builds the index at `core.py:451` from just that group), so the reference date is the newest filing date **in that shard**. In the post-pass it is built over the whole corpus (`core.py:2464-2467`), so it is the newest filing date **in the run**.

  Turning the flag on therefore moves `status` and `status_date` on rows the inference never touched. Demonstrated with zero inferred links: an instrument in an old-filing shard with `maturity_date=2021-06-30` went `active` → `expected_closed` purely because the reference date moved from `2020-01-02` to `2026-06-01`.

  This is arguably a **fix** for a pre-existing shard-locality bug — one "today" per shard is not defensible, and `docs/schema.md:271` already promises the run-wide behavior the pass actually implements. But nothing says so, and it has two consequences the PR does not account for: the flag-on vs flag-off `status` diff is not attributable to lineage, so the measured-effect table understates what the flag does; and the documented reproducibility guarantee ("a rerun over the same inputs reproduces the same statuses") holds only per shard in the normal path, and an incremental run never re-derives a shard it did not touch, so published statuses carry a mix of historical reference dates.

  Fix: compute `reference_date` once per run and pass it into `apply_lifecycle_rollup` explicitly, so both paths agree — which also makes `docs/schema.md:271` true.

- **[correctness] The `expected_active` leg skips the `past_every_end` check the `active` leg applies** — `src/cdt/matcher/core.py`:823-824 vs :815-822 [proposed: Important] [confidence: verified by execution]

  Leg 4 tests `past_every_end` before returning `active`; leg 5 returns `expected_active` without testing it. A row whose planned start **and** whose recorded maturity are both behind the corpus publishes as "probably alive". Confirmed on two rows differing only in whether the start date is confirmed:

  ```
  expected start 2019-06-01, maturity 2023-01-01, ref 2026-01-01 -> ('expected_active',  None, '2019-06-01', None)
  explicit start 2019-06-01, maturity 2023-01-01, ref 2026-01-01 -> ('expected_closed',  None, '2023-01-01', None)
  ```

  Identical facts, opposite lifecycle answer, decided only by whether a filing confirmed the start. Reachable for any never-confirmed announced note carrying a name-derived maturity now in the past ("6.0% Senior Notes due 2023") — and the PR reports 19 `announced → expected_active` transitions, which is exactly where these land.

  Fix: hoist the `past_every_end` return above line 823 so it covers both legs. The existing `expected_active` tests carry no end dates and stay green.

- **[correctness] `ordinal_chain` picks an arbitrary predecessor on a rank tie, and the ambiguity guard cannot see it** — `src/cdt/matcher/lineage_inference.py`:220-228 [proposed: Important] [confidence: verified by execution]

  The ambiguity guard counts *offers per child*, and `ordinal_chain` only ever offers `members[index - 1]` after `members.sort()`. Two equal-rank predecessors are therefore silently reduced to one by the `(rank, first_seen_filing_date, row_id)` sort rather than being refused. Verified: two bare "Amended and Restated Credit Agreement" clusters (both rank 1, dated 2021-01-01 and 2022-01-01) plus a "Second Amended and Restated…" yields one link with `ambiguous=0`; with equal `first_seen_filing_date` the winner is decided by which `debt_instrument_id` sorts larger — deterministic, but semantically arbitrary.

  This contradicts the documented guarantee that the pass "refuses a child with more than one candidate parent". The PR itself notes over-split clusters exist in this corpus, which is precisely what produces same-rank duplicates.

  Fix: offer every member sharing the immediately-preceding rank, so the existing per-child guard refuses the link instead of the sort resolving it.

- **[correctness] `_iso` re-matches case-sensitively, so `IGNORECASE` month spellings drop links silently** — `src/cdt/matcher/lineage_inference.py`:95-100 with :64-72 [proposed: Important] [confidence: verified by execution]

  `DATED_REFERENCE` is compiled `re.IGNORECASE`, so its month group `([A-Z][a-z]+)` matches any capitalization; `_iso` then re-matches case-sensitively and returns `None`:

  | text | `DATED_REFERENCE` | `_iso` |
  |---|---|---|
  | `July 7, 2022` | match | `2022-07-07` |
  | `july 7, 2022` | match | `None` |
  | `JULY 7, 2022` | match | `None` |
  | `Jul 7, 2022` | match | `None` |

  All-caps preambles ("DATED AS OF JULY 7, 2022") are routine in credit-agreement exhibit text, and the drop is silent — no log line, no counter. A silent recall hole in the rule doing 87% of the work.

  Fix: match `([A-Za-z]+)` and look up `.capitalize()`.

- **[correctness] Stale `matured` contract in `docs/schema.md` and `extractor/core.py`; the "`announced` carries no start date" claim is now false** — `docs/schema.md`:207, 213; `src/cdt/extractor/core.py`:248-250 [proposed: Important] [confidence: verified]

  `docs/schema.md:213` still says "`matured` is never extracted; the matcher derives it", and `extractor/core.py:248-250` says "`matured` is deliberately absent … the matcher derives it from `maturity_date` instead." After this PR the matcher never derives `matured`. Separately, `docs/schema.md:207` says an instrument whose status is `announced` "has not started and carries none" — but the new leg at `core.py:830-833` returns `announced` for a row with an explicit `start_date` ahead of the reference date (verified: `start_date=2027-01-01`, ref `2026-01-01` → `('announced', None, '2027-01-01', None)`).

### Verified clean — hypotheses actively refuted

Each of these was a specific suspicion the coordinating reviewer raised before fan-out. All were tested and **do not hold**; they are recorded so the absence of a finding is legible as "checked" rather than "missed".

- **`prior_fact`'s string comparison is NOT type-broken.** This was my strongest pre-fan-out suspicion and it is wrong. `standardized_amount_payload` (`extractor/core.py:3966-3968`) publishes the *parser's* `parsed_amount`, and `normalize_numeric_string` (`:3489-3501`) canonicalizes through `Decimal.normalize()` and `to_integral_value()`, so `"$2.0 billion"`, `"$2,000,000,000"`, `"2000000000"` and `"$2.00 billion"` all become `"2000000000"`. The mention's flat `principal_amount` column *is* that same normalized string (`extractor/core.py:1249`), and parquet round-trips it as `str`. A full end-to-end run — mentions → parquet → `match_pending_mentions` → `apply_lineage_inference_pass` — produced a real `prior_fact` link. **The 1-of-15 yield is conservatism, not a broken comparison.** One caveat that reinforces the `dates_json` finding: because `PreparedMention` has no `dates_json`, the `maturity_date` half of the `terms` set can never match (an amount string never equals an ISO date), so `prior_fact` is measured at half strength.
- **The `prior`-marker asymmetry in `expected_dates_from_dates_json` is moot.** `extractor/core.py:2898-2903` rejects `prior: true` on every member of `EVENT_DATE_KINDS`, which includes `closing`, `retirement`, `termination`, `exchange` and `default`, and the salvage path drops invalid entries rather than repairing them. A `prior`-marked expected retirement cannot reach `dates_json`, so the missing guard on the retirement branch cannot leak a predecessor's plan into a successor's `expected_closed` leg.
- **`planned_retirement_date`'s dependence on `mention.date` is unreachable.** Mention `date` is `item_row["date"]` (`extractor/core.py:1236`) from `str(document["date"])` (`itemizer/core.py:307`), and it doubles as the `date=` partition key of both the classifications and mentions datasets, so it is never null in real data.
- **`offer()`'s missing-date asymmetry is one convention, not a bug.** Both branches treat `""` as sorting earliest: a parent with no date is treated as oldest and accepted, a child with no date is treated as oldest and rejects every dated parent. The permissive direction is a minor soundness gap given the "conservative" framing, but it is coherent. Opinion only.
- **The cycle guard is correct.** `sorted(resolved)` snapshots keys before mutation so there is no `RuntimeError`; `resolved` is fully populated before the guard runs, so `parent_of` sees every inferred edge from the start; and deletion only removes edges, so it can never open a cycle in a link already kept. Tested against 3-node cycles closed through pre-existing extracted pointers in both orderings, and a 2-cycle offered by two different rules — all refused or reduced to one edge correctly.
- **The status cascade has no dead legs and cannot leave the vocabulary.** An exhaustive sweep of 11,664 combinations (start date × maturity × commitment termination × event result × four `RetirementExpectation` shapes × expected start × `superseded_by` × `retired_by` × `is_lineage_head`) returned `{active: 752, closed: 9720, expected_active: 432, expected_closed: 140, announced: 620}` — nothing outside `INSTRUMENT_STATUS_VALUES` and no unreached leg. Both `expected_closed` return sites are reachable. `reference_date is None` collapses to `active`/`announced`/`closed` only, which is conservative and correct.
- **The partition rewrite cannot duplicate or drop rows, and the missing `data_dir` is a non-issue.** `artifact_root` is always supplied by the CLI, so `resolve_artifact_root` never consults `data_dir` and no caller breaks. A null `cik` is impossible because `build_debt_instrument_rows:1577` skips those rows. `shard_for_cik` strips leading zeros (`datasets.py:629-637`), so the padded instrument `cik` and the raw mention `cik` hash to the same shard. An end-to-end run confirmed 3 rows before and after, no duplicate ids, an identical partition set, and identical `part-0000.parquet` filenames. Three consecutive passes were idempotent.
- **`DATED_REFERENCE` has no catastrophic backtracking.** 80 KB of trigger-bearing text with no date: 4 ms. 100 KB single token after a trigger: 8 ms. 5,000 triggers with no date: 12 ms. The alternation branches are mutually exclusive, so the `{0,200}?` bound holds.
- **`read_item_texts`'s glob layout is right for local roots.** `classifier/core.py:245-247` writes `date=*/shard=*/part-0000.parquet`, so the S3 finding is strictly a path-construction failure and not also a layout error.

### Highlights

- The `\.(?=\d)` clause bound in `DATED_REFERENCE` is a genuinely non-obvious fix, and the comment explains why the naive `[^.;]` was wrong rather than just asserting the new form.
- Replacing the latched `retirement_pending` boolean with `RetirementExpectation.unfulfilled_at(reference_date)` is the right shape: a plan re-checked against the corpus instead of a flag that never clears.
- The leg ordering makes the #169 fix fall out of the vocabulary split rather than being special-cased onto it.
- `test_every_derived_status_is_in_the_published_vocabulary` reflecting over the cascade's own source is a good invariant; the 11,664-combination sweep independently agrees with it.

## Factor: cross-cutting — the seam between the two halves, and claim verification (scope: full diff + on-disk artifacts)

Artifact provenance, established before any number was trusted. No artifact on disk was written by HEAD (none carries `status_subtype`), so every #183 figure below is a **re-derivation** executed with HEAD's code over the fixed rows in `data/genwindow-run-branch` — deterministic, and therefore not subject to the ~85% extraction reproducibility bound in #171. The PR body keeps that distinction correctly ("no new model calls", "re-derived"); it is not blurred anywhere.

| dir | `status_subtype`? | `amendment_inferred_by`? | which code wrote it |
|---|---|---|---|
| `data/genwindow-run-branch` | no | no | pre-#183, pre-#170 — the **#155 baseline** |
| `data/lineage-test-off` | no | yes | lineage half only, flag **off** |
| `data/lineage-test-on` / `lineage-verify` | no | yes | lineage half only, flag **on** |
| `data/lineage-probe` | no | mixed | **pre-regex-fix** (12 `dated_reference`, heads 529) |

### Findings

- **[cross-cutting] One replacement clause adopts every open instrument in the same filing, with no cap on children per parent** — `src/cdt/matcher/lineage_inference.py`:236-252 [proposed: High] [confidence: verified by execution on real data]

  This is the same root cause as the sibling-tranche finding above, measured at corpus scale. `item_ids` is the set of items the child's mentions appear in, and the regex runs over the **whole item text**, so every open child in that item inherits that clause's referenced date and is offered the same parent. The ambiguity guard caps parents-per-child at one; nothing caps **children-per-parent**, so fan-in is unbounded.

  ```
  links sharing an item with their parent: 13/15
  links with no shared filing at all:       2/15
  ```

  NMP Acquisition: four children — `First Lien Secured Promissory Note`, two `Secured Promissory Note`, and a `Line of Credit Agreement` — all became amendments of one `Amended and Restated Secured Promissory Note`, and all five clusters come from the single item `000121390026097868-1-01`. PENN Entertainment: a `term loan A facility` and a `revolving facility` both became children of one parent from one item.

  Two consequences at the seam:

  - **The inverse pointer goes null exactly when the status says `superseded`.** 11 rows publish `closed - superseded` after inference; **3 of them have `superseded_by_debt_instrument_id = None`** (EQT `491cf400f`, PENN `e5b73564`, NMP `7dac8a38`), because two or more children make the inverse ambiguous. `docs/schema.md:283` promises the UI "Both pointers are published on the row, so the link needs no extra lookup beyond resolving the target's `name`" — false for 27% of superseded rows, and this PR is what adds that sentence.
  - **Unrelated obligations merge into one lineage family.** Families 479 → 469. The NMP family is now 5 members spanning a line of credit and three promissory notes, so a browse index keyed on `lineage_family_id` collapses them under one head.

- **[cross-cutting] An inferred link can leave a whole lineage family with no live row** — `src/cdt/matcher/core.py`:775-781 with :648-661 [proposed: High] [confidence: verified by execution on real data]

  The precedence in `derive_instrument_status` is sane where it was falsified: an extracted terminal event outranks an inferred `superseded`, so inference cannot overwrite a recorded fact with a guess (confirmed on EQT `dd7eafa77`, which gained two children and kept `closed - terminated`). But the superseded leg sits above every date leg and never asks whether the **child** is a plausible successor.

  ```
  families alive before, entirely closed after: 1
    family dim::139d72f70dcc372cc34d4573
      dim::139d72f70 'MPLX LP' 'unsecured revolving credit facility'
         status=closed/superseded start=2022-07-07 mat=2027-07-07 head=False
  ```

  MPLX's `2022 Credit Agreement` — itself `closed - terminated` — was inferred as the successor of MPLX's `unsecured revolving credit facility`, which starts 2022-07-07 and matures **2027**-07-07 against a reference date of 2026-09-08. The parent is now `closed - superseded`, and the family's only head is the terminated child. **MPLX's live revolver disappears from the browse index entirely** — the precise class of defect #170 was opened about, reproduced by #170's own fix, on one of the two issuers the PR measures. Two rows became `closed - superseded` while carrying a future maturity (MPLX and PENN).

  > **Corrected 2026-09-15 — see the addendum.** The MPLX characterisation here is wrong: those two rows are the same facility in two clusters the matcher failed to merge, and the corpus does record its termination, so closing it is coincidentally the right answer and only the published *relation* is false. **PENN is the clean example** of a live facility published as closed. The finding itself stands; the example does not.

  Relatedly, EQT's `Prior Revolving Credit Agreement` is published as an amendment **child** of the `Third Amended and Restated Credit Agreement` — the direction inverted against the row's own name — and stays `active` and a head.

  Fix: require the child to be a plausible successor before the superseded leg fires — at minimum, refuse a link whose child is already terminal at an earlier date.

- **[cross-cutting] 3 of the 15 inferred links have their direction decided by instrument-ID sort order** — `src/cdt/matcher/lineage_inference.py`:254-286 with :196 [proposed: High] [confidence: verified by execution on real data]

  On the real window the rules produced **both directions** for three pairs. Each child had exactly one candidate parent, so the per-child ambiguity guard never fired; the cycle breaker then deleted whichever link it reached first in `for child_id in sorted(resolved)` — lexicographic order of hash-derived IDs:

  ```
  dropped dim::06e3cd7b… -> dim::9def8324… (dated_reference), would close a cycle   [TIPTREE]
  dropped dim::0b8b5501… -> dim::b7507951… (dated_reference), would close a cycle   [MAXIMUS]
  dropped dim::7dac8a38… -> dim::e092e4c1… (ordinal_chain),   would close a cycle   [NMP]
  ```

  Proved to be the ID by renaming one member of each pair so it sorts last — the surviving direction flipped in all three:

  ```
  pair dim::06e3cd7b7 / dim::9def8324a
    baseline:               06e3cd7b7->None          9def8324a->(06e3cd7b7…, ordinal_chain)
    a renamed to sort last: zzzzcd7b7->(9def8324a…)  9def8324a->None
  ```

  For TIPTREE the two **rules disagreed** — `ordinal_chain` said A amends B, `dated_reference` said B amends A — and the contradiction was settled by ID ordering rather than refused. Direction decides which row publishes `closed - superseded` and drops out of the head-filtered index, so 20% of the measured effect rests on a coin flip.

  Note this does **not** contradict the correctness reviewer's finding that the cycle guard is correct: it is correct as a DAG-preserver and does not crash or leave a cycle. The defect is that reducing a mutual pair to one surviving edge is the wrong *response* — a cycle is prima facie evidence the rules cannot tell which instrument is the predecessor, and the PR's own stated principle is that an ambiguous guess is worse than the status quo.

  Fix: when a mutual pair is detected, drop **both** links rather than keeping a survivor.

- **[cross-cutting] The measured effect does not demonstrate the premise the architectural exception rests on** — PR body "The problem"; `src/cdt/matcher/lineage_inference.py`:1-32 [proposed: Important] [confidence: verified by execution]

  The stated premise is that pointers are missing because "a replacement agreement is named in a *later* filing than the agreement it replaces", and that the matcher can fix it because it "already works across filings within a CIK". Of the 15 links actually produced, **2** connect instruments that never co-occur in a filing (MPLX `74e51184`←`139d72f7`, EQT `3fe2333c`←`63e5b429`). The other **13 share an item with their parent** — inside the single-item scope where `instrument_relation` already operates and declined to emit a pointer.

  So 87% of the effect is in the extractor's territory, not the matcher's. That matters because reading item text in the matcher is the architectural exception the PR asks for, and the measurement offered does not support the reason given for it. The PR's own note that "the right home is the extractor recording the predecessor reference as a fact (#167)" is the more accurate diagnosis than its problem statement.

- **[cross-cutting] The status half is an unflagged breaking change to a published column, with no schema-version signal** — `src/cdt/matcher/core.py`:46; `docs/schema.md`:245-284 [proposed: Important] [confidence: verified by execution]

  The lineage half genuinely is opt-in (confirmed: `match_tables` with no flag publishes no pointers, and the pass is reachable only from `cdt match --infer-lineage`). The status half is not: **125 of 542 rows — 23% — change their `status` string unconditionally**, and the values `matured`, `superseded`, `repaid`, `terminated`, `exchanged` and `defaulted` vanish from the column entirely.

  `MATCHER_SCHEMA_VERSION` is 4 on `origin/dev` and still 4 at HEAD, verified with `git show origin/dev:src/cdt/matcher/core.py`. The in-stack precedent is unambiguous: #178 (`67a4db4`) bumped 3 → 4 with the commit message line "Bumps MATCHER_SCHEMA_VERSION to 4 for the debt-instruments column change", and `f3b1ed5` bumped 2 → 3 before it. This PR adds two columns (40 → 42) **and** swaps the vocabulary, which is strictly more than #178 did. The repo's own test says "The version is how a downstream reader learns a rebuild is required" (`tests/test_file_native_stages.py:5866-5869`). The constant is stamped into the match run manifest (`core.py:392`) and into the `latest.json` snapshot pointer (`pipeline.py:571`) that `docs/architecture.md` tells dashboard consumers to resolve.

  The dashboard is a separate repo reading `status` by value, and per session memory it is the beta gate. Without a bump, the break is silent. Fix: `MATCHER_SCHEMA_VERSION = 5`.

### Claims table

Every headline count in the PR reproduces exactly. The exceptions are listed as DIFFERS.

| lineage claim | verdict |
|---|---|
| lineage heads 537/542 flag off | **REPRODUCED** |
| lineage heads 528/542 flag on | **REPRODUCED** |
| `superseded_by` set 5 → 10 | **REPRODUCED** |
| 15 links; `dated_reference` 13, `prior_fact` 1, `ordinal_chain` 1 | **REPRODUCED** |
| 2 children left alone as ambiguous | **REPRODUCED** |
| the regex period fix "added a link" | **REPRODUCED** (`lineage-probe` pre-fix: 12, heads 529) |
| EQT 68 → 66 heads; MPLX 29 → 28 | **REPRODUCED** |
| reviewed real counts 44 (EQT) / 21 (MPLX) | **NOT CHECKABLE** — longitudinal review lives in the models repo |
| "does not close #170" | **REPRODUCED as honest** — 528/542 is 97.4% heads; EQT still publishes 5 active revolving-named heads |

| status claim | #155 | #183 | verdict |
|---|---|---|---|
| active | 398 | 399 | **REPRODUCED** both |
| announced | 54 | 25 | **REPRODUCED** both |
| closed (all causes) | 65 | 75 | #183 **REPRODUCED**; #155 **DIFFERS — 72** (38 repaid + 12 terminated + 11 exchanged + 7 defaulted + 4 superseded). The PR's #155 column sums to 535, not 542 |
| expected_active | — | 19 | **REPRODUCED** |
| expected_closed | 18 (`matured`) | 24 | **REPRODUCED** both |

| transition / spot-check claim | verdict |
|---|---|
| 19 `announced → expected_active`; 5 `active → expected_closed`; 3 `→ closed - repaid` | **REPRODUCED** |
| 7 `announced → active` (count) | **REPRODUCED** |
| "**exactly** the seven #169 counted as having a completed closing" | **DIFFERS** — both sets are size 7 but overlap in only **6**. #169's own named example `dim::ee2b4…` (EQT 3.000% Senior Notes due 2022) has a completed `closing` and went `announced → expected_closed`; a Cognizant `revolving credit facility` went `announced → active` without one |
| "the rest are straight renames" | **DIFFERS** — one further transition is unlisted: 1 `announced → expected_closed` |
| *Georgia Power Series 2025B* `announced` → `active` | **REPRODUCED** |
| *Orrstown 4.5% Subordinated Notes* `active` → `expected_closed` | **REPRODUCED** for the whole-corpus re-derivation; **DIFFERS for what the pipeline publishes** — with the shard-local reference date (2026-06-30, and the comparison is strict `<`) it stays `active` |
| *Highlander Silver "Interim Credit"* `active` → `expected_closed` | **REPRODUCED** |
| #169 fix: announced rows with a completed closing | **REPRODUCED** — 7 of 54 before, **0 of 25** after |
| "440 pass, 1 pre-existing xfail" | **DIFFERS — 449 passed, 1 xfailed** at HEAD; the claim likely predates the rebase onto #178/#149 |
| "Both pointers are already published on the row, so nothing more is needed from the pipeline" | **DIFFERS** — 3 of 11 `closed - superseded` rows publish a null `superseded_by` |

### Seam answers

- **Is the pass idempotent?** On its own, **yes** — a second `--infer-lineage` pass reports `0 links, heads 528 -> 528` and the output is byte-identical (0 differing cells across 542×42). Across the pass/normal-match boundary, **no**: statuses move on 3 rows.
- **Is `amendment_inferred_by` preserved?** Across a second inference pass, yes. Across a normal `cdt match`, **no** — wiped on all 542 rows while the pointers survive. `--force` clears both (pointers 20 → 5, provenance 15 → 0, heads 528 → 537, ids stable), so `--force` is the only way to revoke an inferred pointer. There is no stale-provenance-pointing-at-nothing case; the failure is the inverse and worse.
- **Does the PR deliver its issues?** **#183 delivered** — all five values present, `status_subtype` published, vocabulary invariant tested, counts reproduce. **#169 delivered** — 7 → 0 announced rows with a completed closing, though the PR's claim about *which* seven is slightly off. **#170 partially delivered**, as the PR itself says — and the MPLX regression above is a new instance of #170's headline symptom. #174 and #155 are context, not asks of this PR.

### Verified clean

- **Precedence where it matters most is sane.** An extracted terminal event outranks an inferred `superseded`, so inference cannot overwrite a recorded fact with a guess.
- **No un-started row became superseded** in this window (0 of 11), so the "future start is now closed" case is latent rather than realised.
- **The #178 strict-xfail is really closed** — the two-children parent reads the child set rather than the pointer, so no row is `active` and head-filtered-out at once.
- **The lineage half is genuinely opt-in.** Verified that `match_tables` with no flag publishes no pointers.

### Highlights

- **Every headline count in the PR reproduces exactly** — 15/13/1/1 links, 2 ambiguous, 537 → 528 heads, 5 → 10 pointers, EQT 68 → 66, MPLX 29 → 28, and all five #183 status counts. That is unusual, and it is the only reason a review this specific was possible.
- **The "Honest limits" section is honest.** It declines to close #170, names the real EQT/MPLX gap against the reviewed counts, flags the missing `dates_json`, and says the architectural home is #167 rather than the matcher. It also records the two non-obvious failures it hit with their mechanisms, not just their fixes. A reader should trust the numbers; the two places to discount are the #155 baseline column (72, not 65) and the premise that the effect is cross-filing.
- **The status cascade is a pure function of the row plus pre-computed signals**, which is why 542 rows could be re-derived in one call with no artifact surgery. That testability is a design achievement, not an accident.
- **`docs/schema.md`'s status section is precise enough to test against** — which is exactly how the reference-date discrepancy was found.

## Factor: repo coherence (scope: full diff)

Severities below are the coordinating reviewer's final calls, not the focused reviewer's proposals; several were downgraded from Important to Personal preference during the false-positive pass, because a defensible-alternative choice is not a maintainability defect. Every finding kept here cites the existing pattern it departs from — uncited coherence findings were dropped.

### Findings

- **[repo-coherence] The pass rewrites all 63 partitions without renewing the writer lease, unlike every other stage** — `src/cdt/matcher/core.py`:2430, 2480-2490 [proposed: Important] [confidence: verified]

  Verified directly. `match_pending_mentions` calls `renew()` before each shard rewrite at `core.py:317-318`, and its docstring says why: "``renew`` is called before each shard is rewritten; a full match pass can outlast the pipeline-writer lease TTL, and it must raise rather than let this run keep rewriting shards a lease thief now owns (#89)." A grep for `renew` or `lease` inside `apply_lineage_inference_pass` (lines 2430-2505) returns **nothing**, while `cli.py:697` holds the `matching` stage lease across the whole call.

  So the one function in the codebase that rewrites the entire published dataset in a loop is the one function with no lease guard, for a hazard the repo wrote a dedicated helper and an issue number for. The signature is also positional `artifact_root: str` with no `data_dir` and no `renew`, against the keyword-only `artifact_root` + `data_dir` + `renew` convention of `match_pending_mentions` (`core.py:262-272`), `classify_pending_items` (`classifier/core.py:184-192`), `itemize_pending_documents` (`itemizer/core.py:124-133`) and `extract_pending_items` (`extractor/core.py:1715-1725`).

  Compounding it, the rewrite is not atomic: if it fails midway, some partitions carry inferred lineage and the rest do not, and the rollup is internally inconsistent across the dataset with nothing recording that.

- **[repo-coherence] Nothing records that lineage was inferred, so an inferred artifact root is indistinguishable from a plain one** — `src/cdt/matcher/core.py`:377-393, 2430-2504 [proposed: Important] [confidence: verified]

  Every stage in this file-native repo writes a run manifest of what it did, and the match manifest records every knob — verified at `core.py:383-393`: `batch_size`, `membership_threshold`, `related_threshold`, `ambiguity_margin`, `schema_version`, `partitions_written`. A grep confirms **nothing** records `infer_lineage` anywhere. `docs/architecture.md:38` names "stage manifests are sidecar metadata" as a design property, and `docs/schema.md:300-306` documents the `runs/match/run_id=latest.json` surface.

  Worse than an omission: the pass runs *after* `match_pending_mentions` has already written the manifest, so the manifest describes a dataset state that the pass then silently changes. `amendment_inferred_by` is per-row and says nothing about rows where no rule fired — and per the High finding above, it does not survive the next ordinary match anyway.

- **[repo-coherence] `docs/schema.md` carries five factual errors about the field this PR is about** — `docs/schema.md`:207, 213, 265, 271, 273 [proposed: Important] [confidence: verified]

  Consolidating the doc errors found by three reviewers. This file is the contract the pending dashboard work will be written against, and the prior review of #178 found four errors in the same section, so it warrants suspicion rather than trust.

  1. **:213 stale vocabulary** — still says "`matured` is never extracted; the matcher derives it". The matcher no longer derives `matured`. The same stale claim appears in `src/cdt/extractor/core.py:248-250`. The line also says a planned retirement "is read by the matcher as pending", but the pending flag became `RetirementExpectation`. The file now contradicts itself 46 lines apart, and the stale words are exactly the terms this PR retired.
  2. **:207 `announced` carries no start date** — says an instrument whose status is `announced` "has not started and carries none". The new leg at `core.py:830-833` returns `announced` for a row with an explicit `start_date` ahead of the reference date (verified: `start_date=2027-01-01`, ref `2026-01-01` → `('announced', None, '2027-01-01', None)`).
  3. **:265 the `announced` table row** — says "the newest decisive event is the announcement", but that same leg publishes `announced` with no announcement event at all and a null `status_source_mention_id`.
  4. **:273 the `status_date` enumeration** — lists four sources and says "Null for a `closed` status derived from lineage rather than an event", but the final leg returns `"active", None, None, None` (`core.py:836`), so an `active` row with no dates and no events has a null `status_date` the enumeration excludes. The same line's "An `announced` status is dated no later than the filing that announced it" is false for the new leg, which dates it to a start the corpus has not reached.
  5. **:271 the reproducibility guarantee** — "The reference date is the newest filing date among this run's mentions, so a rerun over the same inputs reproduces the same statuses". Per the Important finding above, this is true of the post-pass and false of the normal shard path, where it is the newest date *in that shard* across 63 shards spanning 2026-05-05 to 2026-09-08.

  Separately, `docs/architecture.md:197-203` still lists the matcher's inputs as names, dates, amounts, lender signatures and one-hop lineage cues — item text is not among them, and the PR body itself calls that addition "architecturally new". `docs/architecture.md:40` advertises "the same pipeline can target either local paths or `s3://` URIs"; `read_item_texts` is the first place that is untrue.

- **[repo-coherence] `_iso` is a third copy of month-name date parsing, and the weakest of the three** — `src/cdt/matcher/lineage_inference.py`:73-100 [proposed: Important] [confidence: verified]

  Two existing helpers already do exactly this: `normalize_date` (`matcher/core.py:2116-2145`) and `normalized_date_from_text` (`extractor/core.py:3675-3714`) with `MONTH_MAP` at `extractor/core.py:407-419` and `iso_date_from_parts` at `:3793`. The extractor's version is strictly more capable — optional comma, `March 5 , 2026` spacing artefacts, and a validity check via `is_valid_iso_date`.

  This is more than duplication: reusing the existing helper would **fix** the case-sensitivity recall hole reported under correctness, because the extractor's `MONTH_MAP` path does not depend on a capitalized first letter. `matcher/core.py:26-32` already imports from `cdt.extractor.core`, so the import costs no new coupling.

  Caveat on the fix, which is real work rather than a rename: `core.py:33` imports `lineage_inference`, so the new module cannot import these helpers back without an import cycle. The cheap route is for `apply_lineage_inference_pass` to pass normalized values in alongside the rows.

- **[repo-coherence] `_name_rank_and_stem` re-implements the matcher's name canonicalization, divergently** — `src/cdt/matcher/lineage_inference.py`:58, 103-113 [proposed: Personal preference] [confidence: verified]

  `NAME_NOISE = {"the","a","an","that","certain"}` and the `lower().split()` normalization duplicate `normalize_name_fingerprint` (`matcher/core.py:2148-2162`), `NAME_STOPWORDS` (`:2286`) and `name_fingerprint_tokens` (`:2299-2306`). The two stopword sets overlap on only `{the, certain}`. Because the new code works off the raw `name` column rather than a fingerprint, it keeps punctuation and coupon spacing that `normalize_name_fingerprint` deliberately strips, so `"Credit Agreement, dated…"` and `"Credit Agreement dated…"` produce different stems while the matcher's own comparison treats them as the same name. Same import-direction caveat as above.

- **[repo-coherence] The new stage entry point is not exported from `cdt.matcher`, so the CLI reaches into `.core`** — `src/cdt/cli.py`:57 [proposed: Personal preference] [confidence: verified]

  `src/cdt/matcher/__init__.py` is untouched, so this is an omission rather than a decision. Every other stage exports its entry points through the package `__init__`, and `cli.py:49-56` imports the rest of the matcher that way — six lines above the new `from cdt.matcher.core import apply_lineage_inference_pass`. The only existing `.core` reach-in is `pipeline.py:46` for a constant, not an entry point.

- **[repo-coherence] `--infer-lineage` stops at the CLI, with nothing recording that as a deliberate scope choice** — `src/cdt/cli.py`:277-286, `src/cdt/pipeline.py`:107-109, 387-425 [proposed: Personal preference] [confidence: verified on the pattern, medium on intent]

  There is no precedent for a `cdt match` knob that does not reach the pipeline: `--batch-size`, `--force`, `--strong-match-threshold`, `--loose-match-threshold` and `--ambiguity-margin` all have `PipelineConfig` fields (`pipeline.py:92, 101, 107-109`), `pipeline_parser` mirrors (`cli.py:331-337`), and are forwarded from `orchestrator.py:321` and `:393`. A grep for an "experimental" gating idiom across `src/` and `docs/` found none, so that is not an established category here.

  Given the PR states the right long-term home is #167, stopping at the CLI is defensible — but nothing says so, and the deployed daily path can never produce the instruments the PR measured. One clause in the flag help would close it: "not part of `cdt pipeline`; evaluation only until #167 lands".

- **[repo-coherence] Broad `except Exception` around a dataset read has no precedent at this boundary** — `src/cdt/matcher/core.py`:2419-2423 [proposed: Personal preference] [confidence: verified]

  The *idiom* matches the house style — `except Exception:  # noqa: BLE001 - <why one unit must not stop the run>` appears at `sixk/triage.py:455` and `extractor/batch.py:715, 782, 832`. What is novel is swallowing a **dataset read**: tolerance in this repo lives inside `storage.read_table` (`storage.py:448-476`), which already handles the only two realistic failures here — absent file, and a partition written before a column existed (#69) — and lets genuine corruption propagate. Every other partition read in the matcher and extractor goes through it unguarded (`matcher/core.py:289, 324-330`; `extractor/core.py:2038`). The sites that do swallow either log the exception or interpolate the error object; this one drops the cause entirely, so the rule silently degrades to "fewer links". The S3 fix subsumes this.

- **[repo-coherence] `amendment_inferred_by` is appended at the end of the column list, 38 columns from the pointer it explains** — `src/cdt/matcher/core.py`:117 [proposed: Personal preference] [confidence: verified]

  `DEBT_INSTRUMENT_COLUMNS` groups provenance immediately after the value it qualifies — `status`/`status_subtype`/`status_date`/`status_source_mention_id` (`:86-89`), `name`/`name_source_mention_id` (`:94-95`), and the same for seven more fields (`:96-114`) — and the lineage pointers are a block at `:79-85`. This PR's own `status_subtype` follows the convention correctly, which makes the inconsistency internal to the diff. Moving it under `amendment_of_debt_instrument_id` at `:80` would also put it where the schema doc entry belongs.

- **[repo-coherence] The new test fixtures hand-roll column subsets and set a column that no longer exists** — `tests/test_matcher_lineage_inference.py`:13-48 [proposed: Personal preference] [confidence: verified]

  `mention()` and `instrument()` build dicts of 15 and 8 keys. `tests/test_matcher.py:20-26` sets the opposite convention and its docstring says why: "Seeded from `DEBT_INSTRUMENT_MENTION_COLUMNS` so every published column is present: `prepare_mention` reads them all with `row.get`, so a column this fixture forgot silently arrived as None and a rename went unnoticed." The new fixture calls the same `prepare_mention` and reintroduces that exact failure mode — visibly: it sets `"lenders_known_incomplete": False` (`:29`), a column #178 replaced with `lender_disclosure`. `grep -rn lenders_known_incomplete src/` returns nothing.

- **[repo-coherence] The status cascade stayed in `core.py` while lineage got its own module** — `src/cdt/matcher/core.py`:553-927, 1997-2047 [proposed: Personal preference] [confidence: medium]

  The lifecycle code is now ~425 lines of `core.py`'s ~2,500 and is cohesive, touching the rest of the module only through `coerce_optional_text`, `mention_recency_key` and `PreparedMention`. This PR demonstrates the seam by creating `matcher/lineage_inference.py`, and the repo has precedent for stage siblings (`extractor/batch.py`, `itemizer/extract.py`, `sixk/windows.py`). A `matcher/lifecycle.py` would also unify the two halves of one concept that currently sit 1,400 lines apart (`RetirementExpectation` at `:572`, `ExpectedDates` at `:2001`). A defensible "not in this PR" — but worth a sentence or a follow-up issue, since the inconsistency is internal to the diff.

- **[repo-coherence] `INSTRUMENT_STATUS_VALUES` is enforced only by a test, and neither new vocabulary is importable** — `src/cdt/matcher/core.py`:558-568 [proposed: Personal preference] [confidence: verified]

  Placement is right and `CLOSED_STATUS_SUBTYPES` is *derived* from `TERMINAL_STATUS_EVENTS` rather than re-listing it, which is good. But the only consumers are `tests/test_file_native_stages.py:6360-6375`; the cascade returns bare literals at eight sites, and neither constant is in `matcher/__init__.py.__all__`. The repo's precedent is to enforce a published vocabulary at the boundary — `LENDER_DISCLOSURE_VALUES` (`extractor/core.py:217`) is checked in `coerce_lender_disclosure` (`matcher/core.py:2072`) — and to export one for consumers, as `POTENTIALLY_RELEVANT_ITEM_NUMBERS` is. Mitigating: `docs/schema.md:259-272` does table both vocabularies, so a dashboard author reading the schema doc does not have to guess the five strings. Documented, just not importable.

- **[repo-coherence] The UI-contract paragraph breaks the column bullet list** — `docs/schema.md`:276 [proposed: Personal preference] [confidence: verified]

  The new "Presenting `status` (UI contract)" paragraph is inserted between the `status_source_mention_id` bullet and the `first_seen_filing_date` bullet, ending the markdown list and starting a second one. The file's convention is narrative *after* the columns and after the `Primary key:` line — see the `mentions` section at `:229-231`.

### Verified clean

- **New module placement and naming.** `src/cdt/matcher/lineage_inference.py` as a `core.py` sibling matches how every other stage splits, and the module docstring's "problem → why here → three rules → guards" shape mirrors `sixk/__init__.py`.
- **Test file placement.** `tests/test_matcher_lineage_inference.py` follows `test_itemizer_extract.py` / `test_sixk_triage.py`; the status tests went into `tests/test_file_native_stages.py`, where all 51 pre-existing `apply_lifecycle_rollup` assertions already live.
- **Constant placement** for `INSTRUMENT_STATUS_VALUES` / `CLOSED_STATUS_SUBTYPES` (directly beneath `TERMINAL_STATUS_EVENTS`) and `EXPECTED_RETIREMENT_KINDS` (next to its only reader).
- **`status_subtype`** is placed next to `status`, documented in the same bullet voice as its neighbours, and threaded through the rollup rather than joined at publish time.
- **Reuse in the pass.** `apply_lineage_inference_pass` reuses `read_dataset`, `write_partition_table`, `shard_for_cik`, `prepare_mention`, `debt_instruments_root`, `mention_cluster_edges_root` and `apply_lifecycle_rollup`; the partition-write half is consistent with `match_pending_mentions:347-356`.
- **`self: ClassName`** on `RetirementExpectation.unfulfilled_at` matches this file's local convention (`ClusterProfile.add_member`, `CandidateScore.base_match_via`), not the `self: Self` used elsewhere — locally correct.
- **Comment density and voice.** 5.1% comment lines in the new module against 3.8% (`matcher/core.py`), 5.0% (`extractor/core.py`) and 5.8% (`sixk/triage.py`) — in band, and the comments name the motivating issue and explain *why* a branch exists.
- **No dead weight of the usual kinds.** `grep -rn TODO src/cdt/` is empty repo-wide and the new code adds none; no commented-out code, no debug logging.
- **The publisher needs no change.** `write_final_output_tables` (`pipeline.py:490-545`) reads whole datasets via `FINAL_OUTPUT_TABLES`, so the two new columns publish with no publisher edit — correctly relied on.
- **The `expected_closing` rewrite claim at `docs/schema.md:207` is accurate** — `extractor/core.py:4235` does `payload["kind"] = "closing" if kind == "expected_closing" else kind`.
- **`status_subtype`'s doc matches `CLOSED_STATUS_SUBTYPES` exactly**, and the "latest end date governs" and "undated plan blocks indefinitely" claims both check out against `core.py:806-824`.

## Factor: performance and data safety (scope: full diff, measured against real artifacts on disk)

All work done on copies in the scratchpad; `data/` was left unmodified (`find data -newermt '-90 minutes'` → empty).

### Findings

- **[data-safety] A partially readable edges dataset silently zeroes observation columns on rows in unrelated partitions** — `src/cdt/matcher/core.py`:2452-2461, 2476-2478 [proposed: Important, arguably High] [confidence: measured]

  The guard is all-or-nothing: `if instruments.empty or edges.empty or mentions.empty: return`. A *partially* readable edges dataset passes it, and `apply_lifecycle_rollup` then unconditionally overwrites `mention_count`, `document_count`, `first_seen_filing_date`, `last_seen_filing_date`, `status` and `status_subtype` from `member_groups` — which in the pass is derived **only** from the edges dataset, with no fallback and no cross-check against the instrument rows it is about to overwrite.

  Measured by deleting exactly one `mention-cluster-edges/cik_shard=0030/part-0000.parquet`, holding the members of 78 of 542 instruments:

  ```
  rows before 542, after 542           (no row loss)
  mention_count:          changed on 78 rows   (all 78 -> 0; before, 0 rows had 0)
  document_count:         changed on 78 rows
  first_seen_filing_date: changed on 78 rows   (-> None)
  last_seen_filing_date:  changed on 78 rows   (-> None)
  status:                 changed on 25 rows
  status_subtype:         changed on 16 rows
  ```

  14% of the published dataset lost its observation facts, silently, exit code 0. In the normal match path the equivalent damage is bounded to the one shard being rebuilt and membership is re-derived from mentions; here the pass rewrites **every** partition, so one unreadable edges partition corrupts published rows in partitions it has no business touching. That blast radius is a property of the whole-dataset rewrite this PR introduces.

  Fix: a cheap invariant before writing — assert every instrument id in `rows` appears in `member_groups`, or that `sum(mention_count)` did not fall — and abort rather than publish.

- **[data-safety] `amendment_inferred_by` flips physical parquet type between `double` and `string` from run to run** — `src/cdt/matcher/core.py`:117, 2480, 2486 [proposed: Important] [confidence: measured]

  Because no key is ever set for the column in the normal path, `reindex(columns=DEBT_INSTRUMENT_COLUMNS)` materializes it as an all-NaN `float64` → parquet `double`. The pass sets real strings → `null`/`string`.

  ```
  after `cdt match --force`           : {'double': 63}
  after apply_lineage_inference_pass  : {'null': 56, 'string': 7}

  published snapshot (write_final_output_tables path):
    after plain match (no --infer-lineage):  amendment_inferred_by: double
    after --infer-lineage pass:              amendment_inferred_by: string
  ```

  So `debt-instruments/latest.parquet` publishes the same column as float64 on one generation and string on the next — and because the provenance column is wiped by every ordinary match (the High finding above), it flips back and forth on every run. Any typed external table (Glue/Athena, a BigQuery external table, a strict loader) breaks on the flip. Nothing in-repo fails, because `read_dataset` is a pandas concat that tolerates it.

  Fix: the provenance carry-forward fix alone stabilizes this to `string` in both paths.

- **[performance] `read_item_texts` loads the whole corpus's item text when ~17% of it is reachable** — `src/cdt/matcher/core.py`:2407-2427, called at :2469 [proposed: Personal preference] [confidence: measured]

  Measured on the three largest classifications datasets on disk:

  | root | items | files | utf-8 text | dict resident | mean/item | wall |
  |---|---|---|---|---|---|---|
  | `data/genwindow-eval-apr` | 5,794 | 1,554 | 14.1 MB | 28.3 MB (peak 33.3) | 4,884 B | 11.06 s |
  | `data/heldout-eval` | 1,693 | 1,051 | 4.7 MB | 9.5 MB | 5,595 B | 7.75 s |
  | `data/genwindow-run-branch` | 364 | 277 | 1.2 MB | 2.4 MB | 6,622 B | 2.12 s |

  About **5.5 KB resident per item**. Ceiling: 10k items ≈ 55 MB (fine), 100k ≈ 550 MB (uncomfortable alongside the mentions index), 1M ≈ 5.5 GB (fails). The largest corpus on disk is 5,794 items = 33 MB, so **memory is not a problem today**. At ~2.7 classified items per CIK, the 100k-CIK list in `data/ciks/100K-ciks.txt` over a comparable window projects to ~270k items ≈ 1.5 GB, which is where it breaks.

  I/O is the actual current cost and the single most expensive phase of the pass:

  ```
  read debt-instruments 0.579s   read mention-cluster-edges 0.256s
  read mentions         2.142s   read_item_texts            2.221s
  prepare_mention x669  0.251s   infer_amendment_parents    0.274s
  apply_lifecycle_rollup 0.009s  build frame + group        0.008s
  ```

  Streaming per-CIK is **not** a drop-in — `classifications` is partitioned `date=/shard=`, not `cik_shard=`, so there is no per-CIK prefix. The cheap fix that gets most of the win is to keep only items reachable from `mention_index`:

  ```
  data/heldout-eval:         classified=1693  referenced by mentions=295  -> 17.4%
  data/genwindow-run-branch: classified= 364  referenced by mentions=304  -> 83.5%
  ```

  The 83.5% figure is from a window hand-picked for debt filings; 17.4% is the general-corpus number. Passing `keep={m.item_id for m in mention_index.values()}` cuts resident memory ~5.7× at zero I/O cost and removes the ceiling entirely. Three lines.

- **[performance] `DATED_REFERENCE` is re-run on the same item text once per child sharing that item** — `src/cdt/matcher/lineage_inference.py`:236-252 [proposed: Personal preference] [confidence: measured]

  ```
  rule3 (child,item) regex invocations=669 over distinct items=304   redundancy=2.20x
  chars scanned as written=2,867,808 vs distinct item chars=1,024,428 (2.80x)
  as-written    elapsed=0.278s  matches=69
  cached-per-item elapsed=0.112s
  ```

  0.166 s wasted on 542 instruments, scaling linearly at ≈0.31 ms/instrument, so ~31 s wasted at 100k instruments. Fix: hoist to a `matches_by_item` dict computed once over the distinct item ids. Same refactor would remove the sibling-adoption bug's surface area.

- **[data-safety] A degenerate `cik` lands in a different shard than the normal write path chooses** — `src/cdt/matcher/core.py`:2481 vs :304-306 [proposed: Opinion] [confidence: measured]

  Normal path: `mention_rows["cik"].fillna("").map(shard_for_cik)`. Pass: `frame["cik"].map(lambda value: shard_for_cik(str(value)))` — no `fillna`, so a missing value stringifies: `'' -> 0033`, `'nan' -> 0001`, `'None' -> 0049`. Injecting a null-`cik` row relocated it from `cik_shard=0000` to `cik_shard=0049`, with no row loss and no duplication (543 → 543, 0 dups).

  Unreachable through the matcher's own writer — `build_debt_instrument_rows:1576` skips rows with a null `cik`, and across all 14 `debt-instruments` roots on disk `nullcik=0` for every one. Recorded for symmetry only; one word (`.fillna("")`) closes it.

### Verified clean — measured

- **Rows cannot be dropped or duplicated in the normal case.** 542 rows / 63 partitions before → 542 rows / 63 partitions / 0 duplicate `debt_instrument_id` after. Re-verified after a plain rerun and after the null-`cik` injection.
- **No stale-file double-count from filename drift.** This was a specific pre-fan-out hypothesis and it is **refuted**. Both write sites call `write_partition_table` with no `filename=`, so both write `part-0000.parquet` (`storage.py:577`), and empirically every `cik_shard=` partition in all 14 `debt-instruments` roots on disk holds exactly one parquet file.
- **Partition-relocation duplication is real but pre-existing, not introduced here.** Rebuilding a legacy (pre-`67a4db4`, padded-hash) layout and running the pass gave `542 → 550 rows, 8 duplicated ids, 60 → 64 partitions`, because partitions whose entire content relocates are left behind stale and `read_dataset` concatenates everything under the prefix. But a plain `cdt match` on the identical legacy root produces **exactly the same** 550 rows / 8 dups, so this is a `shard_for_cik` `lstrip("0")` hazard (`datasets.py:637`), not this PR's doing. Worth recording separately: `ingest.repair_document_shards` (`ingest.py:885-955`) is the established convention for this class of problem and `debt-instruments` has no equivalent, and the `-dev` eval roots on disk are 100% mismatched under today's `shard_for_cik` (`genwindow-run-dev: rows=573 mismatch=573`).
- **A mid-rewrite failure cannot split a lineage family.** `offer()` refuses cross-CIK candidates, verified on real data: `pointers checked=80 cross-cik=0 families=469 multi-cik families=0` across three roots. Since partitions are CIK-sharded, parent and child always land in the same file. Injecting a failure after 20 of 63 partitions left `children whose parent still says is_lineage_head=True: 0`.
- **The pass is idempotent and converges after a partial failure.** After the interrupted run (1 of 15 links written), re-running gave `links=14, heads 536->528`, then `links=0, heads 528->528`, final `542 rows, 0 dups, inferred 15` — identical to the clean run. Filtering on a non-null pointer makes it naturally re-runnable.
- **No catastrophic backtracking in `DATED_REFERENCE`.** The alternation branches are mutually exclusive, and the behaviour is linear rather than exponential — `'prior'*N` doubling gives ms/MB of 1779, 2166, 2347, 2478, 2104, which is flat. Worst-case constant ~2.5 s/MB against ~14 MB/s on benign prose. The largest real item on disk is 72,853 chars → ≤0.18 s even at pathological density. No ReDoS risk.
- **`infer_amendment_parents` complexity is fine at the real `n`.** Largest single-CIK instrument count on disk is **86**; p90 is 5. Synthetic scaling shows `~O(k²)` per CIK with a tiny constant (`k=800` → 0.604 s) and linear in total rows (0.31 ms/row). At the real `k_max=86` the quadratic term costs ~0.02 s. The whole rule set ran in 0.274 s on 542 instruments; the `O(Σ kᵢ²)` term only becomes visible above roughly `k≈500` for one issuer, 6× the largest real filer. (Minor, Opinion: the inner loop rebuilds the `terms` set per `(child, other)` pair — 7,396 constructions for the k=86 CIK; precomputing per row makes it `O(k)`.)
- **`prepare_mention` over the full mentions dataset is cheap at the measured scale.** 669 rows in 236 files: 2.142 s read (9.1 ms/partition-file, I/O-bound), 0.251 s prepare (375 µs/mention), 1.40 MB resident (2,093 B/mention). Extrapolating: 100k mentions → 37 s / 210 MB; 1M → ~6 min / 2.1 GB, stacking with the item texts for a combined ~3.6 GB. Worth stating in the PR, not worth blocking on.
- **Boolean and numeric dtypes do not drift.** `is_lineage_head` is `bool` in all 63 partitions before and after and in the published snapshot both ways; `mention_count`/`document_count` stay `int64`; `status`, `status_subtype`, `lender_disclosure` stay `string`. Only `amendment_inferred_by` drifts.
- **Per-partition type variance is not a new break.** `pyarrow`/`pandas` directory reads of `debt-instruments` already fail on `origin/dev` roots ("Unsupported cast from string to null"), before and after this PR. The repo's own `read_dataset` tolerates all of it, so the type finding above is about the single published file, not the hive dataset.
- **Backwards compatibility is fine; the forward direction is the problem.** Reading a root that predates the new columns works — the pass added `status_subtype` and `lender_disclosure` cleanly and preserved all 542 rows with 0 duplicates.
- **The S3 finding, quantified:** `dated_reference` supplied **13 of 15 links (87%)** in the measured run, so on an S3 root the feature loses almost all its yield silently — the `LOGGER.warning` is per-unreadable-file and never fires on an empty result.
- **The dead-parameter finding, quantified:** the `infer_lineage=True` no-op on the largest classifications set on disk costs **11.06 s wall, 33.3 MB peak, 1,554 parquet reads, zero effect** — and it is paid *before* the mentions read, so it is paid even when `mention_rows.empty` returns early at `core.py:294`.

## Factor: test coverage and test quality (scope: full diff)

Mutation testing ran in an isolated detached git worktree at `876bc60`; the main checkout was never written to. `PYTHONPATH` was verified to win before any result was trusted — this repo's venv resolves the main checkout otherwise, so a "surviving" mutation could have meant testing unmutated code. Worktree baseline: `449 passed, 1 xfailed`. 50 mutations, applied one at a time, full suite each time, file restored after each.

**25 of 50 mutations killed (50%). The unevenness is the finding:**

| area | killed / total | rate |
|---|---|---|
| `derive_instrument_status` + `apply_lifecycle_rollup` legs (commit `876bc60`) | 17 / 24 | **71%** |
| `lineage_inference.py` rules and guards (commit `43dc6ae`) | 6 / 12 | 50% |
| disk IO and CLI wiring (`apply_lineage_inference_pass`, `read_item_texts`, `--infer-lineage`) | 2 / 14 | **14%** |

**The PR's own testing claim is verified and understated.** It says "five targeted mutations of the new legs each fail at least one test"; 17 mutations of the new status legs were killed, including every returned literal, both new guards, the latest-end-date rule and both directions of `unfulfilled_at`. The status-vocabulary commit is genuinely well tested. Every survivor of consequence is in the lineage commit's plumbing, which the PR body does not claim to have mutation-tested.

### Surviving mutations (25)

| id | mutation | file:line |
|---|---|---|
| M43 | `--infer-lineage` never calls the pass (`if False`) | `cli.py:717` |
| M44 | `--infer-lineage` **inverted** — pass runs when you omit the flag | `cli.py:717` |
| M27 | `read_item_texts` returns `{}` | `core.py:2418` |
| M45 | `read_item_texts` wrong directory (`classificationz`) | `core.py:2419` |
| M46 | `read_item_texts` wrong partition glob | `core.py:2420` |
| M29 | pass never writes the inferred pointer | `core.py:2474` |
| M30 | pass skips the rollup re-derive | `core.py:2478` |
| M47 | pass never writes partitions to disk | `core.py:2484` |
| M48 | pass passes `item_texts=None`, killing `dated_reference` | `core.py:2469` |
| M49 | `match_pending_mentions` stops loading item texts | `core.py:284` |
| M28 | `match_tables(infer_lineage=)` defaults to **True** | `core.py:419` |
| M32 | cycle refusal removed | `lineage_inference.py:275` |
| M42 | `offer()` already-points-here guard removed | `lineage_inference.py:194` |
| M33 | cross-CIK check removed from `offer()` | `lineage_inference.py:186` |
| M38 | ordinal chain `rank > prev_rank` → `>=` | `lineage_inference.py:227` |
| M39 | `MIN_CHAIN_MEMBERS` guard removed | `lineage_inference.py:221` |
| M40 | `prior_fact` ignores the `prior` mark entirely | `lineage_inference.py:132` |
| M41 | `_identity_dates` accepts any date kind | `lineage_inference.py:151` |
| M50 | `INSTRUMENT_STATUS_VALUES` gains a bogus member | `core.py:558` |
| M08 | announced-on-future-explicit-start leg → `active` | `core.py:833` |
| M10 | `past_every_end` boundary `<` → `<=` | `core.py:813` |
| M16 | duplicate expected closings `min` → `max` | `core.py:2037` |
| M20 | `instrument_has_started` boundary `<=` → `<` | `core.py:745` |
| M23 | explicit-start leg boundary `<=` → `<` | `core.py:816` |
| M25 | `planned_retirement_date` `>` → `>=` | `core.py:849` |
| M26 | `expected_start_for_instrument` newest → oldest mention | `core.py:882` |

### Findings

- **[tests] `apply_lineage_inference_pass` has no test, and four mutations make it a no-op with a green suite** — `src/cdt/matcher/core.py`:2430-2500 [proposed: Important, arguably High] [confidence: verified by execution]

  `grep -rn "apply_lineage_inference_pass" tests/` → 0 hits. It is the only function the CLI calls, it rewrites every `debt-instruments` partition on disk, and it re-derives the rollup. Surviving: gutting the pointer write (M29), deleting the rollup re-derive (M30), never writing partitions at all (M47), and disabling `dated_reference` (M48). M30 and M48 matter most — the rollup re-derive is what the PR says keeps `superseded_by`/`is_lineage_head`/`status` consistent, and `dated_reference` is 13 of 15 links. Either could regress to zero silently.

  Every defect in the data-safety and seam findings above lives in this untested function.

  A ~35-line test following the existing `test_match_pending_mentions_writes_match_datasets` pattern (`tests/test_file_native_stages.py:2340`) is sufficient. It was written, run (`1 passed in 0.63s`) and confirmed to kill M29, M30 and M47:

  ```python
  match_pending_mentions(artifact_root=tmp_path, batch_size=5)
  stats = apply_lineage_inference_pass(str(tmp_path))
  assert stats == {"links": 1, "heads_before": 2, "heads_after": 1}
  assert after["m-2"]["amendment_of_debt_instrument_id"] == "m-1"
  assert after["m-2"]["amendment_inferred_by"] == "ordinal_chain"
  assert after["m-1"]["is_lineage_head"] is False      # kills M30
  assert after["m-1"]["status"] == "closed"             # kills M30
  ```

  Graded Important rather than High because the HIPPO scale puts missing tests for new behavior at Important — but noting the doubt, since a demonstrated 35-line test catching three real regressions is unusually cheap for the risk it covers.

- **[tests] The `--infer-lineage` flag can be inverted and all 449 tests still pass** — `src/cdt/cli.py`:281, 717-724 [proposed: Important] [confidence: verified by execution]

  M43 (flag never runs the pass) and M44 (**inverted** — the pass runs when you *omit* the flag and is skipped when you pass it) both survive. `tests/test_cli.py:683` never passes the flag, and its `fake_match_pending_mentions` does not even accept an `infer_lineage` kwarg. The PR's central safety claim — "off by default, so the published contract is unchanged unless a caller opts in" — has zero executable backing.

  Fix: assert the parser default is `False` and `True` with the flag; monkeypatch `cli.apply_lineage_inference_pass` with a recorder and assert it is called exactly once with the flag and **not at all** without it. That kills both.

- **[tests] `test_match_tables_is_unchanged_without_the_flag` asserts nothing about the flag** — `tests/test_matcher_lineage_inference.py`:222-240 [proposed: Important] [confidence: verified by execution]

  Flipping `match_tables`'s default to `infer_lineage: bool = True` (M28) leaves the suite green, because `match_tables` never reads the parameter. The test's title and docstring describe a guarantee it does not touch. What it *does* assert is useful but different — that the two new columns publish as all-null, which is why it also killed M02.

  Fix: either rename it to what it verifies (`test_match_tables_publishes_the_new_lineage_columns_as_null`) and drop the docstring claim, or make it real by asserting the dead parameters are gone — a test that `match_tables(mentions, infer_lineage=True)` raises `TypeError` would at least pin their removal.

- **[tests] The cycle refusal and the `offer()` back-pointer guard mask each other, so neither is tested** — `src/cdt/matcher/lineage_inference.py`:193-195, 263-286; `tests/test_matcher_lineage_inference.py`:188 [proposed: Important] [confidence: verified by execution]

  Both M32 (delete the cycle detection) and M42 (delete the `offer()` "never point at something that already points here" guard) survive, because either guard alone rejects the test's 2-cycle. With the cycle block disabled, the probe returns `{}` — the link was never offered, because `offer()` rejected it at line 194. So the test named for the cycle refusal never reaches it, and the dedicated DAG loop only matters for cycles of length ≥ 3, which no test constructs.

  The assertion is also weak in form: `assert "i2" not in result or result["i2"][0] != "i1"` is a disjunction that passes whenever the key is absent for any reason.

  This matters more given the direction-by-sort-order High finding above: the cycle guard is exactly the code whose *behaviour* is wrong, and it is deletable with a green suite. Fix: add a 3-node case with a positive control in the same test, and tighten to `assert result == {}`.

- **[tests] `ordinal_chain`'s strictness is untested, and the same-rank mutation actively invents lineage** — `src/cdt/matcher/lineage_inference.py`:220-228 [proposed: Important] [confidence: verified by execution]

  M38 (`rank > prev_rank` → `>=`) and M39 (delete `MIN_CHAIN_MEMBERS`) both survive, because the only ordinal test uses three strictly increasing ranks (0, 2, 3). M38 is not benign:

  ```
  unmutated, two same-stem rows with no ordinal:  {}
  with rank >= prev_rank:                         {'b': ('a', 'ordinal_chain')}
  ```

  Two unrelated clusters both named "Credit Agreement", or both "Amended and Restated Credit Agreement" (both rank 1 — confirmed `_name_rank_and_stem` returns `(1, 'credit agreement')`), would be chained into a false lineage. One negative test covers both this and the rank-tie behaviour reported under correctness.

- **[tests] `prior_fact` never tests that the `prior` mark is what makes the link** — `src/cdt/matcher/lineage_inference.py`:116-134; `tests/test_matcher_lineage_inference.py`:76 [proposed: Important] [confidence: verified by execution]

  M40 — dropping `payload.get("prior") and` so every amount counts — survives. The existing test's mention carries a prior-marked `2000000000` and an unmarked `3000000000`; with the guard gone the unmarked value only offers the child to itself, which the `child_id == parent_id` check rejects, so the result is unchanged. The rule's entire premise — "the `prior` mark *is* the predecessor's term" — is unverified. Fix: add a case where an **unmarked** amount equals an earlier instrument's `principal_amount` and assert no link.

- **[tests] `_identity_dates`' date-kind filter is untested** — `src/cdt/matcher/lineage_inference.py`:150-154 [proposed: Important] [confidence: verified by execution]

  M41 — accepting any `normalized_date` instead of only `agreement` and `closing` kinds — survives, because the single `dated_reference` test resolves against `mention.start_date` and never against a `dates_json` payload. Widening the filter would let a predecessor be matched on its *maturity* date, which is one way a dated-as-of reference picks a wrong parent. (Note the interaction with the dead-`dates_json` finding: this filter is currently unreachable in production anyway.)

- **[tests] The vocabulary invariant test skips one leg and does not pin the vocabulary** — `tests/test_file_native_stages.py`:6354-6384 [proposed: Important] [confidence: verified by execution]

  Two gaps, both verified, and this qualifies the praise this test earned elsewhere in the review:

  1. Its regex `r'return "([a-z_]+)", ("[a-z_]+"|None)'` requires a **literal** subtype, so it silently skips the terminal-event leg `return "closed", cause, status_date, source` (`core.py:769`) — the second group there is the variable `cause`. Run against the real source: 10 return statements in the cascade, **9 captured**. M03 (that leg returning `"bogus"`) was caught only by an unrelated behavioural test. Because `assert returned, ...` only requires ≥1 match, the test would also stay green if most legs were refactored to variables.
  2. It pins `CLOSED_STATUS_SUBTYPES` to an exact set — good — but only asserts `status in INSTRUMENT_STATUS_VALUES`. M50, adding `"bogus"` to that set, survives, and nothing anywhere pins it. So the published vocabulary can grow without a consumer-visible failure, though `docs/schema.md` documents exactly five values.

  Fix: relax the regex second group and assert the captured count equals the number of `return "` statements; add `assert INSTRUMENT_STATUS_VALUES == {...}` alongside the existing subtype pin.

- **[tests] The `announced`-on-future-explicit-start leg has no test** — `src/cdt/matcher/core.py`:832-833 [proposed: Important] [confidence: verified by execution]

  M08 — that leg returning `"active"` — survives. This is a distinct behavioural branch, not a boundary nuance: it is the difference between publishing a not-yet-existing instrument as live and as announced. Every other new leg has a dedicated test. One `derive_instrument_status` call with `start_date` after `reference_date` and `event_result=None` closes it.

- **[tests] Boundary and tie-break conditions in the status cascade are uncovered** — `core.py`:745, 813, 816, 849, 879-884, 2037 [proposed: Personal preference] [confidence: verified by execution]

  Six survivors, all off-by-one or tie-break. M10/M20/M23: every date comparison is tested strictly inside its range, never at `reference_date == start_date` or `== terminal_date`. Since the reference date *is* the newest filing date in the run's mentions, equality is routine rather than a corner — an instrument whose start date is the newest filing date flips between `active` and `announced` under M23. M25: at equality (a same-day 8-K announcing and effecting a redemption) `>=` vs `>` flips `closed` to `expected_closed`. M16: `min()` on duplicate expected closings is called "the conservative read" but no test supplies two. M26: `expected_start_for_instrument` reverses to oldest-first with no failure.

  Graded Preference because each is a one-day shift rather than a wrong state, and the surrounding behaviour is well covered. A `parametrize` over `reference_date` ∈ {day before, exact day, day after} would kill M10, M20 and M23 together.

- **[tests] `test_instruments_of_different_issuers_are_never_linked` passes for the wrong reason** — `tests/test_matcher_lineage_inference.py`:207-219 [proposed: Personal preference] [confidence: verified by execution]

  M33 (delete the cross-CIK check in `offer()`) survives. The test uses the ordinal rule, but `stems` is keyed by `(cik, stem)` at line 217, so the two rows land in different single-member groups and no offer is attempted. Rules 1 and 3 iterate `by_cik.get(child's cik)`, so they are CIK-scoped too. The behaviour is correct and doubly enforced; the issue is only that the test's stated subject is not what it exercises.

### Verified clean — deleted and weakened tests

Every `-` line in `git diff origin/dev...HEAD -- tests/` was read. All removals are legitimate:

- `test_lifecycle_status_prefers_terminal_events_and_derives_matured` → renamed to `..._derives_expected_closed`, with assertions **strengthened** from `status == "terminated"` to the full `(status, subtype)` tuple plus a new `status_subtype is None` check.
- `test_a_pending_retirement_blocks_the_matured_leg` → renamed, and the dated half it used to conflate got its own new test. Net stronger — M11, M12 and M13 are all killed.
- `derive_instrument_status(row, [], {}, ...)` → positional `[]`/`{}` dropped: a genuine signature change, not a loosening.
- **The strict xfail was correctly closed, not deleted.** The `@pytest.mark.xfail(strict=True)` was removed from `test_two_amendment_children_leave_no_unreachable_parent` and converted to real assertions, *and* `assert parent["superseded_by_debt_instrument_id"] is None` was added. It passes, and M04/M17 confirm it is load-bearing. This is the right handling, and the honest way to close an xfail.
- The remaining `1 xfailed` (`test_a_planned_retirement_survives_a_newer_amendment_mention`) is pre-existing, unrelated and still strict.

### Highlights

- **`test_an_unstarted_announcement_still_blocks_a_retirement_it_funds`** is the best test in the PR. It drives `apply_lifecycle_rollup` with two interacting instruments and toggles only the expected-closing date across two rollup calls. It single-handedly killed M12, M19, M21 and M22 — pinning the announced-retirer guard, `instrument_has_started`, the `expected_start_date` contribution and `unfulfilled_at` at once, at the integration level where unit tests would have missed the wiring.
- **`test_a_planned_retirement_stops_blocking_once_the_corpus_passes_it`** is the only test distinguishing "re-checked against the reference date" from "latched forever", and the sole killer of M13 — the subtlest of the three fixes the PR claims.
- **`test_the_latest_end_date_governs_expected_closed`** tests the `terminal_dates[-1]` claim with both an included and an excluded end date; killed M09 and M24.
- **`test_expected_dates_reads_planned_starts_and_retirements_apart`** carries five fact shapes in one payload plus a malformed-input loop over `(None, "", "{not json", "[]")`. Killed M14 and M15.
- The unit/integration mix is right: 5 of the new status tests go through `apply_lifecycle_rollup` and 3 call `derive_instrument_status` directly. The integration-level ones are what caught the `announced_ids` and `instrument_has_started` wiring.

---

# Addendum, 2026-09-15 — drill-down on the four High findings

Added after re-examining the four blocking findings in depth at the author's request. This section contains **two corrections to findings recorded above**, three new findings, and one worked trace. Posted to the PR thread as [issuecomment-5680917079](https://github.com/dsi-rse/commercial-debt-tracker/pull/177#issuecomment-5680917079).

Two of the items below surfaced only because the author pushed for precision on terminology — specifically on what "parent" and "child" mean and on what "inherit" was doing in my description of rule 3. Both questions exposed substantive problems rather than wording problems, which is worth recording as a fact about the review: the abstract description of the rule concealed them.

## Terminology, pinned to the code (omitted above, and load-bearing)

If row **X** carries `amendment_of_debt_instrument_id = Y`, that asserts *"X amends Y"*.

| term | which row | meaning | pointer |
|---|---|---|---|
| **child** | X — the row holding the pointer | newer state, the amendment | starts here |
| **parent** | Y — the row pointed at | older state, the thing amended | ends here |

Direction is **child → parent, new → old**. The critical consequence, not stated in the original findings: **becoming a parent is what marks a row dead.** `apply_lifecycle_rollup` sets `is_lineage_head = not children`, so a row *with* children is not a head, and `derive_instrument_status` then returns `closed`/`superseded` for it. The child stays live and in the browse index; the parent is closed and collapsed beneath it.

## Correction 1 — the `first_seen_filing_date` guard is correct and must not be tightened

**What was recorded above:** that the guard is "inert for 12 of 15 links", listed under the `dated_reference` High finding, and a suggested fix to "compare identity dates, not `first_seen_filing_date`".

**Why that was wrong:** the framing implied a defective guard. `parent.first_seen > child.first_seen → reject` correctly permits equality, and permitting same-filing parent/child pairs is **required**:

- **Predecessor objects are same-filing by construction.** This project's extraction rules deliberately mint objects that exist only to carry a pre-amendment figure — the premise of #155 ("a facility amended N times is N+1 index rows"). Those rows are always first seen in the same filing as the successor describing them. Rejecting same-filing links would delete the one lineage signal that already works.
- **The corpus has a left edge.** A 2014 agreement may only ever be named in a 2026 filing. `first_seen_filing_date` records when the pipeline *heard about* an instrument, not when it existed.
- **Filing standards changed.** 8-K Item 1.01 postdates 2004, so older instruments are systematically first-seen late.

**The accurate statement** is narrower: `first_seen_filing_date` answers "did we hear about the parent *after* the child?", a weak sanity check — and it is also the *only* ordering check in the rule. For same-filing pairs it correctly abstains, and then nothing checks ordering at all.

**The fix is additive, not stricter.** Where both rows have a known identity date, require the parent's `<=` the child's. On the 15 links that rejects exactly 3, all genuinely backwards:

```
NMP     'Secured Promissory Note' (2023-02-24) --amendment_of--> 'A&R Secured Promissory Note' (2023-03-06)
NMP     'Secured Promissory Note' (2022-12-09) --amendment_of--> 'A&R Secured Promissory Note' (2023-03-06)
TIPTREE 'Second A&R Credit Agreement' (2022-10-21) --amendment_of--> 'A&R Credit Agreement' (2023-10-06)
```

Four further links have no child identity date, so the check abstains there too. It can only ever be partial, which is the correct shape given the corpus cannot be assumed complete.

## Correction 2 — the suggested `dated_reference` fix covers 2 of 13 links

**What was recorded above:** `referenced -= identity.get(child_id, set())`, presented as the fix.

**Measured reality**, classifying all 13 `dated_reference` links by whether the matched date is the child's own identity date:

```
SELF-REFERENTIAL (what that fix addresses):  2   [MPLX, MAXIMUS]
date distinct from the child's own:         11
```

The other 11 fail for reasons that guard cannot reach. The clauses, pulled from the item text:

- **PENN** — *"entered into an amendment (the "Amendment") to its **Second Amended and Restated Credit Agreement, dated as of May 3, 2022**"*, and later *"The maturity of both the Company's **term loan A facility** and **revolving facility** remains unchanged."* The trigger matched the parent's own **name**; the filing explicitly says the two facilities are unchanged, and the rule concludes they supersede the agreement.
- **NMP** — *"…(iii) the **Amended and Restated Secured Promissory Note, dated as of March 6, 2023**, between Orbital Infrastructu…"* An enumerated list; four instruments in it all attach to item (iii).
- **EQT** — *"repay all outstanding obligations for principal, interest and fees under, and **terminate, the Third Amended and Restated Credit Agreement, dated as of October 31, 2018**"*. A `retired_by` relation pointing the other way; published as `Prior Revolving Credit Agreement amendment_of Third A&R`.

**Revised conclusion: `dated_reference` is not fixable by a guard.** The rule would need to know which instrument a clause is about and what relation it asserts. Both facts are in the sentence; neither survives into what the matcher receives. That is exactly #167's `governing_agreement` property. Defensible options: drop the rule and keep `prior_fact`/`ordinal_chain` (costing 13 of 15 links, but those links are largely spurious), or hold the feature until #167.

## Correction 3 — the MPLX example was mischaracterised

**What was recorded above** (under the "no live row" High finding): that MPLX's live revolver is published as dead.

**Measured reality:**

```
dim::139d72f70  'unsecured revolving credit facility'  start 2022-07-07  mat 2027-07-07  $2B  active      seen 2022-07-12
dim::74e51184   '2022 Credit Agreement'                start 2022-07-07  mat None         $2B  terminated  seen 2026-04-13
```

Same start date, same principal: these are **the same facility in two clusters the matcher failed to merge**, and the corpus does record its termination in the second cluster. Closing it is therefore coincidentally the right *answer*. What is false is the published *relation* — an instrument marked as amending itself — and the surviving head is the stub with no maturity. Still a genuine defect (the site would render "superseded by 2022 Credit Agreement" for a facility that *is* the 2022 Credit Agreement), but not the stronger claim I made.

**PENN is the unambiguous case** and the trace below uses it; Rent the Runway is a second of the same shape (an `Amended and Restated Credit Agreement` superseded by an `incremental term loan facility` drawn under it).

This sharpens the diagnosis: **inference is partly papering over a clustering failure.** MPLX's underlying problem is #48/#122 — bank facilities go unmerged because the fingerprint path needs a coupon. Inference converts an honest "two singletons" into a confident-looking false lineage.

## Worked trace — PENN Entertainment 8-K, filed 2026-05-28

Every value from the artifacts in `data/lineage-test-off` plus a scratch copy run through the pass.

**Extraction output — the extractor was correct:**

```
mention dim::e5b735647ea  raw_id='i-1'  name='Second Amended and Restated Credit Agreement'
   start='2022-05-03'  maturity='2033-05-31'  principal='962500000'  status='amended'  amendment_of=None
mention dim::7e698aa3679  raw_id='i-2'  name='term loan A facility'
   start=None  maturity=None  principal=None  status=None  amendment_of=None
mention dim::92140aac222  raw_id='i-3'  name='revolving facility'
   start=None  maturity=None  principal=None  status=None  amendment_of=None
```

It gave the agreement its real terms, marked it `amended`, and emitted **no** `amendment_of` pointers — its considered answer was "there is no amendment relation here."

**What the regex extracts from the document:**

```
trigger captured : 'Amended and Restated'      <- matched the parent's own NAME
date captured    : 'May 3, 2022'  -> _iso -> '2022-05-03'
```

**Why all three rows reach the same answer.** `referenced` is computed from **item text**, a property of the document rather than of the instrument. All three mentions are in item `000110465926067476-1-01`, so each loop iteration scans identical text and independently arrives at an identical set. ("Inherit", used in the original description, was the wrong word — nothing is passed down; the scan is simply scoped to the item, not the mention.)

```
child candidate 'term loan A facility'   -> referenced={'2022-05-03'} -> intersects Second A&R identity -> OFFER
child candidate 'revolving facility'     -> referenced={'2022-05-03'} -> same document, same set        -> OFFER
child candidate 'Second A&R Credit Agmt' -> referenced={'2022-05-03'} -> self blocked at line 183; the
                                                                         two facilities have NO dates   -> no offer
```

Both guards abstain: each child has exactly one candidate parent, so the one-parent-per-child check sees nothing ambiguous, and nothing caps children-per-parent.

**Published rows, before → after:**

```
[child ] 'term loan A facility'
   amendment_of_debt_instrument_id   None    -> dim::e5b735647ea7d46becd23   <<< CHANGED
   amendment_inferred_by             nan     -> dated_reference              <<< CHANGED
[child ] 'revolving facility'
   amendment_of_debt_instrument_id   None    -> dim::e5b735647ea7d46becd23   <<< CHANGED
   lineage_family_id   dim::92140aac222...  -> dim::7e698aa3679...           <<< CHANGED
[PARENT] 'Second Amended and Restated Credit Agreement'
   is_lineage_head                   True    -> False                        <<< CHANGED
   status                            active  -> closed                       <<< CHANGED
   status_subtype                    None    -> superseded                   <<< CHANGED
   superseded_by_debt_instrument_id  None    -> None
   lineage_family_id   dim::e5b735647ea...  -> dim::7e698aa3679...           <<< CHANGED
```

A $962.5M credit agreement maturing 2033-05-31, described by the filing as amended and continuing, publishes as `closed - superseded` — superseded by two of its own components, which the same filing says are unchanged. The two rows that remain heads have `start_date`, `maturity_date` and `principal_amount` all null, so a head-filtered browse index shows two empty stubs and hides the real facility.

**After the next ordinary `cdt match`** (no `--force`, no `--infer-lineage`): `amendment_inferred_by non-null = 0 ; amendment_of non-null = 20`. All pointers kept, all provenance erased, and `derive_parent_links` re-seeds the wrong link as authoritative extracted evidence on every subsequent run.

## New finding — `superseded_by_debt_instrument_id` is singular where a split needs a list

**[correctness] The inverse amendment pointer cannot express a legitimate multi-successor split** — `src/cdt/matcher/core.py`:656-658 [Important] [confidence: measured]

```python
row["superseded_by_debt_instrument_id"] = next(iter(children)) if len(children) == 1 else None
```

"More than one child" is not necessarily ambiguity. A facility split into two tranches has two legitimate successors, and the correct answer is to name both; publishing null discards information that is available and correct.

The schema already carries the precedent: `retired_by_debt_instrument_ids` is a JSON **array** (`json.loads(retired_by)` at `core.py:786`). Only the amendment inverse is singular.

```
flag OFF: parents with 1 child=5,  with >1 child=0
flag ON : parents with 1 child=10, with >1 child=4
```

Zero multi-child parents before inference, which is why the singular column was survivable until now. Inference creates four and every one publishes null. Three of the four show as `closed - superseded`; the fourth is already `closed - terminated` from an extracted event, which takes precedence — this reconciles the "3 of 11" figure recorded earlier with the 4 counted here.

**Compounding interaction with #183.** Before this PR the status leg read only the pointer, so a two-child parent stayed `active` — the bug the #178 xfail recorded. #183 correctly fixed the *status* by reading the child set, but left the pointer singular. The resulting behaviour is that **the row is closed because it has two children, and unlinkable because it has two children**: the condition that closes it is the condition that makes it impossible to say what closed it. That is the mechanism behind `docs/schema.md:276` being false for 27% of superseded rows.

**Fix:** `superseded_by_debt_instrument_ids` as a JSON array, matching `retired_by_debt_instrument_ids`. A published-schema change, which reinforces the `MATCHER_SCHEMA_VERSION` finding.

## New finding — an inferred pointer is published with no evidence span, against the #154 rule

**[repo-coherence] Rule 3 promotes a document-scoped observation to an object-scoped assertion, with no citable evidence** — `src/cdt/matcher/lineage_inference.py`:236-252 [Important] [confidence: verified]

This is the principled form of the `dated_reference` objection and is stronger than "the regex is imprecise."

The IE step never bound `2022-05-03` to `term loan A facility`. It bound it to `Second Amended and Restated Credit Agreement`, correctly, and emitted `amendment_of=None` for all three objects. The matcher overrode that using a date found by scanning the document, and published a relation the extractor declined to assert.

**The codebase states the rule this breaks**, at `docs/schema.md:12` (#154, implemented by the PR directly below this one in the stack):

> Every evidence payload records `spans`: a list of `{tag_id, char_start, char_end, text}` whose offsets index the source item's `text` **exactly** — the extractor realigns model output whose whitespace drifted.

Every published value points at the characters it came from. The inferred `amendment_of_debt_instrument_id` points at nothing, and cannot, because no span was ever bound to that object.

**The distinction the codebase already draws is refuse vs. assert.** The matcher does parse text-like attributes elsewhere — `name_rates_are_compatible` pulls coupon rates out of name fingerprints — but at the use site (`core.py:1146-1152`) it is a blocker:

```python
if profile.normalized_name_fingerprints and not any(
    name_rates_are_compatible(mention.normalized_name_fingerprint, candidate_name)
    for candidate_name in profile.normalized_name_fingerprints
):
    continue          # refuses a merge
```

A false positive there costs a missed merge — conservative, and visible as a duplicate row. `dated_reference` uses the same class of heuristic to *create* a published relation, where a false positive silently rewrites history.

**Audited for other instances — one only:**

| rule | reads | object-bound? | verdict |
|---|---|---|---|
| `prior_fact` | `amounts_json` with `prior: true` on that mention | **yes** — extractor marked that amount on that object | not an overstep in binding; separate weakness is that amount equality across a CIK is not identity |
| `ordinal_chain` | the `name` column | **yes** — extractor-produced, object-bound | inference over a bound attribute; mild |
| `dated_reference` | raw item text, document-scoped | **no** | the overstep |

Across the whole matcher, `read_item_texts` is the only raw-document read: grepping text access in `src/cdt/matcher/` returns that function plus `span.get("text")` at `core.py:1924-1926`, and the latter reads extractor-produced evidence spans out of `parties_json` — already object-bound. So the PR body's "the matcher has never touched text" is exactly right, and this is the first instance.

## New finding — the rule systematically closes the row that has data

**[correctness] Parent selection requires an identity date, so attribute-rich rows are preferentially closed** — `src/cdt/matcher/lineage_inference.py`:232-252 [Important] [confidence: measured]

Matching a parent requires intersecting *the parent's* identity dates, and `_identity_dates` is effectively `mention.start_date` (the `dates_json` branch being dead). Therefore:

> To be selected as a parent a row must have an identity date. A row with no dates can only ever be a child.

Since being a parent is what closes a row, the rule systematically closes rows carrying data and keeps empty stubs alive. Across all 13 `dated_reference` links, counting populated values among `start_date`, `maturity_date`, `commitment_termination_date`, `principal_amount`, `interest_rate_pct`:

```
mean populated attributes  CHILD (kept as head, stays 'live'): 1.15 of 5
mean populated attributes  PARENT (marked closed-superseded) : 2.00 of 5

links where the PARENT is richer than the CHILD:   7/13
links where the CHILD has NO identity date at all: 4/13
```

Close to twice as likely to close the informative row as the empty one, and in 4 of 13 cases the surviving head has no dates at all. For a browse index whose purpose is "show the live head and collapse the history beneath it", that inverts the intended outcome.

Distinct from the direction errors recorded above: those are cases where the rule picked the wrong *order* between two real states. This is the rule preferring to demote whichever row is better documented, and it follows from how parents are matched — a property of the rule's shape rather than its thresholds.

## Revised sequencing

| | Action | Cost |
|---|---|---|
| Now | Carry `amendment_inferred_by` forward in `build_debt_instrument_rows` (one line) | — |
| Now | Child-viability check before the `closed`/`superseded` leg | — |
| Now | Drop **both** edges of a mutual pair instead of keeping an arbitrary survivor | 3 links |
| Now | *Add* an identity-date ordering check where both dates are known; do **not** tighten `first_seen_filing_date` | catches 3 inverted links |
| Now | `superseded_by_debt_instrument_ids` as a JSON array | schema bump |
| Decide | `dated_reference`: drop the rule, or hold the feature for #167 | 13 of 15 links |
| Keep | `prior_fact` and `ordinal_chain` are sound in principle — tie-break and same-rank fixes, not removal | — |

## Revised severity tally

The three new findings are all Important; the three corrections change no severity, and no finding was withdrawn. Corrections 1 and 2 revise the *suggested fixes* attached to existing High findings, not the findings themselves — `dated_reference` still fabricates links and the family-with-no-live-row defect still stands, now with a cleaner example.

| Severity | Was | Now |
|---|---|---|
| High | 4 | 4 |
| Important | 28 | 31 |
| Personal preference | 15 | 15 |
| Opinion | 4 | 4 |
