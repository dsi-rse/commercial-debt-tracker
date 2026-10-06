# Matching and lineage: design decisions and measurements

The matcher (`src/cdt/matcher/`) groups extracted debt-instrument
mentions into stable clusters within one CIK. It writes membership, related and
ambiguous edges, rolls each cluster up into one published row, and then runs a
corpus-wide pass (`src/cdt/matcher/lineage_inference.py`) that infers amendment
links the item-scoped extractor cannot express. This page records why the
thresholds and rules are what they are, and the measurements behind them. The
docstrings give the current contract. Each section below is headed by the code
name it explains, so you can search for it from the code.

## `DEFAULT_MEMBERSHIP_THRESHOLD`, `DEFAULT_RELATED_THRESHOLD`, `DEFAULT_AMBIGUITY_MARGIN`

A key-matched candidate (amount and start date both agree) scores
`0.75 + 0.25 * support_strength`. That puts the floor at the related threshold
(0.75), and a candidate reaches membership (0.90) only with support from the name
or from lenders. A name-path candidate scores exactly the membership threshold.
Two qualifying candidates within the ambiguity margin (0.05) count as a tie.

## `score_candidates_for_mention`: scoring families

- **amount_start**: the amount and the start date both agree. The support
  family is `name` (a compatible name) or `lenders` (lender similarity
  `>= DEFAULT_LENDER_SUPPORT_THRESHOLD`), whichever is stronger.
- **Lenders cannot vouch across a name conflict.** Distinct facilities under one
  credit agreement share the amount, the start date and the lenders, so shared
  lenders support a membership only when the two names do not actively
  disagree.
- **name_fingerprint**: the keys do not match, but the name is compatible. Launch,
  pricing and closing 8-Ks for one offering drift on the amount (upsizes) and on
  the start date (pricing versus settlement), so an identifying name alone may
  attach a mention whose keys conflict. A non-identifying name needs
  `relaxed_keys_support_membership`, and only while its name class is at or
  below `NAME_CLASS_GATE`.
- **Generic-name clusters cannot claim an identifying mention.** A cluster
  whose every name is generic (`senior notes`) attaching a mention whose name
  individuates a series seeded the tie cascade that shattered GEO's note
  histories.
- **Same-item exclusion.** One item returns one object per instrument (an
  extractor invariant), so two mentions from the same item are two instruments
  by construction. This covers Gray Media's $70M add-on tap (#161), whose parent
  series stated no start date; Longevity Health's same-day twin notes (#131);
  and Kestra's four tranches. Launch, pricing and closing merges across filings
  are different items and are unaffected.

## `relaxed_keys_support_membership`

If amount and start date had to agree together, about half of all mentions
could not join anything: an announcement carries an amount but no closing
date, and an amendment carries dates but no principal. When the name is already
compatible, one agreeing key with no conflicting key is enough evidence.

## `NAME_CLASS_GATE`

FHLB Dallas files 67 `Consolidated Obligation Bonds` mentions with no dates and
repeated round amounts. Without the gate, a single amount collision merges
dozens of distinct bonds. Above two mentions that share one compatible name,
the name counts as generic for that issuer, and the relaxed key rule turns off.

## `name_class_sizes`

Synthesized prior states carry their successor's name verbatim, so they are not
other instruments bearing that name. Counting them pushed the class past the
gate and split a mention out of the cluster it had always joined (#203).

## `name_fingerprints_are_compatible`, `NAME_MIN_SHARED_TOKENS`, `NAME_CLASS_TOKEN`

Requiring equal names is too strict for the filing sequence. An announcement
names `senior notes due 2034`, and the closing names the same debt `7.500%
senior notes due 2034`. The rule is a token subset, with three guards:

- `NAME_MIN_SHARED_TOKENS`: without it, a bare `note` would subsume every note
  the issuer has.
- Coupons present on both sides must intersect.
- The differing tokens must not be only a class or tranche designator.
  Without this guard, Kestra Medical's four tranches (`Tranche A Loan`,
  `Tranche B Loan`, and so on) collapse into one instrument.

## `name_fingerprint_is_identifying`

A maturity year identifies an instrument as well as a coupon does. Within one
CIK, `Convertible Senior Notes due 2031` picks out one debt. If a coupon were
required, an announcement 8-K that has not priced yet could never attach to
its own closing.

## `name_rate_tokens`, `NAME_RATE_PATTERN`

`normalize_name_fingerprint` turns the decimal point into a token break, so the
pattern's separator is optional. The rate is returned as a canonical number,
not as the matched text. Matching the raw token compared only the fractional
digits: `4.375%` and `3.375%` both became `375%`.

## `normalize_name_fingerprint`

Filings write both `4.375%` and `4.375 %`. The punctuation pass would turn the
space into a token break, so the gap is closed first.

## `normalized_end_date_for_matching`, `end_dates_are_compatible`

A year-only maturity (`due 2030`) is synthesized to `2030-12-31` on the way
into the dataset. Comparing that synthesized day would either invent precision
or force every genuine December 31 maturity to be treated loosely (#128).
Name-derived year-ends therefore collapse to the year. Other derived values
(the month-end from `due April 2033` (#164), a full date embedded in a name, or
start plus tenor (#166)) collapse to the month. Stated dates keep their day.

## `canonical_maturity_fields`

Every post-closing `due 2030` mention re-introduces the synthesized year-end.
With selection by recency alone, a name-derived `2030-12-31` outranked the
closing 8-K's stated `2030-07-01` (#162). The newest stated maturity now wins.

## `resolve_candidates`: name-only ties

A mention that ties several existing clusters on its name belongs to at most
one of them. Seeding another cluster is never right, and it guarantees that
every later mention of the series ties too. That cascade produced 263 ambiguous
edges on the 2026-09 window. The tie-break order is: the exact name, then a
live cluster over a retired one, then the largest cluster, then the id.

## `TERMINAL_STATUS_EVENTS`, `apply_lifecycle_rollup`: no lifecycle status

The matcher derives no lifecycle `status`. Answering "is this borrowing still
alive" needs a notion of now. Here, now could only be the run's scope, which
would make the output depend on the run rather than on the filings. The
publisher derives status at publish time against an explicit `asOf` (#196).
Terminal events on mentions are used only to break name-only ties toward the
live obligation.

## `apply_observation_columns`

The rollup and the lineage rules share `first_seen_filing_date`: the lineage
rules read it as both the ordering guard and the chain sort key, and the rollup
rewrites it from the member edges. If the rollup ran only after inference, a
pass could infer against a value that it then overwrote. Pass N+1 would then
see a different corpus than pass N: on a row whose members are gone, pass 1
yields one link and nulls the column, and pass 2 adds a link that pass 1
refused. Rows like that arise naturally. Re-extracting an item mints a new
content-hash mention id, the old member edge is never deleted, and the old
instrument survives with `mention_count` 0. So the observation columns are
recomputed both before and after inference (#211).

## `derive_parent_links`

- **Each pointer kind is judged independently.** Ambiguity within one kind
  (two amendment parents) publishes nothing for that kind. Other kinds are
  unaffected: an instrument split from one predecessor and later retired
  records both. Nulling every column whenever a second kind appeared discarded
  lineage that was unambiguous on its own (#130). Retirers form a list,
  because several instruments jointly retiring one obligation is a legitimate
  state of the world.
- **The carried amendment pointer is a fallback, not a candidate.** Seeding it
  alongside the extracted pointers put a guess and a fact in one set, and the
  ambiguity guard then threw both away. On `data/lineage-verify`, a backfill
  plus one plain match published 19 pointers and 539 heads, against a clean
  rebuild's 22 and 536, and three further matches did not recover them. The
  carried pointer is what keeps an inferred link alive across an ordinary
  rematch, because no mention names it. An ambiguous extracted set is a
  refusal and does not fall back.
- **`amendment_inferred_by` travels with its pointer.** It is cleared when the
  pointer changes. Otherwise a rematch would publish a guess that cannot be
  told apart from an extracted relation (#184).

## `build_debt_instrument_rows`: canonical members

A synthesized prior state carries its successor's filing date, so by recency
it is the newest member and would supply every canonical field. That would
rename a predecessor cluster after the amendment that replaced it, and give
`ordinal_chain` two rows of one rank, which makes it refuse the link (#203).
Model-emitted members therefore decide the canonical values, and synthesized
members decide them only when nothing else is available.

## `mention_sort_key`

Within one item, a synthesized prior state is processed before the amended
object it was minted from. If the amended object went first, the same-item
exclusion refused its own prior state. That left the prior state stranded as a
head, and in a cluster with two amended objects it produced two amendment
parents, and so no parent at all (#203).

## `party_dedupe_key`

Party dedupe keys on the extractor's `canonical_name`. Deriving it again from
the spans was worse: `normalize_party_text` strips legal-form words before the
longest span is chosen, so `NCL Corporation Ltd.` shrank to `ncl` and lost to
its own `NCLC` alias, and `EQT Corporation` lost to `Buyer Parent`. Over the
1,632 party clusters of one eval window, the two methods agreed on 1,595, and
where they differed the span-derived choice was the worse one. `lender_keys`
keeps the span-derived key on purpose. It feeds match scoring, and a dedupe
change must not re-score clusters.

## `_json_text`

`read_table` returns a requested column that a partition's file lacks (it
decides from the footer schema) filled with NaN. NaN breaks the
`str(row.get(col) or "[]")` idiom twice: it is truthy, and `str(nan)` is `nan`.
`_json_text`, and `_json_list` through `storage.json_column`, read such a value
as absent; `retired_by_json` goes through `_json_list`. On
`data/genwindow-run-dev`, 245 mentions partitions were written before
`retired_by_json`, `amounts_json` and `parties_json` existed. With these
guards, `cdt match` completes there, at 679 edge rows and 572 instruments. The sites
that feed `parse_cluster_list` (parties, lender signature) produce the same
answer with or without the guard. Over the twelve column shapes a parquet
column can produce (NaN, None, blank, every `MISSING_TEXT_VALUES` placeholder,
junk, a JSON object, and a list of scalars), the guarded and unguarded
readings agree. Those sites are guarded so that `PreparedMention` never
carries the text `nan` forward.

## `MATCHER_SCHEMA_VERSION`, `_stale_schema_forces_rematch`

Mention ids are content hashes. A schema change that alters the hashed payload
changes every id, so an incremental match over an older root keys surviving
clusters on ids the mentions dataset no longer contains. The rollup then
recomputes lifecycle columns from member lists that are empty after filtering.
The older clusters also hold the slots that freshly minted prior states would
take, so a cluster can end up with two members that name two different
amendment parents. On `lineage-verify` (recorded at version 4, code at
version 7), a backfill plus one plain match published 19 amendment pointers
and 539 heads, against a forced match's 22 and 536, and further plain matches
never recovered them. A rematch is deterministic local compute. Forcing one is
cheaper than refusing (which would wedge the scheduled daily run behind an
operator) and safer than proceeding (which publishes wrong rows with a
successful-looking log) (#208).

## `apply_lineage_inference_pass`

- **Why it is a post-pass.** `match_tables` sees 1 to 7 rows per call on a
  364-item window, so no rule there could see both states of one facility.
- **Why inferred pointers are re-opened.** `infer_amendment_parents` considers
  only null pointers, and a rematch carries pointers forward. Without
  re-opening them, a link that a tightened rule now refuses survives on every
  already-matched root. The EQT/EQM cross-borrower link was one such case: 14
  of 542 pointers differed from a clean rebuild (#204).
- **Shard assignment.** It must match `match_pending_mentions` exactly, null
  CIKs included. Otherwise a rewritten row lands in a second shard and the
  original copy remains.
- **Its own manifest.** The pass rewrites every partition the match manifest
  lists, so without a manifest of its own, the latest record of
  `debt-instruments` would describe a state that has since changed.

## `lineage_inference` module: scope

`instrument_relation` resolves `amendment_of` against the `raw_id`s of a
single item, so a pointer can only name an object in the same filing. A
replacement agreement is named in a later filing than the agreement it
replaces, so the lifecycle rollup gets no pointer to work from. On a 364-item
window, 537 of 542 instruments were lineage heads, and EQT published three
active revolvers at once (#170).

- **No text rules.** A value that the matcher derives from text carries no
  evidence span and cannot acquire one (#154). Item text is also scoped to a
  document, not to an object, so a text rule attributes a document-level
  observation to every instrument the filing names. A `dated_reference` rule
  resolved the "dated as of" date in a replacement clause against other
  clusters. On real filings it linked sibling tranches of one agreement, and
  whole enumerated lists, to a single parent. The evidence it needed (which
  object a clause is about) exists only at extraction time, so it belongs in
  the extractor as a `governing_agreement` property (#167).
- **No `prior_fact` rule.** The extractor mints the predecessor of a
  `prior`-marked amount as its own mention (#203), and the successor's
  `amendment_of` names it directly. The extractor also sees prior dates, which
  `PreparedMention` does not carry. As an inference rule, `prior_fact` produced
  1 link in 542.
- **Refusal over guessing.** A wrong pointer silently rewrites a published
  history, so any candidate set that does not single out one parent is
  refused.

## `_borrowers_disagree`, `BORROWER_SUFFIXES`, `GENERIC_BORROWER_PHRASES`

- **Why a borrower check.** The CIK check uses the filer's CIK, and that cannot
  separate two issuers' agreements named in one 8-K. EQT's own Third Amended
  and Restated Credit Agreement and EQM Midstream Partners' agreement both
  appear in EQT's 2024-07-22 filing, with the same name stem and ordinal. Both
  were offered as children of EQT's Second A&R agreement. That welded an
  acquired subsidiary's terminated facility into the parent's chain, and the
  two-child rule then nulled the real predecessor's `superseded_by` (#197).
- **Exact key match, not prefix.** Finance subsidiaries are usually named after
  their parent, so a prefix rule treats `EQT Corporation` and `EQT Midstream
  Partners, LP` as one party. `BORROWER_SUFFIXES` already makes `EQT` equal to
  `EQT Corporation` (#205).
- **Placeholders are silence.** The extractor records a party as the longest
  span it saw, so a filing that never uses the company's name records the
  borrower as `Issuer`. Read as a name, that turned `HSBC Holdings plc` versus
  `Issuer` into evidence of two different companies (#205).
- **Silence is not disagreement.** Refusing a link whenever a party is missing
  would drop ordinary links to the many mentions that never name a borrower.
- **Local JSON guard.** `_borrowers` parses `parties_json` itself rather than
  calling `matcher.normalize._json_text`. That was forced when both lived in
  one module that imported this one; since the split it is not, and the two
  could be unified. It returns the same answer on every input the helper
  handles.

## `infer_amendment_parents`: ordering guards

- **The `first_seen_filing_date` guard allows equality.** A minted predecessor
  is always first seen in the same filing as its successor. The corpus also has
  a left edge, and 8-K Item 1.01 postdates 2004. So a real predecessor is
  routinely first heard of no earlier than its successor.
- **`start_date` outranks the filing-date guard** when both rows carry one,
  because a filing date cannot order two instruments named in the same filing.
- **Ties are refused.** Every member of the preceding rank is offered, so a tie
  reaches the ambiguity check and is refused, rather than being broken by id
  order.
- **Same-rank states.** One restatement can have several published states,
  because the extractor mints a prior state and an amendment within a
  restatement keeps the ordinal. Those states already point at each other, so
  the replaced one steps aside.
- **Mutual pairs drop both links.** Keeping whichever direction iteration
  reaches first would publish a coin flip.
