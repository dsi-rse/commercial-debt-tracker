# Matching and lineage: design decisions and measurements

The matcher (`src/cdt/matcher/`) groups extracted debt-instrument
mentions into clusters within one CIK, as a pure function of that CIK's current
mentions. It writes membership, related and ambiguous edges, gives each cluster
an id (`src/cdt/matcher/identity.py`), rolls each cluster up into one published
row, and infers the amendment links the item-scoped extractor cannot express
(`src/cdt/matcher/lineage_inference.py`). This page records why the
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
- **`repeats_a_same_item_member`.** The invariant does not always hold: the
  extractor sometimes returns one instrument twice in an item (#319). A
  same-item mention may therefore join a cluster when its name is identifying
  and exactly equals a same-item member's, and neither its amount nor its start
  date conflicts with that member's. Gray Media's tap and Longevity's twins
  conflict on principal, and Kestra's tranches differ in name, so they still
  stay apart. A generic name keeps the exclusion, since one item can describe
  two term loans. The relation checks still apply. On the 2026-10-23 beta
  (`database/cdt-beta` run 20261023T204636541016Z), 93 items hold an
  identifying same-name pair with no conflicting key. 67 of those pairs are
  exchanges or refinancings in which one object is `retired_by` its namesake,
  and they stay apart. The rule joins 18.

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

## `name_fingerprints_are_compatible`, `NAME_MIN_SHARED_TOKENS`, `NAME_CLASS_TOKEN`

Requiring equal names is too strict for the filing sequence. An announcement
names `senior notes due 2034`, and the closing names the same debt `7.500%
senior notes due 2034`. Equal informative tokens (names that differ only by a
stopword, `the senior notes due 2034`) are compatible outright. Otherwise the
rule is a token subset, with three guards:

- `NAME_MIN_SHARED_TOKENS`: without it, a bare `note` would subsume every note
  the issuer has.
- Coupons present on both sides must intersect.
- The differing tokens must not be only a class or tranche designator.
  Without this guard, Kestra Medical's four tranches (`Tranche A Loan`,
  `Tranche B Loan`, and so on) collapse into one instrument.

The equal-token case used to fall through to the class guard, which found no
differing token and refused the pair (#239). A forced rematch of
`data/genwindow-run-dev` merges one split instrument (572 to 571 instruments);
`data/genwindow-run-branch` is unchanged. `MATCHER_SCHEMA_VERSION` was not
bumped, since no published column or hashed payload changes, so a root that
already split such a pair keeps it until `cdt match --force`.

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

## `mention_coupons`, `ClusterProfile.coupons`: the coupon guard

A cluster keeps every coupon its model-emitted members carry, and a mention
whose coupons miss that set is refused. The guard used to pass when any one
member *name* was rate-compatible with the mention. Because a coupon-less name
is compatible with every coupon, one `Senior Notes due 2027` member opened the
cluster to all series due that year (#318). EQT's cluster held the 3.900% and
the 6.500% notes due 2027. Once filings named both series in one item, the
same-item exclusion sent one of them to a second cluster, and the two series
then alternated between the two clusters from filing to filing.

A mention's coupons are those in its name. When the name has none, the
extracted `interest_rate_pct` stands in, unless the rate is floating. The name
wins because, where both exist, they disagree on 58 of 4,099 notes, and in
those the extracted rate is a step-up or current rate: the coupon plus one
point.

A synthesized prior state uses only its name's coupons. Its extracted rate is
its successor's, inherited because no `prior` rate is extracted, so after a
repricing it would refuse the prior state from the cluster it belongs to.

On the 2026-10-23 beta, multi-mention note clusters that mix extracted rates
fall from 64 to 11. All 11 remaining carry one coupon in every name; only the
extracted rate holds the step-up.

## `normalize_name_fingerprint`

Filings write both `4.375%` and `4.375 %`. The punctuation pass would turn the
space into a token break, so the gap is closed first.

Three further spellings of one name are folded (#320):

- **Worded coupons:** `5.25 per cent.` becomes `5.25%`. The 2026-10-23 beta has
  229 such names, mostly from 6-K issuers. No coupon was read from any of them,
  and only 5 had an extracted rate, so the coupon guard could not see them.
- **Leading currency amounts** are dropped: `$500 million 3.75% Notes`,
  `€1,000,000,000 2.125% Fixed Rate Notes`, `US$2 billion revolving credit
  facility`. There were 373 such names. An amount's tokens kept the name from
  being a subset of the same note's plain name. Only an amount with a currency
  mark is dropped. A bare letter prefix needs a currency symbol (`A$`, `C$`,
  `US$`), and a rand `R` with no symbol needs a magnitude word or grouped
  digits (`R1 000 million`). A leading coupon, year or class designator
  (`US 2027 Notes`, `A1 Notes`) therefore stays.
- **Plural instrument nouns** are singularized (note, bond, debenture, loan,
  facility): `Series B Senior Note Due 2032` against `Series B Senior Notes due
  2032`.

## The 2026-10-23 rematch: #318, #319, #320

A forced rematch of the beta mentions (7,184 mentions, 82 CIKs), with each fix
in turn switched off:

| Measure | Before | All three | Without #319 | Without #320 |
|---|---|---|---|---|
| Instruments | 4,035 | 3,985 | 3,996 | 4,068 |
| Note clusters mixing extracted rates | 64 | 11 | 11 | 11 |
| (CIK, coupon, maturity year) note keys over 2+ instruments | 254 | 208 | 210 | 229 |
| Excess note instruments on those keys | 329 | 248 | 253 | 303 |
| Ambiguous-candidate edges | 684 | 590 | 598 | 613 |

The coupon guard alone adds instruments, because it separates mixed series;
the name folding and the repeat rule take them back out. A read of 14 of the
123 new merges and 10 of the 71 splits found no wrong call. Each merge joined
spellings of one series. Each split separated distinct coupons, such as
Murphy's 4.875% and 7.500% notes due 2031, and Petrobras's 8.375%, 5.375% and
3.750% notes due 2021. Facilities are unchanged: 92% of them still have a
single mention.

`MATCHER_SCHEMA_VERSION` is not bumped, because no published column or hashed
payload changes. A root matched before this change keeps its clusters until
`cdt match --force`.

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
live cluster over a retired one, then the largest cluster, then the earliest
seeded (`seed_sort_key`).

## `TERMINAL_STATUS_EVENTS`, `apply_lifecycle_rollup`: no lifecycle status

The matcher derives no lifecycle `status`. Answering "is this borrowing still
alive" needs a notion of now. Here, now could only be the run's scope, which
would make the output depend on the run rather than on the filings. The
publisher derives status at publish time against an explicit `asOf` (#196).
Terminal events on mentions are used only to break name-only ties toward the
live obligation.

## `apply_observation_columns`

The lineage rules read `first_seen_filing_date` as both the ordering guard and
the chain sort key, so the observation columns are set from each row's members
before inference runs (#211): inference must read the dates of the rows it is
about to publish. Rows are built only from a CIK's current mentions, so every
row has members.

## `derive_parent_links`

- **Each pointer kind is judged independently.** Ambiguity within one kind
  (two amendment parents) publishes nothing for that kind. Other kinds are
  unaffected: an instrument split from one predecessor and later retired
  records both. Nulling every column whenever a second kind appeared discarded
  lineage that was unambiguous on its own (#130). Retirers form a list,
  because several instruments jointly retiring one obligation is a legitimate
  state of the world.
- **Only extracted pointers.** Inferred links are added afterwards
  (`apply_inferred_lineage`), so `amendment_inferred_by` is null on every
  pointer this function returns and names the rule on every pointer inference
  adds; a guess can never be told apart from an extracted relation (#184).
  Nothing is carried from a previous match: carrying the previous row's
  pointer as a fallback was what let a guess and a fact cancel out (19
  pointers and 539 heads after a backfill on `data/lineage-verify`, against a
  clean rebuild's 22 and 536).
- **`break_amendment_cycles`.** Extracted amendment pointers can
  form a cycle (A amends B, B amends A); every member then has a child, so the
  family has no head and drops out of head-only views (#267). One pointer per
  cycle is dropped: the one into the member first seen latest (earliest member
  mention by `mention_sort_key`, ties by id), since nothing amends an
  instrument that appeared after it. That member becomes the head of a pure
  cycle; in a chain that joins the cycle, the family still has exactly one
  head. Each break is logged. Inferred links refuse to close a cycle, so the
  final rows stay acyclic. No stored root under `data/` had a cycle (14 roots).

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

## `match_pending_mentions`: one CIK's mentions in, one CIK's rows out

Every matching rule works within one CIK: candidates must share the CIK, name
classes are counted per CIK, and the lender signatures, extracted pointers and
lineage rules are all per CIK. So the CIK is the unit, and the 64 `cik_shard`
files are only an I/O grouping. A run re-clusters, from all of their current
mentions, exactly the CIKs whose mentions changed (a digest per normalized CIK
in `runs/match/cik-digests/`), then runs lineage for those same CIKs. It never
extends clusters an earlier run built.

The digest covers each mention's id and every other column the matcher reads
that the id does not hash (`DIGEST_COLUMNS`: the mention-id columns are
`MENTION_ID_COLUMNS`). A digest of ids alone missed a change to
`amendment_of`, `retired_by_json`, `company_name` and the like, so the CIK was
never re-matched and its published rows went stale until a `--force` (#290).
All of those columns are short, so the scan stays cheap.

A shard's tables are written redirects first, then instruments, then edges,
then the shard's digests. Retired ids are worked out from the stored edges, so
a run that stops before the edges are written retires the same ids again;
writing the edges first lost that run's redirects for good (#291). A full
rebuild (`--force`, `--remint` or a stale schema) writes
`runs/match/rebuild-pending.json` before it starts and deletes it after the
manifest. Shards it finished already have current digests, so without the
marker the next ordinary run would skip the rest and publish a mix of content
and carried ids (#294).

The matcher used to extend existing clusters, scoring only mentions with no
member edge yet and then running a separate lineage pass over the whole
corpus. Nobody recorded why; it dates from the SQLite matcher's
`pending_only`/`batch_size` mode (`c752f63`), and there are no human edits for
history to protect. Its costs, measured by replaying stored roots in
different orders:

- `name_class_sizes` counted an existing cluster once through its mentions and
  again through its instrument row, so two halves matched forward gave 550
  clusters where a full match gave 542 on `genwindow-run-branch` (#258).
- Backfilling the newer half first gave 213 of 901 mentions a different
  instrument id on the pre-beta root, because an id was its seed mention's id
  (#259).
- Re-extraction mints new mention ids; the old member edge was never deleted,
  so the old instrument stayed published with zero members (#225).
- The lineage pass held every instrument, edge and mention of the corpus in
  one task: about 3.7 KB per mention on top of the frames, estimated at 4.5 GB
  or more at full scope, on a 4 GB task.

With the per-CIK design, on `genwindow-run-branch` (669 mentions, 193 CIKs) a
full match, two halves forward and the newer half first all give the same 542
clusters. Split at the median month, the newer half first gives 40 of 669
mentions a different carried id and reverse month order 138; after `--remint`
every order gives identical ids. A full rematch takes 1.8 s. A daily run whose mentions did not change reads the digest columns of
the mentions dataset and writes nothing.

**Ties never depend on ids.** Scoring ties are broken by the cluster's seed
sort key (its smallest member `mention_sort_key`), not by its id, so how ids
are assigned can never change a clustering decision.

**The unit key is the normalized CIK.** Local roots mix zero-padded and
unpadded CIKs; keyed on the raw string, one issuer would be two units. Reads
filter on every stored spelling of the CIK.

## `cdt.matcher.identity`: ids carried by overlap, minted from content

Options considered: the seed mention's id (not deterministic, changes with
arrival order and re-extraction); a pure content id such as a hash of the
earliest member (deterministic, but a backfilled older filing becomes the
earliest member and changes every URL); a separate persisted id map (another
store to keep consistent); and the chosen one, a content id when a cluster is
first seen, carried forward by member overlap.

- Overlap is counted on mention ids first. 8-K item ids survive
  re-extraction while mention ids do not, so a cluster that shares no
  mentions with a previous instrument can still pair with it through shared
  items. An item often holds several instruments, and every old instrument of
  a re-extracted item then shares that item with every new cluster from it;
  pairing on items alone swapped a revolver's and a term loan's ids in one
  item with no redirect (on `genwindow-run-branch`, renaming every mention id
  moved 246 of 542 instruments to another's id, #289). So an item-only pair
  also needs content agreement: how many of the previous instrument's name
  fingerprint, type, start date, maturity, rate and principal amount some
  member states (at least one), ranked before shared items. Without the
  amount, same-named tranches such as $1.6B and $690M convertible notes in one
  item tied and lost their ids (#302). With it, renaming every mention id
  moves no id and keeps 534 of 542; the 8 that get new ids agree on every
  compared field with another instrument of their item. A pairing still tied at its level
  (two clusters for one old id, or two old ids for one cluster) is ambiguous:
  neither side carries, and the old id gets no redirect, because a new id is
  safe and a wrong one silently opens a different instrument. 6-K item ids
  name the snippet span, so a re-windowed 6-K filing shares neither and gets a
  new id.
- Pairs are taken greedily, largest overlap first, each previous id used at
  most once, so when two clusters merge the larger keeps its id.
- An id no longer used gets a redirect to its best admissible pair's cluster;
  redirects follow a retired target, so every target is live.
- Carried ids depend on history: two roots built in different orders get the
  same clusters but can label them differently. `cdt match --remint` replaces
  every id with its content id, once, before the first public publish.
- The `di::` prefix keeps instrument ids distinct from mention ids (`dim::`).

## `name_class_sizes`

Synthesized prior states carry their successor's name verbatim, so they are not
other instruments bearing that name. Counting them pushed the class past the
gate and split a mention out of the cluster it had always joined (#203).

**One comparison per pair of distinct names.**
Counting compatible names pairwise over an issuer's mentions was O(n²), and
one 4,048-mention CIK took 31 s to match (#269). The count now compares each
distinct fingerprint once against the issuer's distinct fingerprints. That
alone did not make a large CIK fast: replicating the largest
`genwindow-run-branch` CIK to 4,048 mentions still took 36.6 s, almost all of
it the candidate loop calling the name rules for the same few hundred
fingerprints millions of times. Those rules are pure, so `compat` memoizes
them: 253 mentions 0.07 s, 1,012 mentions 0.30 s, 4,048 mentions 3.0 s.

## `MATCHER_SCHEMA_VERSION`, `_stale_schema_forces_rematch`

A schema change can change what the matcher writes for a CIK whose mentions
did not change, so the digests alone would leave that CIK stale; a plain match
over a root recorded at an older version re-clusters every CIK. A rematch is
deterministic local compute. Forcing one is cheaper than refusing (which would
wedge the scheduled daily run behind an operator) and safer than proceeding
(which publishes wrong rows with a successful-looking log) (#208).

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

## `_borrowers_disagree`, `BORROWER_SUFFIXES`, role-only borrowers

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
- **Placeholders are silence.** A filing that never uses the company's name
  records the borrower as `Issuer`. Read as a name, that turned `HSBC Holdings
  plc` versus `Issuer` into evidence of two different companies (#205). The
  extractor's `is_role_only_text` decides what is a placeholder, the same
  check that picks party `canonical_name` and lender keys (#323).
- **Silence is not disagreement.** Refusing a link whenever a party is missing
  would drop ordinary links to the many mentions that never name a borrower.
- **One JSON guard.** `_borrowers` reads `parties_json` through
  `matcher.normalize._json_list`, so an absent, missing, unparseable or
  non-list value means no borrower named.

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
