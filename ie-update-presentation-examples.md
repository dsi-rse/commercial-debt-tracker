# IE schema update: single-slot fields → kind-typed fact lists

Evidence from the A/B pair `data/genwindow-run-dev` (old schema) vs
`data/genwindow-run-branch` (new schema). Same generalization window, same
filings, same model: 312 filing items, 676 old-schema mentions, 669 new-schema
mentions, 298 items in common. Every example below was checked against the
source filing text.

**The change.** Old schema gave each instrument one `amount`, one `start_date`,
one `end_date`, and two untyped party buckets (`lenders`,
`other_interested_parties`). New schema gives it `amounts[]`, `dates[]`, and
`parties[]` — lists of typed facts, each with a `kind`/`role`, its own evidence
span, and flags (`prior`, `expected`, `as_of_date`). The flat published columns
are then *derived* from the lists by the post-processor instead of being
arbitrated by the model.

---

## Example 1 — the requested case: a draw published as the facility's size

**Cognizant Technology Solutions, 8-K Item 2.03, 2026-05-21**
(`000105829026000020-2-03`)

> "the Company provided notice to the lenders **to borrow $1 billion** to be
> funded on May 20, 2026 **under the revolving credit facility of the Credit
> Agreement, dated as of October 6, 2022** and as amended by Amendment No. 1
> dated as of April 18, 2024 ... and JPMorgan Chase Bank, N.A., as
> administrative agent"

The filing never states how big the facility is. $1B is what Cognizant is
drawing on it.

| | Old schema | New schema |
|---|---|---|
| amount | `1000000000` | `principal_amount = None` |
| amounts[] | — | `draw $1,000,000,000` |
| start_date | `2022-10-06` | `2022-10-06` |
| dates[] | — | `agreement 2022-10-06`, `amendment 2024-04-18`, `announcement 2026-05-15`, `closing 2026-05-20 (expected)` |
| parties | lenders: `[]`; other: `[JPMorgan Chase Bank, N.A.]` | `borrower` Cognizant Technology Solutions Corp., `borrower` Cognizant Worldwide Ltd., `lender` "financial institutions" *(collective)*, `agent` JPMorgan Chase Bank, N.A. |

The old row reads as "a $1B revolving credit facility." With only one amount
slot the model had to put *some* number there, and the only number in the
document was the draw. The new schema lets it say "this is a draw" and leave
the instrument's size unknown — which is the truth.

## Example 2 — a balance published as the note's face amount

**EQT Corp, 8-K Item 8.01, 2024-11-25** (`000110465924122342-8-01`)

> "**As of November 25, 2024, the outstanding aggregate principal amount of the
> 2025 Notes was $400.0 million** and the outstanding aggregate principal
> amount of the 2026 Notes was $500.0 million."

| | Old schema | New schema |
|---|---|---|
| 6.000% Senior Notes due 2025 | `amount = 400000000` | `outstanding_balance $400.0M`; `principal_amount = None` |
| 4.125% Senior Notes due 2026 | `amount = 500000000` | `outstanding_balance $500.0M`; `principal_amount = None` |

Across the window there are **48 mentions** where the old schema's `amount` was
a figure the new schema classifies as a balance, draw, repayment, or proceeds —
i.e. not the instrument's size at all.

## Example 3 — one facility became two instruments (the best single slide)

**Global Water Resources, 8-K Item 1.01, 2026-09-01** (`000162828026059647-1-01`)

> "on April 30, 2020, ... the Company entered into an agreement ... with The
> Northern Trust Company ... for a revolving line of credit that currently
> provides the Company up to a maximum of $20.0 million ... On August 28, 2026,
> the Company and Northern Trust entered into the [Eighth Modification
> Agreement] to ... (i) **extend the scheduled maturity date from May 18, 2028
> to August 30, 2028** and (ii) **increase the maximum principal amount ... from
> $20.0 million to $30.0 million**; provided, however, that if the Company
> completes any capital market activity ... the maximum ... will be $25.0
> million."

One facility, amended. The old schema emitted **two mentions**:

| | name | start_date | end_date | amount | lender |
|---|---|---|---|---|---|
| row 1 | revolving line of credit | 2020-04-30 | 2028-08-30 | 30,000,000 | Northern Trust |
| row 2 | revolving line of credit | 2020-04-30 | 2028-05-18 | 20,000,000 | Northern Trust |

Same name, same lender, same start date -- two phantom instruments, with nothing
in the output marking which is current. Because `amount` and `end_date` each held
exactly one value, duplicating the row was the *only* way to express "changed from
X to Y" -- and the old prompt mandated exactly that, instructing the model to
"return the predecessor as its own additional object" recording "its prior
commitment or maturity when stated." So the duplicate was a designed workaround for
the single-value slot, not a model slip, which makes it a cleaner argument: the old
schema had to put a phantom instrument in the output on purpose.

New schema: **one mention.**

```
name             revolving line of credit      instrument_type  revolving_credit
status           amended (2026-08-28)
principal_amount 30000000  kind=commitment  currency=USD
maturity_date    2028-08-30
amounts[]  commitment $20.0M   prior=True
           commitment $30.0M   prior=False
           commitment $25.0M   prior=False
dates[]    closing    2020-04-30
           amendment  2026-08-28
           maturity   2028-08-30  prior=False
           maturity   2028-05-18  prior=True
parties[]  borrower Global Water Resources, Inc.
           lender   The Northern Trust Company
```

`prior: true` is what collapses the duplicate. The superseded $20.0M commitment
and the superseded 2028-05-18 maturity are still *recorded* — they just aren't
published as current. There are **25 items** in the window where the old schema
produced same-name/different-amount duplicate mentions, and 12 mentions where
the new schema marks an amount `prior` (Hilton Grand Vacations $850M→$1.0B,
Sensient $105M→$115M, Ares Strategic Income $3.25B→$4.1B, Allbirds $50M→$44.2M,
LyondellBasell $900M→$700M, Hornbeck $75M→$125M, Nexentis €6M→€10M, …).

## Example 4 — every published field was wrong, in one filing

**comScore, 8-K Item 7.01, 2026-05-27** (`000115817226000032-7-01`)

> "the Company used a portion of proceeds from the Transaction to **repay in
> full** all of its obligations under the **Financing Agreement, dated as of
> December 31, 2024**, by and among the Company, **certain subsidiaries of the
> Company as guarantors**, **Blue Torch Finance LLC**, and the lenders from time
> to time party thereto ... Upon receipt of such repayment, **which totaled
> approximately $40.1 million**, the Credit Agreement and related obligations
> ... **were terminated**."

| | Old schema | New schema |
|---|---|---|
| name | Credit Agreement | Financing Agreement |
| start_date | **2026-05-27** | `2024-12-31` |
| amount | **40100000** | `principal_amount = None`; amounts[]: `repayment $40.1M` |
| status | *(no such field)* | `repaid` (2026-05-27) |
| dates[] | — | `agreement 2024-12-31`, `retirement 2026-05-27`, `termination 2026-05-27` |
| parties | lenders: `[Blue Torch Finance LLC]`; other: `[Company]` | `borrower` comScore Inc., `guarantor` "subsidiaries of the Company" *(collective)*, `lender` Blue Torch Finance LLC |

The old row says comScore **entered into** a $40.1M credit agreement on
2026-05-27. The filing says a facility from 2024-12-31 was **paid off and
killed** that day for $40.1M. The start slot absorbed the retirement date and
the amount slot absorbed the payoff — because those were the only slots
available.

## Example 5 — the announcement-date-as-start pattern

**Georgia Power, 8-K, 2026-05-22** (`000004109126000021`) — and the same pattern
in EQT 2020-04-23, EQT 2021-05-10 (×3), Hexcel, Onto Innovation, Pilgrim's
Pride. Notes that had been *announced* but not yet issued got the announcement
date written into `start_date`, so an unissued note looked like a live one.

New schema: `announcement 2026-05-19` in `dates[]`, `status = announced`,
`start_date` left null until a closing exists. The window has **90 mentions**
with `status = announced` and **85 dates** flagged `expected: true` — states the
old schema had no way to represent.

**Maximus, 8-K Item 1.01, 2026-05-28** (`000103222026000030-1-01`) is the
amendment flavor of the same problem: old `start_date = 2026-05-27` (the
amendment date); new schema keeps `agreement 2024-05-30` as the start, records
`amendment 2026-05-27`, and sets `status = amended`.

Across the window, the old schema's `start_date` holds something the new schema
types as *not* a start (announcement, amendment, retirement, repayment,
maturity, default, exchange) in **18 verified cases**; `end_date` holds a
commitment-termination or retirement rather than a maturity in **5** more
(EQT 2022-12-27 and Sensient: a commitment-termination date published as the
maturity — two genuinely different facts about a revolver).

## Example 6 — facts that used to be silently discarded

**Brookfield Business Corp, 6-K, 2026-05-11** (`000162828026033635-6K-1-30`)

| | Old schema | New schema |
|---|---|---|
| amount | `2350000000` | `principal_amount = 2350000000 (commitment, USD)` |
| amounts[] | — | `commitment $2,350M`<br>`outstanding_balance $1,485M as_of 2026-03-31`<br>`outstanding_balance $1,325M as_of 2025-12-31` |

Here the old schema picked the *right* number — and still threw away both
period-end balances, because there was nowhere to put them. This is the purely
additive half of the story: **46 amounts** in the window carry an `as_of_date`,
which the old schema could not express at all.

## Example 7 — parties: one untyped bucket → six roles

**MPLX LP** (`000155200016000155-1-01`): old schema put Wells Fargo Bank, N.A.
in `other_interested_parties` with no role and no canonical name. New schema:
`role=agent`, `canonical_name="Wells Fargo Bank, National Association"` — and it
also identifies `MPLX LP` as the `borrower`, a concept the old schema didn't
have.

Where the old schema's two buckets landed in the new role vocabulary
(792 matched party spans):

| old bucket | agent | borrower | guarantor | lender | other | trustee | underwriter |
|---|---|---|---|---|---|---|---|
| `lenders` | 11 | 1 | 0 | **153** | 1 | 0 | 6 |
| `other_interested_parties` | 119 | 8 | 112 | 24 | 14 | 141 | 202 |

The `other_interested_parties` bucket was a junk drawer: trustees, underwriters,
administrative agents, and guarantors all went in undifferentiated — plus 24
actual lenders (Bank of America N.A. ×7, PNC ×6, Royal Bank of Canada, …). And
19 parties the old schema called lenders were really agents or underwriters
(JPMorgan Chase, Wells Fargo, PNC as *agent*; Goldman Sachs, Barclays Capital,
Deutsche Bank Securities as *underwriter*) — a materially wrong claim about who
holds the debt.

Also replaced: the `lenders_known_incomplete` boolean, which was `true` on
**377 of 676** old-schema mentions (56%) for two different reasons — a
collective lender phrase, or no named lender at all. A consumer reading the flag
could not tell "something is undisclosed" from "nothing was disclosed." The new
schema separates `kind=collective` (232 party facts, e.g. "lenders party
thereto", "financial institutions", "the holders") from a three-valued
disclosure field.

---

## Aggregate numbers for a summary slide

Same 312-item window, old schema → new schema:

| | old | new |
|---|---|---|
| amount facts captured | 437 (1 slot/instrument) | **584** typed |
| mentions stating >1 amount | not representable | **104** (16%) |
| amounts that aren't the instrument's size | published as `amount` anyway | **173** typed as balance/draw/repayment/proceeds |
| amounts with an `as_of` date | 0 | **46** |
| superseded amounts marked `prior` | 0 (emitted as duplicate rows) | **12** |
| date facts captured | 727 (`start_date` + `end_date`) | **1,173** typed |
| mentions stating >1 date | not representable | **385** (58%) |
| lifecycle-event dates | no slot | **361** (amendment 131, announcement 100, retirement 68, repayment 21, termination 19, exchange 15, default 7) |
| mention-level `status` | no field | **475** derived |
| party facts captured | 927 in 2 untyped buckets | **1,632** across 7 roles |
| borrowers identified | 0 (no concept) | **668** |

Amount kind mix: principal 284, commitment 127, outstanding_balance 69,
repayment 50, proceeds 28, draw 26.
Date kind mix: maturity 407, closing 258, agreement 132, amendment 131,
announcement 100, retirement 68, repayment 21, termination 19,
commitment_termination 15, exchange 15, default 7.

**The through-line:** the old schema didn't just lose the extra facts — it
forced a *wrong* value into the one slot it had, because the model always had to
put something there. The typed list lets the model record what the document
actually says and leaves the published columns to be derived, so "we don't know
this instrument's size" became an expressible answer.

## Reproducing

```
data/genwindow-run-dev/mentions/     # old: name, start_date, end_date, amount,
                                     #      lenders_json, other_interested_parties_json
data/genwindow-run-branch/mentions/  # new: amounts_json, dates_json, parties_json,
                                     #      status, instrument_type + derived flat columns
```

Join on `item_id`, then match instruments by name within the item. Source text
for any item is in `<run>/extractor-runs/*/full.jsonl` under
`stage_responses.ner`. Kind vocabularies are defined in
`src/cdt/extractor/core.py` (`AMOUNT_KINDS`, `DATE_KINDS`, `PARTY_ROLES`,
`PARTY_KINDS`).

## Example 8 -- computed values: arithmetic the old schema could not record

The new schema tags every fact with `derived_from`: `stated` (read off the page),
`name` (the value is inside the instrument's own name), or `computed` (deterministic
arithmetic over cited spans). `computed` is the interesting one -- it covers facts the
document implies but never writes down, which the old single-value slots had no way
to hold.

### A computed amount: base offering + over-allotment

**SiTime Corp, 8-K Item 8.01, 2026-05-22** (`000119312526237180-8-01`)

> "the Company agreed to sell **$1.2 billion** aggregate principal amount of Notes
> and, at the option of the Underwriters, up to an additional **$150.0 million**
> aggregate principal amount of Notes, solely to cover over-allotments, **which was
> exercised in full** by the Underwriters on May 20, 2026."

The over-allotment was exercised in full, so the issuance is $1.35 billion. The
filing never says that: the only two dollar figures in the whole document are
`$1.2 billion` and `$150.0 million` (checked -- "1.35", "1,350" and "1350" appear
nowhere in the text).

| | Old schema | New schema |
|---|---|---|
| mentions for this issuance | **2** | **1** |
| row 1 | `name='Notes', amount=1200000000` | `principal_amount = 1350000000` |
| row 2 | `name='Notes', amount=150000000` | -- |
| amounts[] | -- | `principal $1,350,000,000`, `derived_from: computed`, citing `'$1.2 billion'` + `'$150.0 million'` |

One $1.35B issuance was published as two separate note instruments. This is the
`000119312526237180-8-01 | notes | n=2` row in the duplicate list in Example 3, where
I hedged that it might be two legitimate tranches -- the source text settles it: base
offering plus its own over-allotment, one instrument.

The guard (`computed_sum_amount`, #165) is deliberately narrow: at least two
parseable cited spans, refuses when any single span already equals the model's number
(that is agreement, not computation), and refuses if any cited span reads as a rate.
Summation is the only arithmetic accepted for amounts.

### A computed date: closing date + tenor

**Ashland Inc., 8-K Item 1.01, 2026-05-29** (`000119312526246229-1-01`)

> "The Credit Agreement provides for a $500 million **five-year** revolving credit
> facility (including a $125 million letter of credit sublimit)"

The facility closes **May 28, 2026**. The string `2031` never appears in the filing,
so its maturity exists only as arithmetic.

| | Old schema | New schema |
|---|---|---|
| start_date | `2026-05-28` | `2026-05-28` |
| maturity / `end_date` | **`None`** -- lost | **`2031-05-28`** |
| dates[] | -- | `maturity 2031-05-28`, `derived_from: computed`, citing `'May 28, 2026'` + `'five-year'` |

`computed_maturity_date` (#166) publishes the model's date only when some cited
`date` span plus some cited `duration` span actually lands on it -- the model must
cite both, and the arithmetic has to check out.

### All 12 computed dates on this window

| company | cited date + tenor | computed maturity | old-schema `end_date` |
|---|---|---|---|
| Ashland | May 28, 2026 + five-year | 2031-05-28 | none in item |
| Tractor Supply | May 19, 2026 + five-year | 2031-05-19 | none in item |
| Octave Intelligence | April 27, 2026 + five-year | 2031-04-27 | none in item |
| Octave Intelligence | April 27, 2026 + four-year | 2030-04-27 | none in item |
| Dorian LPG | September 2, 2026 + seven year | 2033-09-02 | none in item |
| Highlander Silver | January 22, 2024 + 5-year | 2029-01-22 | none in item |
| Classover Holdings | May 28, 2026 + two-year | 2028-05-28 | none in item |
| Bluerock Homes | August 27, 2026 + 36-month | 2029-08-27 | none in item |
| Medicus Pharma (Note A-1) | May 27, 2026 + eighteen months | 2027-11-27 | none in item |
| Medicus Pharma (Note B) | May 27, 2026 + eighteen months | 2027-11-27 | none in item |
| Spire | August 31, 2026 + 364 days | 2027-08-30 | none in item |
| PAID Inc | October 13, 2022 + 9-month | 2023-07-13 *(prior)* | none in item |

**In all 12, the old-schema run published no `end_date` anywhere in the filing
item** -- not a wrong maturity, no maturity at all. Without the arithmetic a stated
tenor produced nothing, so these are pure recovery rather than corrected errors.

Two details worth a sentence on stage. Octave Intelligence resolves two *different*
tenors off the same closing date -- a five-year revolver maturing 2031-04-27 and a
four-year term loan maturing 2030-04-27, from one April 27, 2026 closing -- which a
single `end_date` column could not have distinguished even in principle. And the
tenor parser spans words, digits, months and days: `five-year`, `5-year`, `36-month`,
`eighteen months`, `364 days`. PAID Inc's is a computed *prior* maturity, recorded
but correctly not published.

**Frequency.** Computed values are rare: 12 of 1,173 date facts (1%) and 1 of 584
amount facts. Present them as a capability, not a volume driver. The `name`-derived
category is far larger -- 190 of 1,173 dates (16%) come from a name like
`3.875% senior notes due 2028`, while no amount on this window was name-derived.

---

# Substitution vs. abstention: how often did the old schema fill a slot it should have left null?

**Method.** The new-schema run gives a usable proxy for "was the correct value even
stated?" For each paired mention (541 of 676 dev mentions pair by name inside the
same item), classify by what the new run found for that dimension:

- **A — correct value stated:** the new run has a non-`prior` fact of the kind the
  old slot needed (`commitment`/`principal` for `amount`; `closing` for
  `start_date`; `maturity` for `end_date`).
- **B — no correct value, but a tempting one present:** the new run extracted facts
  for that dimension, but none of the required kind. *This is the population the
  question is about.*
- **C — the new run found no facts of that dimension at all.** **Excluded as
  uninformative** — inspection shows these are mostly new-run misses or
  pairing mismatches, not documents that state nothing. (E.g. Sysco `start_date`
  2026-09-04 where the new run captured no dates whatsoever.) Counting C as
  "nothing was stated" would be reading a new-schema miss as an old-schema win.

"Confirmed substitution" is the strong version: the old value is *exactly* a
figure/date the new run typed as a **different** kind, so the borrowed fact is
identifiable rather than merely suspected.

## Results

| old slot | B: no correct value, tempting value present | filled anyway | left null | confirmed substitution | borrowed from |
|---|---|---|---|---|---|
| `amount` | 54 | **47 (87%)** | 7 (13%) | **47 of 47** | outstanding_balance 22, repayment 20, draw 5 |
| `start_date` *(strict: closing only)* | 262 | 73 raw / **57 real (22%)** | 189 (72%) | 57 of 73 | agreement 41, announcement 10, amendment 4, retirement 2 |
| `start_date` *(agreement allowed as start)* | 203 | **23 (11%)** | 180 (89%) | 14 of 23 | announcement 10, amendment 3, retirement 1 |
| `end_date` | 144 | **6 (4%)** | 139 (96%) | 3 of 6 | commitment_termination 2, retirement 1 |

**The 16 non-confirmed `start_date` fills are method noise, not substitution.** Of
the 73 raw strict-definition fills, 57 are confirmed and the remaining 16 were
audited individually:

- **12 are pairing crossings.** The value dev put in `start_date` *is* present
  elsewhere in the same filing item -- typed by the new run as `agreement` (10),
  `closing` (1), or `amendment` (1) -- just attached to a different mention than my
  greedy name-matcher chose. NMP Acquisition Corp is the clean illustration: two
  identically named "Secured Promissory Note" mentions whose dates (2022-12-09 and
  2023-02-24) got matched in swapped order, so both scored as errors when both are
  right. These are artifacts of the join, not model behavior.
- **4 are unresolvable**: dev's value appears nowhere in the new run for that item
  (Southern Co Gas 2026-05-20, EagleRock 2026-05-04, Americold 2022-08-23, Aveanna
  2017-03-16). These are most likely new-run misses of an original agreement date,
  so they cannot be scored against either arm.

So the strict `start_date` substitution rate is **57/262 = 22%**, not 28%, and
**none** of the 16 leftovers is a demonstrated substitution. Netting out the 41
agreement-date cases that the new schema ratifies (below), genuinely wrong-kind
strict substitutions number just **16** (announcement 10, amendment 4, retirement 2)
-- consistent with the 14 found under the permissive definition.

**The `amount` figure is immune to this problem.** In the `amount` bucket B, all 47
fills were confirmed -- zero untraceable -- because the borrowed value sits in the
*same* paired mention. And the item-level check below uses no name matching at all
and lands at 85%. The 87% does not depend on the join.

**Contrast rows.** When the correct value *was* stated (bucket A), the old schema
filled the slot 96% (`amount`, 300/313), 95% (`start_date`, 250/263), 92%
(`end_date`, 296/322) of the time — so low fill rates in bucket B are genuine
abstention, not a slot the model ignored generally.

**It does not invent from nothing.** In bucket C, where the new run found no money
facts at all, the old schema still published an `amount` in only 5% of mentions
(9/174) and an `end_date` in 1% (1/75). The failure mode is *substitution of an
adjacent wrong-kind value*, not fabrication — the model needs a plausible number
sitting in the text to misfile.

**Item-level robustness check** (no name matching, 298 shared items): of 33 items
where the new run found money but no stated size, the old schema published an
amount in 28 (**85%**), and in all 28 the published value is one the new run typed
as a non-size kind. That independently reproduces the 87% mention-level figure.

## The answer, in one line

**The behavior is almost entirely an `amount` problem.** Faced with a document that
stated money but no principal or commitment, the old schema published a wrong-kind
figure **87%** of the time and correctly abstained only **13%**. Dates were handled
conservatively by comparison -- a genuinely wrong-kind `start_date` in 6% of
opportunities (16/262 strict, or 14/203 = 7% permissive, once the ratified
agreement-date cases and the 16 audited join artifacts are removed) and a wrong-kind
`end_date` in 4% (6/144).

## Why this is the strongest argument for the change

The old prompt already told the model to abstain, four separate times:

> "`amount` is the principal or commitment amount only. ... **Omit `amount` when the
> document states no principal or commitment amount.**"
> "Omit `amount` unless the document states that loan's principal or commitment amount."
> "When the document states only a combined total for a group ... omit `amount` on the individual instruments."
> "If a property is absent, omit it." / "Do not invent ids, parties, dates, or amounts."

It still substituted 87% of the time. **Instructing the model to leave the slot
empty did not work; giving the value a correct home did.** Under the new schema the
same figures come back typed as `draw`, `repayment`, and `outstanding_balance`, with
`principal_amount` correctly null — because the model no longer has to choose
between misfiling a fact and discarding it.

The `start_date` rows carry a second version of the same lesson. The old prompt
declared "the `dated as of` date of an indenture, purchase agreement, or amendment
**is not** the instrument's start date" — and the model used the agreement date as
`start_date` anyway in **41** of the 57 confirmed strict-definition substitutions.
The new schema doesn't re-argue the rule: it records `agreement` as its own kind and
then defines the precedence in code (`start_date` ← `closing`, falling back to
`agreement`, in `select_date_payload`). Those 41 cases stop being violations because
the derivation moved out of the prompt. That is why the strict and permissive rows
differ so much: 28% → 11% is almost entirely the agreement-date question, which the
new design settled rather than restated.

## Caveats

- The "correct value" proxy is the new run's own extraction, not human ground truth.
  It is reliable for *confirmed* substitutions (the borrowed fact is identified
  explicitly) and weaker for bucket membership.
- Counts are from one run per arm. Per issue #171 ~15% of units change on re-run, so
  quote these as measured on this window, not as stable population rates. The 87/13
  split for `amount` survives the independent item-level check, so the qualitative
  ordering (`amount` ≫ `start_date` > `end_date`) is safe to assert; the exact
  percentages are not.
- 135 of 676 dev mentions did not pair by name and are outside this analysis.

## Sample size and scope (read before quoting any percentage)

The window as a whole: **312 items** from 273 filings and 198 companies (old-schema
run); 304 items / 265 filings / 193 companies on the new-schema run; **298 items in
common**. The 541 paired mentions the substitution analysis runs on span **271
items, 237 filings, 168 companies**.

But each slot's percentage rests on a much smaller base — only the mentions in
bucket B:

| slot | bucket B | items | filings | companies | confirmed substitutions | items | companies |
|---|---|---|---|---|---|---|---|
| `amount` | 54 | **40** | 39 | 21 | 47 | 36 | 20 |
| `start_date` (strict) | 262 | 134 | 122 | 75 | 57 | 47 | 36 |
| `start_date` (permissive) | 203 | 93 | 82 | 47 | 14 | 11 | 9 |
| `end_date` | 144 | 89 | 86 | 73 | 3 | 3 | 3 |

Consequences for how to present these:

- **The 87% `amount` figure rests on 54 mentions across 40 items and 21 companies.**
  That is a real but small base. Quote it as "87% of 54 opportunities," not as a rate
  with implied precision.
- **The `end_date` "4%" should not be quoted as a percentage at all.** It is **3
  confirmed cases** (EQT, Sensient, HSBC). Say "we found 3 instances" — a 3/144 rate
  invites a false sense of measurement.
- **`start_date` is the only slot with a comfortable base**: 262 opportunities across
  134 items and 75 companies.

### Issuer concentration: EQT

The window is not a uniform sample. It contains an **EQT Corp historical backfill** —
67 of 312 items (21%) are EQT, spanning 2017–2026, while nearly everything else sits
in 2026. EQT is therefore heavily overrepresented in the example lists, and supplies
**28 of the 47** confirmed `amount` substitutions (60%).

The effect survives removing it:

| | mentions | items | companies | filled | null |
|---|---|---|---|---|---|
| all | 54 | 40 | 21 | 47 (**87%**) | 7 |
| EQT only | 31 | 20 | 1 | 28 (**90%**) | 3 |
| excluding EQT | 23 | 20 | 20 | 19 (**83%**) | 4 |

83% vs 90% — EQT contributes volume, not the effect. And **19 of the 20** non-EQT
companies in the bucket substituted at least once, so the behavior is broad-based
rather than one issuer's filing style. Still, say "83% excluding EQT, on 23
opportunities across 20 companies" if anyone asks; the honest headline is that the
direction is solid and the exact percentage is not.
