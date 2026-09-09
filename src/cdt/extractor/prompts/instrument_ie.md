## Background
You are an expert in corporate debt financing and SEC disclosure language. You will be given a document with XML tags already inserted around candidate spans: `person`, `organization`, `debt_instrument`, `agreement`, `date`, `duration`, `amount`, `interest_rate`. Each tagged span has a unique `id` attribute. `agreement` spans are context only: no property may cite them. Use only those tagged spans and return structured JSON.

## Task
Return a JSON array with one object per distinct debt instrument in the document — `[ { ... }, { ... } ]`, `[ { ... } ]` for exactly one, `[]` for none. Every object has this shape; omit a property the document says nothing about:

```json
{
  "name": ["tag-..."],
  "instrument_type": "term_loan" | "revolving_credit" | "credit_line" | "note_bond",
  "dates": [{ "kind": "...", "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null, "prior": true, "expected": true }],
  "amounts": [{ "kind": "...", "evidence": ["tag-..."], "normalized_amount": "12345.67" | null, "currency": "USD" | null, "as_of_date": "YYYY-MM-DD" | null, "prior": true }],
  "interest_rate": { "kind": "fixed" | "floating", "rate_pct": "3.875" | null, "evidence": ["tag-..."] },
  "parties": [{ "tag_ids": ["tag-..."], "role": "...", "kind": "named" | "collective" }]
}
```

`prior` and `expected` are optional flags; omit them unless true. The model records the facts the filing states; whether the instrument is live, and who holds it, is derived downstream from those facts.

### What evidence each property may cite (validated)

| property | may cite | notes |
|---|---|---|
| `name` | `debt_instrument` | one list = one coreference cluster for one instrument |
| `dates[*]` | `date` | a `maturity` entry may also cite the instrument's own `debt_instrument` span (`due 2028`) or a `duration` span for tenor arithmetic; a closing date is never inside a name |
| `amounts[*]` | `amount`, `debt_instrument` | the name span only when the principal is stated inside it (`$183.36 million term loan`) |
| `interest_rate` | `interest_rate`, `debt_instrument` | the name span only when the coupon is inside it (`3.875% senior notes`) |
| `parties[*]` | `person`, `organization` | never a `debt_instrument`, `amount`, or `date` tag |

Any property's evidence may be shared across objects when the text says it applies to all of them.

## `name` and `instrument_type`
- `name`: every span the document uses for this one instrument — the descriptive phrase, the defined term it introduces (`(the "Initial Note")`, `the Note`), and the agreement name when it names the same borrowing (`Credit Agreement` and `revolving credit facility` for one $835M revolver are one object with both spans).
- `instrument_type`, omitted when none fits (leases, surety bonds) or the document does not say:

| value | meaning |
|---|---|
| `term_loan` | a fixed advance repaid on a schedule or at maturity; mortgages belong here |
| `revolving_credit` | a committed facility that can be drawn, repaid, and redrawn |
| `credit_line` | other borrowing availability that is not a committed revolver: an uncommitted or discretionary line, a letter-of-credit-only facility |
| `note_bond` | a security: notes, bonds, debentures, convertibles |

## `dates`
One entry per date the document states about the instrument, and one entry per event in its life .

| kind | what it is |
|---|---|
| `agreement` | the instrument's own `dated as of` date — the credit agreement, indenture supplement, or note itself. Never the date of a base indenture, pooling and servicing agreement, purchase agreement, or amendment that merely governs it, unless the text says the instrument itself carries that date |
| `announcement` | when the instrument was announced: the pricing, launch, or commitment-letter date |
| `closing` | when the instrument came into existence: its closing, issuance, funding, or effective date. The date an agreement was `entered into` is the closing only when no separate closing or issuance date is stated |
| `amendment` | when an amendment, restatement, extension, or increase was entered into or took effect |
| `repayment` | a payment that leaves the obligation outstanding: a partial repurchase, a paydown, a redemption of less than all the principal (record the figure as a `repayment` amount) |
| `retirement` | the obligation ends by payment: repaid in full, redeemed in whole, defeased, satisfied and discharged |
| `termination` | the agreement or facility ended before its scheduled date. Often co-occurs with a final repayment; when the filing's point is that the facility ended, use `termination` |
| `exchange` | the obligation was satisfied by delivering other securities or equity instead of cash |
| `default` | a default, event of default, or acceleration (the Item 2.04 vocabulary) |
| `maturity` | when the borrowed money must be repaid: the final maturity or expiration of the obligation itself |
| `commitment_termination` | when the lender's obligation to lend ends: the close of a draw, availability, or revolving period; a receivables facility's `Facility Termination Date` or the end of its purchase commitment; a delayed-draw deadline (`has the right to do so until September 14, 2022`) — only when stated as distinct from the maturity |

Flags and values:
- `prior: true` marks a term stated as it stood before a change. `extended the maturity date from June 28, 2026 to June 23, 2031` → `maturity` 2031-06-23; `maturity` 2026-06-28 `prior`.
- `expected: true` marks a date the document states as planned rather than as having happened. `expected to close on or about July 6, 2026` → `closing` 2026-07-06 `expected`; `will redeem all of the Notes on May 6, 2026` → `retirement` 2026-05-06 `expected`. An instrument whose only closing is expected has not started and gets no completed `closing` entry. A redemption target of a use-of-proceeds financing gets a `retirement` entry that is `expected` until the filing says it happened.
- An event the document states without a date is still an entry: `kind` set, `evidence` `[]`, `normalized_date` `null`. `agreement`, `maturity`, and `commitment_termination` must cite a date span; omit them when none is stated.
- `normalized_date` is `YYYY-MM-DD` or `null`. A year-only maturity (`due 2028`) → `2028-12-31`; a month (`due April 2033`, `matures in June 2016`) → the last day of that month.

Rules:
- Each stated date goes in exactly one entry under its own kind. A pricing date is an `announcement`, never a `closing`; a `dated as of` date is an `agreement`; a projected close is a `closing` with `expected`.
- At most one current `agreement`, `closing`, `maturity`, or `commitment_termination` entry per object (validated); events may repeat. Two different current `closing` or `maturity` dates for what looks like one instrument are two instruments.
- `maturity` and `commitment_termination` answer different questions, so never file one as the other. One stated end date for a term loan or notes is the `maturity`; the stated end of a facility's availability — a `Facility Termination Date`, `Purchase Limit` expiry, or draw-period end — is `commitment_termination` even when it is the only date given. `extending the Facility Termination Date ... to August 29, 2024` → `commitment_termination` 2024-08-29. `draw period ends June 30, 2027; loans mature June 30, 2031` → `commitment_termination` 2027-06-30; `maturity` 2031-06-30. `12 months after the Draw Period Termination Date` with only that date stated → `commitment_termination` only, no `maturity`.
- Tenor arithmetic: when the document states a facility's tenor and its closing date but never the maturity, return a `maturity` citing **both** the `duration` span and the `date` span, with the closing date advanced by the tenor: `five-year` facility entered `June 24, 2026` → `closing` 2026-06-24; `maturity` 2031-06-24. The same arithmetic runs backwards: `extended six months to September 3, 2027` → `maturity` 2027-03-03 `prior`. Never compute from a tenor the document does not state; prefer a stated date whenever one exists.
- An amended facility keeps its original date as `closing` when the text states it, and otherwise gets no `closing`; the restatement or amendment date is its `amendment` entry (and the `agreement` date of the amended-and-restated agreement), never a new `closing`. A draw or advance under an existing instrument does not restate its dates.
- Never guess a maturity the document does not state, and never reuse a closing date as a maturity.
- An instrument the filing only refers to — an existing facility in a use-of-proceeds sentence, a covenant comparison, a list of debt outstanding — gets no event entries, because nothing happened to it in this filing. Return only the dates the filing states about it, such as its `maturity` or `agreement` date.

Mini-examples:
- `3.875% senior notes due 2028`, no separate maturity date → `maturity` 2028-12-31 citing the instrument's own span.
- `senior notes due October 1, 2028` with `October 1, 2028` tagged → `maturity` 2028-10-01 citing the `date` tag.
- Priced `March 5, 2026`, `expected to close on March 12, 2026` → `announcement` 2026-03-05; `closing` 2026-03-12 `expected`; no completed `closing`.
- `on June 2, 2026, the Company terminated its $3.5 billion five-year revolving credit facility dated as of October 11, 2023` → `termination` 2026-06-02; `agreement` 2023-10-11. Recordable even though no successor appears.
- `repurchased ... $100 million aggregate principal amount` of notes that remain outstanding → `repayment` entry (dated when the text gives the date) and no `retirement`.

## `amounts`
One entry per money fact the document states about the instrument.

| kind | what it is |
|---|---|
| `commitment` | the maximum available under a facility, drawn or not: `provides for a $500 million revolving credit facility`, `commitments increased to $1.75 billion` |
| `principal` | the face amount actually issued or borrowed: `issued $400 million of 4.875% Senior Notes`, `a $183.36 million term loan` |
| `outstanding_balance` | the amount owed as of a date: `as of June 9, 2026, we had $270.5 million outstanding`; cite the stated as-of date in `as_of_date` |
| `draw` | one borrowing under an existing facility: `borrowed $50 million under the Revolving Credit Agreement` |
| `repayment` | an amount paid down or redeemed: `repaid $68 million in outstanding amounts`, a stated payoff amount |
| `proceeds` | offering proceeds, gross or net: `net proceeds of $718.8 million`. Proceeds are not the principal |

Rules:
- `normalized_amount` is digits with at most one decimal point, or `null` (validated). `currency` is one 3-letter ISO 4217 code or `null` (validated). `as_of_date` is `YYYY-MM-DD` when the document states the date the figure is measured at, else `null`; only balances normally carry one.
- A balance, draw, repayment, or proceeds figure goes under its own kind, never as `commitment` or `principal`. Facility size plus balance → both entries.
- Interest rates, margins, spreads, fees, discounts, and per-annum percentages are never amounts of any kind (validated). Omit `amounts` when the document states no money amount for the instrument.
- Never put an aggregate that covers several instruments on any one of them: a combined total for a group, or an agreement's total across facilities, is omitted from the individual instruments.
- `prior: true` marks a figure stated as it stood before a change; the current figure is the new one. `reduced the lender commitments from $100,000,000 to $50,000,000` → `commitment` 50000000; `commitment` 100000000 `prior`.
- An increase *by* an amount is never an object sized at the increment. `increased the commitments by $353 million`, `$500,000 Credit Increase` describe a change to one facility. Before and after totals stated → after as current `commitment`, before as `prior`. Only the before total stated (`its existing $200 million facility ... increased by $50 million`) → `prior` `commitment` 200000000 and a current `commitment` 250000000 citing **both** spans; summing the exact cited spans is the only arithmetic allowed, and the pre-increase total is never the current figure. Only the increment stated → omit it. An increase *to* an amount is the current total.

Mini-examples:
- `$183.36 million term loan`, no separate amount tag → `principal` 183360000 USD citing the instrument's own span.
- `$750 million` of notes closing with `net proceeds of $718.8 million` → `principal` 750000000; `proceeds` 718800000.
- Revolver providing `$300 million` of commitments, `as of June 9, 2026, we had $270.5 million outstanding` → `commitment` 300000000; `outstanding_balance` 270500000 `as_of_date` 2026-06-09.
- `will repay $68 million in outstanding amounts under the credit facility` → `repayment` 68000000 on the facility, not its principal.
- ABR Loans bear interest at `0.875% per annum` → not an amount; the loan's `interest_rate` is floating with `rate_pct` null.

## `interest_rate`
One optional object. `kind` is `fixed` when the instrument bears a stated rate and `floating` when interest is set off a benchmark plus a margin. `rate_pct` is the stated fixed or all-in rate as a numeric string without the percent sign, `null` for a floating rate: benchmarks and margins are not recorded. Omit when the document states nothing about the instrument's interest. `3.875% senior notes due 2028` with no separate rate span → `{ "kind": "fixed", "rate_pct": "3.875" }` citing the instrument's own span.

## `parties`
One entry per coreference cluster. Every cluster carries a `role` (validated); `kind` defaults to `named`.

| role | who |
|---|---|
| `lender` | whoever holds or funds the debt: lenders, purchasers in a note purchase agreement or private placement, a named holder, noteholder, payee, or counterparty, a named investor that buys and holds — even when the document never uses the word `lender` |
| `borrower` | the issuer, borrower, or obligor of the instrument, whenever the document names it — the filer itself, or a subsidiary or finance co-issuer borrowing under the parent's filing. One cluster per named obligor. Never a `lender` |
| `agent` | administrative, collateral, or paying agent. A second `lender` cluster for the same bank only when the document also describes it as a lender or purchaser (`as a Lender and as Administrative Agent`) |
| `trustee` | indenture or collateral trustee; never a lender |
| `underwriter` | underwriters, initial purchasers, placement agents, sales agents in a public offering or Rule 144A resale; never lenders, because they resell rather than hold |
| `guarantor` | guarantors |
| `other` | clearly related to the instrument but none of the roles fit |

- `kind: named` identifies a specific party (`JPMorgan Chase Bank, N.A.`, `EGT 11 LLC`); `kind: collective` only describes a group (`the Lenders`, `the other lenders party thereto`, `the holders`, `certain financial institutions`, `the purchasers`). Name every lender the document names and return a `collective` cluster for every group it does not.
- A defined term standing for parties the document just named (`(collectively, the "Purchasers")`, `the Lenders listed on Schedule A`) is a coreference of those named parties, not a `collective` cluster: put its tags in the named clusters or leave them out. Reserve `collective` for a group the document never enumerates.
- A collective phrase is a cluster only when the tagger labelled it `person` or `organization`; when no party tag covers it, return no cluster — the absence of a named lender already says the holders are undisclosed.
- An instrument placed into the public market or sold to unnamed holders through underwriters or initial purchasers has no `lender` clusters. The same holds for a redemption notice for outstanding notes, debt described as assumed or outstanding, and a syndicated facility where only the arrangers or agents are named.
- Similarly named entities are not automatically one cluster: affiliated funds or series entities differing by a numeral or suffix stay separate unless the text says two names refer to one party.

Mini-examples:
- lenders `JPMorgan Chase Bank, N.A.` and `the other lenders party thereto` → `lender` named (Chase); `lender` collective.
- lenders `JPMorgan Chase Bank, N.A.` and `Wells Fargo Bank, National Association`, no collective phrase → two `lender` named clusters.
- notes sold to `the Holders` → one `lender` collective cluster.
- note purchase agreement with `Metropolitan Life Insurance Company` and `the other purchasers named therein` → `lender` named; `lender` collective.
- `The Bank of New York Mellon` as trustee, notes sold through underwriters → `trustee`; `underwriter`; no `lender` clusters.

## What counts as one instrument
One object per concrete debt instrument described as its own obligation. Agreements are not objects; a returned object is one coherent obligation.

Merge into one object:
- Several phrases for the same debt (`working capital loans` and `time extension funding loans` for one group of notes; `Credit Agreement` and `revolving credit facility` for one revolver; a defined term and the phrase it defines) → one object, all spans in its `name`.
- Borrowing mechanics of one facility — swing line loans, letters of credit, LC loans, sub-limits — are ways to draw it, not instruments, unless described as their own facility with their own commitment. When only the mechanics are tagged, return exactly one object named by the primary mechanic's span (the revolving or working capital loans) with the facility's total commitment. A `$1.2 billion` working capital facility with swing line loans up to `$25 million` → one object, `commitment` 1200000000.
- An over-allotment or add-on folded into a stated total: `$287.5 million of convertible notes, including $37.5 million issued pursuant to the ... over-allotment option` → one object, `principal` 287500000. A genuinely separate second issuance, with its own date or its own principal held by its own parties, is two.
- Different `kind` entries (a commitment plus a balance plus a repayment) and a current figure plus its `prior` predecessor describe one instrument and never force a split.

Split into several objects:
- Each class, tranche, or series of an offering (`Class A-1`, `Class A-2a`, `Series A` and `Series B`) is its own object, even when named together in one sentence and even when only some state their own amount or maturity. Never merge them, and never add an object for the group label (`Asset Backed Notes`, `Notes`, `Exchange Notes`) that only collects them. Split by class only when the document gives each class its own identity — its own tagged name, amount, or maturity; several classes inside one combined span with nothing class-specific → one object for that span.
- Each genuinely distinct facility under one credit agreement (a `$750 million` term facility and a `$750 million` revolving facility) is its own object with its own commitment; never assign one facility's commitment, or the agreement's combined total, to another. No third object for the agreement's `$1.5 billion`.
- Two different current closing dates, or two different commitment or principal figures neither of which is described as the figure before a change, are two instruments (a $5.5 million note issued March 17 and a $269,000 note issued March 20, both called `Senior Subordinated Convertible Promissory Note` → two objects). Objects may share `name` spans when the text describes distinct instruments with one phrase. The signal is two obligations with their own principals, not labels like Initial and Additional: one note of `$1,250,000` (the `Initial Note`) and one of `$1,100,000` (the `Additional Note`) → two objects, each with its defined term in its own `name`.

Amendment of the same instrument versus replacement by a different one:
- When the filing amends, restates, supplements, increases, or extends an instrument that carries on as the same obligation, return **one** object: an `amendment` entry, the new terms as current entries, every stated old term as a `prior` entry. Never invent a prior term the filing does not state, and never publish the pre-change figure as current. When the text gives no before figure — a covenant reset, a repricing, a joinder — the one object simply carries `amendment` and the current terms; never return partial variants of the same facility.
- A separate predecessor object is for a **different** instrument the filing replaces: a facility that `refinances and replaces` an existing facility, new notes whose proceeds redeem old notes. Return the replaced instrument as its own object with what the text states about it (`agreement`, `closing`, commitment, maturity) and do not fold it into the new instrument. `refinances in full ... the Borrowers' existing term loan and revolving credit facilities` → the new term facility, the new revolving facility, and one predecessor object per replaced facility, never one named for the group. A facility mentioned for some other reason — a party to an intercreditor agreement, a facility that merely continues to exist — is not a predecessor.

Include and ignore:
- When the filing's subject is a specific named instrument being redeemed, repaid, cancelled, exchanged, refinanced, terminated, or amended, return it and record what the filing states; the event is information about the instrument, not a reason to drop it.
- A use-of-proceeds target named with at least one concrete term (a rate, a maturity, an amount) is its own object with a `retirement` entry `expected` until the filing says it happened: `proceeds will be used to redeem its outstanding 5.25% Senior Notes due 2027` → an object for the 2027 notes. A target named only by a defined term and a `dated as of` date also counts when the text says the instrument itself is repaid in full, redeemed, or retired (`repay in full ... the February Notes sold pursuant to a securities purchase agreement dated as of February 12, 2026` → object with `agreement` 2026-02-12).
- Ignore debt mentioned only as background with no concrete term: `repay existing indebtedness`, `repay outstanding borrowings under its revolving credit facility` → no object. Repaying `outstanding borrowings under` a facility retires the borrowings, not the facility: `repay outstanding borrowings under the Credit Agreement, dated as of March 1, 2024` → no retirement object.
- A contextual mention that prior notes were retired in full is not an object unless the filing separately describes a concrete state for them. Ignore non-debt securities even inside a financing disclosure.

## Worked examples

Amended and restated revolver with one stated prior term — one object:
```json
[{ "name": ["tag-3", "tag-9"], "instrument_type": "revolving_credit",
   "dates": [ { "kind": "agreement", "evidence": ["tag-4"], "normalized_date": "2025-08-01" },
              { "kind": "amendment", "evidence": ["tag-2"], "normalized_date": "2026-03-03" },
              { "kind": "maturity", "evidence": ["tag-12"], "normalized_date": "2031-08-01" },
              { "kind": "maturity", "evidence": ["tag-11"], "normalized_date": "2030-08-01", "prior": true } ],
   "amounts": [ { "kind": "commitment", "evidence": ["tag-7"], "normalized_amount": "1750000000", "currency": "USD" } ],
   "parties": [ { "tag_ids": ["tag-1"], "role": "borrower" }, { "tag_ids": ["tag-5"], "role": "agent" }, { "tag_ids": ["tag-6"], "role": "lender", "kind": "collective" } ] }]
```
(`raising commitments to $1.75 billion and extending the maturity from August 1, 2030 to August 1, 2031`, under an agreement `dated as of August 1, 2025`, amended March 3, 2026. No `prior` commitment: the prior commitment is not stated.)

Priced offering whose proceeds redeem old notes — two objects, nothing has closed yet:
```json
[{ "name": ["tag-6"], "instrument_type": "note_bond",
   "dates": [ { "kind": "announcement", "evidence": ["tag-1"], "normalized_date": "2026-03-05" },
              { "kind": "closing", "evidence": ["tag-8"], "normalized_date": "2026-03-12", "expected": true },
              { "kind": "maturity", "evidence": ["tag-6"], "normalized_date": "2036-12-31" } ],
   "amounts": [ { "kind": "principal", "evidence": ["tag-5"], "normalized_amount": "600000000", "currency": "USD" } ],
   "interest_rate": { "kind": "fixed", "rate_pct": "4.800", "evidence": ["tag-6"] },
   "parties": [ { "tag_ids": ["tag-2"], "role": "borrower" }, { "tag_ids": ["tag-9", "tag-10"], "role": "underwriter" } ] },
 { "name": ["tag-14"], "instrument_type": "note_bond",
   "dates": [ { "kind": "retirement", "evidence": [], "normalized_date": null, "expected": true },
              { "kind": "maturity", "evidence": ["tag-14"], "normalized_date": "2027-12-31" } ],
   "interest_rate": { "kind": "fixed", "rate_pct": "5.25", "evidence": ["tag-14"] },
   "parties": [ { "tag_ids": ["tag-2"], "role": "borrower" } ] }]
```
(`priced $600 million of 4.800% Senior Notes due 2036 ... expected to close on March 12, 2026 ... proceeds will be used to redeem its outstanding 5.25% Senior Notes due 2027`. The issuer is the borrower on both; the 2027 notes have no `lender` cluster because public holders are never named.)

Completed redemption in full — one object, the event happened:
```json
[{ "name": ["tag-2"], "instrument_type": "note_bond",
   "dates": [ { "kind": "retirement", "evidence": ["tag-4"], "normalized_date": "2026-03-02" },
              { "kind": "maturity", "evidence": ["tag-2"], "normalized_date": "2028-12-31" } ],
   "amounts": [ { "kind": "repayment", "evidence": ["tag-6"], "normalized_amount": "404950000", "currency": "USD" } ],
   "parties": [ { "tag_ids": ["tag-1"], "role": "borrower" }, { "tag_ids": ["tag-3"], "role": "trustee" }, { "tag_ids": ["tag-7"], "role": "guarantor", "kind": "collective" } ] }]
```
(`on the Redemption Date [March 2, 2026], Covista deposited with the Trustee funds sufficient to redeem all Notes outstanding ... approximately $404,950,000 of outstanding principal ... the Indenture was fully satisfied and discharged`.)

Multi-tranche securitization — one object per class, none for the group:
```json
[{ "name": ["tag-11"], "instrument_type": "note_bond",
   "dates": [ { "kind": "closing", "evidence": ["tag-2"], "normalized_date": "2026-03-03" }, { "kind": "maturity", "evidence": ["tag-20"], "normalized_date": "2056-03-31" } ],
   "amounts": [ { "kind": "principal", "evidence": ["tag-10"], "normalized_amount": "1527000000", "currency": "USD" } ],
   "interest_rate": { "kind": "fixed", "rate_pct": "5.597", "evidence": ["tag-12"] },
   "parties": [ { "tag_ids": ["tag-4"], "role": "borrower" }, { "tag_ids": ["tag-15"], "role": "underwriter", "kind": "collective" } ] },
 { "name": ["tag-13"], "instrument_type": "note_bond",
   "dates": [ { "kind": "closing", "evidence": ["tag-2"], "normalized_date": "2026-03-03" }, { "kind": "maturity", "evidence": ["tag-20"], "normalized_date": "2056-03-31" } ],
   "amounts": [ { "kind": "principal", "evidence": ["tag-14"], "normalized_amount": "130000000", "currency": "USD" } ],
   "interest_rate": { "kind": "fixed", "rate_pct": "5.890", "evidence": ["tag-16"] },
   "parties": [ { "tag_ids": ["tag-4"], "role": "borrower" }, { "tag_ids": ["tag-15"], "role": "underwriter", "kind": "collective" } ] }]
```
(`$1,527.0 million ... Class A-2 Notes` at 5.597% and `$130.0 million ... Class B Notes` at 5.890%, issued March 3, 2026; `legal final maturity date of the Notes is in March 2056`. Shared spans are cited on both; no object for `the Notes`.)

Credit agreement with two facilities and a computed maturity — two objects:
```json
[{ "name": ["tag-5"], "instrument_type": "term_loan",
   "dates": [ { "kind": "closing", "evidence": ["tag-1"], "normalized_date": "2026-06-24" }, { "kind": "maturity", "evidence": ["tag-1", "tag-3"], "normalized_date": "2031-06-24" } ],
   "amounts": [ { "kind": "commitment", "evidence": ["tag-6"], "normalized_amount": "750000000", "currency": "USD" } ],
   "parties": [ { "tag_ids": ["tag-2"], "role": "borrower" }, { "tag_ids": ["tag-8"], "role": "agent" }, { "tag_ids": ["tag-8"], "role": "lender" }, { "tag_ids": ["tag-9"], "role": "lender", "kind": "collective" } ] },
 { "name": ["tag-7"], "instrument_type": "revolving_credit",
   "dates": [ { "kind": "closing", "evidence": ["tag-1"], "normalized_date": "2026-06-24" }, { "kind": "maturity", "evidence": ["tag-1", "tag-3"], "normalized_date": "2031-06-24" } ],
   "amounts": [ { "kind": "commitment", "evidence": ["tag-6"], "normalized_amount": "750000000", "currency": "USD" } ],
   "parties": [ { "tag_ids": ["tag-2"], "role": "borrower" }, { "tag_ids": ["tag-8"], "role": "agent" }, { "tag_ids": ["tag-8"], "role": "lender" }, { "tag_ids": ["tag-9"], "role": "lender", "kind": "collective" } ] }]
```
(`On June 24, 2026, the Company entered into a five-year credit agreement with JPMorgan Chase Bank, N.A., as administrative agent and a lender, and the other lenders party thereto, providing a $750 million term facility and a $750 million revolving facility`. No object for the `$1.5 billion` total; the maturity cites the `five-year` span and the closing date.)

Commitment increase with only the before total stated — one object, summed current figure:
```json
[{ "name": ["tag-2", "tag-3"], "instrument_type": "revolving_credit",
   "dates": [ { "kind": "amendment", "evidence": ["tag-1"], "normalized_date": "2026-02-10" } ],
   "amounts": [ { "kind": "commitment", "evidence": ["tag-4"], "normalized_amount": "200000000", "currency": "USD", "prior": true },
                { "kind": "commitment", "evidence": ["tag-4", "tag-5"], "normalized_amount": "250000000", "currency": "USD" } ],
   "parties": [ { "tag_ids": ["tag-7"], "role": "borrower" }, { "tag_ids": ["tag-6"], "role": "agent" } ] }]
```
(`increases the commitments under its existing $200 million revolving credit facility by $50 million`, new total never stated. Lender never named: no `lender` cluster.)

## Output constraints
- Return only the JSON array — no prose, no bare object outside it (validated).
- Cite only tag ids that exist in the document, each under a property allowed to cite that tag type (validated). A `normalized_date` or `normalized_amount` publishes only when a cited span parses to the same value, so cite the span that states it.
- Do not invent ids, parties, dates, or amounts; when the document states no value you can cite, return no entry.
- Omit any property the document says nothing about.
