## Background
You are an expert in corporate debt financing and SEC disclosure language. You will be given a document with XML tags already inserted around candidate spans. The tags are:

- `person`
- `organization`
- `debt_instrument`
- `date`
- `duration`
- `amount`
- `interest_rate`

Each tagged span has a unique `id` attribute. Use only those tagged spans and return structured JSON.

## Task
Return a JSON array with one object per distinct debt instrument mention cluster in the document: `[ { ... }, { ... } ]`. The array wrapper is required even when there is exactly one instrument (`[ { ... } ]`); return `[]` when there is none.

For each object, extract these properties when present:
- `name`
- `instrument_type`
- `dates`
- `amounts`
- `interest_rate`
- `status_event`
- `lenders`
- `lenders_known_incomplete`
- `other_interested_parties`

For dates, return one `dates` list per object, one entry per date the document states about that instrument:
- `dates`: `[{ "kind": "agreement" | "announcement" | "expected_closing" | "closing" | "maturity" | "commitment_termination", "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null, "prior": true }]` (`prior` is optional; omit it unless true)

The `kind` labels what the date is:
- `agreement`: the instrument's own `dated as of` date — the credit agreement, indenture supplement, or note itself. Not the date of a base indenture, purchase agreement, or amendment that merely governs it.
- `announcement`: when the instrument was announced — the pricing, launch, or commitment-letter date.
- `expected_closing`: a projected closing, as in `expected to close on or about July 6, 2026`. A projection, never a `closing`.
- `closing`: the date the instrument came into existence — its closing, issuance, funding, or effective date. For a facility that is being amended, this is the original facility's date when the text states it; the amendment's own date is the `status_date` of the `amended` event, not a new `closing`.
- `maturity`: when the borrowed money must be repaid — the final maturity or expiration of the obligation itself.
- `commitment_termination`: when the lender's obligation to lend ends — the close of a draw, availability, or revolving period — only when the document states one distinct from the maturity.
- `prior: true` marks a term stated as it stood before a change: in `extended the maturity date from June 28, 2026 to June 23, 2031`, `2026-06-28` is a `maturity` entry with `prior: true` and `2031-06-23` is the current `maturity` entry, both on the same object.

For the instrument's category, return one optional plain string:
- `instrument_type`: `"term_loan" | "revolving_credit" | "credit_line" | "note_bond"`
  - `term_loan`: a fixed advance repaid on a schedule or at maturity. Mortgages belong here.
  - `revolving_credit`: a committed facility that can be drawn, repaid, and redrawn.
  - `credit_line`: other borrowing availability that is not a committed revolver, such as an uncommitted or discretionary line, or a letter-of-credit-only facility.
  - `note_bond`: a security — notes, bonds, debentures, convertibles.
  Omit `instrument_type` when none of the four fits (leases, surety bonds) or the document does not say.

For the instrument's interest rate, return one optional object:
- `interest_rate`: `{ "kind": "fixed" | "floating", "rate_pct": "3.875" | null, "evidence": ["tag-..."] }`
  - `kind` is `fixed` when the instrument bears a stated rate, and `floating` when interest is set off a benchmark plus a margin.
  - `rate_pct` is the stated fixed or all-in rate as a numeric string of digits and at most one decimal point, without the percent sign. Leave it `null` for a floating rate: benchmarks and margins are not recorded.
  - `evidence` may contain `interest_rate` tag ids, or the instrument's own `debt_instrument` tag id when the coupon is embedded in the name, such as `3.875% senior notes due 2028`.
  Omit `interest_rate` when the document states nothing about the instrument's interest.

For the instrument's state, return one optional event:
- `status_event`: `{ "status": "announced" | "entered_into" | "amended" | "terminated" | "repaid" | "exchanged" | "defaulted", "status_date": { "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null } | null }`

The `status` labels what this mention says happened to the instrument:
- `announced`: the instrument is disclosed before it exists — a priced or launched offering, a signed commitment letter, `expected to close on or about July 6`. An `announced` instrument gets no `closing` date; it has not started. Its projected close is an `expected_closing` entry.
- `entered_into`: the instrument closed, was issued, or became effective. The normal case for a new agreement or issuance.
- `amended`: the instrument's terms were modified. The one object for the amended instrument carries `amended`, its new terms as current entries, and the stated old terms as `prior: true` entries.
- `terminated`: the agreement or facility was ended before its scheduled date, as in `On June 2, 2026, the Company terminated its $3.5 billion revolving credit facility`. Termination often co-occurs with a final repayment; when the filing's point is that the facility ended, use `terminated`.
- `repaid`: the obligation was or will be satisfied **in full** by payment — repaid in full, redeemed in whole, defeased. A redemption target of a use-of-proceeds financing is `repaid`. A repurchase or redemption of less than all outstanding principal is not `repaid`: record the figure as a `repayment` entry in `amounts` and return no `status_event`, unless the text says the remaining obligation was also satisfied.
- `exchanged`: the obligation was satisfied by delivering other securities or equity instead of cash.
- `defaulted`: the filing reports a default, event of default, or acceleration of the obligation (the Item 2.04 vocabulary).

`status_date` is the date the event happened or takes effect, when the text states one: the termination date, the redemption date, the closing date. Cite `date` tag ids. Omit `status_event` entirely when the mention states no event — a facility merely described in passing has no status.

For money, return one `amounts` list per object, one entry per money fact the document states about that instrument:
- `amounts`: `[{ "kind": "commitment" | "principal" | "outstanding_balance" | "draw" | "repayment" | "proceeds", "evidence": ["tag-..."], "normalized_amount": "12345.67" | null, "currency": "USD" | null, "as_of_date": "YYYY-MM-DD" | null, "prior": true }]` (`prior` is optional; omit it unless true)

The `kind` labels what the money fact is:
- `commitment`: the maximum available under a facility, drawn or not, as in `provides for a $500 million revolving credit facility` or `commitments increased to $1.75 billion`.
- `principal`: the face amount actually issued or borrowed, as in `issued $400 million of 4.875% Senior Notes` or `a $183.36 million term loan`.
- `outstanding_balance`: the amount owed as of a date, as in `as of June 9, 2026, we had $270.5 million outstanding`. Cite the stated as-of date in `as_of_date` when the document gives one.
- `draw`: one borrowing under an existing facility, as in `borrowed $50 million under the Revolving Credit Agreement`.
- `repayment`: an amount paid down or redeemed, as in `repaid $68 million in outstanding amounts` or a stated payoff amount.
- `proceeds`: offering proceeds, gross or net, as in `net proceeds of $718.8 million`. Proceeds are not the principal: discounts and fees separate them.

For party properties, return one object per coreference cluster:
- `lenders`: `[{ "tag_ids": ["tag-..."], "kind": "named" | "collective" }]`
- `lenders_known_incomplete`: `true` | `false`
- `other_interested_parties`: `[{ "tag_ids": ["tag-..."], "role": "agent" | "trustee" | "underwriter" | "guarantor" | "borrower" | "other" }]`

## Hard Rules
- Return one JSON object per concrete debt instrument described as its own obligation in the document.
- Do not return agreements as objects.
- `name` may contain only `debt_instrument` tag ids.
- Each `dates` entry's `evidence` may contain only `date` tag ids, except a `maturity` entry, which may also cite the instrument's own `debt_instrument` tag id when the maturity is embedded in the name, such as `3.875% senior notes due 2028`, or a `duration` span for the tenor arithmetic below. A closing date is never stated inside a name, so never cite the name as `closing` evidence; when the document states no date you can cite, return no entry rather than guessing. An instrument whose `status_event` is `announced` has not yet come into existence and never gets a `closing` entry.
- `maturity` and `commitment_termination` answer different questions — when the money must be repaid versus when the lender stops lending — so never file one as the other. When the text states one date for a facility's end, it is the `maturity`. When it states both an availability or draw-period end and a repayment date, return both entries. When it states only a draw-period or commitment-termination end, return `commitment_termination` and no `maturity`.
- When the document states a facility's tenor and its closing date but never the maturity — `entered into a five-year revolving credit facility` on a stated date — return a `maturity` entry citing **both** the `duration` span and the closing `date` span, with `normalized_date` equal to the closing date advanced by the tenor. The same arithmetic runs backwards: `extended six months to September 3, 2027` states the prior maturity as `2027-03-03`. Like summed amounts, this is arithmetic on exactly the cited spans; never compute a date from a tenor the document does not state, and prefer a stated date over the computation whenever one exists.
- At most one current entry per `kind` on one object. Two different current `closing` or `maturity` dates for what looks like one instrument are two instruments.
- Each `amounts` entry's `evidence` may contain `amount` tag ids, or the instrument's own `debt_instrument` tag id when the principal is stated inside the name, such as `$183.36 million term loan`.
- `lenders` and `other_interested_parties` cluster `tag_ids` may contain only `person` or `organization` tag ids. Never cite a `debt_instrument`, `agreement`, `amount`, or `date` tag id in a party cluster.
- For `name`, return one list of tag ids representing a single coreference cluster.
- Every `lenders` cluster must carry a `kind`, and every `other_interested_parties` cluster must carry a `role`.
- `lenders_known_incomplete` is optional and must be `true` or `false`.
- Do not use an aggregate amount that covers several instruments in any single instrument's `amounts`. When the document states only a combined total for a group, such as the total principal subject to one amendment, omit that figure on the individual instruments.
- Interest rates, margins, spreads, fees, discounts, and per-annum percentages are never `amounts` entries of any kind. Omit `amounts` when the document states no money amount for the instrument.
- A stated balance, draw, repayment, or proceeds figure belongs in `amounts` under its own `kind`, never as `commitment` or `principal`. When the document states both a facility size and a balance, return both entries.
- For each entry's `normalized_amount`, return only digits and at most one decimal point, or `null`.
- For each entry's `currency`, return one 3-letter ISO 4217 currency code or `null`.
- For each entry's `as_of_date`, return `YYYY-MM-DD` when the document states the date the figure is measured at, and `null` otherwise. Only balances normally carry one.
- For each `dates` entry's `normalized_date`, return `YYYY-MM-DD` or `null`. When a maturity gives only a year, such as `due 2028`, return `2028-12-31`; when it gives a month, such as `due April 2033` or `matures in June 2016`, return the last day of that month.
- Each stated date goes in exactly one entry under its own kind. A pricing date is an `announcement`, never a `closing`; a `dated as of` date is an `agreement`; a projected close is an `expected_closing`. When the document states the instrument's own closing or issuance date, that is the `closing`; the date the filing says an agreement was `entered into` is the `closing` only when no separate closing or issuance date is stated.
- The `dated as of` date of a base indenture, pooling and servicing agreement, purchase agreement, or amendment is never the instrument's `agreement` or `closing` date unless the text says the instrument itself carries that date.
- In an amended and restated agreement, a facility the text describes as `existing` keeps its original date as `closing` when the text states one, and otherwise gets no `closing`. The restatement date is the `agreement` date of the amended instrument and the `status_date` of its `amended` event, never its `closing`.
- A draw or advance under an existing note or facility does not restate that instrument's dates: the instrument keeps its own stated dates, not the draw date.
- Never guess a maturity that the document does not state, and never reuse a closing date as a maturity.
- If a property is absent, omit it.
- Do not invent ids, parties, dates, or amounts.
- Return only the JSON array, with no extra text and no bare object outside it.

Selection rules:
- Ignore debt-like mentions that are only passing background to some other transaction, such as proceeds used to `repay existing indebtedness` or `repay outstanding borrowings` where the older debt is never named with any concrete term.
- When a filing says the proceeds of a new financing will redeem, repay, or retire an older instrument that is named with at least one concrete term — a rate, a maturity, or an amount — return that older instrument as its own object recording what the text states about it. Being the target of a use-of-proceeds redemption is a debt instrument state this schema records, not ignorable background.
- A redemption target named only by a defined term and a `dated as of` date also counts, but only when the text says the instrument itself is repaid in full, redeemed, or retired. Repaying `outstanding borrowings under` a facility retires the borrowings, not the facility, and is not a retirement of that instrument.
- When the filing's subject is a specific named instrument being redeemed, repaid, cancelled, exchanged, refinanced, terminated, or amended, return that instrument and record what the filing states about it. The retirement or amendment itself is information about that instrument, not a reason to drop it.
- When the filing's subject amends, restates, supplements, increases, or extends an instrument that carries on as the same obligation, return **one** object for it: `status_event` `amended`, the new terms as current `amounts` and `dates` entries, and every stated old term as an entry with `prior: true`. `increasing the Revolving Credit Commitment from $25,000,000 to $50,000,000` is one object with a `commitment` of `50000000` and a `prior: true` `commitment` of `25000000`; `extended the maturity date from June 28, 2026 to June 23, 2031` is one object with a `maturity` of `2031-06-23` and a `prior: true` `maturity` of `2026-06-28`. Never invent a prior term the filing does not state, and never publish the pre-change figure as the current one.
- When the text gives no before figure, the amendment changed nothing this schema records — a covenant reset, a repricing, a joinder — and the one object simply carries `amended` and the current terms.
- A separate predecessor object is for a **different** instrument the filing replaces: a new facility that `refinances and replaces` an existing facility, new notes whose proceeds redeem old notes. Return the replaced instrument as its own object recording what the text states about it — its `dated as of` date as `agreement`, its stated closing as `closing`, its commitment or maturity — and do not fold it into the new instrument. A facility the filing mentions for some other reason, such as a party to an intercreditor agreement or a facility that merely continues to exist, is not a predecessor.
- When the text names a group of replaced instruments, such as `its existing term loan and revolving credit facilities`, return one object per facility rather than one object for the phrase, and give each successor facility its own counterpart. Never return an object whose `name` covers more than one facility.
- Return one object per instrument, not one per way of describing it. When the document offers several phrases for the same debt, such as `working capital loans` and `time extension funding loans` for one group of notes, put all of those tag ids in that object's single `name` cluster rather than repeating the object once per phrase.
- Naming a borrowing by its agreement and naming it by what the agreement provides describes one instrument, not two. When an item calls the same $835 million revolver both the `Credit Agreement` and the `revolving credit facility`, return one object with both spans in its `name` cluster. Return a separate object for the agreement only when it is a predecessor being amended, restated, refinanced, or replaced, or when it establishes more than one facility, in which case each facility is its own object.
- When the document introduces a defined term for an instrument it has just described, such as `(the "Initial Note")`, `(the "Prior Credit Agreement")`, or a later bare `the Note`, include that defined-term span in the same object's `name` cluster as the descriptive span. Do not return a separate object for the defined term.
- Ignore collective labels that only group multiple concrete instruments described elsewhere in the same document, such as `Exchange Notes` or generic `Notes`, when the underlying instruments can be extracted separately.
- Ignore non-debt securities even if they appear in the same financing disclosure.
- A returned object should correspond to one coherent debt instrument.
- A single debt instrument should have at most one current `closing` date and at most one current `commitment` or `principal` entry. If the document presents two different closing dates, or two different commitment or principal figures for what looks like one instrument and neither is described as the figure before a change, that is strong evidence there are two separate debt instruments and you should return two objects. Different `kind` entries — a commitment plus a balance plus a repayment — and a current figure plus its `prior: true` predecessor describe one instrument and never force a split.
- An increase *by* an amount is never an object sized at the increment. `increased the commitments by $353 million` and a `$500,000 Credit Increase` describe a change to one facility, which is one object. What its `amounts` hold depends on which totals the text states:
  - Before and after totals stated: the after total as the current `commitment`, the before total as a `prior: true` `commitment`.
  - Only the before total stated (`its existing $200 million facility ... increased by $50 million`): the before total as a `prior: true` `commitment`, and the arithmetic result as the current `commitment` — `normalized_amount` `250000000` — citing **both** the prior-total span and the increment span as `evidence`. Summing the exact cited spans is the only arithmetic you may ever do; never carry the pre-increase total as the current figure.
  - Only the increment stated: omit the increment from `amounts`.
  An increase *to* an amount is the current total.
- An over-allotment or add-on folded into a stated total is one instrument. When the text says notes were issued `including` an over-allotment exercise, or gives an add-on `bringing the total to` one figure, return one object carrying the total. Contrast a genuinely separate second issuance, with its own date or its own stated principal held by its own parties, which is two objects.
- Multiple returned objects may share the same `name` evidence tags when the text clearly describes multiple distinct instruments using the same name phrase.
- A securities offering that lists multiple classes, tranches, or series, such as `Class A-1`, `Class A-2a`, `Class A-3`, or `Series A` and `Series B`, is multiple debt instruments. Return one object per class, tranche, or series, even when the document names them together in one sentence, and even when only some of them state their own amount or maturity.
- Each object's `name` must refer to a single class, tranche, or series. Never merge several of them into one object, and never return an extra object for the group label, such as `Asset Backed Notes` or `Notes`, that only collects them.
- Split by class only when the document gives each class its own identity, such as its own tagged name, amount, or maturity. When several classes appear only inside one combined tagged span and the document states nothing specific to any single class, return one object for that span rather than repeating the same object several times.
- A credit agreement that establishes genuinely distinct facilities, such as a term loan facility and a revolving credit facility, is multiple debt instruments. Return one object per facility, each with its own commitment amount when stated, and never assign one facility's commitment, or the agreement's combined total, to another facility.
- A single facility's borrowing mechanics are not separate debt instruments. Swing line loans, letters of credit, LC loans, and similar sub-limits available under a revolving or working capital facility are ways to draw that facility. Return one object for the facility rather than one object per mechanic, unless the document describes a mechanic as its own facility with its own commitment. When only the mechanics are tagged as `debt_instrument` spans, still return exactly one object: name it with the primary mechanic's span, such as the revolving or working capital loans, and give it the facility's total commitment.
- Any property evidence may be shared across multiple returned objects when the text says the property applies to all of them, including `name`, `dates`, `amounts`, `lenders`, and `other_interested_parties`.

Party rules:
- Use `kind: "named"` for a cluster that identifies a specific lender by name, such as `JPMorgan Chase Bank, N.A.` or `EGT 11 LLC`.
- Use `kind: "collective"` for a cluster whose surface text only describes the group without identifying anyone, such as `the Lenders`, `the other lenders party thereto`, `the holders`, `certain financial institutions`, or `the purchasers`.
- A defined term that stands for a list of parties the document just named, such as `(collectively, the "Purchasers")` or `the Lenders listed on Schedule A`, is a coreference of those named parties rather than a `collective` cluster. Put its tag ids in the `named` clusters they refer to, or leave them out. Reserve `collective` for a group the document never enumerates.
- Return `lenders_known_incomplete: true` when the document signals lenders it does not name, which is the normal case when any `lenders` cluster is `collective` or when the text hedges with wording such as `certain lenders`, `including`, or `and others`. Otherwise omit it or return `false`.
- A collective phrase is a `collective` cluster only when the tagger labelled it `person` or `organization`. When the document refers to lenders it does not name and no party tag covers that phrase, return no cluster for it and set `lenders_known_incomplete: true`.
- An instrument placed into the public market, or sold to unnamed holders through underwriters or initial purchasers, has lenders the document does not name. Return no `lenders` clusters for the underwriters and set `lenders_known_incomplete: true`. Reserve `false` for an instrument whose counterparties the document names in full, such as a bilateral loan from one bank.
- The same holds beyond new offerings: a redemption notice for outstanding notes, debt described as assumed or outstanding, and a syndicated facility where only the arrangers or agents are named all concern holders or lenders the document never names. Set `lenders_known_incomplete: true` on those instruments too.
- The filer, issuer, borrower, or obligor is never its own lender. Put it in `other_interested_parties` with `role: "borrower"` only when the document treats it as a distinct party worth recording, and otherwise omit it.
- An administrative agent, collateral agent, or paying agent belongs in `other_interested_parties` with `role: "agent"`. Include it in `lenders` only when the document also describes it as a lender or purchaser of that instrument, for example `as a Lender and as Administrative Agent`.
- An indenture trustee or collateral trustee belongs in `other_interested_parties` with `role: "trustee"`, never in `lenders`.
- Underwriters, initial purchasers, placement agents, and sales agents in a public offering or Rule 144A resale belong in `other_interested_parties` with `role: "underwriter"`, never in `lenders`, because they resell the debt rather than hold it.
- In a note purchase agreement or private placement sold directly to investors, the `purchasers` ARE the lenders. Return them in `lenders`, using `kind: "named"` when they are named and `kind: "collective"` when the document only refers to `the Purchasers`.
- Guarantors belong in `other_interested_parties` with `role: "guarantor"`.
- A named party the document identifies as the holder, noteholder, payee, purchaser, or counterparty of the debt is a `named` lender, even when the document never uses the word `lender`. The rules above about parties that are never lenders cover agents, trustees, and underwriters only.
- A named party the document identifies as an initial holder or purchaser that will hold the debt rather than resell it is a `named` lender. `Initial purchasers` in a Rule 144A resale are underwriters, because they resell; a named investor that buys and holds is not.
- Use `role: "other"` only when the party is clearly related to the instrument but none of the other roles fit.
- Similarly named entities are not automatically one cluster. A filing can enumerate affiliated funds or series entities whose names differ only by a numeral or suffix; keep each in its own cluster unless the text says two names refer to the same party.

Examples:
- If a document describes `3.875% senior notes due 2028` and gives no separate maturity date, return a `maturity` entry citing that instrument's `debt_instrument` tag id with `normalized_date` `2028-12-31`.
- If a document describes a `$183.36 million term loan` and tags no separate amount, return one `amounts` entry with `kind` `principal`, the instrument's `debt_instrument` tag id as `evidence`, `normalized_amount` `183360000`, and `currency` `USD`.
- If a document describes `senior notes due October 1, 2028` and tags `October 1, 2028` as a date, the `maturity` entry cites the `date` tag id with `normalized_date` `2028-10-01`.
- If a delayed-draw facility's draw period ends `June 30, 2027` and its loans mature `June 30, 2031`, return a `commitment_termination` entry `2027-06-30` and a `maturity` entry `2031-06-30`.
- If a company enters into a `five-year` senior secured revolving credit facility on `June 24, 2026` and the item never states the maturity, return a `closing` entry `2026-06-24` and a `maturity` entry with `normalized_date` `2031-06-24` citing the `five-year` duration span and the `June 24, 2026` date span.
- If a filing states only a Draw Period Termination Date and defines the maturity relative to it, such as `12 months after the Draw Period Termination Date`, return the stated date as `commitment_termination` and no `maturity` entry: never publish an availability end as the maturity.
- If a company prices `$600 million of 4.800% Senior Notes due 2036` on `March 5, 2026` in an offering `expected to close on March 12, 2026`, return one object with `status_event` `announced` dated `2026-03-05`, an `announcement` entry `2026-03-05`, an `expected_closing` entry `2026-03-12`, a `maturity` entry `2036-12-31`, and no `closing` entry.
- If a credit agreement says ABR Loans bear interest at `0.875% per annum`, do not return `0.875` in `amounts`. That margin belongs nowhere: the loan's `interest_rate` is `{ "kind": "floating", "rate_pct": null }`, citing the tagged rate span. Omit `amounts` unless the document states a money amount for that loan.
- If a document describes `3.875% senior notes due 2028` with no separate rate span, return `interest_rate` `{ "kind": "fixed", "rate_pct": "3.875" }`, citing the instrument's own tag id.
- If a company closes a `$1.2 billion` working capital facility that provides revolving loans, swing line loans up to `$25 million`, and letters of credit, return one object with one `commitment` entry of `1200000000`, named by the facility's tagged span when present and otherwise by the `Revolving Loans` span. Do not return additional objects for `Swing Line Loans` or `Letters of Credit`, and never give any single mechanic the `$1.2 billion` total.
- If a credit agreement provides a `$750 million` term facility and a `$750 million` revolving facility, return two objects, each with its own `commitment` entry of `750000000`. Do not return a third object for the agreement's `$1.5 billion` combined total.
- If a company enters into a commitment increase and maturity extension agreement for its revolving credit agreement dated as of August 1, 2025, raising commitments to `$1.75 billion` and extending the maturity from August 1, 2030 to August 1, 2031, return exactly one object: `status_event` `amended`, an `agreement` entry `2025-08-01`, a `commitment` entry of `1750000000`, a `maturity` entry `2031-08-01`, and a `prior: true` `maturity` entry `2030-08-01`. No `prior` commitment, because the prior commitment is not stated.
- If an amendment `reduced the lender commitments from $100,000,000 to $50,000,000` and `extended the maturity date from June 28, 2026 to June 23, 2031`, return one object with a current `commitment` of `50000000`, a `prior: true` `commitment` of `100000000`, a current `maturity` of `2031-06-23`, and a `prior: true` `maturity` of `2026-06-28`.
- If an amendment only resets a financial covenant, reprices a margin, or adds a guarantor, and states no prior commitment or maturity, return one object for the facility with `status_event` `amended` and no `prior` entries. Never return extra partial variants of the same facility.
- If a new credit facility `refinances and replaces` the company's existing revolving credit facility dated as of May 29, 2019, return the new facility and a second object for the replaced facility with an `agreement` entry `2019-05-29`.
- If a trust issues `Class A-1 Asset Backed Notes`, `Class A-2a Asset Backed Notes`, `Class A-2b Asset Backed Notes`, `Class A-3 Asset Backed Notes`, and `Class A-4 Asset Backed Notes` in one offering, return five objects, one per class, each with its own amount and maturity when stated. Do not return one object naming all five, and do not return a sixth object for `Asset Backed Notes`.
- If a document says the company issued an initial note on March 17, 2025 for $5.5 million and a subsequent note on March 20, 2025 for $269,000, both called `Senior Subordinated Convertible Promissory Note`, return two objects, each with its own `principal` entry.
- If a document later refers collectively to those instruments as `Exchange Notes`, do not return a third `Exchange Notes` object.
- If a document says prior notes were retired in full, do not return a new object just for that contextual mention unless the filing separately describes a concrete debt instrument state for it.
- If a company issues new senior notes and states that the proceeds will be used to redeem its outstanding `5.25% Senior Notes due 2027`, return an object for the 2027 notes as well. The redemption target is named with concrete terms, and its redemption is a state this schema records.
- If a company issues new senior notes and states that the proceeds will be used to `repay existing indebtedness` or to `repay outstanding borrowings under its revolving credit facility`, stating no rate, maturity, or amount for what is repaid, do not return an object for the repaid debt.
- If a 1.02 item says `on June 2, 2026, the Company terminated its $3.5 billion five-year revolving credit facility dated as of October 11, 2023`, return that facility with `status_event` `terminated`, `status_date` `2026-06-02`, and an `agreement` entry `2023-10-11`. The termination is recordable even though no successor instrument appears in the item.
- If a company issues new notes whose proceeds will redeem its `5.25% Senior Notes due 2027`, the 2027 notes' object carries `status_event` `repaid`, with the stated redemption date as `status_date` when the text gives one. The new notes carry `entered_into`, or `announced` if the offering has not yet closed.
- If a company prices an offering `expected to close on or about July 6, 2026`, return the notes with `status_event` `announced`, an `expected_closing` entry `2026-07-06`, and no `closing` entry. The expected closing date is not a `closing` until the filing says the closing happened.
- If a company states that offering proceeds were used to repay in full its outstanding senior convertible notes (the `February Notes`) sold pursuant to a securities purchase agreement dated as of February 12, 2026, return an object for the February Notes with an `agreement` entry `2026-02-12`: the notes themselves are repaid, and the dated agreement identifies them even though no rate, maturity, or amount is stated.
- If a company states that proceeds were used to `repay outstanding borrowings under the Credit Agreement, dated as of March 1, 2024`, do not return a retirement object for the Credit Agreement: paying down borrowings leaves the facility in place.
- If a credit agreement says the lenders are `JPMorgan Chase Bank, N.A.` and `the other lenders party thereto`, return both clusters, `kind: "named"` for Chase and `kind: "collective"` for the other lenders, with `lenders_known_incomplete: true`.
- If the document says the lenders are `JPMorgan Chase Bank, N.A.` and `Wells Fargo Bank, National Association` with no collective phrase, return two `named` clusters and omit `lenders_known_incomplete`.
- If the document only says the notes were sold to `the Holders`, return one `collective` cluster with `lenders_known_incomplete: true`.
- If a note purchase agreement says the company sold notes to `Metropolitan Life Insurance Company` and `the other purchasers named therein`, return Metropolitan Life as `named` and the other purchasers as `collective`, with `lenders_known_incomplete: true`.
- If an indenture names `The Bank of New York Mellon` as trustee and the notes were sold through underwriters, return no `lenders` clusters, return the trustee with `role: "trustee"` and the underwriters with `role: "underwriter"`, and set `lenders_known_incomplete: true`, because the holders of the notes are never named.
- If a new credit agreement `refinances in full and extends the maturities of the Borrowers' existing term loan and revolving credit facilities`, return the new term facility, the new revolving facility, and two predecessor objects — one for the existing term loan facility and one for the existing revolving facility. Do not return a single predecessor named `term loan and revolving credit facilities`.
- If a filing describes a settlement in which one note is exchanged for a `10% Senior Secured Convertible Note` of `$1,250,000` (the `Initial Note`) and a warrant for a second of `$1,100,000` (the `Additional Note`), return two objects, each with its defined-term span in the same `name` cluster as its descriptive span. Do not return separate objects for `Initial Note` and `Additional Note`. The signal for two objects is two obligations with their own principals, not the words Initial and Additional.
- If a company issues `$287.5 million of convertible notes, including $37.5 million issued pursuant to the initial purchasers' over-allotment option`, return one object with one `principal` entry of `287500000`. The over-allotment is part of the same series, not a sibling note, and neither `250000000` nor `37500000` is its own object.
- If a filing says the notes are `working capital loans and time extension funding loans` totalling `$6.9 million`, consisting of `$2.9 million`, `$2.2 million`, and `$1.8 million` held by three parties, return three objects, one per note, each naming both phrases in one `name` cluster and carrying its own `principal` entry. Do not return six, and do not give any note the `$6.9 million` total.
- If a filing describes a `Credit Agreement` providing an `$835,000,000` `revolving credit facility` maturing `June 18, 2031`, return one object naming both spans, with a `commitment` entry of `835000000` and a `maturity` entry `2031-06-18`. Do not return one object for the agreement and another for the facility.
- If a 2.03 item says the company's revolving credit facility provides `$300 million` of commitments and that `as of June 9, 2026, we had $270.5 million outstanding`, return one object with two `amounts` entries: `kind` `commitment` `300000000`, and `kind` `outstanding_balance` `270500000` with `as_of_date` `2026-06-09`. The balance is never the `commitment` or `principal`.
- If a company states it will `repay $68 million in outstanding amounts under the credit facility`, that figure is a `repayment` entry on the facility's object, not its principal.
- If a company `repurchased, in a privately negotiated transaction, $100 million aggregate principal amount` of its `10.500% senior secured first lien notes due 2029`, and the series remains outstanding, return the notes with a `repayment` entry of `100000000` and no `status_event`. A partial repurchase is a payment, not the end of the obligation.
- If an amendment increases the commitments under a company's `existing $200 million revolving credit facility` by `$50 million` and never states the new total, return one object with `status_event` `amended`, a `prior: true` `commitment` entry of `200000000`, and a current `commitment` entry of `250000000` citing both the `$200 million` and `$50 million` spans. The facility's capacity is the sum; publishing the pre-increase `200000000` as current would be wrong.
- If an offering closes with `net proceeds of $718.8 million` from `$750 million` of notes, the notes' object carries a `principal` entry of `750000000` and a `proceeds` entry of `718800000`.
