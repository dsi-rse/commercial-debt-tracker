## Background
You are an expert in corporate debt financing and SEC disclosure language. You will be given a document with XML tags already inserted around candidate spans. The tags are:

- `person`
- `organization`
- `debt_instrument`
- `date`
- `amount`

Each tagged span has a unique `id` attribute. Use only those tagged spans and return structured JSON.

## Task
Return one JSON object per distinct debt instrument mention cluster in the document.

For each object, extract these properties when present:
- `name`
- `start_date`
- `end_date`
- `amount`
- `lenders`
- `lenders_known_incomplete`
- `other_interested_parties`

For standardized single-value properties, use these object shapes:
- `start_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null }`
- `end_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null }`
- `amount`: `{ "evidence": ["tag-..."], "normalized_amount": "12345.67" | null, "currency": "USD" | null }`

For party properties, return one object per coreference cluster:
- `lenders`: `[{ "tag_ids": ["tag-..."], "kind": "named" | "collective" }]`
- `lenders_known_incomplete`: `true` | `false`
- `other_interested_parties`: `[{ "tag_ids": ["tag-..."], "role": "agent" | "trustee" | "underwriter" | "guarantor" | "borrower" | "other" }]`

## Hard Rules
- Return one JSON object per concrete debt instrument described as its own obligation in the document.
- Do not return agreements as objects.
- `name` may contain only `debt_instrument` tag ids.
- `start_date.evidence` may contain only `date` tag ids. The `debt_instrument` allowance below is specific to `end_date` and `amount`, because a maturity or a principal can be stated inside a name while an issuance date never is. When the document states no date you can cite, omit `start_date` rather than citing the instrument's name.
- `end_date.evidence` may contain `date` tag ids, or the instrument's own `debt_instrument` tag id when the maturity is embedded in the name, such as `3.875% senior notes due 2028`.
- `amount.evidence` may contain `amount` tag ids, or the instrument's own `debt_instrument` tag id when the principal is stated inside the name, such as `$183.36 million term loan`.
- `lenders` and `other_interested_parties` cluster `tag_ids` may contain only `person` or `organization` tag ids. Never cite a `debt_instrument`, `agreement`, `amount`, or `date` tag id in a party cluster.
- For `name`, return one list of tag ids representing a single coreference cluster.
- Every `lenders` cluster must carry a `kind`, and every `other_interested_parties` cluster must carry a `role`.
- `lenders_known_incomplete` is optional and must be `true` or `false`.
- Do not use an aggregate amount that covers several instruments as the `amount` of any one of them. When the document states only a combined total for a group, such as the total principal subject to one amendment, omit `amount` on the individual instruments.
- `amount` is the principal or commitment amount only. Interest rates, margins, spreads, fees, discounts, and per-annum percentages are never `amount`. Omit `amount` when the document states no principal or commitment amount.
- For `amount.normalized_amount`, return only digits and at most one decimal point, or `null`.
- For `amount.currency`, return one 3-letter ISO 4217 currency code or `null`.
- For `start_date.normalized_date` and `end_date.normalized_date`, return `YYYY-MM-DD` or `null`.
- When a maturity gives only a year, such as `due 2028`, return `2028-12-31` as `end_date.normalized_date`.
- `start_date` is the date the instrument came into existence: the closing, issuance, or effective date the document states for it. When the filing opens with `On <date>, the Company entered into`, `issued`, or `closed`, that is the `start_date` unless the text gives the instrument its own different date. The `dated as of` date of an indenture, purchase agreement, or amendment is not the instrument's start date unless the text says the instrument itself carries that date.
- Never guess a maturity that the document does not state, and never reuse the start date as the end date.
- If a property is absent, omit it.
- Do not invent ids, parties, dates, or amounts.
- Return only valid JSON with no extra text.

Selection rules:
- Ignore debt-like mentions that are only passing background to some other transaction, such as proceeds used to `repay existing indebtedness` or `repay outstanding borrowings` where the older debt is never named with any concrete term.
- When a filing says the proceeds of a new financing will redeem, repay, or retire an older instrument that is named with at least one concrete term — a rate, a maturity, or an amount — return that older instrument as its own object recording what the text states about it. Being the target of a use-of-proceeds redemption is a debt instrument state this schema records, not ignorable background.
- When the filing's subject is a specific named instrument being redeemed, repaid, cancelled, exchanged, refinanced, terminated, or amended, return that instrument and record what the filing states about it. The retirement or amendment itself is information about that instrument, not a reason to drop it.
- When the filing's subject amends, restates, supplements, increases, extends, refinances, or replaces an identified predecessor instrument, return the predecessor as its own additional object recording what the text states about it: its `dated as of` date as `start_date`, and its prior commitment or maturity when stated. Do not fold the predecessor into the object for the instrument as amended or replaced, and do not treat the predecessor reference as ignorable background. The object for the instrument as amended records the new terms.
- The predecessor is the instrument the filing says is being amended, restated, refinanced, or replaced, normally identified by a `dated as of` date. A facility the filing mentions for some other reason, such as a party to an intercreditor agreement or a facility that merely continues to exist, is not the predecessor. When the filing names the replaced agreement and its date, that date is the predecessor's `start_date`.
- When the text names a group of predecessors, such as `its existing term loan and revolving credit facilities`, return one predecessor object per facility rather than one object for the phrase, and give each successor facility its own counterpart. Never return a predecessor object whose `name` covers more than one facility.
- When the filing states the predecessor's prior commitment or maturity, as in `commitments will be decreased from $100,000,000 to $54,000,000`, the predecessor object carries the prior figure and the object for the instrument as amended carries the new one. Do not omit the predecessor when its prior terms are stated, and never give the predecessor the new figure.
- A term stated both before and after the change is the signal that there are two objects to return. `increasing the Revolving Credit Commitment from $25,000,000 to $50,000,000` and `extended the maturity date from June 28, 2026 to June 23, 2031` each describe two states: the earlier figure is the predecessor's and the later figure is the amended instrument's. Return both objects whenever the text gives a before figure.
- When the text gives no before figure, the amendment changed nothing this schema records — a covenant reset, a repricing, a joinder — and one object is correct. Do not invent a predecessor whose terms the filing never states.
- Return one object per instrument, not one per way of describing it. When the document offers several phrases for the same debt, such as `working capital loans` and `time extension funding loans` for one group of notes, put all of those tag ids in that object's single `name` cluster rather than repeating the object once per phrase.
- Naming a borrowing by its agreement and naming it by what the agreement provides describes one instrument, not two. When an item calls the same $835 million revolver both the `Credit Agreement` and the `revolving credit facility`, return one object with both spans in its `name` cluster. Return a separate object for the agreement only when it is a predecessor being amended, restated, refinanced, or replaced, or when it establishes more than one facility, in which case each facility is its own object.
- When the document introduces a defined term for an instrument it has just described, such as `(the "Initial Note")`, `(the "Prior Credit Agreement")`, or a later bare `the Note`, include that defined-term span in the same object's `name` cluster as the descriptive span. Do not return a separate object for the defined term.
- Ignore collective labels that only group multiple concrete instruments described elsewhere in the same document, such as `Exchange Notes` or generic `Notes`, when the underlying instruments can be extracted separately.
- Ignore non-debt securities even if they appear in the same financing disclosure.
- A returned object should correspond to one coherent debt instrument.
- A single debt instrument should have at most one start date and one principal amount. If the document presents two different start dates or two different principal amounts, that is strong evidence there are two separate debt instruments and you should return two objects.
- Multiple returned objects may share the same `name` evidence tags when the text clearly describes multiple distinct instruments using the same name phrase.
- A securities offering that lists multiple classes, tranches, or series, such as `Class A-1`, `Class A-2a`, `Class A-3`, or `Series A` and `Series B`, is multiple debt instruments. Return one object per class, tranche, or series, even when the document names them together in one sentence, and even when only some of them state their own amount or maturity.
- Each object's `name` must refer to a single class, tranche, or series. Never merge several of them into one object, and never return an extra object for the group label, such as `Asset Backed Notes` or `Notes`, that only collects them.
- Split by class only when the document gives each class its own identity, such as its own tagged name, amount, or maturity. When several classes appear only inside one combined tagged span and the document states nothing specific to any single class, return one object for that span rather than repeating the same object several times.
- A credit agreement that establishes genuinely distinct facilities, such as a term loan facility and a revolving credit facility, is multiple debt instruments. Return one object per facility, each with its own commitment amount when stated, and never assign one facility's commitment, or the agreement's combined total, to another facility.
- A single facility's borrowing mechanics are not separate debt instruments. Swing line loans, letters of credit, LC loans, and similar sub-limits available under a revolving or working capital facility are ways to draw that facility. Return one object for the facility rather than one object per mechanic, unless the document describes a mechanic as its own facility with its own commitment. When only the mechanics are tagged as `debt_instrument` spans, still return exactly one object: name it with the primary mechanic's span, such as the revolving or working capital loans, and give it the facility's total commitment.
- Any property evidence may be shared across multiple returned objects when the text says the property applies to all of them, including `name`, `start_date`, `end_date`, `amount`, `lenders`, and `other_interested_parties`.

Party rules:
- Use `kind: "named"` for a cluster that identifies a specific lender by name, such as `JPMorgan Chase Bank, N.A.` or `EGT 11 LLC`.
- Use `kind: "collective"` for a cluster whose surface text only describes the group without identifying anyone, such as `the Lenders`, `the other lenders party thereto`, `the holders`, `certain financial institutions`, or `the purchasers`.
- A defined term that stands for a list of parties the document just named, such as `(collectively, the "Purchasers")` or `the Lenders listed on Schedule A`, is a coreference of those named parties rather than a `collective` cluster. Put its tag ids in the `named` clusters they refer to, or leave them out. Reserve `collective` for a group the document never enumerates.
- Return `lenders_known_incomplete: true` when the document signals lenders it does not name, which is the normal case when any `lenders` cluster is `collective` or when the text hedges with wording such as `certain lenders`, `including`, or `and others`. Otherwise omit it or return `false`.
- A collective phrase is a `collective` cluster only when the tagger labelled it `person` or `organization`. When the document refers to lenders it does not name and no party tag covers that phrase, return no cluster for it and set `lenders_known_incomplete: true`.
- An instrument placed into the public market, or sold to unnamed holders through underwriters or initial purchasers, has lenders the document does not name. Return no `lenders` clusters for the underwriters and set `lenders_known_incomplete: true`. Reserve `false` for an instrument whose counterparties the document names in full, such as a bilateral loan from one bank.
- The filer, issuer, borrower, or obligor is never its own lender. Put it in `other_interested_parties` with `role: "borrower"` only when the document treats it as a distinct party worth recording, and otherwise omit it.
- An administrative agent, collateral agent, or paying agent belongs in `other_interested_parties` with `role: "agent"`. Include it in `lenders` only when the document also describes it as a lender or purchaser of that instrument, for example `as a Lender and as Administrative Agent`.
- An indenture trustee or collateral trustee belongs in `other_interested_parties` with `role: "trustee"`, never in `lenders`.
- Underwriters, initial purchasers, placement agents, and sales agents in a public offering or Rule 144A resale belong in `other_interested_parties` with `role: "underwriter"`, never in `lenders`, because they resell the debt rather than hold it.
- In a note purchase agreement or private placement sold directly to investors, the `purchasers` ARE the lenders. Return them in `lenders`, using `kind: "named"` when they are named and `kind: "collective"` when the document only refers to `the Purchasers`.
- Guarantors belong in `other_interested_parties` with `role: "guarantor"`.
- A named party the document identifies as the holder, noteholder, payee, purchaser, or counterparty of the debt is a `named` lender, even when the document never uses the word `lender`. The rules above about parties that are never lenders cover agents, trustees, and underwriters only.
- A named party the document identifies as an initial holder or purchaser that will hold the debt rather than resell it is a `named` lender. `Initial purchasers` in a Rule 144A resale are underwriters, because they resell; a named investor that buys and holds is not.
- Use `role: "other"` only when the party is clearly related to the instrument but none of the other roles fit.

Examples:
- If a document describes `3.875% senior notes due 2028` and gives no separate maturity date, return that instrument's `debt_instrument` tag id as `end_date.evidence` with `normalized_date` `2028-12-31`.
- If a document describes a `$183.36 million term loan` and tags no separate amount, return that instrument's `debt_instrument` tag id as `amount.evidence` with `normalized_amount` `183360000` and `currency` `USD`.
- If a document describes `senior notes due October 1, 2028` and tags `October 1, 2028` as a date, cite the `date` tag id with `normalized_date` `2028-10-01`.
- If a credit agreement says ABR Loans bear interest at `0.875% per annum`, do not return `0.875` as the `amount`. Omit `amount` unless the document states that loan's principal or commitment amount.
- If a company closes a `$1.2 billion` working capital facility that provides revolving loans, swing line loans up to `$25 million`, and letters of credit, return one object with `normalized_amount` `1200000000`, named by the facility's tagged span when present and otherwise by the `Revolving Loans` span. Do not return additional objects for `Swing Line Loans` or `Letters of Credit`, and never give any single mechanic the `$1.2 billion` total.
- If a credit agreement provides a `$750 million` term facility and a `$750 million` revolving facility, return two objects, each with its own `750000000` amount. Do not return a third object for the agreement's `$1.5 billion` combined total.
- If a company enters into a commitment increase and maturity extension agreement for its revolving credit agreement dated as of August 1, 2025, raising commitments to `$1.75 billion` and extending the maturity from August 1, 2030 to August 1, 2031, return exactly two objects for that facility: the facility as amended, with `normalized_amount` `1750000000`, `start_date` `2025-08-01`, and `end_date` `2031-08-01`, and the predecessor, with `start_date` `2025-08-01`, `end_date` `2030-08-01`, and no `amount` because the prior commitment is not stated.
- If an amendment `reduced the lender commitments from $100,000,000 to $50,000,000` and `extended the maturity date from June 28, 2026 to June 23, 2031`, return two objects for that facility: the predecessor with `normalized_amount` `100000000` and `end_date` `2026-06-28`, and the instrument as amended with `normalized_amount` `50000000` and `end_date` `2031-06-23`. Both before figures are stated, so both belong to the predecessor.
- If an amendment only resets a financial covenant, reprices a margin, or adds a guarantor, and states no prior commitment or maturity, return one object for the facility. There is no second state to record. Never give the predecessor the increased total, and never return extra partial variants of the same facility.
- If a new credit facility `refinances and replaces` the company's existing revolving credit facility dated as of May 29, 2019, return the new facility and a second object for the replaced facility with `start_date` `2019-05-29`.
- If a trust issues `Class A-1 Asset Backed Notes`, `Class A-2a Asset Backed Notes`, `Class A-2b Asset Backed Notes`, `Class A-3 Asset Backed Notes`, and `Class A-4 Asset Backed Notes` in one offering, return five objects, one per class, each with its own amount and maturity when stated. Do not return one object naming all five, and do not return a sixth object for `Asset Backed Notes`.
- If a document says the company issued an initial note on March 17, 2025 for $5.5 million and a subsequent note on March 20, 2025 for $269,000, both called `Senior Subordinated Convertible Promissory Note`, return two objects.
- If a document later refers collectively to those instruments as `Exchange Notes`, do not return a third `Exchange Notes` object.
- If a document says prior notes were retired in full, do not return a new object just for that contextual mention unless the filing separately describes a concrete debt instrument state for it.
- If a company issues new senior notes and states that the proceeds will be used to redeem its outstanding `5.25% Senior Notes due 2027`, return an object for the 2027 notes as well. The redemption target is named with concrete terms, and its redemption is a state this schema records.
- If a company issues new senior notes and states that the proceeds will be used to `repay existing indebtedness` or to `repay outstanding borrowings under its revolving credit facility`, stating no rate, maturity, or amount for what is repaid, do not return an object for the repaid debt.
- If a credit agreement says the lenders are `JPMorgan Chase Bank, N.A.` and `the other lenders party thereto`, return both clusters, `kind: "named"` for Chase and `kind: "collective"` for the other lenders, with `lenders_known_incomplete: true`.
- If the document says the lenders are `JPMorgan Chase Bank, N.A.` and `Wells Fargo Bank, National Association` with no collective phrase, return two `named` clusters and omit `lenders_known_incomplete`.
- If the document only says the notes were sold to `the Holders`, return one `collective` cluster with `lenders_known_incomplete: true`.
- If a note purchase agreement says the company sold notes to `Metropolitan Life Insurance Company` and `the other purchasers named therein`, return Metropolitan Life as `named` and the other purchasers as `collective`, with `lenders_known_incomplete: true`.
- If an indenture names `The Bank of New York Mellon` as trustee and the notes were sold through underwriters, return no `lenders` clusters, return the trustee with `role: "trustee"` and the underwriters with `role: "underwriter"`, and set `lenders_known_incomplete: true`, because the holders of the notes are never named.
- If a new credit agreement `refinances in full and extends the maturities of the Borrowers' existing term loan and revolving credit facilities`, return the new term facility, the new revolving facility, and two predecessor objects — one for the existing term loan facility and one for the existing revolving facility. Do not return a single predecessor named `term loan and revolving credit facilities`.
- If a filing describes a settlement in which one note is exchanged for a `10% Senior Secured Convertible Note` of `$1,250,000` (the `Initial Note`) and a warrant for a second of `$1,100,000` (the `Additional Note`), return two objects, each with its defined-term span in the same `name` cluster as its descriptive span. Do not return separate objects for `Initial Note` and `Additional Note`.
- If a filing says the notes are `working capital loans and time extension funding loans` totalling `$6.9 million`, consisting of `$2.9 million`, `$2.2 million`, and `$1.8 million` held by three parties, return three objects, one per note, each naming both phrases in one `name` cluster. Do not return six.
- If a filing describes a `Credit Agreement` providing an `$835,000,000` `revolving credit facility` maturing `June 18, 2031`, return one object naming both spans, with `normalized_amount` `835000000` and `end_date` `2031-06-18`. Do not return one object for the agreement and another for the facility.
