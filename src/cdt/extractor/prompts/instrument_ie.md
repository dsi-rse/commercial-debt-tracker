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
- `lenders_complete`
- `other_interested_parties`

For standardized single-value properties, use these object shapes:
- `start_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null }`
- `end_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null, "derived_from_name": true | false }`
- `amount`: `{ "evidence": ["tag-..."], "normalized_amount": "12345.67" | null, "currency": "USD" | null }`

For party properties, return one object per coreference cluster:
- `lenders`: `[{ "tag_ids": ["tag-..."], "kind": "named" | "collective" }]`
- `lenders_complete`: `true` | `false`
- `other_interested_parties`: `[{ "tag_ids": ["tag-..."], "role": "agent" | "trustee" | "underwriter" | "guarantor" | "borrower" | "other" }]`

## Hard Rules
- Return one JSON object per concrete debt instrument described as its own obligation in the document.
- Do not return agreements as objects.
- `name` may contain only `debt_instrument` tag ids.
- `start_date.evidence` may contain only `date` tag ids.
- `end_date.evidence` may contain `date` tag ids, or the instrument's own `debt_instrument` tag id when the maturity is embedded in the name, such as `3.875% senior notes due 2028`.
- `end_date.derived_from_name` is optional. Set it to `true` when the maturity comes from the instrument name rather than a standalone date mention.
- `amount.evidence` may contain only `amount` tag ids.
- `lenders` and `other_interested_parties` cluster `tag_ids` may contain only `person` or `organization` tag ids.
- For `name`, return one list of tag ids representing a single coreference cluster.
- Every `lenders` cluster must carry a `kind`, and every `other_interested_parties` cluster must carry a `role`.
- `lenders_complete` must be `true` or `false`, and may be returned only when `lenders` is present.
- `amount` is the principal or commitment amount only. Interest rates, margins, spreads, fees, discounts, and per-annum percentages are never `amount`. Omit `amount` when the document states no principal or commitment amount.
- For `amount.normalized_amount`, return only digits and at most one decimal point, or `null`.
- For `amount.currency`, return one 3-letter ISO 4217 currency code or `null`.
- For `start_date.normalized_date` and `end_date.normalized_date`, return `YYYY-MM-DD` or `null`.
- When a maturity gives only a year, such as `due 2028`, return `2028-12-31` as `end_date.normalized_date`.
- Never guess a maturity that the document does not state, and never reuse the start date as the end date.
- If a property is absent, omit it.
- Do not invent ids, parties, dates, or amounts.
- Return only valid JSON with no extra text.

Selection rules:
- Ignore debt-like mentions that are only contextual references to older debt being retired, repaid, cancelled, exchanged, refinanced, or discussed as background.
- Ignore collective labels that only group multiple concrete instruments described elsewhere in the same document, such as `Exchange Notes` or generic `Notes`, when the underlying instruments can be extracted separately.
- Ignore non-debt securities even if they appear in the same financing disclosure.
- A returned object should correspond to one coherent debt instrument.
- A single debt instrument should have at most one start date and one principal amount. If the document presents two different start dates or two different principal amounts, that is strong evidence there are two separate debt instruments and you should return two objects.
- Multiple returned objects may share the same `name` evidence tags when the text clearly describes multiple distinct instruments using the same name phrase.
- A securities offering that lists multiple classes, tranches, or series, such as `Class A-1`, `Class A-2a`, `Class A-3`, or `Series A` and `Series B`, is multiple debt instruments. Return one object per class, tranche, or series, even when the document names them together in one sentence, and even when only some of them state their own amount or maturity.
- Each object's `name` must refer to a single class, tranche, or series. Never merge several of them into one object, and never return an extra object for the group label, such as `Asset Backed Notes` or `Notes`, that only collects them.
- Any property evidence may be shared across multiple returned objects when the text says the property applies to all of them, including `name`, `start_date`, `end_date`, `amount`, `lenders`, and `other_interested_parties`.

Party rules:
- Use `kind: "named"` for a cluster that identifies a specific lender by name, such as `JPMorgan Chase Bank, N.A.` or `EGT 11 LLC`.
- Use `kind: "collective"` for a cluster whose surface text only describes the group without identifying anyone, such as `the Lenders`, `the other lenders party thereto`, `the holders`, `certain financial institutions`, or `the purchasers`.
- Return `lenders_complete: true` when the document names every lender, which is the normal case when all `lenders` clusters are `named`.
- Return `lenders_complete: false` when the document signals additional undisclosed lenders, which is the normal case when any `lenders` cluster is `collective` or when the text hedges with wording such as `certain lenders`, `including`, or `and others`.
- The filer, issuer, borrower, or obligor is never its own lender. Put it in `other_interested_parties` with `role: "borrower"` only when the document treats it as a distinct party worth recording, and otherwise omit it.
- An administrative agent, collateral agent, or paying agent belongs in `other_interested_parties` with `role: "agent"`. Include it in `lenders` only when the document also describes it as a lender or purchaser of that instrument, for example `as a Lender and as Administrative Agent`.
- An indenture trustee or collateral trustee belongs in `other_interested_parties` with `role: "trustee"`, never in `lenders`.
- Underwriters, initial purchasers, placement agents, and sales agents in a public offering or Rule 144A resale belong in `other_interested_parties` with `role: "underwriter"`, never in `lenders`, because they resell the debt rather than hold it.
- In a note purchase agreement or private placement sold directly to investors, the `purchasers` ARE the lenders. Return them in `lenders`, using `kind: "named"` when they are named and `kind: "collective"` when the document only refers to `the Purchasers`.
- Guarantors belong in `other_interested_parties` with `role: "guarantor"`.
- Use `role: "other"` only when the party is clearly related to the instrument but none of the other roles fit.

Examples:
- If a document describes `3.875% senior notes due 2028` and gives no separate maturity date, return that instrument's `debt_instrument` tag id as `end_date.evidence` with `normalized_date` `2028-12-31` and `derived_from_name` `true`.
- If a document describes `senior notes due October 1, 2028` and tags `October 1, 2028` as a date, cite the `date` tag id with `normalized_date` `2028-10-01`.
- If a credit agreement says ABR Loans bear interest at `0.875% per annum`, do not return `0.875` as the `amount`. Omit `amount` unless the document states that loan's principal or commitment amount.
- If a trust issues `Class A-1 Asset Backed Notes`, `Class A-2a Asset Backed Notes`, `Class A-2b Asset Backed Notes`, `Class A-3 Asset Backed Notes`, and `Class A-4 Asset Backed Notes` in one offering, return five objects, one per class, each with its own amount and maturity when stated. Do not return one object naming all five, and do not return a sixth object for `Asset Backed Notes`.
- If a document says the company issued an initial note on March 17, 2025 for $5.5 million and a subsequent note on March 20, 2025 for $269,000, both called `Senior Subordinated Convertible Promissory Note`, return two objects.
- If a document later refers collectively to those instruments as `Exchange Notes`, do not return a third `Exchange Notes` object.
- If a document says prior notes were retired in full, do not return a new object just for that contextual mention unless the filing separately describes a concrete debt instrument state for it.
- If a credit agreement says the lenders are `JPMorgan Chase Bank, N.A.` and `the other lenders party thereto`, return both clusters, `kind: "named"` for Chase and `kind: "collective"` for the other lenders, with `lenders_complete: false`.
- If the document says the lenders are `JPMorgan Chase Bank, N.A.` and `Wells Fargo Bank, National Association` with no collective phrase, return two `named` clusters with `lenders_complete: true`.
- If the document only says the notes were sold to `the Holders`, return one `collective` cluster with `lenders_complete: false`.
- If a note purchase agreement says the company sold notes to `Metropolitan Life Insurance Company` and `the other purchasers named therein`, return Metropolitan Life as `named` and the other purchasers as `collective`, with `lenders_complete: false`.
- If an indenture names `The Bank of New York Mellon` as trustee and the notes were sold through underwriters, return no `lenders` clusters, and return the trustee with `role: "trustee"` and the underwriters with `role: "underwriter"`.
