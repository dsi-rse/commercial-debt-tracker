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
- `other_interested_parties`

For standardized single-value properties, use these object shapes:
- `start_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null }`
- `end_date`: `{ "evidence": ["tag-..."], "normalized_date": "YYYY-MM-DD" | null, "derived_from_name": true | false }`
- `amount`: `{ "evidence": ["tag-..."], "normalized_amount": "12345.67" | null, "currency": "USD" | null }`

## Hard Rules
- Return one JSON object per concrete debt instrument described as its own obligation in the document.
- Do not return agreements as objects.
- `name` may contain only `debt_instrument` tag ids.
- `start_date.evidence` may contain only `date` tag ids.
- `end_date.evidence` may contain `date` tag ids, or the instrument's own `debt_instrument` tag id when the maturity is embedded in the name, such as `3.875% senior notes due 2028`.
- `end_date.derived_from_name` is optional. Set it to `true` when the maturity comes from the instrument name rather than a standalone date mention.
- `amount.evidence` may contain only `amount` tag ids.
- `lenders` and `other_interested_parties` may contain only `person` or `organization` tag ids.
- For `name`, return one list of tag ids representing a single coreference cluster.
- For multi-value properties, return a list of lists, where each inner list is one coreference cluster.
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
- Any property evidence may be shared across multiple returned objects when the text says the property applies to all of them, including `name`, `start_date`, `end_date`, `amount`, `lenders`, and `other_interested_parties`.

Examples:
- If a document describes `3.875% senior notes due 2028` and gives no separate maturity date, return that instrument's `debt_instrument` tag id as `end_date.evidence` with `normalized_date` `2028-12-31` and `derived_from_name` `true`.
- If a document describes `senior notes due October 1, 2028` and tags `October 1, 2028` as a date, cite the `date` tag id with `normalized_date` `2028-10-01`.
- If a document says the company issued an initial note on March 17, 2025 for $5.5 million and a subsequent note on March 20, 2025 for $269,000, both called `Senior Subordinated Convertible Promissory Note`, return two objects.
- If a document later refers collectively to those instruments as `Exchange Notes`, do not return a third `Exchange Notes` object.
- If a document says prior notes were retired in full, do not return a new object just for that contextual mention unless the filing separately describes a concrete debt instrument state for it.
