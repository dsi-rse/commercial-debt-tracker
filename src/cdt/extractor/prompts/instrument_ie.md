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

## Hard Rules
- Return one JSON object per distinct debt instrument mention cluster.
- Do not return agreements as objects.
- Every `debt_instrument` tag in the input must appear in exactly one `name` list.
- `name` may contain only `debt_instrument` tag ids.
- `start_date` and `end_date` may contain only `date` tag ids.
- `amount` may contain only `amount` tag ids.
- `lenders` and `other_interested_parties` may contain only `person` or `organization` tag ids.
- For single-value properties, return one list of tag ids representing a single coreference cluster.
- For multi-value properties, return a list of lists, where each inner list is one coreference cluster.
- If a property is absent, omit it.
- Do not invent ids, parties, dates, or amounts.
- Return only valid JSON with no extra text.
