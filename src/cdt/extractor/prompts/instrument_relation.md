## Background
You will be given HTML that contains only `debt_instrument` tags. Each tag has an `instrument-id` attribute such as `i-1`, `i-2`, and so on. Each `instrument-id` refers to one already-extracted debt instrument mention cluster.

Your task is to identify lineage relationships between these mention clusters only.

## Relationship Types
- `amendment_of`: the `from` mention cluster is the newer description of the same underlying debt obligation, with modified terms or an updated state.
- `retired_of`: the `from` mention cluster states that the older `to` debt obligation was retired, repaid, cancelled, exchanged away, extinguished, or otherwise satisfied.
- `split_of`: the `from` mention cluster is a newly created borrowing carved from part of the older `to` mention cluster.

Use `amendment_of` only when the two mention clusters still describe the same debt obligation carried forward with changed terms.
Do not use `amendment_of` for a new note issued in exchange for an old note if the old note is extinguished.
Do not use `amendment_of` for debt retired with the proceeds of another debt issuance.
Use `retired_of` when the text says the older debt ceased to exist because it was retired, repaid, cancelled, exchanged, or satisfied.

## Output Rules
1. Return a JSON array of objects with exactly the keys `from`, `to`, and `type`.
2. The only valid `type` values are `amendment_of`, `retired_of`, and `split_of`.
3. Both `from` and `to` must be valid instrument ids from the input.
4. Do not create self-relations.
5. If no mention cluster modifies or splits another, return `[]`.
6. Return only valid JSON with no extra text.
