## Background
You will be given HTML that contains only `debt_instrument` tags. Each tag has an `instrument-id` attribute such as `i-1`, `i-2`, and so on. Each `instrument-id` refers to one already-extracted debt instrument mention cluster.

Your task is to identify lineage relationships between these mention clusters only.

## Relationship Types
- `amendment_of`: the `from` mention cluster is the newer borrowing and modifies, replaces, extends, refinances, or otherwise updates the older `to` mention cluster.
- `split_of`: the `from` mention cluster is a newly created borrowing carved from part of the older `to` mention cluster.

## Output Rules
1. Return a JSON array of objects with exactly the keys `from`, `to`, and `type`.
2. The only valid `type` values are `amendment_of` and `split_of`.
3. Both `from` and `to` must be valid instrument ids from the input.
4. Do not create self-relations.
5. If no mention cluster modifies or splits another, return `[]`.
6. Return only valid JSON with no extra text.
