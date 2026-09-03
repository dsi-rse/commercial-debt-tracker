## Background
You will be given an `<instruments>` list followed by HTML that contains only `debt_instrument` tags. Each tag has an `instrument-id` attribute such as `i-1`, `i-2`, and so on. Each `instrument-id` refers to one already-extracted debt instrument mention cluster.

The `<instruments>` list gives the terms already extracted for each id: its `name`, and its `amount`, `start_date`, and `end_date` where those were found. Use it to tell ids apart. Two ids often point at the same tagged text, because two objects were built from one name span, and then the list is the only thing that distinguishes them. An id whose `start_date` is later, or whose `amount` and `end_date` match the post-change figures in the text, is the newer state.

Your task is to identify lineage relationships between these mention clusters only.

## Relationship Types
- `amendment_of`: the `from` mention cluster is the newer description of the same underlying debt obligation, with modified terms or an updated state.
- `retired_of`: the `from` mention cluster states that the older `to` debt obligation was retired, repaid, cancelled, exchanged away, extinguished, or otherwise satisfied.
- `split_of`: the `from` mention cluster is a newly created borrowing carved from part of the older `to` mention cluster.

Use `amendment_of` only when the two mention clusters still describe the same debt obligation carried forward with changed terms.
Do not use `amendment_of` for a new note issued in exchange for an old note if the old note is extinguished.
Do not use `amendment_of` for debt retired with the proceeds of another debt issuance.
Use `retired_of` when the text says the older debt ceased to exist because it was retired, repaid, cancelled, exchanged, or satisfied.
Use `retired_of` also when the text says the older debt will be redeemed, repaid, or retired with the proceeds of the `from` instrument. Proceeds-financed retirement counts, whether or not the new instrument structurally replaces the old one.

## Examples
- A commitment increase, maturity extension, amendment, or amendment and restatement filing that also describes the predecessor instrument: the mention cluster for the instrument as amended is `amendment_of` the predecessor's mention cluster. The cluster for the instrument as amended, carrying the newer terms, is always `from`; the predecessor, carrying the older terms, is always `to`.
- Read "newer terms" off the figures the text gives, not off which id comes first. When a filing says commitments were reduced `from $100,000,000 to $50,000,000` and the maturity extended `from June 28, 2026 to June 23, 2031`, the id holding `$100,000,000` and `2026-06-28` is the predecessor and belongs in `to`; the id holding `$50,000,000` and `2031-06-23` is the instrument as amended and belongs in `from`.
- A new facility that `refinances and replaces` an existing facility: the new facility is `retired_of` the replaced facility, not `amendment_of`, because the old facility ceased to exist.
- New notes whose stated use of proceeds is to redeem the company's outstanding `5.25% Senior Notes due 2027`: the new notes are `retired_of` the 2027 notes, even though the new notes do not structurally replace them.

## Output Rules
1. Return a JSON array of objects with exactly the keys `from`, `to`, and `type`.
2. The only valid `type` values are `amendment_of`, `retired_of`, and `split_of`.
3. Both `from` and `to` must be valid instrument ids from the input.
4. Do not create self-relations.
5. If no mention cluster modifies or splits another, return `[]`.
6. Return only valid JSON with no extra text.
