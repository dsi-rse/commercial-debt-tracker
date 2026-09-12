## Background
You will be given an `<instruments>` list followed by HTML that contains only `debt_instrument` tags. Each tag has an `instrument-id` attribute such as `i-1`, `i-2`, and so on. Each `instrument-id` refers to one already-extracted debt instrument mention cluster.

The `<instruments>` list gives the terms already extracted for each id: its `name`, and its `amount`, `start_date`, `maturity_date`, and `status` where those were found, plus `expected_retirement="true"` when the extraction recorded that the filing plans to redeem, repay, exchange or terminate it — the usual mark of a use-of-proceeds target. Use it to tell ids apart when two ids point at the same tagged text.

Your task is to identify lineage relationships between these mention clusters only.

## Relationship Types
- `amendment_of`: the `from` mention cluster is the newer description of the same underlying debt obligation, with modified terms or an updated state.
- `retired_by`: the `from` mention cluster is a debt obligation that the text says was retired, repaid, cancelled, exchanged away, extinguished, or otherwise satisfied by the newer `to` mention cluster.
- `split_of`: the `from` mention cluster is a newly created borrowing carved from part of the older `to` mention cluster.

Use `amendment_of` only when the two mention clusters still describe the same debt obligation carried forward with changed terms.
Do not use `amendment_of` for a new note issued in exchange for an old note if the old note is extinguished.
Do not use `amendment_of` for debt retired with the proceeds of another debt issuance.
Use `retired_by` when the text says a debt obligation ceased to exist because it was retired, repaid, cancelled, exchanged, or satisfied: the obligation that ended is `from`, and the instrument that ended it is `to`.
Use `retired_by` also when the text says the older debt will be redeemed, repaid, or retired with the proceeds of the newer instrument. Proceeds-financed retirement counts, whether or not the new instrument structurally replaces the old one.

## Examples
- An amendment of one facility is normally a single mention cluster carrying its old terms as prior figures, so it needs no relation. Only when the extraction produced two clusters for one obligation carried forward with changed terms — an amended-and-restated agreement described alongside the facility it restates — is the newer cluster `amendment_of` the older one. The cluster for the instrument as amended is `from`; the predecessor is `to`.
- When two ids for one obligation do exist, read which is newer off the figures the text gives, not off which id comes first: the id holding the pre-change `$100,000,000` and `2026-06-28` is the predecessor (`to`); the id holding `$50,000,000` and `2031-06-23` is the instrument as amended (`from`).
- A new facility that `refinances and replaces` an existing facility: the replaced facility is `retired_by` the new facility, not `amendment_of` it, because the old facility ceased to exist. The replaced facility is `from`; the new facility is `to`.
- New notes whose stated use of proceeds is to redeem the company's outstanding `5.25% Senior Notes due 2027`: the 2027 notes (`expected_retirement="true"` in the list) are `retired_by` the new notes, even though the new notes do not structurally replace them.

## Output Rules
1. Return a JSON array of objects with exactly the keys `from`, `to`, and `type`.
2. The only valid `type` values are `amendment_of`, `retired_by`, and `split_of`.
3. Both `from` and `to` must be valid instrument ids from the input.
4. Do not create self-relations.
5. If no mention cluster modifies or splits another, return `[]`.
6. Return only valid JSON with no extra text.
