# Extraction: design decisions and measurements

The extractor (`src/cdt/extractor/core.py`, with the OpenAI Batch backend in
`src/cdt/extractor/batch.py`) runs three LLM stages over each relevant 8-K item
or 6-K snippet: NER tags entity spans in the item text, instrument IE turns the
tagged text into kind-typed facts per debt instrument, and the relation stage
links instruments by lineage. Every model value is verified against a
deterministic parse of the spans it cites before it publishes. This page records
why the validation, normalization, salvage and abort-handling rules are what they
are, and the measurements behind them. The docstrings give the current contract.
Each section is headed by the code name it explains, so you can search for it
from the code. Windows named here (the 2026-09 window, the generalization window,
PR #57's held-out window) are scoring sets, not training data.

## Attempt budget and provider aborts

### `DEFAULT_MAX_ATTEMPTS`

One attempt budget for every stage (`--max-attempts`). #127 proposed giving NER more, on the reasoning that it is the only stage that must reproduce the item verbatim, so it is the only one whose failures more attempts fix. The stored corpora do not support it: the NER-calls-per-row histogram over all 761 rows is {1: 756, 2: 3, 3: 2}, so only two rows ever reached the cap. One of those is MPLX, whose third call is the give-up #176 rejects; it would run to a larger budget and still fail. That leaves one row (`000133146326000103-2-03`, FHLB Boston), and the evidence for it is that it passed on a *rerun* of the same arm: cross-run variance, not a fourth attempt recovering anything within a run.

Against that, a bigger NER budget is paid on the most expensive call in the pipeline (the one stage that echoes the whole item back) by every row that legitimately exhausts, and by every row #176's guards correctly reject. Raise it only with a measurement of attempts 4-6 on a fresh window, and only after re-reading #176's ordering note: each extra attempt is another chance for the model to pass NER by returning the input untagged.

### `MAX_CONTENT_FILTER_RESENDS`

A `content_filter` abort is not a verdict on the row and appears not to be billed: eleven of thirteen live NER calls against `openai/gpt-5.6-terra` came back `finish_reason=content_filter` with `completion_tokens=0`, `prompt_tokens=0` and `cost=0.0` (aborted upstream, unbilled), and the cut point is nondeterministic: three repeats of one item gave 1,163 / 280 / 1,375 characters (#127, #135). So resending is both the correct remedy and a free one.

It is classified by the *callers*, before `handle_response`, for the same reason every other unbilled failure is: a connection reset, a timeout, a 429 or a 5xx raises and is caught by `is_infrastructure_error`, so it never becomes an attempt. A filtered response differs only in arriving as a normal 200 with a body (see `completion_result_from_batch_line`), not in kind. Scoring it and then exempting it from the budget would make every cross-attempt check (the give-up check and the high-water mark) read a call the model never answered as earlier work the model is now regressing against.

The cap is not about the cost of retrying. It exists because the non-billing claim is *unverified*: thirteen calls against one model, and internally inconsistent (280-1,375 characters of text alongside `completion_tokens=0`). The batch route reports no `cost` field at all, so it cannot be checked there. The cap bounds the damage if the assumption is wrong, and each resend is logged so it can be noticed.

`max_tokens` is deliberately unset: the evidence says the length cap was never the constraint (it rescued one item and broke another).

### `AttemptRecord.finish_reason`

Without `finish_reason` a response the provider aborted is indistinguishable in the audit log from one the model chose to end, and those need opposite remedies: four items lost to NER in one held-out window turned out to be `content_filter` aborts rather than the length cap or a model stop (#135).

### `ExtractionRowState.record_unbilled_abort`

The outstanding attempt is left untouched (no response, no `attempt_index` bump, no validation), so the caller resends the identical request and the model's next real answer is scored as the attempt it actually is. There is nothing for the model to correct, so the repair conversation `retry` builds (the aborted response as an assistant turn plus a complaint) would be noise every later call in the row pays prompt tokens for.

The abort is still appended to `all_attempts`, so the audit log keeps a true record of calls made (#135 telemetry), and so the resend cap survives a process exit: the batch backend folds one response per tick, so the count has to come from state `to_state_dict` already round-trips rather than a local. `messages` is left empty on the abort record: the request is by construction identical to the one on the attempt that eventually gets scored, and copying it per abort made a persistently filtered row's state grow several-fold.

### `terminate_on_provider_aborts`

Reaching `MAX_CONTENT_FILTER_RESENDS` means "stop resending", not "the model answered badly". Scoring the abort instead (falling through to `handle_response`) would charge the row for a call it never got an answer to, grow the retry conversation with an empty assistant turn plus a complaint, and leave a `FAILED` attempt that every cross-attempt check reads as the model having failed; one abort past the cap was enough to reject the honest answer that followed it.

Salvage applies to what the row already earned as `_salvage_or_fail` applies it after a scored failure: a stage the provider will not run costs the item what that stage would have added, not what earlier stages validated. The two paths had diverged on `instrument_ie` (PARTIAL with mentions after three rejected answers, FAILED with none after aborts), so both ask the same two questions in the same order. One difference remains: an `instrument_ie` salvage here finishes PARTIAL without lineage and needs at least one mention, where `_salvage_or_fail` advances the row to `instrument_relation`.

The note names both causes when there are two. `count_content_filter_aborts` is a per-stage lifetime count with no reset, deliberately (it bounds the damage one item can do and survives a process exit), so the aborts need not have been consecutive. `summarize_failure` prefers salvage notes over `validation_errors`, so a note claiming "no attempt was scored" on a row that was also answered badly would hide the model's actual error from the registry and send an operator to the vendor.

### `LIVE_REQUEST_TIMEOUT_SECONDS`

Without a client timeout, one non-responsive provider socket wedges the whole synchronous run, and the ECS task hosting it, indefinitely (#93). Generous because reasoning models legitimately take minutes per response.

### `completion_result_from_batch_line`

The batch route reaches OpenAI directly rather than through OpenRouter, so the usage block carries token counts but no `cost`; spend must be derived from the counts, and the content-filter "unbilled" claim cannot be verified there.

## NER stage

### `NERStage.validate`: the give-up checks

The six structural checks are each correct alone, and together they admit a bare `<body>{input}</body>` with zero tags: well-formed, correctly rooted, no disallowed tag, no attributes, no empty tag, and text identical to the input. `early_stop` then reads zero `debt_instrument` tags as "no debt disclosed" and the row publishes SUCCESS with no mentions, indistinguishable downstream from a genuine zero, and its completion record makes a re-run skip it (#176).

The case: MPLX item 2.03 (`000119312519257376-2-03`). Attempts 1 and 2 tagged 91 `debt_instrument` spans each and failed only copy fidelity; attempt 3 returned its input byte for byte, passed, and published 0 mentions against six note series and a term loan. `000121390026025721-1-01` is the same story at 15 spans.

Two cross-attempt checks close it, because the row already holds the evidence to tell a give-up from a genuine zero:

* **High-water mark** (`prior_debt_instrument_high_water`): zero `debt_instrument` tags fails when an earlier attempt on this row found some. It cannot misfire on a debt-free item, which found none on attempt 1 either.
* **Untagged response after tagged work** (`prior_attempt_tagged`): a response with no entity tags at all fails once an earlier attempt on this row tagged anything. The condition is "the model found something here before", not the attempt number and not the exact bytes.
  * It is stated as a tag count, not "byte-identical to the input". Copy fidelity already guarantees the text is the input, so "no tags" is the whole question, and a tag count cannot be sidestepped by whitespace (an extra space, a newline inside `<body>`, doubled spacing all cleared a byte comparison while `collapse_whitespace` let them clear copy fidelity). It also needs no rebuilt copy of the request.
  * It is gated on prior tagged work, not `attempt_index > 1`. An item with nothing to tag whose first attempt failed for a reason that itself tagged nothing (malformed XML, a wrong wrapper) answers honestly with a bare echo; the attempt-number gate rejected that on every remaining attempt. Measured against `dev`, such a row went from SUCCESS in 2 calls to FAILED in 3. A prior failure that *did* tag something, truncation included, still counts as earlier work (`test_an_echo_after_a_truncated_tagged_attempt_is_still_rejected`).
  * It is not unconditional. Over the three stored corpora with attempt logs (761 attempt-1 NER responses in `genwindow-run-branch`, `genwindow-run-dev`, `genwindow-sol-retried`), no attempt-1 response was a byte-identical echo, and the 63 carrying no `debt_instrument` tag all carried some other tag. Rejecting every untagged echo would buy nothing and cost legitimate zero-instrument items. The single echo in the corpus is MPLX's attempt 3.
  * It counts any entity tag, not just `debt_instrument`: a response that tagged an organization and a date and then regressed to a bare echo has given up just as surely, and the high-water mark reads zero for it.

### `prior_attempt_tagged` / `prior_debt_instrument_high_water`: aborted attempts excluded

The text on an `ABORTED` record is whatever the provider emitted before it cut, and it is tagged: all 58 `content_filter` responses in the stored corpora carry between 9 and 107 entity tags and at least one `debt_instrument` tag, with no empty bodies. Counting them turns an honest untagged echo into a rejection the model cannot act on: the abort contributes no assistant turn, so "re-emit your previous tagged output" names work that is not in its context, and the row exhausts its budget and fails.

For the high-water mark the judgement is closer: a truncated abort that tagged four instruments really is evidence the item discloses debt, so reading it would catch a give-up this misses. It is excluded anyway, because the failure tells the model "an earlier attempt tagged N -- keep every tag you found", and on an aborted call the model neither produced those tags nor can see them. A guard the model cannot satisfy costs the row its whole budget. If this is reconsidered, the retry message must change with it.

### `count_debt_instrument_tags` / `count_ner_entity_tags`

Regex counts rather than `parse_tag_details`: the callers include failed attempts, whose responses are the ones that did not parse (11 of 27 NER failures on the PR #57 window were `Response is not valid XML`, #127). A truncated response with an unclosed tag still shows the model was tagging, which is the only question the high-water mark asks.

### `NERStage.build_retry_message`

The retry turn asks for a repair, not a redo. A version that listed only what the *text* had to satisfy, and made tagging sound optional ("only add the allowed bare tags"), made adding nothing the cheapest compliant response; on MPLX item 2.03 that is what came back (#176). The preserve clause goes first so the instruction the model is most likely to follow is the one it was missing.

### `_advance_after_stage`: zero-tag NER finishes SUCCESS

A zero-tag NER response that passed validation is the honest "this filing disclosed no debt" and finishes SUCCESS whether or not the row retried. Filing a retried zero as PARTIAL ("a model that failed once may be giving up") bought a registry entry and no re-extraction, because a PARTIAL row is terminal and the next run skips it, while asserting a loss nothing had evidence for. A give-up the row *does* hold evidence for is a validation failure instead (the high-water mark and the untagged-response check), so it retries to budget and terminates FAILED, which is counted and re-extractable (#176).

### `repair_unescaped_ampersands`

`NERStage.preprocess` wraps the item text in `<body>` unescaped, so an item containing `A&R Registration Rights Agreement` reaches the model as invalid XML, and the response must both reproduce the text exactly and be well-formed XML, which conflict unless the model escapes on its own. Bare ampersands appear in 44 of 342 relevant items in one held-out window across 37 issuers, and in 13% to 17% of relevant items in each of three windows: a standing tax, not one filer's quirk. Repairing the response rather than escaping the input keeps what the model sees unchanged, so its tagging behaviour does not move (#127). Only `&` is repaired: a stray `<` or `>` never occurs in source text, so one in a response is a real malformation.

### `realign_tag_details`

Evidence `char_start`/`char_end` must index the item's own `text` exactly; span highlighting rests on it (#154). NER validation only checks whitespace-collapsed equality, so the model may add or drop whitespace; non-whitespace characters are identical in order and each span is snapped along that alignment.

## Instrument IE: schema and validation

### `DATE_KINDS`

Dates are a kind-typed list of facts rather than single-value `start_date` / `maturity_date` / `commitment_termination_date` slots, so the model records each stated date once instead of arbitrating between competing dates; the post-processor chooses the published columns (`DATE_COLUMN_KINDS`). Events (amendment, repayment, retirement, ...) are dated facts too.

### `SINGLE_CURRENT_DATE_KINDS`

`closing` is an event kind but still singular: two current closings describe two instruments, exactly as two maturities do. The set is listed explicitly rather than derived as `not in EVENT_DATE_KINDS`, which would exempt `closing`.

### `validate_dates_property`: `expected` entries are not current

`select_date_payload` publishes the entry that is neither prior nor expected, so a real closing beside a planned one is one current closing, not two. Costamare's 6-K is the case: it states a 2026-04-30 closing and a 2026-06-30 expected closing for one facility, and counting the planned one rejected the response.

### `validate_cross_field_semantics`

Each check is a rule the prompt states that no single-field check can see. A repayment *amount* is dated by the payment event that produced it: a partial paydown (`repayment`) or the retirement/termination/exchange that paid the rest. Requiring a separate `repayment` date beside a `retirement` made the model manufacture undated placeholder events (HASI, Smith Micro), so a terminal event satisfies the requirement.

### `instrument_entries_from_response`

The model returns one object instead of a one-element array on most single-instrument items (73 of 300 on the 2026-09 window, every one recovered on retry). The object is the same entry, so it is read as such rather than paid for twice.

### `LENDER_DISCLOSURE_VALUES`

A three-valued disclosure replaces a boolean that was true for two different reasons (a collective lender phrase, or no named lender at all). 395 of 517 true values on the 2026-09 window were the second case, so a consumer reading the flag could not tell "something is undisclosed" from "nothing was disclosed here".

### `LENDER_DISCLOSURE_PRECEDENCE`

`collective_present` wins outright: one filing showing `the other lenders party thereto` means holders are hidden however many other filings name some. `complete` beats `none_named` because a filing that named every lender supersedes one that named none; the reverse would let a passing reference erase a full syndicate list.

### `party_payloads_and_disclosure`

The model labels every cluster, and the labels persist rather than only steering what to drop (#150). The borrower's identity matters exactly when a subsidiary is the obligor under the parent filer's 8-K. The model gives a `kind` for every party and the code keeps it, but only lender kinds feed `lender_disclosure`.

### `canonical_instrument_name`

`ner.md` rule 11 has NER tag both the facility phrase and its agreement name, and says the descriptive phrase "must never be dropped in favour of the agreement name". Plain longest-span selection did exactly that on 9 of the 476 multi-span names in the 2026-09 window (`Second Amended and Restated Credit Agreement` over `term loan B facility`; the `Super-Priority Senior Secured Priming ...` title over `DIP Facility`). It is not only display: the published name feeds the matcher's `normalize_name_fingerprint`, and an amendment title is the generic, near-duplicate string that `NAME_CLASS_GATE` and the identifying-name guard have to defend against.

The alternative must actually name an obligation (`INSTRUMENT_NOUN_PATTERN`): preferring any non-agreement span published `Local Currency Addendums` over `Credit Agreement (2025 364-Day Facility)` and `RFA` over `receivables financing agreement`, trading the agreement-title problem for vacuous names.

## Normalization: amounts, rates and dates

### `DERIVED_FROM_*`

Downstream consumers key on the provenance marker: the matcher treats a name-synthesized `YYYY-12-31` maturity as year-resolution only (#128), and the site can explain a value whose evidence list is empty. `scaled` is a separate marker from `computed`, which means arithmetic over addends and is consumed as such on the maturity side (`DERIVED_MATURITY_KINDS` in `matcher/core.py`); `src/` has no amount-side consumer of `derived_from`, so a new value costs nothing and records more honestly how the figure was reached. `inherited` marks a term carried onto a synthesized predecessor because the filing marked no `prior` value for that kind.

### `RATE_SUFFIX_PATTERN`

Includes `bps`, the abbreviation filings actually write. Without it `is_rate_like_amount_text` reads `50 bps` as money and `validate_amount_is_not_rate` lets a basis-point margin through as an amount. `amounts_agree` does not catch it: it only rejects a model figure that disagrees with the span's number, and the model reports the basis-point figure itself, so a 50bp margin would publish as a principal of 50 (#228).

### `is_rate_like_amount_text`

Every number in the span must carry a rate marker: `500,000,000 (100% of principal)` states a principal then a percentage of it, and the parser reads the first number, so treating the whole span as a rate would discard a real amount (#103).

### `AMOUNT_MULTIPLIERS`

Magnitude words, spelled out and abbreviated (#182). The abbreviations matter because the name-derived principal (#129) reads the instrument's own name, and names use them: `Citibank $382.5 mil. Revolving Credit Facility`, `Syndicated $850.0 mil. Facility` (Costamare's 6-K facility schedules). On a cited span a missing magnitude only publishes null, because `amounts_agree` rejects the mismatch; on the name-derived path there is no model value to disagree with, so a value six orders of magnitude off would publish. `AMOUNT_SCALE_ALTERNATION` is built from this table so every magnitude the pattern recognizes is one the parser can apply.

### `magnitude_in_amount_text`

The single definition of a magnitude word, shared by the parser and `scaled_amount_from_sibling` (which asks whether a span carries its own magnitude and which one its neighbour carries), so the two cannot disagree.

### `canonical_amount_value`

An amount cluster often pairs the figure with its label, and the label is longer: `['$2,000,000', 'Principal Amount']` resolved to `Principal Amount`, which parses to nothing, so the amount published null (#120). The rate guard reads the same text, so choosing the parseable span keeps it pointed at the right words.

### `normalize_numeric_string` / `amounts_agree`

Decimal, not float: `float("372246148.11")` is not that number, and `f"{value:.12f}"` renders it as `372246148.110000014305`, so every amount carrying cents failed the agreement check and published null (#119). Agreement is numeric, so `500000.00` against a parsed `500000` agrees.

### `currency_candidates_from_text`

A qualified dollar sign (`C$`, `A$`, `HK$`) is a different currency. Reading `C$300 million` as USD mislabelled the amount and kept CAD out of the candidate set, so the model's correct currency was rejected (#121).

### `standardized_amount_payload`: document currency

A table cell (`35,000,000` under a `BANK PAR ($)` header) carries no marker of its own, so the model's `USD` had been rejected (59 of 901 mentions on the 2026-09 window, all FHLB schedules and one private-credit filer). When the cited span shows no currency and the whole item shows exactly one, that one is accepted.

### `normalized_amount_from_name` / `standardized_amount_payload` name fallback

`$183.36 million term loan` carries its own principal, and NER tags the whole phrase as one `debt_instrument`, so there is no `amount` span to cite (#129). A name stating more than one figure names no single principal.

### `name_derived_principal_payload`

There is no model value to agree with here (the parser's reading is the value), so it cannot route through `standardized_amount_payload`, whose `amounts_agree` gate rejects a null model amount.

### `standardized_amounts_payloads`: no name principal when every principal is prior

When every stated commitment or principal is `prior`, the head's current figure is unstated, and the figure in the name is the prior one. Reading it back off the name would publish the pre-amendment figure as current, the stale head of #165 by a second route (#206). Null is the honest answer; the minted prior state carries that figure.

### `scaled_amount_from_sibling`

Filing English writes a magnitude word once, after the second of two figures. Crescent Capital BDC's Loan and Security Agreement amendment (`000119312526241887-1-01`) says it "increased the facility size **from $400.0 to $500.0 million**": NER tags `$400.0` (tag-15, chars 654-660) and `$500.0 million` (tag-16, 664-678) as two `amount` spans, the model reads the shared `million` and returns `400000000` for the `prior` commitment, and `amounts_agree` compares that against the parser's reading of the cited span alone, `400`. The correct value published null with `validation_errors: []`, and because `mint_prior_state_rows` builds a predecessor only from `prior` facts with a value, the null suppressed the minted row (`skipped_unparsed_prior`). The parser already reads the phrase (`normalized_amount_from_text("$400.0 to $500.0 million")` is `400000000`); the failure is only that the cited span is tighter than the phrase carrying the magnitude (#213).

**A sibling fact's cited span is acceptable evidence** (decided 2026-09-19). The span is cited, just by the neighbouring amount fact, so the arithmetic stays deterministic and anchored to spans the model pointed at. Rejected alternatives: an `instrument_ie` rule requiring both spans on the `prior` fact (#165's precedent, which needed a fresh-window eval); and the *wider* "nearest magnitude-bearing `amount` span in the item", which needs no plumbing (`tag_details` is the whole item's map) but admits spans no fact cites. This is why the rescue runs as a post-pass in `standardized_amounts_payloads`, the narrowest place "cited by a sibling amount fact of *this object*" is expressible; `standardized_amount_payload` sees one entry and cannot see its neighbour.

Refusals mirror `computed_sum_amount`'s, and the two cannot both land: one needs the sum of the cited spans to equal the model's value, the other one span's reading times a factor of at least a thousand; it is the exact-product comparison that separates them, not a span count. The product must equal the model's value exactly and the return is the model's own value re-normalized, so the rescue only ever *confirms* the model. The two leading type refusals are not load-bearing (mutation testing: `decimal_from_amount_string` rejects non-strings itself, and the exact-product comparison subsumes both); they are kept for the contract and symmetry.

A fact whose own evidence already carries a magnitude is never rescaled, checked over *every* own span, not just the canonical one: a fact citing both `aggregate principal amount of $400.0` and `$500.0 million` would otherwise be rescaled by a sibling's `million` while holding the magnitude in its own evidence, because the longer bare span wins selection. This is reachable because `validate_amounts_property` does not run `validate_standardized_single_value_cardinality`, so an `amounts[*]` entry may cite spans with distinct values. It is also why `sibling_texts` needs no filtering against `own_texts`. More than one distinct sibling magnitude refuses: guessing between `million` and `billion` is a three-orders error. The currency stays the one read from the fact's own span.

Measured by re-deriving `data/genwindow-run-branch` from stored responses: one fact rescued of 587, moving `skipped_unparsed_prior` 1 -> 0 and `minted` 15 -> 16, every other refusal counter unchanged, and `lineage-verify`'s amendment pointers 22 -> 23 on unchanged heads and families. The rescue fires on no other fact in any stored root. #214's Blue Owl `($ in thousands)` table is untouched: its magnitude lives in an untagged header no fact cites, so all 18 are refused.

### `computed_maturity_date` / `computed_sum_amount`

The only arithmetic accepted, both anchored to cited spans and confirming the model's value rather than originating one. Computed maturity (#166): a filing stating a closing date and a tenor but never the maturity supports exactly one answer; it runs both directions because `extended six months to September 3, 2027` states the prior maturity as the new one minus the tenor. Computed sum (#165): an increase-by amendment states a prior total and an increment but often never the result.

### `date_plus_tenor`

Returns None rather than raising for an out-of-calendar result: an exception would unwind out of postprocess past the driver (which catches only `InfrastructureError`) and kill the run before the failure registry, mentions and audit log were written.

### `rate_tokens_in_rate_span`

A table cell under a `COUPON PCT` header is the bare number `4.125`; the tagger's `interest_rate` type is the marker, so a span that is nothing but a number is that rate. FHLB consolidated-obligation schedules lost every coupon without this (47 mentions with kind fixed and no pct on the 2026-09 window).

### `standardized_interest_rate_payload`

The model's `rate_pct` publishes only when a rate token in the cited evidence (or, failing that, the instrument name) parses to the same number (#157). The canonical form is published rather than the model's spelling: verbatim persistence produced 141 distinct strings for 115 distinct rates on the generalization window (`5`, `5.00`, `5.000`), which splits any group-by and makes equality filters miss rows.

### `normalized_date_from_text`

The parser must read every spelling a filing uses, because `standardized_date_payload` keeps the model's date only when it matches what the parser reads. `M/D/YYYY` alone cost 67 start dates and 67 end dates on one held-out window, all from tabular schedules (#133). Two-digit years stay unparsed: a null beats a wrong decade.

### `MONTH_YEAR_DATE_PATTERN`

Reads month-resolution dates outside an instrument name (`matures in June 2016`, `legal final maturity date is in March 2056`), for maturities only, normalized to the month's last day.

### `normalized_month_year_from_text` / `iso_month_end_from_parts`

Only maturities accept month resolution (#164): `matures in June 2016` states the maturity as precisely as the filing ever will, while a start or status date at month resolution would be a guess. Month-end is strictly better than the year-end synthetic the name would produce without the month; the matcher compares name-derived values at their true resolution.

### `standardized_date_payload`: every cited span is a candidate

Checking only the longest span threw away a correct `March 2, 2026` whenever the model also cited the defined term `Redemption Date` (the longer string) for the same date. The multiple-distinct-values validator already guarantees the parseable spans agree.

## Relation stage and lineage

### `relation_instrument_manifest`

Two objects built from one name span render as the same tagged text twice, so `Third Amended and Restated Loan Agreement` and its predecessor are indistinguishable in the body; the amounts and dates that tell them apart live in the `instrument_ie` output. Every inverted lineage pair on the held-out run was a same-name pair (#138), so the relation stage gets a manifest of each id's extracted terms, plus `expected_retirement`, without which it cannot tell a use-of-proceeds target from a note merely mentioned.

### `oriented_lineage_pair`

`amendment_of` runs from the amended instrument to its predecessor (source is later); `retired_by` runs from the retired obligation to its retirer (source is earlier). When both sides carry start dates on the wrong side of that order, the model named the pair backwards and the pointer is flipped. Alclear's revolver is the confirmed case: a Credit Agreement dated as of 2020-03-31, amended 2026-06-23 to cut commitments from $100,000,000 and extend maturity from 2026-06-28 to 2031-06-23; every figure on the `$100,000,000` object is pre-amendment, yet it was the object carrying `amendment_of` (#138). The dates have to disagree for the flip to fire.

### `InstrumentRelationStage.postprocess`: `retired_by` is a list

One obligation may be retired jointly by several instruments (a dual-tranche offering funding one redemption), so `retired_by_json` accumulates every edge rather than keeping the last.

## Salvage

### `_salvage_or_fail`

Terminal salvage (#152): one invalid entry should not cost every valid one, and a relation-stage failure should not discard mentions that already passed `instrument_ie`. Both salvages finish PARTIAL: mentions publish and the failure registry records the loss. NER has nothing to salvage.

## Prior-state minting

### `mint_prior_state_rows`

An amendment 8-K extracts as **one** object: its terms as amended, plus every old term the filing states marked `prior` (`instrument_ie.md`). That is the shape the model gets right: asking it to emit the predecessor as its own object (#81) pointed at the wrong instrument in three of five spot-checks (#125). But one object leaves nothing for `amendment_of` to name, so the pipeline published no amendment lineage: 537 of 542 instruments were lineage heads (#170). The mint expands the shape in code (#203). Nothing *chooses* a predecessor; it is built from the object's own cited `prior` facts, so #125's failure class cannot occur.

The predecessor P shares every property of its successor M except the terms marked `prior`, which replace their kind. A `prior` mark means "this term changed"; a current term with no prior sibling of its kind is, on the filing's evidence, unchanged and is carried onto P marked `derived_from: "inherited"`. The one failure that rule cannot see (a filing stating a *new* value with no before-figure, "increased commitments **to** $250M", alongside some other `prior` term) is why the marker exists. Parties carry the borrower only: a joinder adds and removes lenders, so the filing never states who lent under the earlier terms. `amendment_of` is not hashed into M's id, so no existing row re-keys.

Origin date: a `prior` agreement (the predecessor's own dated-as-of, the best evidence), else the current `closing` or `agreement`. An origin equal to an `amendment` date is the restatement's own dated-as-of (MPLX: agreement 2019-07-31 == amendment 2019-07-31), not the predecessor's; P is still minted, with no start date, rather than carry a date the filing did not state for that state. Measured on one 364-item window: 22 objects carry a prior term, 17 mint.

Counters: every refusal increments a named counter so the rate is measurable before anyone loosens the rule. They partition the population: each object with a `prior` claim increments exactly one of `minted`, `minted_shared`, or a `skipped_*`. `minted_no_origin` tags a subset of `minted` and is bumped only after the append, so it can never sit beside a skip with no synthesized row (#211).

Kind sets come from the *claims*, values from what *parsed*. Using the parsed subset for both inverted the inheritance rule: a `prior` term whose value did not parse was invisible to "did this kind change?", so the post-amendment value was copied onto P as inherited, asserting the current figure as the prior state's own term (#211). `skipped_unparsed_prior` exists so the counters sum to the population (on the 364-item window they had summed to 21 against 22).

The function copies its input: it writes `amendment_of` onto the successor, and a caller passing `row_state.debt_instrument_mentions` directly would otherwise persist a minted pointer into `state.jsonl`.

Runs at write time (`published_mention_rows`) and over existing partitions (`backfill_mentions`). Rows read back from parquet carry NaN where the writer had None, so every copied field is coerced, or a mint built at write time and one built by the backfill would hash differently.

### `published_mention_rows`

The one seam between what the model returned for an item and what the pipeline writes. Every publish path (the live loop, the batch finalize, `extract_tables`, the `full.jsonl` audit record) goes through it, so the prior-state mint applies identically on every backend, including rows of an in-flight batch job whose IE postprocess ran under older code, while `state.jsonl` keeps only what the model returned. It takes no `counters`: the mint counters are read off `cdt backfill-mentions`, where the pre-registered yield is measured; a live-path counter should arrive with the manifest field that would carry it.

### `backfill_mentions`

Re-derives synthesized rows over existing partitions, so partitions written before the prior-state mint gain prior states with no re-extraction. `renew` extends the writer lease per rewritten partition: the job rewrites the whole canonical mentions dataset, and on a corpus that outlasts the lease TTL the next orchestrator tick would otherwise legitimately steal the lease and start extract/match into the same objects (#89).

## Writing mentions (live and batch backends)

### `_mentions_partition_needs_write`

There are two reasons to write a mentions partition: new mentions to add, or stored ids to remove. The second matters when an item that is still relevant is re-extracted and this time yields *no* mentions: its previous rows must be withdrawn or the pipeline keeps publishing facts the newer pass retracted (#209). `retired_item_ids` cannot carry that case (it is `done_item_ids - relevant_item_ids`, and the item is still relevant), and `replaced_item_ids & pending.done_item_ids` is empty under `--force` by construction, because `pending_extract_partitions` sets `done_item_ids=frozenset()` for a forced partition. Both backends had each failed one of those ways, so the rule lives in one place and asks the stored partition directly. "Ids to remove" is not simply `replaced_item_ids`: on a first-ever extraction that yields nothing every claimed id is replaced and there is nothing to purge, and writing then would create an empty parquet file where the contract is to create none (`test_extract_pending_items_skips_empty_outputs_on_rerun`).

### `sampling_params`, `handle_response`, `handle_provider_abort`

The live backend (OpenRouter chat completions, `run_extraction_workflow`) and the batch backend (OpenAI Batch, `cdt.extractor.batch`) drive the same stage objects and share every decision that could make their output differ: sampling parameters (reasoning models reject `temperature != 1`, so both decide from `REASONING_MODEL_PREFIXES` against the native model id), scoring (`handle_response`), abort handling (`handle_provider_abort`), what publishes (`published_mention_rows`), and when a mentions partition is rewritten (`_mentions_partition_needs_write`). Each backend only classifies transport outcomes before handing a response to the shared path.
