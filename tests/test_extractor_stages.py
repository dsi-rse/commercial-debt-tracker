"""Tests for the extractor stages, validation, retries, provider aborts and salvage."""

from __future__ import annotations

import json
from types import SimpleNamespace

from support import (
    PARTY_ROLE_XML,
    _dates_tag_details,
    maturity_row_state,
    party_row_state,
)

from cdt.extractor.llm import (
    completion_result_from_batch_line,
    completion_result_from_response,
    load_prompt,
)
from cdt.extractor.schema import INSTRUMENT_RELATION_TYPES
from cdt.extractor.stages import (
    InstrumentIEStage,
    InstrumentRelationStage,
    NERStage,
    oriented_lineage_pair,
)
from cdt.extractor.state import CompletionResult, ExtractionRowState
from cdt.extractor.tags import (
    parse_tag_details,
    realign_tag_details,
    repair_unescaped_ampersands,
)
from cdt.extractor.validate import (
    validate_amount_is_not_rate,
    validate_dates_property,
    validate_instrument_entry,
    validate_interest_rate,
    validate_parties_property,
)


def _ner_row(text: str) -> ExtractionRowState:
    return ExtractionRowState(
        item_row={"item_id": "item-1", "text": text}, stage_name="ner"
    )


def test_instrument_ie_validate_allows_shared_evidence_and_skipped_collective_tags() -> (
    None
):
    """Shared evidence and skipped collective labels should validate successfully."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-2">Initial Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-1">$5.5 million</amount>
to <organization id="tag-o-1">EGT 11 LLC</organization>.
On <date id="tag-d-2">March 20, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1b">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-3">Subsequent Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-2">$269,000</amount>
to <organization id="tag-o-1b">EGT 11 LLC</organization>.
Together, the <debt_instrument id="tag-i-4">Exchange Notes</debt_instrument> were outstanding.
</body>
""".strip()
    response = """
[
  {
    "name": [
      "tag-i-1",
      "tag-i-2"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-1"
        ],
        "normalized_date": "2025-03-17"
      }
    ],
    "amounts": [
      {
        "evidence": [
          "tag-a-1"
        ],
        "normalized_amount": "5500000",
        "currency": "USD",
        "kind": "principal",
        "as_of_date": null
      }
    ],
    "parties": [
      {
        "tag_ids": [
          "tag-o-1"
        ],
        "role": "lender",
        "kind": "named"
      }
    ]
  },
  {
    "name": [
      "tag-i-1b",
      "tag-i-3"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-2"
        ],
        "normalized_date": "2025-03-20"
      }
    ],
    "amounts": [
      {
        "evidence": [
          "tag-a-2"
        ],
        "normalized_amount": "269000",
        "currency": "USD",
        "kind": "principal",
        "as_of_date": null
      }
    ],
    "parties": [
      {
        "tag_ids": [
          "tag-o-1b"
        ],
        "role": "lender",
        "kind": "named"
      }
    ]
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert failures == []


def test_instrument_ie_validate_accepts_party_kinds_and_roles() -> None:
    """Annotated lender and other-party clusters should validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "parties": [
                    {"tag_ids": ["tag-o-named"], "role": "lender", "kind": "named"},
                    {
                        "tag_ids": ["tag-o-collective"],
                        "role": "lender",
                        "kind": "collective",
                    },
                    {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(party_row_state(), response) == []


def test_instrument_ie_validate_rejects_unannotated_party_clusters() -> None:
    """Bare tag-id lists and unknown roles should fail validation."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "parties": [
                    ["tag-o-named"],
                    {"tag_ids": ["tag-o-agent"], "role": "servicer"},
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "must be an object with 'tag_ids' and 'role'" in failure for failure in failures
    )
    assert any("'role' must be one of" in failure for failure in failures)


def test_instrument_ie_validate_rejects_non_boolean_lenders_known_incomplete() -> None:
    """The retired lenders_known_incomplete flag is rejected outright."""
    response = json.dumps([{"name": ["tag-i-1"], "lenders_known_incomplete": "yes"}])

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders_known_incomplete' is not a property of this schema" in failure
        for failure in failures
    )


def test_instrument_ie_validate_rejects_conflicting_start_dates() -> None:
    """One extracted object cannot carry two distinct start dates."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The <debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
was issued on <date id="tag-d-1">March 17, 2025</date> and <date id="tag-d-2">March 20, 2025</date>.
</body>
""".strip()
    response = """
[
  {
    "name": [
      "tag-i-1"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-1",
          "tag-d-2"
        ],
        "normalized_date": "2025-03-17"
      }
    ]
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert any("multiple distinct normalized values" in failure for failure in failures)


def test_instrument_ie_validate_accepts_name_span_as_end_date_evidence() -> None:
    """The instrument name span is valid end_date evidence when maturity is embedded."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "maturity",
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(maturity_row_state(), response) == []


def test_instrument_ie_validate_still_rejects_name_span_as_start_date_evidence() -> (
    None
):
    """Only end_date may cite a debt_instrument tag."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    }
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(maturity_row_state(), response)

    assert any("expected date" in failure for failure in failures)


def test_instrument_ie_postprocess_leaves_end_date_null_without_maturity() -> None:
    """Names without a maturity phrase should not gain an invented end date."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The Company issued <debt_instrument id="tag-i-1">3.875% senior notes</debt_instrument>
on <date id="tag-d-1">March 17, 2025</date>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps([{"name": ["tag-i-1"]}])

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] is None
    assert json.loads(str(mention["maturity_date_json"]))["normalized_date"] is None


RATE_AMOUNT_XML = """
<body>
<debt_instrument id="tag-i-1">ABR Loan</debt_instrument> borrowings bear interest at
<amount id="tag-a-rate">0.875% per annum</amount> and the facility provides for
<amount id="tag-a-principal">$500.0 million</amount> of commitments.
</body>
""".strip()


def rate_amount_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state with both a rate and a principal amount."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_AMOUNT_XML
    return row_state


def test_instrument_ie_validate_rejects_rate_only_amount_evidence() -> None:
    """An amount citing only a rate should fail validation and retry."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-rate"],
                        "normalized_amount": "0.875",
                        "currency": None,
                    }
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(rate_amount_row_state(), response)

    assert any(
        "describes an interest rate, margin, or fee" in failure for failure in failures
    )


def test_instrument_ie_validate_rejects_wrapped_basis_point_evidence() -> None:
    """Basis-point evidence is rejected, and the retry quotes readable text (#102).

    The evidence span wraps across a line, as filings do. Judging whitespace-free
    text hid the `basis point` marker from the predicate entirely, so the retry
    never fired and the amount was silently nulled instead.
    """
    tag_details = {"tag-a-bps": {"type": "amount", "text": "100 basis\npoints"}}
    failures = validate_amount_is_not_rate(
        index=0,
        value={"evidence": ["tag-a-bps"]},
        tag_details=tag_details,
    )

    assert len(failures) == 1
    assert "'100 basis points'" in failures[0]
    assert "describes an interest rate, margin, or fee" in failures[0]


def test_instrument_ie_validate_accepts_spelled_currency_principal() -> None:
    """A principal whose currency and scale are words, not symbols, validates (#102)."""
    tag_details = {
        "tag-a-spelled": {
            "type": "amount",
            "text": "500 million U.S. dollars, or 5% of assets",
        }
    }

    assert (
        validate_amount_is_not_rate(
            index=0,
            value={"evidence": ["tag-a-spelled"]},
            tag_details=tag_details,
        )
        == []
    )


def test_instrument_ie_validate_accepts_principal_amount_evidence() -> None:
    """A principal amount should still validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-principal"],
                        "normalized_amount": "500000000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(rate_amount_row_state(), response) == []


def test_instrument_ie_postprocess_drops_rate_amount() -> None:
    """A rate that slips past validation should not be stored as an amount."""
    row_state = rate_amount_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-rate"],
                        "normalized_amount": "0.875",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] is None
    assert payload["normalized_amount"] is None
    assert payload["currency"] is None
    # Evidence is preserved so the dropped value stays auditable.
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-a-rate"]


def test_lineage_pair_is_oriented_by_relation_type() -> None:
    """Each lineage type has its own side that must hold the later date (#138, #142).

    `amendment_of` runs from the amended instrument to the predecessor, so its
    source is the later of the two; `retired_by` runs from the retired
    obligation to the instrument that retired it, so its source is the earlier.
    Both directions of both types are asserted here: flipping unconditionally
    is as wrong as never flipping, and only one of the two used to be covered
    (#180).

    Alclear's revolver is the confirmed case: a Credit Agreement dated as of
    2020-03-31, amended 2026-06-23 to cut commitments from $100,000,000 and
    extend maturity from 2026-06-28 to 2031-06-23. Every figure on the
    `$100,000,000` object is pre-amendment, so it is the predecessor, and the
    pointer must run the other way.
    """
    by_raw_id = {
        "i-1": {"raw_id": "i-1", "start_date": "2020-03-31", "amount": "100000000"},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23", "amount": "50000000"},
    }
    # The model named the predecessor first; the pointer is flipped.
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", by_raw_id) == (
        "i-2",
        "i-1",
    )
    # Already the right way round, so it is left alone.
    assert oriented_lineage_pair("i-2", "i-1", "amendment_of", by_raw_id) == (
        "i-2",
        "i-1",
    )
    # `retired_by` runs the other way: the retired obligation is the earlier
    # side, so a pair the model named retirer-first is flipped.
    assert oriented_lineage_pair("i-2", "i-1", "retired_by", by_raw_id) == (
        "i-1",
        "i-2",
    )
    # And a `retired_by` pair already named retired-first is left alone, which
    # is what an unconditional flip would break.
    assert oriented_lineage_pair("i-1", "i-2", "retired_by", by_raw_id) == (
        "i-1",
        "i-2",
    )


def test_lineage_pair_is_left_alone_without_two_dates_to_compare() -> None:
    """Nothing to orient on means nothing is changed (#138)."""
    same = {
        "i-1": {"raw_id": "i-1", "start_date": "2025-08-01"},
        "i-2": {"raw_id": "i-2", "start_date": "2025-08-01"},
    }
    # Equal dates carry no ordering, which is exactly the state the old prompt
    # convention produced, and why the direction went unchecked for so long.
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", same) == ("i-1", "i-2")
    # Equal dates are not a contradiction for `retired_by` either: a filing
    # that issues and redeems on the same day gives nothing to orient on.
    assert oriented_lineage_pair("i-1", "i-2", "retired_by", same) == ("i-1", "i-2")
    missing = {
        "i-1": {"raw_id": "i-1", "start_date": None},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23"},
    }
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", missing) == (
        "i-1",
        "i-2",
    )
    assert oriented_lineage_pair("i-2", "i-1", "retired_by", missing) == (
        "i-2",
        "i-1",
    )
    # `split_of` is not a before/after relation, so it is never reoriented.
    ordered = {
        "i-1": {"raw_id": "i-1", "start_date": "2020-03-31"},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23"},
    }
    assert oriented_lineage_pair("i-1", "i-2", "split_of", ordered) == ("i-1", "i-2")


def test_relation_postprocess_flips_an_inverted_amendment() -> None:
    """The published mention carries the corrected direction (#138)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "x"}, stage_name="instrument_relation"
    )
    row_state.debt_instrument_mentions = [
        {
            "raw_id": "i-1",
            "debt_instrument_mention_id": "dim::pred",
            "start_date": "2020-03-31",
            "amendment_of": None,
        },
        {
            "raw_id": "i-2",
            "debt_instrument_mention_id": "dim::amended",
            "start_date": "2026-06-23",
            "amendment_of": None,
        },
    ]
    row_state.stage_responses["instrument_relation"] = json.dumps(
        [{"from": "i-1", "to": "i-2", "type": "amendment_of"}]
    )

    InstrumentRelationStage().postprocess(row_state)

    by_raw = {m["raw_id"]: m for m in row_state.debt_instrument_mentions}
    assert by_raw["i-2"]["amendment_of"] == "dim::pred"
    assert by_raw["i-1"]["amendment_of"] is None


def test_completion_result_captures_a_filtered_live_response() -> None:
    """A provider abort is recorded, not just its partial text (#135).

    This is the shape observed live: `content_filter`, partial content, and a
    zeroed unbilled usage block. Without `finish_reason` the audit log cannot
    tell it from a model that chose to stop, and the two need opposite remedies.
    """
    usage = SimpleNamespace(
        completion_tokens=0, prompt_tokens=0, total_tokens=0, cost=0.0
    )
    message = SimpleNamespace(content="<body>partial", refusal=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="content_filter", message=message)],
        usage=usage,
        id="gen-1788206728-iWdiE2p9PV0",
        model="openai/gpt-5.6-terra",
    )

    result = completion_result_from_response(response)

    assert result.text == "<body>partial"
    assert result.finish_reason == "content_filter"
    assert result.response_id == "gen-1788206728-iWdiE2p9PV0"
    assert result.served_model == "openai/gpt-5.6-terra"
    assert result.usage == {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def test_completion_result_captures_a_filtered_batch_line() -> None:
    """The batch route carries the same fields on the same path (#135).

    A filtered batch response arrives as a normal 200 with a body, so it never
    reaches the infrastructure-error path and would otherwise be indistinguishable
    from ordinary bad output.
    """
    line = {
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-abc",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": "<body>partial", "refusal": None},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 0,
                    "total_tokens": 11,
                },
            },
        }
    }

    result = completion_result_from_batch_line(line)

    assert result.text == "<body>partial"
    assert result.finish_reason == "content_filter"
    assert result.response_id == "chatcmpl-abc"
    assert result.served_model == "gpt-5.4"
    assert result.usage["total_tokens"] == 11


def test_attempt_records_provider_metadata() -> None:
    """The metadata reaches the attempt, and so `full.jsonl` (#135)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "x"}, stage_name="ner"
    )
    row_state.add_response(
        "<body>x</body>",
        CompletionResult(
            text="<body>x</body>",
            finish_reason="length",
            usage={"total_tokens": 42},
            response_id="gen-9",
            served_model="openai/gpt-5.4",
        ),
    )

    recorded = row_state.current_attempt.to_dict()
    assert recorded["finish_reason"] == "length"
    assert recorded["usage"] == {"total_tokens": 42}
    assert recorded["response_id"] == "gen-9"
    assert recorded["served_model"] == "openai/gpt-5.4"
    # An attempt recorded without provider metadata stays null rather than absent.
    other = ExtractionRowState(item_row={"item_id": "i", "text": "x"}, stage_name="ner")
    other.add_response("<body>x</body>")
    assert other.current_attempt.to_dict()["finish_reason"] is None


def test_repair_unescaped_ampersands_leaves_real_entities_alone() -> None:
    """A bare ampersand is escaped; anything already an entity is untouched (#127)."""
    assert repair_unescaped_ampersands("A&R Agreement") == "A&amp;R Agreement"
    assert repair_unescaped_ampersands("Smith & Wesson & Co") == (
        "Smith &amp; Wesson &amp; Co"
    )
    for already_valid in (
        "A&amp;R",
        "a &lt; b",
        "a &gt; b",
        "&quot;x&quot;",
        "&apos;x&apos;",
        "&#8217;s",
        "&#x2019;s",
    ):
        assert repair_unescaped_ampersands(already_valid) == already_valid


def test_ner_validate_accepts_a_response_carrying_a_bare_ampersand() -> None:
    """The exact-text and well-formed-XML requirements conflict without this (#127).

    `NERStage.preprocess` wraps the item text in `<body>` unescaped, so an item
    naming an `A&R Registration Rights Agreement` is handed to the model as
    invalid XML. Reproducing it verbatim then yields invalid XML, and 44 of 342
    relevant items in one held-out window carry a bare ampersand.
    """
    text = "The Company entered into the A&R Registration Rights Agreement."
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": text}, stage_name="ner"
    )
    response = (
        "<body>The Company entered into the <agreement>A&R Registration Rights "
        "Agreement</agreement>.</body>"
    )

    assert NERStage().validate(row_state, response) == []

    row_state.stage_responses["ner"] = response
    NERStage().postprocess(row_state)
    _, plain_text, tag_details = parse_tag_details(str(row_state.ner_tagged_xml))
    # The ampersand round-trips, so the text invariant still holds downstream.
    assert plain_text == text
    assert [d["text"] for d in tag_details.values()] == [
        "A&R Registration Rights Agreement"
    ]


def test_ner_validate_still_rejects_a_stray_angle_bracket() -> None:
    """Only `&` is repaired; a malformed tag is still a failure (#127)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "The Company borrowed."},
        stage_name="ner",
    )
    failures = NERStage().validate(row_state, "<body>The Company <borrowed.</body>")
    assert failures and "not valid XML" in failures[0]


# MPLX item 2.03 (`000119312519257376-2-03`) in miniature: one note series the
# model tags, in an item whose verbatim reproduction it then gets wrong.
MPLX_TEXT = "The Company issued 6.250% Senior Notes due 2022."
MPLX_TAGGED_BUT_UNFAITHFUL = (
    "<body>The Company issued <debt_instrument>6.250% Senior Notes due "
    "2022</debt_instrument>!</body>"
)
MPLX_IE = json.dumps([{"name": ["tag-1"]}])
# The same tagging with the text left exactly as it was given.
MPLX_TAGGED = (
    "<body>The Company issued <debt_instrument>6.250% Senior Notes due "
    "2022</debt_instrument>.</body>"
)


def test_ner_validate_rejects_a_zero_tag_retry_after_an_earlier_attempt_tagged() -> (
    None
):
    """The high-water mark: dropping every tag on retry is a failure (#176)."""
    from cdt.extractor.state import AttemptRecord

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response=MPLX_TAGGED_BUT_UNFAITHFUL,
            status="FAILED",
        )
    )

    # Valid, faithful, and carrying no `debt_instrument` -- it passes all six
    # structural checks. Tagged with a `date` so it is not also a byte echo,
    # which isolates the high-water check from the echo check.
    failures = NERStage().validate(
        row_state,
        "<body>The Company issued 6.250% Senior Notes due <date>2022</date>.</body>",
    )

    assert failures
    assert any("no <debt_instrument> tags" in failure for failure in failures)
    # The message quotes the count so the retry turn names what was lost.
    assert any("tagged 1" in failure for failure in failures)


def test_ner_validate_high_water_never_misfires_on_a_debt_free_item() -> None:
    """An item with no debt found none on attempt 1 either, so nothing regresses (#176).

    This is the check that makes the high-water mark safe to apply
    unconditionally: it is a comparison against the row's own history, not a
    floor on how many tags a response must carry.
    """
    from cdt.extractor.state import AttemptRecord

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response="not xml at all",
            status="FAILED",
        )
    )
    response = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert NERStage().validate(row_state, response) == []


def test_ner_high_water_counts_tags_in_attempts_that_never_parsed() -> None:
    """A truncated prior attempt still proves the model was tagging (#176, #127).

    11 of 27 NER failures on the PR #57 window were `Response is not valid
    XML`, so a high-water mark built on `parse_tag_details` would read zero for
    exactly the attempts that matter most.
    """
    from cdt.extractor.stages import (
        count_debt_instrument_tags,
        prior_debt_instrument_high_water,
    )
    from cdt.extractor.state import AttemptRecord

    truncated = (
        "<body>The Company issued <debt_instrument>6.250% Senior Notes"
        "</debt_instrument> and <debt_instrument>5.250% Notes"
    )
    assert count_debt_instrument_tags(truncated) == 2

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(stage_name="ner", response=truncated, status="FAILED")
    )
    # Another stage's attempts are not this stage's history.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="instrument_ie",
            response="<debt_instrument><debt_instrument><debt_instrument>",
            status="FAILED",
        )
    )

    assert prior_debt_instrument_high_water(row_state, "ner") == 2


def test_ner_validate_rejects_a_byte_identical_echo_after_the_model_tagged() -> None:
    """Returning the input verbatim is a regression against the model's own work (#176).

    MPLX attempt 3 was byte-identical to the model's own input, `<body>`
    wrapper included, and passed. Driven through `handle_response` rather than
    hand-setting `attempt_index`: what makes the echo a give-up is that an
    earlier attempt on this row tagged something, and only a driven row builds
    that history the way production does.
    """
    from cdt.extractor.stages import ner_input_body
    from cdt.extractor.workflow import handle_response

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)

    echo = ner_input_body(row_state)
    assert echo == f"<body>{MPLX_TEXT}</body>"

    failures = NERStage().validate(row_state, echo)

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_validate_rejects_an_echo_that_drops_non_debt_tags() -> None:
    """Giving up is giving up even when the item has no debt (#176).

    The high-water mark asks the narrower question that drives `early_stop`:
    did an earlier attempt find a `debt_instrument`? A response that tagged an
    organization and a date and then regressed to a bare echo reads zero there,
    but the model has still discarded everything it found.
    """
    from cdt.extractor.workflow import handle_response

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    # Tags an organization and a date, and fails only copy fidelity.
    unfaithful = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>!</body>"
    )
    assert handle_response(row_state, unfaithful, max_attempts=3)

    failures = NERStage().validate(row_state, f"<body>{text}</body>")

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_validate_accepts_an_untagged_first_attempt() -> None:
    """On attempt 1 an untagged echo is the honest answer for a debt-free item (#176).

    The echo check is gated on earlier tagging for this reason, and was never
    made unconditional for it. Measured over the three
    stored corpora carrying attempt logs (`genwindow-run-branch`,
    `genwindow-run-dev`, `genwindow-sol-retried`): of 761 attempt-1 NER
    responses, zero were byte-identical echoes, and the 63 with no
    `debt_instrument` tag all carried some other tag. The only echo in the
    corpus is MPLX's attempt 3.

    Driven through `handle_response` rather than calling `validate` directly,
    because the boundary is exactly where `add_response` leaves
    `attempt_index`: 1 on a first attempt, not 0. A direct `validate` call on a
    freshly built row state sees 0, so it would pass an off-by-one guard that
    rejects every genuine first attempt.
    """
    from cdt.extractor.workflow import handle_response

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    assert row_state.all_attempts[0].attempt_index == 1
    assert row_state.all_attempts[0].validation_errors == []
    # An item with nothing to tag is a clean zero, not a give-up.
    assert row_state.state == "SUCCESS"
    assert row_state.debt_instrument_mentions == []


def test_ner_high_water_is_the_most_any_attempt_found_not_the_least() -> None:
    """The mark is the *maximum* across attempts, which is what makes it a guard (#176).

    MPLX is the shape that needs it: attempt 1 tagged 91 spans, attempt 2 came
    back malformed and tagged none, and attempt 3 was the untagged echo. Taking
    the minimum -- or just the last -- would read zero off attempt 2, switch the
    guard off, and let attempt 3 publish as a clean zero, which is the defect
    #176 describes, verbatim.
    """
    from cdt.extractor.stages import prior_debt_instrument_high_water
    from cdt.extractor.state import AttemptRecord

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response=MPLX_TAGGED_BUT_UNFAITHFUL,
            status="FAILED",
        )
    )
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=2,
            response="not xml at all",
            status="FAILED",
        )
    )

    # Tag counts across this row's history are 1 then 0: the max is 1, while
    # the min and the most recent are both 0.
    assert prior_debt_instrument_high_water(row_state, "ner") == 1


def test_ner_give_up_check_is_not_escaped_by_whitespace() -> None:
    """Perturbing the whitespace must not buy a give-up a pass (#176).

    The check was first written as "byte-identical to the input", and every
    one of these clears that comparison: a trailing newline, a space after
    `<body>`, newlines inside it, a space in the opening tag, doubled
    inter-word spacing. None of them clears copy fidelity either way, because
    that check runs on `collapse_whitespace`, so each one used to reach
    `early_stop` as an accepted zero-tag response. Asking for a tag count
    instead makes the whole family unreachable rather than enumerable.
    """
    from cdt.extractor.state import AttemptRecord

    for response in (
        f"<body>{MPLX_TEXT}</body>",
        f"<body>{MPLX_TEXT}</body>\n",
        f"<body> {MPLX_TEXT}</body>",
        f"<body>\n{MPLX_TEXT}\n</body>",
        f"<body >{MPLX_TEXT}</body>",
        "<body>" + MPLX_TEXT.replace(" ", "  ") + "</body>",
    ):
        row_state = _ner_row(MPLX_TEXT)
        row_state.all_attempts.append(
            AttemptRecord(
                stage_name="ner",
                attempt_index=1,
                response=MPLX_TAGGED_BUT_UNFAITHFUL,
                status="FAILED",
            )
        )

        failures = NERStage().validate(row_state, response)

        assert any(
            "drops every one of them" in failure for failure in failures
        ), response


def test_ner_high_water_accepts_a_reduced_but_nonzero_tag_count() -> None:
    """The high-water check is exact-zero by design, and the boundary is deliberate.

    A model that merges two adjacent spans into one has not given up, and no
    false-positive rate has been measured for any ratio threshold. Pinned so
    that tightening this to "fewer tags than before" is a visible decision
    rather than a quiet one (#176).
    """
    from cdt.extractor.state import AttemptRecord

    text = "The Company issued 6.250% Notes due 2022 and 5.250% Notes due 2025."
    two_tags = (
        "<body>The Company issued <debt_instrument>6.250% Notes due "
        "2022</debt_instrument> and <debt_instrument>5.250% Notes due "
        "2025</debt_instrument>.</body>"
    )
    one_tag = (
        "<body>The Company issued <debt_instrument>6.250% Notes due 2022 and "
        "5.250% Notes due 2025</debt_instrument>.</body>"
    )
    row_state = _ner_row(text)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner", attempt_index=1, response=two_tags, status="FAILED"
        )
    )

    assert NERStage().validate(row_state, one_tag) == []


def test_prior_attempt_tagged_reads_the_row_the_way_the_echo_guard_needs() -> None:
    """The three properties the echo gate rests on (#176).

    Mirrors `test_ner_high_water_counts_tags_in_attempts_that_never_parsed`
    for the wider any-tag question the echo check asks.
    """
    from cdt.extractor.stages import count_ner_entity_tags, prior_attempt_tagged
    from cdt.extractor.state import AttemptRecord

    # 1. `<body>` is the wrapper this stage supplies, not something the model
    #    found, so an untagged echo must not count as having tagged anything.
    assert count_ner_entity_tags("<body>plain text</body>") == 0
    # 2. A truncated attempt that never parsed still proves the model was
    #    tagging, which is why this counts by regex rather than by parse.
    assert count_ner_entity_tags("<body>x <debt_instrument>Term Loa") == 1
    assert count_ner_entity_tags("not xml at all") == 0

    row_state = _ner_row(MPLX_TEXT)
    # An earlier attempt that wrapped the input and tagged nothing.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner", response="<body>wrong text</body>", status="FAILED"
        )
    )
    assert prior_attempt_tagged(row_state, "ner") is False

    # 3. Another stage's attempts are not this stage's history.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="instrument_ie",
            response="<organization>A</organization>",
            status="FAILED",
        )
    )
    assert prior_attempt_tagged(row_state, "ner") is False

    row_state.all_attempts.append(
        AttemptRecord(stage_name="ner", response="<body>x <date>2022", status="FAILED")
    )
    assert prior_attempt_tagged(row_state, "ner") is True


def test_an_echo_is_accepted_when_the_earlier_attempt_also_tagged_nothing() -> None:
    """A wrapper is not a tag: the `<body>` the stage supplies proves nothing (#176).

    Attempt 1 wraps the input and tags nothing, failing only copy fidelity. It
    gives no evidence the model can find anything in this item, so attempt 2's
    honest echo is still the honest answer.
    """
    from cdt.extractor.workflow import handle_response

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(
        row_state, "<body>wrong text entirely</body>", max_attempts=3
    )
    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    assert row_state.all_attempts[-1].validation_errors == []
    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_an_untagged_echo_is_accepted_after_a_failure_that_found_nothing() -> None:
    """The regression the old `attempt_index > 1` gate caused (#176).

    A debt-free item whose first attempt failed for a reason unrelated to
    tagging -- malformed XML here -- answers honestly with a bare echo on its
    second. The old gate rejected that answer on every remaining attempt and
    the row died FAILED after three whole-item calls -- the stage's whole
    budget -- losing the item. Nothing
    about the first attempt suggests the model can find anything here, so
    there is no earlier work for the echo to regress against.

    Nothing on this row is evidence of a give-up, so it finishes SUCCESS. The
    zero used to be filed as a possible loss instead; see
    `_advance_after_stage` for why that was dropped.
    """
    from cdt.extractor.prior_state import published_mention_rows
    from cdt.extractor.state import PUBLISHABLE_ROW_STATES
    from cdt.extractor.workflow import handle_response

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(row_state, "not xml at all", max_attempts=3)
    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    # Two calls, not three, and the echo itself was accepted.
    assert len([a for a in row_state.all_attempts if a.response is not None]) == 2
    assert row_state.all_attempts[0].status == "FAILED"
    assert row_state.all_attempts[-1].validation_errors == []
    assert row_state.all_attempts[-1].status == "SUCCESS"
    assert row_state.state == "SUCCESS"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert published_mention_rows(row_state) == []
    assert row_state.salvage_notes == []


def test_a_zero_tag_row_with_no_earlier_tagging_is_a_clean_zero() -> None:
    """With no earlier tags to compare against, a zero is a finding, not a loss.

    Attempt 1 failed for a reason unrelated to tagging and itself found no
    `debt_instrument`, so neither the high-water mark nor the give-up check has
    anything to fire on -- and nothing else on the row suggests the model can
    find debt here. The row finishes SUCCESS with no failure record.

    This is the case that used to finish PARTIAL as a "possible loss". It was
    dropped because PARTIAL is terminal like any other state, so it bought a
    registry entry and no re-extraction while asserting a loss nothing had
    evidence for. A give-up the row *can* evidence is a validation failure, so
    it retries and then fails for real -- see
    `test_mplx_untagged_echo_no_longer_publishes_as_a_clean_success`.
    """
    from cdt.extractor.prior_state import published_mention_rows
    from cdt.extractor.state import PUBLISHABLE_ROW_STATES
    from cdt.extractor.workflow import handle_response

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert handle_response(row_state, "not xml at all", max_attempts=3)
    assert handle_response(row_state, tagged_no_debt, max_attempts=3) is None

    assert row_state.state == "SUCCESS"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert published_mention_rows(row_state) == []
    assert row_state.salvage_notes == []


def test_a_give_up_is_retried_and_the_row_recovers_if_a_later_answer_passes() -> None:
    """A give-up spends an attempt, it does not end the row (#176).

    The give-up check is an ordinary validation failure, so the model is told
    what was wrong and asked again, and a row that answers properly on the
    next attempt finishes SUCCESS with its mentions -- the same as any other
    row that needed a retry. Only running out of attempts ends it.
    """
    row_state, client = _run_live(
        MPLX_TEXT,
        [
            # Tagged, but it rewrote the text: fails copy fidelity.
            _stopped(MPLX_TAGGED_BUT_UNFAITHFUL),
            # Gives up -- drops every tag it had found.
            _stopped(f"<body>{MPLX_TEXT}</body>"),
            # Then answers properly, and the row carries on to instrument_ie.
            _stopped(MPLX_TAGGED),
            _stopped(MPLX_IE),
        ],
    )

    assert len(client.requests) == 4
    assert row_state.state == "SUCCESS"
    assert len(row_state.debt_instrument_mentions) == 1
    assert row_state.salvage_notes == []
    # The give-up was scored as a failed attempt, not as a terminal verdict.
    statuses = [a.status for a in row_state.all_attempts if a.stage_name == "ner"]
    assert statuses == ["FAILED", "FAILED", "SUCCESS"]


def test_an_echo_after_a_truncated_tagged_attempt_is_still_rejected() -> None:
    """A response cut mid-tag still proves the model was tagging (#176, #127)."""
    from cdt.extractor.workflow import handle_response

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    # Truncated mid-name, so it never parses -- a parse-based count reads zero.
    truncated = "<body>The Company issued <debt_instrument>6.250% Senior No"
    assert handle_response(row_state, truncated, max_attempts=3)

    failures = NERStage().validate(row_state, f"<body>{MPLX_TEXT}</body>")

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_retry_message_tells_the_model_to_keep_its_tags() -> None:
    """The retry turn must ask for a repair, not invite a redo (#176)."""
    message = NERStage().build_retry_message(["stripped text mismatch"])

    assert "Keep every tag from your previous output" in message
    assert "untagged is not a valid fix" in message
    # The preserve clause leads, ahead of the copy-fidelity bullets that the
    # model previously satisfied by discarding its work.
    assert message.index("Keep every tag") < message.index(
        "Return the original input text exactly"
    )


def test_mplx_untagged_echo_no_longer_publishes_as_a_clean_success() -> None:
    """End to end on #176's headline case: a counted loss, not a silent zero.

    Before this, attempts 1 and 2 failed the copy-fidelity check with 91
    `debt_instrument` spans each, attempt 3 returned the input byte for byte,
    validated clean, and the row published `SUCCESS` with 0 mentions against
    six note series and a term loan -- writing a completion record that makes a
    re-run skip the item.
    """
    from cdt.extractor.outputs import summarize_failure
    from cdt.extractor.workflow import handle_response

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    # Attempts 1 and 2 tag densely and fail only copy fidelity, as MPLX's did.
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)

    # Attempt 3 is the give-up that used to pass, and so is every attempt after
    # it: the row exhausts NER's budget being rejected rather than early-stopping
    # SUCCESS on the first one.
    echo = f"<body>{MPLX_TEXT}</body>"
    calls = 2
    result: list[dict[str, str]] | None = [{}]
    while result is not None and calls < 20:
        result = handle_response(row_state, echo, max_attempts=3)
        calls += 1

    assert result is None
    # One budget for every stage (#127).
    assert calls == 3
    assert row_state.state == "FAILED"
    # FAILED exactly: a give-up the row can evidence retries to the stage's
    # budget and then terminates like any other exhausted stage.
    assert row_state.state == "FAILED"
    assert row_state.debt_instrument_mentions == []
    assert "drops every one of them" in summarize_failure(row_state)
    # No attempt on this row was ever accepted.
    assert [a.status for a in row_state.all_attempts] == ["FAILED"] * 3


def test_a_genuinely_debt_free_item_still_early_stops_success() -> None:
    """The no-retry path is untouched: a clean zero stays a clean SUCCESS (#176)."""
    from cdt.extractor.workflow import handle_response

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert handle_response(row_state, tagged_no_debt, max_attempts=3) is None

    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_ner_high_water_survives_the_resumable_batch_state() -> None:
    """Batch resumability needed no schema change: `all_attempts` already round-trips.

    A batch row can cross a process exit between its failed attempt and its
    retry, so the guard has to be rebuildable from `state.jsonl` alone (#176).
    """
    from cdt.extractor.stages import prior_debt_instrument_high_water

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    from cdt.extractor.workflow import handle_response

    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)
    assert prior_debt_instrument_high_water(row_state, "ner") == 1

    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())

    assert prior_debt_instrument_high_water(restored, "ner") == 1
    # And the restored row rejects the give-up just as the live one would. A
    # zero-tag response that is not also a byte echo, so this exercises the
    # high-water mark rather than the echo check, which returns first.
    failures = NERStage().validate(
        restored,
        "<body>The Company issued 6.250% Senior Notes due <date>2022</date>.</body>",
    )
    assert failures and any("no <debt_instrument> tags" in f for f in failures)


# The debt-free item and its honest answer: an untagged echo is correct here,
# which is what makes it the right probe for the echo guard's rule that an
# echo is only a give-up once the model has tagged something on this row.
NODEBT_TEXT = "This is the extracted event text."
NODEBT_NER = f"<body>{NODEBT_TEXT}</body>"

# A two-instrument item, so the relation stage actually runs on it.
MULTI_TEXT = "Company entered into a Term Loan and a Revolver on January 1, 2024."
MULTI_NER = (
    "<body>Company entered into a <debt_instrument>Term Loan</debt_instrument> "
    "and a <debt_instrument>Revolver</debt_instrument> on "
    "<date>January 1, 2024</date>.</body>"
)
# One entry that validates and one that cites a tag id the NER output never
# produced, so `instrument_ie` rejects the response as a whole while
# `salvage_instrument_ie_entries` keeps the first entry (#152).
MULTI_IE_ONE_BAD = json.dumps(
    [
        {
            "name": ["tag-1"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
        {"name": ["tag-99"]},
    ]
)
MULTI_IE = json.dumps(
    [
        {
            "name": ["tag-1"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
        {
            "name": ["tag-2"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
    ]
)

# As observed live: aborted upstream, nothing generated, nothing billed.
# A real `content_filter` body, trimmed, from item 000114036126024567-8-01 in
# `ie_review/runs/followups/full.jsonl` of the models repo. An abort arrives as
# a normal 200 carrying whatever the provider had emitted before it cut, so it
# is partially-tagged XML ending mid-tag -- not an empty string. All 58
# `content_filter` responses in the stored corpora carry 9-107 entity tags and
# at least one `debt_instrument` tag, and none is empty, so a `text=""` fixture
# exercises a shape that has never occurred. It mattered: with an empty body
# the cross-attempt guards in #176 could read an aborted call as the model's
# own tagged work and no test objected.
CONTENT_FILTER_PARTIAL = "<body>Item 8.01\nOther Events.\nOn <date>June 9, 2026</date>, the <organization>Company</organization> commenced an offering of <amount>$500.0 million</amount> in aggregate principal amount of its <debt_instrument>senior secured notes due 2031</debt_instrument> (the “<debt_instrument>Notes"
CONTENT_FILTERED = CompletionResult(
    text=CONTENT_FILTER_PARTIAL,
    finish_reason="content_filter",
    usage={"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0},
)
# The degenerate shape, kept so both are covered: some aborts may carry nothing.
CONTENT_FILTERED_EMPTY = CompletionResult(
    text="",
    finish_reason="content_filter",
    usage={"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0},
)


def _stopped(text: str) -> CompletionResult:
    return CompletionResult(text=text, finish_reason="stop")


class _ScriptedCompletionClient:
    """Fake chat client replaying scripted `CompletionResult`s.

    Unlike the text-only scripted client in `test_extractor_batch.py`, this one
    scripts the whole completion, so a `content_filter` abort can be injected
    where it really occurs -- at the provider boundary, above
    `handle_response` -- rather than simulated by hand-feeding the scorer
    (#127, #135).
    """

    def __init__(
        self: _ScriptedCompletionClient, completions: list[CompletionResult]
    ) -> None:
        """Store the completions to hand back in order."""
        self.completions = list(completions)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(
        self: _ScriptedCompletionClient,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> CompletionResult:
        """Record the request and return the next scripted completion."""
        del model, reasoning_effort
        self.requests.append([dict(message) for message in messages])
        return self.completions[len(self.requests) - 1]


def _run_live(
    text: str, completions: list[CompletionResult], max_attempts: int = 3
) -> tuple[ExtractionRowState, _ScriptedCompletionClient]:
    """Drive the real live loop, returning (row_state, client)."""
    import asyncio

    from cdt.extractor.workflow import run_extraction_workflow

    client = _ScriptedCompletionClient(completions)
    row_state = asyncio.run(
        run_extraction_workflow(
            item_row={"item_id": "item-1", "text": text},
            model="m",
            reasoning_effort="none",
            max_attempts=max_attempts,
            client=client,
        )
    )
    return row_state, client


def test_every_stage_gets_the_same_attempt_budget() -> None:
    """`--max-attempts` means the same thing for every stage (#127).

    #127 proposed a larger NER budget; the stored corpora do not support the
    number (only two of 761 rows ever reached the cap, and one of those is the
    give-up #176 now rejects), so NER is back in line with the rest. Driven
    end to end rather than asserted against a helper, because the budget is
    only real if the loop stops there.
    """
    from cdt.extractor.state import DEFAULT_MAX_ATTEMPTS

    # Pinned as a literal for the same reason MAX_CONTENT_FILTER_RESENDS is:
    # the budget *is* the number, so restating the constant asserts nothing
    # about its value.
    assert DEFAULT_MAX_ATTEMPTS == 3

    # NER: three scored attempts, then the row terminates.
    row_state, client = _run_live(MPLX_TEXT, [_stopped("not xml")] * 10)
    assert len(client.requests) == 3
    assert row_state.state == "FAILED"

    # instrument_ie: the same three, after a NER pass.
    row_state, client = _run_live(
        MULTI_TEXT, [_stopped(MULTI_NER)] + [_stopped("not json")] * 10
    )
    assert len(client.requests) == 1 + 3
    assert row_state.state == "FAILED"

    # And the operator's knob still moves it.
    row_state, client = _run_live(MPLX_TEXT, [_stopped("not xml")] * 10, max_attempts=5)
    assert len(client.requests) == 5


def test_a_content_filtered_attempt_is_resent_as_the_original_request() -> None:
    """An upstream abort has nothing for the model to correct (#127, #135).

    The ordinary retry path would append the aborted response as an assistant
    turn plus a validation complaint about it, which every later call in the
    row then pays prompt tokens for. A `content_filter` abort is not the
    model answering badly, so the same request goes back out unchanged.
    """
    row_state, client = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(NODEBT_NER)])

    assert len(client.requests) == 2
    # Byte-identical resend: no assistant turn, no complaint, same request.
    assert client.requests[1] == client.requests[0]
    assert all(message["role"] != "assistant" for message in client.requests[1])
    assert row_state.state == "SUCCESS"


def test_a_provider_abort_is_recorded_but_not_scored() -> None:
    """The audit log keeps the call; the scorer never sees it (#127, #135).

    `attempt_index` stays a true count of *scored* attempts, so the model's
    first real answer is attempt 1 even when the provider aborted first -- which
    is what keeps #176's cross-attempt checks from reading an abort as a retry.
    """
    row_state, _ = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(NODEBT_NER)])

    statuses = [
        (a.stage_name, a.attempt_index, a.status) for a in row_state.all_attempts
    ]
    assert statuses == [("ner", 0, "ABORTED"), ("ner", 1, "SUCCESS")]
    aborted = row_state.all_attempts[0]
    assert aborted.finish_reason == "content_filter"
    assert aborted.usage == {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0}
    # The request is identical to the scored attempt's, so it is not stored twice.
    assert aborted.messages == []


def test_content_filter_aborts_do_not_consume_the_stage_budget() -> None:
    """Unscored calls must not spend the row's attempts (#127, #135).

    Eleven of thirteen live NER calls came back `content_filter` with
    `completion_tokens=0` and `cost=0.0`, and the cut point is nondeterministic
    (1,163 / 280 / 1,375 characters on three repeats of one item), so resending
    is both correct and free.
    """
    # Three aborts, then the stage's full budget of three real attempts, all
    # rejected. The aborts cost the row nothing: it still gets all three.
    completions = [CONTENT_FILTERED] * 3 + [_stopped("not xml")] * 3
    row_state, client = _run_live(MPLX_TEXT, completions)

    assert len(client.requests) == 6
    assert row_state.state == "FAILED"
    scored = [a for a in row_state.all_attempts if a.status != "ABORTED"]
    assert [a.attempt_index for a in scored] == [1, 2, 3]


def test_persistent_content_filtering_terminates_at_the_resend_cap() -> None:
    """Free retries still have to stop, and stopping must not blame the model (#127).

    Past the cap the abort used to fall through to `handle_response` and be
    scored as an ordinary bad answer: the row paid the stage's whole budget
    again in whole-item calls,
    each one growing the retry conversation with a junk assistant turn, and
    the resulting `FAILED` attempt made #176's checks read a provider abort as
    a model failure. The row now terminates instead.
    """
    from cdt.extractor.outputs import summarize_failure
    from cdt.extractor.state import MAX_CONTENT_FILTER_RESENDS

    # Pinned as a literal: the cap bounds spend on a filtered row, so restating
    # the constant here would assert nothing about its value.
    assert MAX_CONTENT_FILTER_RESENDS == 6

    row_state, client = _run_live(MPLX_TEXT, [CONTENT_FILTERED] * 40)

    # Six re-sends plus the call that trips the cap, and no more.
    assert len(client.requests) == 7
    assert row_state.state == "FAILED"
    # Not one call was scored, so the model is never blamed for the abort.
    # The trailing "incomplete" is the unused outstanding attempt that
    # `finish` always appends, not a call that was made.
    assert [a.status for a in row_state.all_attempts] == ["ABORTED"] * 7 + [
        "incomplete"
    ]
    assert row_state.current_attempt.attempt_index == 0
    # Every request was the pristine one: no conversation growth past the cap.
    assert all(request == client.requests[0] for request in client.requests)
    summary = summarize_failure(row_state)
    assert "aborted by the provider on 7 of its calls" in summary
    # Nothing was scored on this row, so saying so is accurate here.
    assert "no attempt was scored" in summary


def test_the_abort_note_reports_the_model_failure_that_happened_too() -> None:
    """When both the provider and the model failed, the registry says both (#127).

    The abort count is a per-stage lifetime count with no reset, so aborts need
    not be consecutive: this row is aborted six times, answers badly once, and
    is aborted again past the cap. The note used to call all seven calls
    "consecutive" and add "no attempt was scored", and because
    `summarize_failure` prefers salvage notes over `validation_errors`, the
    model's own error never reached the registry. An operator was told to go
    and talk to the provider while the extraction defect stayed invisible.
    """
    from cdt.extractor.outputs import summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [CONTENT_FILTERED] * 6 + [_stopped("not xml at all")] + [CONTENT_FILTERED] * 5,
    )

    assert len(client.requests) == 6 + 1 + 1
    assert row_state.state == "FAILED"
    statuses = [a.status for a in row_state.all_attempts if a.stage_name == "ner"]
    assert statuses == ["ABORTED"] * 6 + ["FAILED", "ABORTED", "incomplete"]

    summary = summarize_failure(row_state)
    # The count is right, and no longer claims the calls were back to back.
    assert "aborted by the provider on 7 of its calls" in summary
    assert "consecutive" not in summary
    # One call *was* scored, so the opposite claim is gone...
    assert "no attempt was scored" not in summary
    assert "1 scored attempt also failed" in summary
    # ...and the model's own error reaches the registry alongside the abort.
    assert "not valid XML" in summary


def test_the_abort_note_only_reports_failures_from_the_stage_that_aborted() -> None:
    """An earlier stage's failure is not this stage's story (#127).

    NER fails once and then recovers, so the row reaches `instrument_ie` with a
    scored failure already in its history. `instrument_ie` is then aborted to
    its cap without ever being answered, and its note must say so -- citing
    NER's error here would send an operator to the wrong stage.
    """
    from cdt.extractor.outputs import failed_stage_name, summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [_stopped("not xml at all"), _stopped(MPLX_TAGGED)] + [CONTENT_FILTERED] * 10,
    )

    assert len(client.requests) == 2 + 7
    assert row_state.state == "FAILED"
    assert failed_stage_name(row_state) == "instrument_ie"

    summary = summarize_failure(row_state)
    assert "instrument_ie aborted by the provider on 7 of its calls" in summary
    # instrument_ie itself was never answered, so that claim is true of it...
    assert "no attempt was scored" in summary
    # ...and NER's failure, which belongs to a stage that went on to succeed,
    # stays out of it.
    assert "not valid XML" not in summary


def test_the_abort_note_reports_the_most_recent_scored_failure() -> None:
    """Of several scored failures, the latest is the one worth reporting (#127).

    The model is shown its error and asked again, so the last rejection is the
    state the row actually died in; an earlier one has already been superseded.
    """
    from cdt.extractor.outputs import summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [
            _stopped("not xml at all"),
            _stopped("<body>wrong text entirely</body>"),
        ]
        + [CONTENT_FILTERED] * 10,
    )

    assert len(client.requests) == 2 + 7
    assert row_state.state == "FAILED"

    summary = summarize_failure(row_state)
    assert "2 scored attempts also failed" in summary
    assert "must match the input text exactly" in summary
    assert "not valid XML" not in summary


def test_a_filtered_row_is_registered_against_the_stage_that_was_aborted() -> None:
    """An operator retrying the row needs the stage, not a generic failure (#127)."""
    from cdt.extractor.outputs import failed_stage_name

    row_state, _ = _run_live(MPLX_TEXT, [CONTENT_FILTERED] * 40)

    assert failed_stage_name(row_state) == "ner"


def test_aborts_at_the_relation_stage_still_publish_the_items_mentions() -> None:
    """#152's rule holds: a stage the provider will not run costs lineage, not instruments.

    The relation stage is the last one, and its output is only lineage. A row
    whose instruments already validated must not lose them because the
    provider refused to run the final call.
    """
    from cdt.extractor.outputs import summarize_failure
    from cdt.extractor.state import PUBLISHABLE_ROW_STATES

    row_state, client = _run_live(
        MULTI_TEXT,
        [_stopped(MULTI_NER), _stopped(MULTI_IE)] + [CONTENT_FILTERED] * 40,
    )

    # Two scored calls for ner and instrument_ie, then the relation stage is
    # aborted to its cap.
    assert len(client.requests) == 2 + 7
    assert row_state.state == "PARTIAL"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert len(row_state.debt_instrument_mentions) == 2
    assert "mentions published without lineage relations" in summarize_failure(
        row_state
    )


def test_aborts_at_the_ie_stage_still_publish_the_entries_that_validated() -> None:
    """The abort cap applies #152 the same way a scored failure does (#127, #152).

    An `instrument_ie` response rejected as a whole can still hold valid
    entries, and when the provider then aborts the stage to its cap those
    entries are already sitting in `stage_responses` -- the aborts came after
    the answer, not instead of it. Terminating FAILED there threw them away,
    while the same row reaching the same dead end through three *scored*
    failures published them. Same loss, same remedy, so the two terminal paths
    now agree.
    """
    from cdt.extractor.outputs import summarize_failure
    from cdt.extractor.state import PUBLISHABLE_ROW_STATES

    row_state, client = _run_live(
        MULTI_TEXT,
        [_stopped(MULTI_NER), _stopped(MULTI_IE_ONE_BAD)] + [CONTENT_FILTERED] * 40,
    )

    # One NER call, one scored instrument_ie answer, then the stage is aborted
    # to its cap.
    assert len(client.requests) == 1 + 1 + 7
    assert row_state.state == "PARTIAL"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    # The entry that validated publishes; the one citing a missing tag does not.
    assert len(row_state.debt_instrument_mentions) == 1
    assert "published without lineage relations" in summarize_failure(row_state)
    # And the invalid entry is gone from the stored response, not merely
    # ignored downstream: `salvage_instrument_ie_entries` rewrites what the
    # audit log keeps, so the kept set is recorded rather than inferred.
    # `postprocess` would have skipped the bad entry either way, so the mention
    # count alone cannot tell a salvage from an unfiltered publish.
    assert "tag-99" not in row_state.stage_responses["instrument_ie"]


def test_content_filter_resend_cap_survives_the_resumable_batch_state() -> None:
    """A batch row can cross a process exit between an abort and its resend (#127).

    The cap is read off `all_attempts` rather than a counter held by the
    caller, precisely because the batch backend folds one response per tick.
    """
    from cdt.extractor.workflow import count_content_filter_aborts

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    for _ in range(2):
        row_state.record_unbilled_abort("", CONTENT_FILTERED)

    assert count_content_filter_aborts(row_state, "ner") == 2

    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())

    assert count_content_filter_aborts(restored, "ner") == 2
    # And the outstanding request is unchanged, so the resend is byte-identical.
    assert restored.current_attempt.messages == NERStage().preprocess(row_state)
    assert restored.current_attempt.attempt_index == 0


def test_an_abort_does_not_make_an_honest_zero_tag_row_partial() -> None:
    """A provider abort is not the model failing, so the row is a clean zero (#176, #127).

    An aborted call carries its own status rather than `FAILED`, so nothing
    downstream treats the row as one the model answered badly: a debt-free item
    whose first call was aborted publishes its honest zero with no failure
    record attached.
    """
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )
    row_state, _ = _run_live(
        "Acme Corp filed this report on January 1, 2024.",
        [CONTENT_FILTERED, _stopped(tagged_no_debt)],
    )

    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_the_ner_guards_do_not_read_an_aborted_calls_partial_output() -> None:
    """An abort is not the model's work, so it is not history to regress against.

    `record_unbilled_abort` stores whatever the provider emitted before it cut,
    and that text is tagged: all 58 `content_filter` responses in the stored
    corpora carry between 9 and 107 entity tags and at least one
    `debt_instrument` tag, and none of them is empty. Counted as the model's
    own earlier work it makes the echo guard and the high-water mark fire on an
    item the model has never successfully tagged, and the row then dies FAILED
    on a complaint it cannot act on -- the abort adds no assistant turn, so
    "keep every tag you found" names work that is not in its context
    (#176, #127).
    """
    from cdt.extractor.stages import (
        count_debt_instrument_tags,
        count_ner_entity_tags,
        prior_attempt_tagged,
        prior_debt_instrument_high_water,
    )

    # The fixture really is tagged, so the assertions below are not vacuous --
    # this is exactly what a `text=""` fixture failed to exercise.
    assert count_ner_entity_tags(CONTENT_FILTER_PARTIAL) == 5
    assert count_debt_instrument_tags(CONTENT_FILTER_PARTIAL) == 2

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    row_state.record_unbilled_abort(CONTENT_FILTER_PARTIAL, CONTENT_FILTERED)

    assert prior_attempt_tagged(row_state, "ner") is False
    assert prior_debt_instrument_high_water(row_state, "ner") == 0

    # And the same row driven end to end: the honest echo publishes in two
    # calls instead of looping to FAILED on an unanswerable complaint.
    row_state, client = _run_live(
        NODEBT_TEXT, [CONTENT_FILTERED, _stopped(f"<body>{NODEBT_TEXT}</body>")]
    )

    assert len(client.requests) == 2
    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_an_empty_aborted_body_behaves_the_same_as_a_truncated_one() -> None:
    """Both abort shapes are unscored calls, so neither changes the verdict (#127)."""
    for aborted in (CONTENT_FILTERED, CONTENT_FILTERED_EMPTY):
        row_state, client = _run_live(
            NODEBT_TEXT, [aborted, _stopped(f"<body>{NODEBT_TEXT}</body>")]
        )

        assert len(client.requests) == 2
        assert row_state.state == "SUCCESS"
        assert row_state.salvage_notes == []


def test_abort_counts_are_kept_per_stage_not_per_row() -> None:
    """Each stage gets its own resend budget (#127).

    Filtering is nondeterministic per call, so one row can be aborted on an
    early stage, recover, and be aborted again later. Counted per row, the
    earlier stage's aborts would eat the later stage's budget and terminate a
    row that still had resends coming to it.
    """
    from cdt.extractor.workflow import count_content_filter_aborts

    row_state, client = _run_live(
        MULTI_TEXT,
        [CONTENT_FILTERED] * 3
        + [_stopped(MULTI_NER), _stopped(MULTI_IE)]
        + [CONTENT_FILTERED] * 40,
    )

    # Three NER aborts, a NER pass, an instrument_ie pass, and then the
    # relation stage still gets its own full seven before terminating.
    assert len(client.requests) == 3 + 2 + 7
    assert count_content_filter_aborts(row_state, "ner") == 3
    assert count_content_filter_aborts(row_state, "instrument_relation") == 7
    # #152 still applies: the mentions it already earned publish.
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 2


def test_after_a_resend_the_next_answer_is_still_a_first_attempt() -> None:
    """The echo guard must not fire on an answer the model was never corrected on.

    The model has been shown no error after a provider abort -- the pristine
    request went back out -- so an untagged echo is still the honest answer for
    an item with nothing to tag (#176, #127).
    """
    echo = f"<body>{NODEBT_TEXT}</body>"
    row_state, client = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(echo)])

    assert len(client.requests) == 2
    scored = [a for a in row_state.all_attempts if a.status != "ABORTED"]
    assert [a.validation_errors for a in scored] == [[]]
    assert row_state.state == "SUCCESS"
    assert row_state.debt_instrument_mentions == []


def test_instrument_ie_postprocess_keeps_a_slash_format_date() -> None:
    """A tabular schedule row publishes its dates (#133)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>Schedule A lists <debt_instrument id="tag-i-1">Consolidated '
        'Obligation Bonds</debt_instrument> settling <date id="tag-d-1">7/28/2026'
        '</date> and maturing <date id="tag-d-2">7/28/2028</date>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2026-07-28",
                    },
                    {
                        "kind": "maturity",
                        "evidence": ["tag-d-2"],
                        "normalized_date": "2028-07-28",
                    },
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["start_date"] == "2026-07-28"
    assert mention["maturity_date"] == "2028-07-28"


def test_instrument_ie_postprocess_recovers_a_principal_from_the_name() -> None:
    """An uncited principal inside the name still publishes (#129)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The Company prepaid its <debt_instrument id="tag-i-1">$183.36 million '
        "term loan</debt_instrument>.</body>"
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                # No `amount` span exists, so the model cites nothing.
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": [],
                        "normalized_amount": "183360000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] == "183360000"
    assert payload["currency"] == "USD"
    assert payload["derived_from"] == "name"
    # Nothing was cited, so the evidence list stays empty, as it does for a
    # name-derived maturity.
    assert payload["spans"] == []


def test_instrument_ie_validate_accepts_the_name_span_as_amount_evidence() -> None:
    """Citing the instrument's own name span for a name-embedded amount is valid (#129)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">$183.36 million term loan'
        "</debt_instrument> was prepaid.</body>"
    )
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-i-1"],
                        "normalized_amount": "183360000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(row_state, response) == []

    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["principal_amount"] == "183360000"


def test_instrument_ie_postprocess_keeps_an_amount_with_cents() -> None:
    """A principal with cents survives the model/parser agreement gate (#119)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">construction loan facility'
        "</debt_instrument> provides for "
        '<amount id="tag-a-1">$372,246,148.11</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        # The model reports the value it read, without the float
                        # artifact the parser used to produce.
                        "normalized_amount": "372246148.11",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] == "372246148.11"
    assert payload["normalized_amount"] == "372246148.11"
    assert payload["currency"] == "USD"


def test_instrument_ie_postprocess_accepts_a_differently_formatted_amount() -> None:
    """`500000.00` and `500000` are the same amount, so neither is lost (#119)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">promissory note'
        '</debt_instrument> is for <amount id="tag-a-1">$500,000.00</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "500000.00",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    # The parser's canonical string is what gets published.
    assert mention["principal_amount"] == "500000"
    # A genuinely different value is still rejected.
    assert json.loads(str(mention["amounts_json"]))[0]["normalized_amount"] == "500000"


def test_instrument_ie_postprocess_keeps_an_amount_clustered_with_its_label() -> None:
    """The figure survives being clustered with a longer label span (#120)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">Secured Convertible Promissory Note'
        '</debt_instrument> has a <amount id="tag-a-label">Principal Amount</amount> '
        'of <amount id="tag-a-figure">$2,000,000</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-figure", "tag-a-label"],
                        "normalized_amount": "2000000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] == "2000000"
    # Both spans stay in the payload as provenance.
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-a-figure", "tag-a-label"]


def test_instrument_ie_postprocess_keeps_a_canadian_dollar_currency() -> None:
    """A C$ principal publishes CAD rather than a null currency (#121)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">4.200% Senior Notes due 2033'
        '</debt_instrument> total <amount id="tag-a-1">C$300 million</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "300000000",
                        "currency": "CAD",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    payload = json.loads(str(row_state.debt_instrument_mentions[0]["amounts_json"]))[0]
    assert payload["normalized_amount"] == "300000000"
    assert payload["currency"] == "CAD"


def test_instrument_ie_postprocess_normalizes_wrapped_names() -> None:
    """A name wrapped across lines is stored collapsed, verbatim in name_json."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        "<body>The Company entered into a "
        '<debt_instrument id="tag-i-1">revolving credit\nfacility</debt_instrument> '
        "and issued "
        '<debt_instrument id="tag-i-2">4.85% Remarketable\tSenior\xa0Notes '
        "due 2032</debt_instrument>.</body>"
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-2"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    names = [mention["name"] for mention in row_state.debt_instrument_mentions]
    assert names == [
        "revolving credit facility",
        "4.85% Remarketable Senior Notes due 2032",
    ]
    # Provenance keeps the verbatim span, since the char offsets index into it.
    verbatim = [
        json.loads(str(mention["name_json"]))["spans"][0]["text"]
        for mention in row_state.debt_instrument_mentions
    ]
    assert verbatim == [
        "revolving credit\nfacility",
        "4.85% Remarketable\tSenior\xa0Notes due 2032",
    ]


def test_instrument_ie_postprocess_drops_duplicate_identical_mentions() -> None:
    """Objects that differ in no extracted property must not duplicate a mention row."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The trust issued
<debt_instrument id="tag-i-1">Class A-1 Notes, Class A-2 Notes, and Class A-3 Notes</debt_instrument>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    mentions = row_state.debt_instrument_mentions
    assert len(mentions) == 1
    assert mentions[0]["raw_id"] == "i-1"


def test_instrument_ie_prompt_requires_one_object_per_class() -> None:
    """The IE prompt must keep telling the model to split multi-class offerings."""
    prompt = load_prompt("instrument_ie")

    assert "Each class, tranche, or series of an offering" in prompt
    assert "Class A-1" in prompt


def test_instrument_relation_stage_accepts_retired_by() -> None:
    """Relation validation and postprocessing should support retired_by."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_relation",
    )
    row_state.debt_instrument_mentions = [
        {
            "debt_instrument_mention_id": "m-1",
            "raw_id": "i-1",
        },
        {
            "debt_instrument_mention_id": "m-2",
            "raw_id": "i-2",
        },
    ]
    response = '[{"from": "i-2", "to": "i-1", "type": "retired_by"}]'

    failures = InstrumentRelationStage().validate(row_state, response)
    assert failures == []

    row_state.stage_responses["instrument_relation"] = response
    InstrumentRelationStage().postprocess(row_state)

    assert row_state.debt_instrument_mentions[1]["retired_by_json"] == '["m-1"]'


def test_instrument_relation_stage_accumulates_every_retirer() -> None:
    """Several edges out of one obligation all survive on its mention (#180).

    Venture Global's two new series jointly redeem one earlier obligation, so
    the relation stage emits two `retired_by` edges sharing a `from`. A scalar
    column kept only the last one, which is the bug this guards: with one edge
    per test, an append and an overwrite look identical. The repeated edge
    covers the deduplication guard at the same time.
    """
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_relation",
    )
    row_state.debt_instrument_mentions = [
        {"debt_instrument_mention_id": "m-old", "raw_id": "i-old"},
        {"debt_instrument_mention_id": "m-2034", "raw_id": "i-2034"},
        {"debt_instrument_mention_id": "m-2036", "raw_id": "i-2036"},
    ]
    response = json.dumps(
        [
            {"from": "i-old", "to": "i-2034", "type": "retired_by"},
            {"from": "i-old", "to": "i-2036", "type": "retired_by"},
            {"from": "i-old", "to": "i-2034", "type": "retired_by"},
        ]
    )

    failures = InstrumentRelationStage().validate(row_state, response)
    assert failures == []

    row_state.stage_responses["instrument_relation"] = response
    InstrumentRelationStage().postprocess(row_state)

    assert (
        row_state.debt_instrument_mentions[0]["retired_by_json"]
        == '["m-2034", "m-2036"]'
    )


def test_instrument_relation_prompt_matches_the_accepted_relation_types() -> None:
    """The prompt's type vocabulary must track the validator's (#180).

    `validate` rejects any type outside `INSTRUMENT_RELATION_TYPES`, so a
    prompt naming a type the code no longer accepts fails every relation the
    model returns and loses all lineage silently -- with nothing else in the
    suite noticing.
    """
    prompt = load_prompt("instrument_relation")

    for relation_type in INSTRUMENT_RELATION_TYPES:
        assert f"`{relation_type}`" in prompt
    assert "retired_of" not in prompt


def test_retry_includes_prior_response_as_assistant_turn() -> None:
    """Retry conversations must include the failed output the retry references."""
    row_state = ExtractionRowState(item_row={"item_id": "item-1"}, stage_name="ner")
    row_state.add_messages([{"role": "user", "content": "tag this filing"}])
    row_state.add_response("<bad-xml>")
    row_state.add_validation(["unclosed tag"])
    row_state.retry("Your previous NER output failed validation: unclosed tag")
    assert row_state.current_attempt.messages == [
        {"role": "user", "content": "tag this filing"},
        {"role": "assistant", "content": "<bad-xml>"},
        {
            "role": "user",
            "content": "Your previous NER output failed validation: unclosed tag",
        },
    ]


def test_infrastructure_error_classification() -> None:
    """Status- and name-shaped provider errors classify as infrastructure."""
    from cdt.extractor.state import is_infrastructure_error

    class PaymentRequiredResponseError(Exception):
        pass

    class WithStatus(Exception):
        status_code = 503

    assert is_infrastructure_error(PaymentRequiredResponseError())
    assert is_infrastructure_error(WithStatus())
    assert is_infrastructure_error(ConnectionResetError())
    assert not is_infrastructure_error(ValueError("bad xml"))


def test_realign_tag_details_maps_offsets_onto_the_original_text() -> None:
    """Evidence offsets index the item's own text, not the model's echo (#154)."""
    from cdt.extractor.tags import realign_tag_details

    original = "The  $5,000,000\tTerm Loan closed."
    roundtrip = "The $5,000,000 Term Loan closed."
    tag_details = {
        "tag-1": {
            "type": "amount",
            "text": "$5,000,000",
            "char_start": roundtrip.index("$5,000,000"),
            "char_end": roundtrip.index("$5,000,000") + len("$5,000,000"),
        },
        "tag-2": {
            "type": "debt_instrument",
            "text": "Term Loan",
            "char_start": roundtrip.index("Term Loan"),
            "char_end": roundtrip.index("Term Loan") + len("Term Loan"),
        },
    }
    realigned = realign_tag_details(tag_details, roundtrip, original)
    for tag_id in tag_details:
        start = realigned[tag_id]["char_start"]
        end = realigned[tag_id]["char_end"]
        assert original[start:end] == realigned[tag_id]["text"]
    assert realigned["tag-1"]["text"] == "$5,000,000"
    assert realigned["tag-2"]["text"] == "Term Loan"


def test_realign_tag_details_is_identity_when_texts_match() -> None:
    """The common exact-echo case pays no alignment cost."""
    from cdt.extractor.tags import realign_tag_details

    text = "A $10 note."
    details = {
        "tag-1": {"type": "amount", "text": "$10", "char_start": 2, "char_end": 5}
    }
    assert realign_tag_details(details, text, text) is details


def test_realign_tag_details_handles_model_deleted_whitespace() -> None:
    """collapse-equality permits dropped whitespace; spans still land right."""
    from cdt.extractor.tags import realign_tag_details

    original = "Senior Notes due 2028\nwere issued."
    roundtrip = "Senior Notes due 2028 were issued."
    details = {
        "tag-1": {
            "type": "debt_instrument",
            "text": "Senior Notes due 2028",
            "char_start": 0,
            "char_end": len("Senior Notes due 2028"),
        }
    }
    realigned = realign_tag_details(details, roundtrip, original)
    start = realigned["tag-1"]["char_start"]
    end = realigned["tag-1"]["char_end"]
    assert original[start:end] == "Senior Notes due 2028"


def test_terminal_ie_failure_salvages_the_valid_entries() -> None:
    """One invalid entry no longer drops the whole item (#152)."""
    from cdt.extractor.workflow import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": None,
                    }
                ],
            },
            {"name": ["tag-o-named"]},
        ]
    )
    # attempt_index reaches max_attempts on this response, forcing terminal
    # handling of the validation failure from the second entry's bad name tag.
    result = handle_response(row_state, response, max_attempts=1)

    assert result is None
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 1
    assert row_state.debt_instrument_mentions[0]["name"] == "Term Loan"
    assert row_state.salvage_notes
    assert "dropped 1" in row_state.salvage_notes[0]


def test_terminal_ie_failure_drops_an_entry_with_a_retired_property() -> None:
    """Salvage applies the retired-property check per entry, like validation."""
    from cdt.extractor.workflow import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The Company has a <debt_instrument id="tag-i-1">Term Loan</debt_instrument>'
        ' and a <debt_instrument id="tag-i-2">Revolving Credit Facility'
        "</debt_instrument>.</body>"
    )
    response = json.dumps(
        [
            {"name": ["tag-i-1"]},
            {"name": ["tag-i-2"], "status_event": {"status": "repaid"}},
        ]
    )
    result = handle_response(row_state, response, max_attempts=1)

    assert result is None
    assert row_state.state == "PARTIAL"
    assert [m["name"] for m in row_state.debt_instrument_mentions] == ["Term Loan"]
    assert "dropped 1" in row_state.salvage_notes[0]


def test_terminal_relation_failure_publishes_mentions_without_lineage() -> None:
    """A relation-stage failure keeps the already-validated mentions (#152)."""
    from cdt.extractor.workflow import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    ie_response = json.dumps(
        [
            {"name": ["tag-i-1"]},
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": None,
                    }
                ],
            },
        ]
    )
    next_messages = handle_response(row_state, ie_response, max_attempts=3)
    assert next_messages is not None
    assert row_state.current_attempt.stage_name == "instrument_relation"
    assert len(row_state.debt_instrument_mentions) == 2

    result = handle_response(row_state, "not json at all", max_attempts=1)

    assert result is None
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 2
    assert any("without lineage" in note for note in row_state.salvage_notes)


def test_terminal_ie_failure_with_nothing_valid_still_fails() -> None:
    """Salvage never invents output: no valid entry means FAILED as before."""
    from cdt.extractor.workflow import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    response = json.dumps([{"name": ["tag-unknown"]}])

    result = handle_response(row_state, response, max_attempts=1)

    assert result is None
    assert row_state.state == "FAILED"
    assert row_state.debt_instrument_mentions == []


def test_salvage_notes_round_trip_through_batch_state() -> None:
    """PARTIAL provenance survives the resumable batch state (#152)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.salvage_notes.append("instrument_ie dropped 1 entry")
    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())
    assert restored.salvage_notes == ["instrument_ie dropped 1 entry"]


BALANCE_ITEM_XML = """
<body>
The <debt_instrument id="tag-i-1">Revolving Credit Facility</debt_instrument> provides
<amount id="tag-a-commitment">$300 million</amount> of commitments. As of
<date id="tag-d-asof">June 9, 2026</date>, the Company had
<amount id="tag-a-balance">$270.5 million</amount> outstanding.
</body>
""".strip()


def balance_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state with a commitment and a balance."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = BALANCE_ITEM_XML
    return row_state


def test_amounts_are_kind_typed_and_the_balance_never_becomes_principal() -> None:
    """A commitment and a balance publish as two entries; principal is the commitment (#140)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-a-commitment"],
                        "normalized_amount": "300000000",
                        "currency": "USD",
                    },
                    {
                        "kind": "outstanding_balance",
                        "evidence": ["tag-a-balance"],
                        "normalized_amount": "270500000",
                        "currency": "USD",
                        "as_of_date": "2026-06-09",
                    },
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payloads = json.loads(str(mention["amounts_json"]))
    assert [(entry["kind"], entry["normalized_amount"]) for entry in payloads] == [
        ("commitment", "300000000"),
        ("outstanding_balance", "270500000"),
    ]
    assert payloads[1]["as_of_date"] == "2026-06-09"
    assert mention["principal_amount"] == "300000000"
    assert mention["principal_currency"] == "USD"
    assert mention["principal_amount_kind"] == "commitment"


def test_a_balance_only_mention_publishes_no_principal() -> None:
    """A balance observation alone never becomes the headline amount (#140)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "outstanding_balance",
                        "evidence": ["tag-a-balance"],
                        "normalized_amount": "270500000",
                        "currency": "USD",
                        "as_of_date": "2026-06-09",
                    }
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] is None
    assert mention["principal_amount_kind"] is None
    payloads = json.loads(str(mention["amounts_json"]))
    assert payloads[0]["kind"] == "outstanding_balance"


def test_instrument_ie_validate_rejects_unknown_amount_kinds() -> None:
    """An amounts entry must carry a known kind and a valid as_of_date (#140)."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "headline",
                        "evidence": ["tag-a-commitment"],
                        "normalized_amount": "300000000",
                        "as_of_date": "June 9, 2026",
                    }
                ],
            }
        ]
    )
    failures = InstrumentIEStage().validate(balance_row_state(), response)
    assert any("'amounts[0].kind' must be one of" in failure for failure in failures)
    assert any(
        "'amounts[0].as_of_date' must be YYYY-MM-DD" in failure for failure in failures
    )


DRAW_PERIOD_XML = """
<body>
The <debt_instrument id="tag-i-1">Delayed Draw Term Loan</debt_instrument> draw period
ends on <date id="tag-d-draw">June 30, 2027</date> and the loans mature on
<date id="tag-d-maturity">June 30, 2031</date>.
</body>
""".strip()


def test_commitment_termination_date_is_its_own_field() -> None:
    """Draw-period ends publish separately from the maturity (#158)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = DRAW_PERIOD_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "maturity",
                        "evidence": ["tag-d-maturity"],
                        "normalized_date": "2031-06-30",
                    },
                    {
                        "kind": "commitment_termination",
                        "evidence": ["tag-d-draw"],
                        "normalized_date": "2027-06-30",
                    },
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] == "2031-06-30"
    assert mention["commitment_termination_date"] == "2027-06-30"
    payload = json.loads(str(mention["commitment_termination_date_json"]))
    assert payload["derived_from"] == "stated"


TERMINATION_ITEM_XML = """
<body>
On <date id="tag-d-term">June 2, 2026</date>, the Company terminated its
<debt_instrument id="tag-i-1">$3.5 billion five-year revolving credit facility</debt_instrument>,
dated as of <date id="tag-d-dated">October 11, 2023</date>.
</body>
""".strip()


def test_no_event_publishes_null_status() -> None:
    """A mention that states no event carries no status."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TERMINATION_ITEM_XML
    row_state.stage_responses["instrument_ie"] = json.dumps([{"name": ["tag-i-1"]}])
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["status"] is None
    assert mention["status_date"] is None


def test_instrument_type_persists_and_rejects_unknown_values() -> None:
    """instrument_type is one of four categories or absent (#156)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"], "instrument_type": "revolving_credit"}]
    )
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["instrument_type"] == (
        "revolving_credit"
    )

    failures = InstrumentIEStage().validate(
        balance_row_state(),
        json.dumps([{"name": ["tag-i-1"], "instrument_type": "surety_bond"}]),
    )
    assert any("'instrument_type' must be one of" in failure for failure in failures)


RATE_TAGGED_XML = """
<body>
The <debt_instrument id="tag-i-1">3.875% senior notes due 2028</debt_instrument> were
issued, and revolver borrowings bear interest at
<interest_rate id="tag-r-1">SOFR plus 0.875% per annum</interest_rate>.
</body>
""".strip()


def test_interest_rate_is_parser_verified_from_name_or_evidence() -> None:
    """rate_pct publishes only when a cited or name-embedded rate matches (#157)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {
                    "kind": "fixed",
                    "rate_pct": "3.875",
                    "evidence": ["tag-i-1"],
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["interest_rate_kind"] == "fixed"
    assert mention["interest_rate_pct"] == "3.875"
    payload = json.loads(str(mention["interest_rate_json"]))
    assert payload["derived_from"] == "name"


def test_interest_rate_mismatch_publishes_null_rate() -> None:
    """A model rate contradicted by every cited span keeps kind but no number."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {
                    "kind": "floating",
                    "rate_pct": "5.5",
                    "evidence": ["tag-r-1"],
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["interest_rate_kind"] == "floating"
    assert mention["interest_rate_pct"] is None


def test_interest_rate_validation_rejects_bad_kind_and_evidence() -> None:
    """Kind must be fixed or floating; evidence must be rate or own-name tags."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {"kind": "variable", "evidence": ["tag-r-1"]},
            }
        ]
    )
    failures = InstrumentIEStage().validate(row_state, response)
    assert any("'interest_rate.kind' must be one of" in failure for failure in failures)


TENOR_FACILITY_XML = """
<body>
On <date id="tag-d-close">June 24, 2026</date>, the Company entered into a
<duration id="tag-t-1">five-year</duration>
<debt_instrument id="tag-i-1">senior secured revolving credit facility</debt_instrument>
of <amount id="tag-a-1">$1.0 billion</amount>.
</body>
""".strip()


def test_computed_maturity_from_start_plus_tenor() -> None:
    """A cited closing date plus a cited tenor verifies the maturity (#166)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TENOR_FACILITY_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-close"],
                        "normalized_date": "2026-06-24",
                    },
                    {
                        "kind": "maturity",
                        "evidence": ["tag-t-1", "tag-d-close"],
                        "normalized_date": "2031-06-24",
                    },
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] == "2031-06-24"
    payload = json.loads(str(mention["maturity_date_json"]))
    assert payload["derived_from"] == "computed"
    assert {s["tag_id"] for s in payload["spans"]} == {"tag-t-1", "tag-d-close"}


def test_computed_maturity_rejects_arithmetic_that_misses() -> None:
    """A model date that is not start plus tenor stays null."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TENOR_FACILITY_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "maturity",
                        "evidence": ["tag-t-1", "tag-d-close"],
                        "normalized_date": "2030-06-24",
                    }
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["maturity_date"] is None


def test_instrument_ie_accepts_a_bare_object_as_one_entry() -> None:
    """A bare object is the one-instrument case, not a validation failure."""
    from cdt.extractor.stages import instrument_entries_from_response

    assert instrument_entries_from_response('{"name": ["tag-1"]}') == [
        {"name": ["tag-1"]}
    ]
    assert instrument_entries_from_response('[{"name": ["tag-1"]}]') == [
        {"name": ["tag-1"]}
    ]
    assert instrument_entries_from_response("[]") == []


def test_dates_property_validation_rejects_bad_kind_and_two_current_maturities() -> (
    None
):
    """Two current maturities are two instruments; a prior one is history."""
    from cdt.extractor.validate import validate_dates_property

    tags = _dates_tag_details()
    bad_kind = validate_dates_property(
        index=0,
        obj={
            "dates": [{"kind": "issue", "evidence": ["tag-2"], "normalized_date": None}]
        },
        tag_details=tags,
    )
    assert any("kind" in failure for failure in bad_kind)
    two = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-4"],
                    "normalized_date": "2026-06-28",
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-5"],
                    "normalized_date": "2031-06-23",
                },
            ]
        },
        tag_details=tags,
    )
    assert any("2 current entries of kind 'maturity'" in failure for failure in two)
    ok = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-4"],
                    "normalized_date": "2026-06-28",
                    "prior": True,
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-5"],
                    "normalized_date": "2031-06-23",
                },
            ]
        },
        tag_details=tags,
    )
    assert ok == []
    name_as_closing = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {"kind": "closing", "evidence": ["tag-1"], "normalized_date": None}
            ]
        },
        tag_details=tags,
    )
    assert any("expected date" in failure for failure in name_as_closing)


def _semantic_tags() -> dict[str, dict[str, object]]:
    return {
        "tag-1": {
            "text": "5% Senior Notes due 2031",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 24,
        },
        "tag-2": {
            "text": "March 5, 2026",
            "type": "date",
            "char_start": 30,
            "char_end": 43,
        },
        "tag-3": {
            "text": "$100 million",
            "type": "amount",
            "char_start": 50,
            "char_end": 62,
        },
    }


def test_semantic_validators_reject_misplaced_flags_and_kinds() -> None:
    """Stage 2: `expected` only on events, `prior` only on terms, kinds fit the type, repayments pair."""
    from cdt.extractor.validate import (
        validate_cross_field_semantics,
        validate_dates_property,
    )

    tags = _semantic_tags()
    expected_maturity = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "expected": True,
                }
            ]
        },
        tag_details=tags,
    )
    assert any("cannot be `expected`" in f for f in expected_maturity)
    prior_event = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "retirement",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "prior": True,
                }
            ]
        },
        tag_details=tags,
    )
    assert any("cannot be `prior`" in f for f in prior_event)
    ok = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "expected": True,
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-1"],
                    "normalized_date": "2031-12-31",
                    "prior": True,
                },
            ]
        },
        tag_details=tags,
    )
    assert ok == []
    prior_balance = validate_cross_field_semantics(
        index=0,
        obj={
            "amounts": [
                {
                    "kind": "outstanding_balance",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                    "prior": True,
                }
            ]
        },
    )
    assert any("cannot be `prior`" in f for f in prior_balance)
    unpaired_date = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [{"kind": "repayment", "evidence": [], "normalized_date": None}],
            "amounts": [],
        },
    )
    assert any("needs the repaid figure" in f for f in unpaired_date)
    unpaired_amount = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                }
            ],
            "amounts": [
                {
                    "kind": "repayment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert any("is an event" in f for f in unpaired_amount)
    paired = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [{"kind": "repayment", "evidence": [], "normalized_date": None}],
            "amounts": [
                {
                    "kind": "repayment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert paired == []
    mismatch = validate_cross_field_semantics(
        index=0,
        obj={
            "instrument_type": "note_bond",
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert any("does not fit instrument_type" in f for f in mismatch)
    assert (
        validate_cross_field_semantics(
            index=0,
            obj={
                "instrument_type": "revolving_credit",
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-3"],
                        "normalized_amount": "100000000",
                    }
                ],
            },
        )
        == []
    )


def test_validation_rejects_legacy_properties() -> None:
    """A response reverting to status_event/lenders/start_date fails validation."""
    from cdt.extractor.validate import validate_no_legacy_properties

    legacy = {
        "name": ["tag-1"],
        "status_event": {"status": "repaid"},
        "lenders": [],
        "start_date": {"evidence": [], "normalized_date": None},
    }
    failures = validate_no_legacy_properties(0, legacy)
    assert len(failures) == 3 and all(
        "is not a property of this schema" in f for f in failures
    )
    assert (
        validate_no_legacy_properties(
            0, {"name": ["tag-1"], "dates": [], "parties": []}
        )
        == []
    )


def test_relation_manifest_marks_expected_retirement() -> None:
    """The relation stage sees a planned retirement it cannot read off the body tags."""
    from cdt.extractor.stages import relation_instrument_manifest
    from cdt.extractor.state import ExtractionRowState

    state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_relation"
    )
    state.debt_instrument_mentions = [
        {
            "raw_id": "i-1",
            "name": "New Notes",
            "principal_amount": "600000000",
            "start_date": None,
            "maturity_date": "2036-12-31",
            "status": "announced",
            "dates_json": json.dumps([{"kind": "closing", "expected": True}]),
        },
        {
            "raw_id": "i-2",
            "name": "5.25% Senior Notes due 2027",
            "principal_amount": None,
            "start_date": None,
            "maturity_date": "2027-12-31",
            "status": None,
            "dates_json": json.dumps(
                [{"kind": "retirement", "expected": True, "normalized_date": None}]
            ),
        },
    ]
    manifest = relation_instrument_manifest(state)
    assert 'id="i-2"' in manifest and 'expected_retirement="true"' in manifest
    assert manifest.count('expected_retirement="true"') == 1
    assert 'status="announced"' in manifest


def test_repayment_amount_is_dated_by_a_terminal_event() -> None:
    """A repayment figure beside a retirement needs no separate repayment date."""
    from cdt.extractor.validate import validate_cross_field_semantics

    obj = {
        "dates": [
            {
                "kind": "retirement",
                "evidence": ["tag-2"],
                "normalized_date": "2026-03-05",
            }
        ],
        "amounts": [
            {
                "kind": "repayment",
                "evidence": ["tag-3"],
                "normalized_amount": "100000000",
            }
        ],
    }
    assert validate_cross_field_semantics(index=0, obj=obj) == []


def test_published_evidence_spans_index_the_item_text_exactly() -> None:
    """#154's contract, asserted on a published payload rather than the helper.

    Every other span assertion in this suite projects to `tag_id` or `text`, so
    stripping the offsets out of `cluster_payload` entirely went unnoticed.
    """
    item_text = (
        "On March 5, 2026 the Company entered into a $500,000,000 term loan "
        "under the Credit Agreement."
    )
    tagged = (
        '<document>On <date id="tag-d-1">March 5, 2026</date> the Company '
        'entered into a <amount id="tag-a-1">$500,000,000</amount> '
        '<debt_instrument id="tag-i-1">term loan</debt_instrument> under the '
        "Credit Agreement.</document>"
    )
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": item_text, "date": "2026-03-10"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = tagged
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "instrument_type": "term_loan",
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "500000000",
                        "currency": "USD",
                    }
                ],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2026-03-05",
                    }
                ],
                "parties": [],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)
    mention = row_state.debt_instrument_mentions[0]

    checked = 0
    for column in ("name_json", "amounts_json", "dates_json", "start_date_json"):
        payload = json.loads(str(mention[column]))
        for entry in payload if isinstance(payload, list) else [payload]:
            for span in entry.get("spans", []):
                assert (
                    item_text[span["char_start"] : span["char_end"]] == span["text"]
                ), f"{column} span does not index the item text"
                checked += 1
    assert checked >= 3


def test_two_current_closing_dates_are_rejected() -> None:
    """`closing` is singular too, though it is an event kind.

    The cap counted only kinds outside `EVENT_DATE_KINDS`, so `closing` escaped
    it while `maturity`, `agreement` and `commitment_termination` did not —
    contradicting the prompt's own "(validated)" claim. One of the two dates
    then published and the other was silently dropped.
    """
    tags = _dates_tag_details()
    date_ids = [tag_id for tag_id, detail in tags.items() if detail["type"] == "date"][
        :2
    ]
    assert len(date_ids) == 2

    def two_of(kind: str) -> list[str]:
        return validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": kind, "evidence": [date_ids[0]]},
                    {"kind": kind, "evidence": [date_ids[1]]},
                ]
            },
            tag_details=tags,
        )

    for kind in ("closing", "maturity", "agreement", "commitment_termination"):
        failures = two_of(kind)
        assert any(
            f"'dates' has 2 current entries of kind '{kind}'" in failure
            for failure in failures
        ), f"{kind} should be capped at one current entry"
    # Events genuinely may repeat: two amendments are two amendments.
    assert not any(
        "current entries of kind" in failure for failure in two_of("amendment")
    )
    # A planned closing beside a real one is one *current* closing, not two:
    # `select_date_payload` skips `expected` entries, so counting them rejected
    # Costamare's 6-K, which states both for one facility.
    assert not any(
        "current entries of kind" in failure
        for failure in validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": "closing", "evidence": [date_ids[0]]},
                    {"kind": "closing", "evidence": [date_ids[1]], "expected": True},
                ]
            },
            tag_details=tags,
        )
    )
    # A `prior` term is likewise not current.
    assert not any(
        "current entries of kind" in failure
        for failure in validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": "maturity", "evidence": [date_ids[0]]},
                    {"kind": "maturity", "evidence": [date_ids[1]], "prior": True},
                ]
            },
            tag_details=tags,
        )
    )


def test_interest_rate_without_an_evidence_key_is_accepted() -> None:
    """The prompt's own `3.875% senior notes due 2028` example omits `evidence`.

    Rejecting it cost a full retry cycle on a shape postprocess already handles
    by verifying the rate against the instrument's name.
    """
    assert (
        validate_interest_rate(
            index=0,
            obj={"interest_rate": {"kind": "fixed", "rate_pct": "3.875"}},
            tag_details={},
        )
        == []
    )
    # A present-but-wrong evidence value is still a failure.
    assert validate_interest_rate(
        index=0,
        obj={"interest_rate": {"kind": "fixed", "rate_pct": "3.875", "evidence": "x"}},
        tag_details={},
    )


def test_realign_tag_details_leaves_unalignable_text_untouched() -> None:
    """The documented degradation path: degrade to old offsets, never guess.

    Proceeding with a partial alignment map would emit realigned-but-wrong
    offsets for some tags, which is worse than the stale ones.
    """
    details = {
        "tag-1": {
            "type": "date",
            "text": "March 5, 2026",
            "char_start": 3,
            "char_end": 16,
        }
    }
    # Differs by more than whitespace, so no alignment exists.
    assert (
        realign_tag_details(details, "on March 5, 2026", "on April 5, 2026") is details
    )
    # A span whose recorded offsets include surrounding whitespace still snaps
    # onto the non-whitespace run.
    padded = {
        "tag-1": {
            "type": "date",
            "text": " March 5, 2026 ",
            "char_start": 2,
            "char_end": 17,
        }
    }
    realigned = realign_tag_details(padded, "on  March 5, 2026 .", "on March 5, 2026.")
    span = realigned["tag-1"]
    assert "on March 5, 2026."[span["char_start"] : span["char_end"]] == span["text"]
    assert span["text"] == "March 5, 2026"


def test_validate_parties_property_rejects_every_bad_shape() -> None:
    """Each malformed `parties` cluster gets its own failure message."""
    tags = {
        "tag-p-1": {
            "type": "organization",
            "text": "Acme Bank",
            "char_start": 0,
            "char_end": 9,
        },
        "tag-d-1": {
            "type": "date",
            "text": "March 5, 2026",
            "char_start": 10,
            "char_end": 23,
        },
    }

    def failures(parties: object) -> list[str]:
        return validate_parties_property(
            index=0, obj={"parties": parties}, tag_details=tags
        )

    assert any("must be a list of cluster objects" in f for f in failures({}))
    assert any("must be an object with" in f for f in failures([["tag-p-1"]]))
    assert any(
        "'role' must be one of" in f
        for f in failures([{"tag_ids": ["tag-p-1"], "role": "financier"}])
    )
    assert any(
        "'kind' must be named or collective" in f
        for f in failures(
            [{"tag_ids": ["tag-p-1"], "role": "lender", "kind": "anonymous"}]
        )
    )
    assert any(
        "'tag_ids' must be a list" in f
        for f in failures([{"tag_ids": "tag-p-1", "role": "lender"}])
    )
    assert any(
        "string tag IDs only" in f
        for f in failures([{"tag_ids": [7], "role": "lender"}])
    )
    assert any(
        "unknown tag ID" in f
        for f in failures([{"tag_ids": ["tag-missing"], "role": "lender"}])
    )
    assert any(
        "must be person or organization" in f
        for f in failures([{"tag_ids": ["tag-d-1"], "role": "lender"}])
    )
    # The shape the prompt describes passes.
    assert failures([{"tag_ids": ["tag-p-1"], "role": "lender", "kind": "named"}]) == []


def test_a_salvaged_row_registers_the_salvage_note_not_the_last_stage() -> None:
    """A PARTIAL row's registry entry must say what salvage dropped (#152).

    `failure_record` read `current_attempt`, so a row salvaged at
    `instrument_ie` that then completed `instrument_relation` published
    `stage: instrument_relation` and the invented error "Unexpected response at
    stage instrument_relation" — naming a stage that succeeded and describing a
    failure that never happened. The one PARTIAL row in the 364-unit 2026-09 run
    published exactly that, so the row here keeps two mentions in order to
    advance past the stage it was salvaged at.
    """
    from cdt.extractor.outputs import failed_stage_name, failure_record
    from cdt.extractor.workflow import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "accession_number": "0001", "cik": "0000320193"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    # Two valid entries plus one whose name tag is an organization: salvage keeps
    # the pair, which is enough mentions to run the relation stage. The two differ
    # on `instrument_type` so they hash to distinct mention ids rather than
    # collapsing into one.
    advanced = handle_response(
        row_state,
        json.dumps(
            [
                {"name": ["tag-i-1"], "instrument_type": "term_loan"},
                {"name": ["tag-i-1"], "instrument_type": "revolving_credit"},
                {"name": ["tag-o-named"]},
            ]
        ),
        max_attempts=1,
    )
    assert advanced is not None, "salvage should advance to instrument_relation"
    assert len(row_state.debt_instrument_mentions) == 2

    # The relation stage then succeeds, so the row's last attempt is clean.
    assert handle_response(row_state, json.dumps([]), max_attempts=1) is None
    assert row_state.state == "PARTIAL"
    assert row_state.current_attempt.stage_name == "instrument_relation"
    assert not row_state.current_attempt.validation_errors

    record = failure_record(
        row_state,
        partition_date="2026-09-04",
        shard="0025",
        run_id="run-1",
        backend="batch",
    )
    assert record["state"] == "PARTIAL"
    # The stage salvage fired in, not the last stage the row ran.
    assert record["stage"] == "instrument_ie"
    assert failed_stage_name(row_state) == "instrument_ie"
    assert "Unexpected response" not in str(record["error"])
    assert record["error"] == "; ".join(row_state.salvage_notes)
    assert "dropped 1" in str(record["error"])


_WRONG_TYPE_VALUES: tuple[object, ...] = (["term_loan"], {"kind": "x"}, 7, None, True)


def _with_each_value_replaced(value: object) -> list[object]:
    """Copies of ``value`` with one nested value at a time swapped for a wrong type."""
    variants: list[object] = []
    if isinstance(value, dict):
        for key, child in value.items():
            for replacement in (*_WRONG_TYPE_VALUES, *_with_each_value_replaced(child)):
                variants.append({**value, key: replacement})
    elif isinstance(value, list):
        for position, child in enumerate(value):
            for replacement in (*_WRONG_TYPE_VALUES, *_with_each_value_replaced(child)):
                variants.append(
                    [*value[:position], replacement, *value[position + 1 :]]
                )
    return variants


def test_wrong_type_model_json_is_a_validation_failure_not_an_exception() -> None:
    """A list or object where the schema wants a string must fail validation.

    Testing an unhashable value against a set raises TypeError, which no caller
    catches: live extract would die before saving completion and the batch fold
    would re-crash on the same response every tick.
    """
    tag_details = {
        "tag-i-1": {"type": "debt_instrument", "text": "Term Loan"},
        "tag-a-1": {"type": "amount", "text": "$5.5 million"},
        "tag-d-1": {"type": "date", "text": "March 17, 2025"},
        "tag-r-1": {"type": "interest_rate", "text": "5.25%"},
        "tag-o-1": {"type": "organization", "text": "EGT 11 LLC"},
    }
    entry = {
        "name": ["tag-i-1"],
        "instrument_type": "term_loan",
        "amounts": [
            {"kind": "principal", "evidence": ["tag-a-1"], "prior": False},
            {"kind": "repayment", "evidence": ["tag-a-1"]},
        ],
        "dates": [
            {"kind": "closing", "evidence": ["tag-d-1"], "expected": False},
            {"kind": "repayment", "evidence": []},
        ],
        "interest_rate": {"kind": "fixed", "rate_pct": "5.25", "evidence": ["tag-r-1"]},
        "parties": [{"tag_ids": ["tag-o-1"], "role": "lender", "kind": "named"}],
    }
    assert validate_instrument_entry(0, entry, tag_details) == []

    variants = _with_each_value_replaced(entry)
    assert len(variants) > 100
    for variant in variants:
        assert isinstance(validate_instrument_entry(0, variant, tag_details), list)
    assert validate_instrument_entry(
        0, {**entry, "instrument_type": ["term_loan"]}, tag_details
    )


def test_relation_stage_rejects_a_non_string_relation_type() -> None:
    """A list ``type`` is a validation failure, not a TypeError."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_relation"
    )
    row_state.debt_instrument_mentions = [{"raw_id": "a"}, {"raw_id": "b"}]
    response = json.dumps([{"from": "a", "to": "b", "type": ["amendment_of"]}])

    failures = InstrumentRelationStage().validate(row_state, response)

    assert any("Invalid relation type" in failure for failure in failures)
