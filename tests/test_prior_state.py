"""Tests for minting the prior state of an amended instrument."""

from __future__ import annotations

import json

from support import _fact, amended_row, build_mention_row

from cdt.extractor.stages import InstrumentIEStage
from cdt.extractor.state import ExtractionRowState


def test_published_mention_rows_is_the_single_publish_seam() -> None:
    """Every publish path reads through one helper, which hands out a copy.

    The live loop, batch finalize, `extract_tables` and the audit record all
    call `published_mention_rows`; a derivation attached there (#203) reaches
    every backend at once. The helper returns a fresh list so a caller that
    extends its result cannot mutate the state persisted to `state.jsonl`.
    """
    from cdt.extractor.prior_state import published_mention_rows
    from cdt.extractor.state import ExtractionRowState

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_ie"
    )
    row_state.debt_instrument_mentions = [{"debt_instrument_mention_id": "m-1"}]

    published = published_mention_rows(row_state)
    assert published == [{"debt_instrument_mention_id": "m-1"}]
    published.append({"debt_instrument_mention_id": "m-2"})
    assert row_state.debt_instrument_mentions == [{"debt_instrument_mention_id": "m-1"}]
    assert row_state.to_audit_dict()["debt_instrument_mentions"] == published[:1]


def test_mint_builds_the_prior_state_from_the_prior_marked_terms() -> None:
    """The amended object's `prior` terms become the predecessor's current ones.

    The predecessor keeps the agreement's dated-as-of (the same agreement), its
    unchanged maturity and rate marked `inherited`, the borrower and nothing
    the joinder may have changed; the successor is untouched apart from the
    pointer, and its id — which never hashed `amendment_of` — is unchanged.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    successor = amended_row()
    counters: dict[str, int] = {}
    published = mint_prior_state_rows([dict(successor)], counters)

    assert counters == {"minted": 1}
    assert len(published) == 2
    after, minted = published
    assert (
        after["debt_instrument_mention_id"] == successor["debt_instrument_mention_id"]
    )
    assert after["amendment_of"] == minted["debt_instrument_mention_id"]
    assert json.loads(str(after["amounts_json"]))[0]["prior"] is True  # no aliasing

    assert minted["synthesized_by"] == "prior_state"
    assert (
        minted["synthesized_from_mention_id"] == successor["debt_instrument_mention_id"]
    )
    assert minted["item_id"] == "item-1"
    assert minted["raw_id"] == "i-1-prior"
    assert minted["name"] == "Credit Agreement"
    assert minted["instrument_type"] == "revolving_credit"
    assert minted["principal_amount"] == "300000000"
    assert minted["principal_amount_kind"] == "commitment"
    assert minted["start_date"] == "2020-02-03"
    assert minted["maturity_date"] == "2029-02-03"
    assert minted["status"] == "entered_into"
    assert minted["amendment_of"] is None
    assert minted["lender_disclosure"] == "none_named"
    assert [p["role"] for p in json.loads(str(minted["parties_json"]))] == ["borrower"]

    amounts = json.loads(str(minted["amounts_json"]))
    assert [
        (a["kind"], a["normalized_amount"], a["prior"], a["derived_from"])
        for a in amounts
    ] == [
        ("commitment", "300000000", False, "stated")
    ]  # the new $250M is gone; the balance observation stays with the successor
    dates = {d["kind"]: d for d in json.loads(str(minted["dates_json"]))}
    assert set(dates) == {"maturity", "agreement"}  # no event kinds
    assert dates["maturity"]["derived_from"] == "inherited"
    assert dates["agreement"]["derived_from"] == "stated"
    assert json.loads(str(minted["interest_rate_json"]))["derived_from"] == "inherited"


def test_mint_refusals_are_counted_and_leave_the_rows_alone() -> None:
    """Each way the trigger can fail is named, and nothing is minted."""
    from cdt.extractor.prior_state import mint_prior_state_rows

    def run(
        rows: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[str, int]]:
        counters: dict[str, int] = {}
        return mint_prior_state_rows([dict(r) for r in rows], counters), counters

    # no prior term at all: not an amendment with a before-figure
    plain, counts = run(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="250000000", prior=False)
                ]
            )
        ]
    )
    assert len(plain) == 1 and counts == {}

    # amendment date only: no origin to place the predecessor at
    rows, counts = run(
        [
            amended_row(
                dates=[
                    _fact(kind="amendment", normalized_date="2024-06-01", prior=False)
                ]
            )
        ]
    )
    assert len(rows) == 1 and counts == {"skipped_no_origin": 1}

    # the relation stage already paired it with a model-emitted predecessor
    rows, counts = run([amended_row(amendment_of="m-model-predecessor")])
    assert len(rows) == 1 and counts == {"skipped_model_paired": 1}

    # a sibling in the item already *is* the predecessor
    sibling = build_mention_row(
        mention_id="m-sibling",
        item_id="item-1",
        accession_number="0002",
        cik="0000320193",
        date="2024-06-01",
        name="Credit Agreement",
        start_date="2020-02-03",
        amount="300000000",
    )
    rows, counts = run([amended_row(), sibling])
    assert len(rows) == 2 and counts == {"skipped_sibling_is_predecessor": 1}

    # two before-values of one kind
    rows, counts = run(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="300000000", prior=True),
                    _fact(kind="commitment", normalized_amount="200000000", prior=True),
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                ]
            )
        ]
    )
    assert len(rows) == 1 and counts == {"skipped_ambiguous_prior": 1}


def test_mint_places_the_predecessor_at_the_right_origin() -> None:
    """A prior agreement is the predecessor's own date and wins outright.

    An origin equal to the amendment date is the restatement's own dated-as-of
    (MPLX: agreement 2019-07-31 == amendment 2019-07-31), so the predecessor
    is minted with no start date rather than a date the filing did not state
    for that state of the facility.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    restated = amended_row(
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="closing",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=True,
                expected=False,
            ),
        ]
    )
    counters: dict[str, int] = {}
    _, minted = mint_prior_state_rows([restated], counters)
    assert counters == {"minted": 1}
    assert minted["start_date"] == "2020-02-03"
    kinds = [d["kind"] for d in json.loads(str(minted["dates_json"]))]
    assert kinds == ["agreement"]  # the restatement's closing/agreement were not copied

    mplx = amended_row(
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2019-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2019-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2024-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2020-12-04",
                prior=True,
                expected=False,
            ),
        ]
    )
    counters = {}
    _, minted = mint_prior_state_rows([mplx], counters)
    assert counters == {"minted": 1, "minted_no_origin": 1}
    assert minted["start_date"] is None
    assert minted["maturity_date"] == "2020-12-04"
    assert minted["principal_amount"] == "300000000"


def test_minted_no_origin_is_not_counted_when_the_mint_is_then_refused() -> None:
    """The tag names a subset of `minted`, so it cannot outlive a refusal.

    Bumped where the origin was resolved, `minted_no_origin` fired ahead of the
    two guards that still stand between that point and the append. A successor
    whose only origin candidate is its own amendment date, sitting beside the
    sibling that *is* its predecessor, then reported `minted_no_origin: 1` with
    no synthesized row anywhere — a mint that never happened, inside the
    counters this docstring calls the pre-registered yield (#211).
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    successor = amended_row(
        "m-successor",
        dates=[
            # The only origin candidate is the restatement's own dated-as-of,
            # which is what empties `origin_payloads`.
            _fact(kind="agreement", normalized_date="2024-06-01", prior=False),
            _fact(kind="amendment", normalized_date="2024-06-01", prior=False),
        ],
    )
    # The model returned the predecessor as its own object: it carries the
    # successor's prior commitment and states no start date of its own.
    predecessor = amended_row(
        "m-predecessor",
        amounts=[],
        dates=[],
        principal_amount="300000000",
        start_date=None,
    )

    counters: dict[str, int] = {}
    published = mint_prior_state_rows([successor, predecessor], counters)

    assert counters == {"skipped_sibling_is_predecessor": 1}
    assert not [row for row in published if row.get("synthesized_by") == "prior_state"]


def test_mint_from_a_prior_maturity_alone_inherits_the_amount() -> None:
    """`extended the maturity from 2029 to 2031`: the commitment is unchanged."""
    from cdt.extractor.prior_state import mint_prior_state_rows

    extended = amended_row(
        amounts=[
            _fact(
                kind="commitment",
                normalized_amount="250000000",
                currency="USD",
                prior=False,
            )
        ],
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2031-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2029-02-03",
                prior=True,
                expected=False,
            ),
        ],
    )
    _, minted = mint_prior_state_rows([extended])
    assert minted["maturity_date"] == "2029-02-03"
    assert minted["principal_amount"] == "250000000"
    amounts = json.loads(str(minted["amounts_json"]))
    assert amounts[0]["derived_from"] == "inherited"


def test_two_successors_sharing_one_prior_state_point_at_one_mint() -> None:
    """Byte-identical mints collapse to one row; both pointers still land."""
    from cdt.extractor.prior_state import mint_prior_state_rows

    first = amended_row("m-a", raw_id="i-1")
    second = amended_row("m-b", raw_id="i-1", interest_rate_pct="5.25")
    counters: dict[str, int] = {}
    published = mint_prior_state_rows([first, second], counters)
    assert counters == {"minted": 1, "minted_shared": 1}
    assert len(published) == 3
    assert (
        published[0]["amendment_of"]
        == published[1]["amendment_of"]
        == published[2]["debt_instrument_mention_id"]
    )


def test_mint_is_id_stable_and_idempotent() -> None:
    """Model-emitted ids never change, and minting its own output adds nothing."""
    from cdt.extractor.prior_state import mint_prior_state_rows

    rows = [
        amended_row(),
        amended_row(
            "m-plain",
            amounts=[_fact(kind="commitment", normalized_amount="1", prior=False)],
        ),
    ]
    before = {r["debt_instrument_mention_id"] for r in rows}
    published = mint_prior_state_rows([dict(r) for r in rows])
    real_after = {
        r["debt_instrument_mention_id"]
        for r in published
        if r.get("synthesized_by") is None
    }
    assert real_after == before

    again = mint_prior_state_rows([dict(r) for r in published])
    assert again == published


def test_an_only_prior_amount_publishes_no_current_principal() -> None:
    """The figure in the name is the prior figure; it must not come back as current.

    `Amendment No. 2 to the $100 million Credit Agreement ... from $100,000,000`
    states only the before-figure. With every commitment `prior`, the head's
    current capacity is unstated, and reading `$100 million` back off the name
    published the pre-amendment figure as current — #165's stale head by a
    second route (#206). The honest answer is null; the minted prior state is
    where that figure belongs.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "date": "2024-06-01"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        "<body>Amendment No. 2 to the "
        '<debt_instrument id="tag-i-1">$100 million Credit Agreement</debt_instrument>, '
        'dated as of <date id="tag-d-1">February 3, 2020</date>, reduced the '
        'commitments from <amount id="tag-a-1">$100,000,000</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "100000000",
                        "currency": "USD",
                        "prior": True,
                    }
                ],
                "dates": [
                    {
                        "kind": "agreement",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2020-02-03",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] is None
    amounts = json.loads(str(mention["amounts_json"]))
    assert [(a["normalized_amount"], a["prior"]) for a in amounts] == [
        ("100000000", True)
    ]

    published = mint_prior_state_rows([dict(mention)])
    assert len(published) == 2
    assert published[1]["principal_amount"] == "100000000"
    assert published[1]["start_date"] == "2020-02-03"
    assert published[0]["amendment_of"] == published[1]["debt_instrument_mention_id"]


def test_an_unparsed_prior_term_suppresses_inheritance_rather_than_licensing_it() -> (
    None
):
    """A stated before-figure the parser could not resolve still says this changed.

    The kind sets that answer "did this term change?" were built from the
    *parsed* prior entries, so a `prior: true` term whose value did not resolve
    was invisible to them, and the current value of that kind was copied onto
    the predecessor marked `derived_from: "inherited"` — asserting the
    post-amendment figure as the prior state's own term, the one thing the rule
    must never do. The claims decide the kinds; only the values come from what
    parsed (#211).

    Incidence of this shape on the reference corpus is 0, so no published row
    was ever wrong because of it.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    counters: dict[str, int] = {}
    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    amounts=[
                        _fact(
                            kind="commitment",
                            normalized_amount="250000000",
                            prior=False,
                        ),
                        # stated, but the parser could not resolve it
                        _fact(kind="commitment", normalized_amount=None, prior=True),
                    ],
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2021-03-01", prior=True
                        ),
                        _fact(
                            kind="maturity", normalized_date="2029-05-01", prior=False
                        ),
                        _fact(kind="maturity", normalized_date=None, prior=True),
                    ],
                )
            ],
            counters,
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    assert counters == {"minted": 1}
    assert len(minted) == 1
    # neither post-amendment value is laundered onto the predecessor
    assert minted[0]["principal_amount"] is None
    assert minted[0]["maturity_date"] is None
    assert json.loads(minted[0]["amounts_json"]) == []
    inherited = [
        entry
        for entry in json.loads(minted[0]["dates_json"])
        if entry.get("derived_from") == "inherited"
    ]
    assert inherited == []


def test_a_prior_claim_that_never_parsed_is_counted_not_silently_dropped() -> None:
    """The counters are the pre-registered yield, so a refusal cannot be silent.

    An object whose *only* prior term failed to parse fell through the combined
    "no prior amounts and no prior dates" guard with no counter at all, which
    is why the window's 22 objects carrying a prior term summed to 21 across
    the counters. On `data/genwindow-run-branch` the swallowed object is
    `dim::5542bb4c…`, a Loan and Security Agreement whose prior commitment has
    a null amount; the counters now sum to 22 (#211).
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                amounts=[
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                    _fact(kind="commitment", normalized_amount=None, prior=True),
                ],
                dates=[
                    _fact(kind="agreement", normalized_date="2020-02-03", prior=False),
                ],
            )
        ],
        counters,
    )

    assert len(rows) == 1
    assert counters == {"skipped_unparsed_prior": 1}


def test_two_before_figures_are_ambiguous_even_when_one_did_not_parse() -> None:
    """Ambiguity is judged on the claims: two stated before-values are two states."""
    from cdt.extractor.prior_state import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="300000000", prior=True),
                    _fact(kind="commitment", normalized_amount=None, prior=True),
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                ]
            )
        ],
        counters,
    )

    assert len(rows) == 1
    assert counters == {"skipped_ambiguous_prior": 1}


def test_mint_does_not_write_the_pointer_onto_the_rows_it_was_handed() -> None:
    """`amendment_of` belongs on the published row, never on the persisted state.

    A caller passing `row_state.debt_instrument_mentions` straight in would
    otherwise persist a minted pointer into `state.jsonl`.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    caller_rows = [amended_row()]

    published = mint_prior_state_rows(caller_rows)

    assert caller_rows[0]["amendment_of"] is None
    successor = next(
        row
        for row in published
        if row["debt_instrument_mention_id"]
        == caller_rows[0]["debt_instrument_mention_id"]
    )
    assert successor["amendment_of"] is not None
    assert successor is not caller_rows[0]


def test_an_unhashable_date_value_does_not_kill_the_whole_mint_pass() -> None:
    """One malformed partition row must not abort an extract.

    `{"kind": "amendment", "normalized_date": ["2020-01-01"]}` raised
    `TypeError: cannot use 'list' as a set element` out of the amendment-date
    set, taking down every remaining item in the run. No model output can reach
    it — `standardized_date_payload` overwrites `normalized_date` with this
    repo's own parser output, always `str | None` — so this is hardening for a
    tampered or hand-edited partition, and it is the failure class the
    `_borrowers` guard was written for.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                dates=[
                    _fact(kind="agreement", normalized_date="2020-02-03", prior=True),
                    _fact(kind="amendment", normalized_date=["2024-06-01"]),
                ]
            )
        ],
        counters,
    )

    assert counters == {"minted": 1}
    assert len(rows) == 2


def test_a_prior_commitment_termination_mints_and_is_not_inherited_over() -> None:
    """`commitment_termination` is in both date frozensets, and both halves matter.

    Every mint test used `maturity` and `agreement` only, so dropping
    `commitment_termination` from `PRIOR_TERM_DATE_KINDS` or from
    `INHERITED_DATE_KINDS` left the suite green (#211). The two sets answer
    different questions: the first is which prior dates can trigger and carry
    onto the predecessor, the second is which current dates are carried forward
    as unchanged when the filing states no before-value for them.
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    counters: dict[str, int] = {}
    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=False
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2026-02-03",
                            prior=True,
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2029-02-03",
                            prior=False,
                        ),
                        _fact(
                            kind="maturity", normalized_date="2030-02-03", prior=False
                        ),
                    ]
                )
            ],
            counters,
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    assert counters == {"minted": 1}
    dates = {entry["kind"]: entry for entry in json.loads(minted[0]["dates_json"])}
    # the prior value triggers and lands on the predecessor as its own, stated
    assert dates["commitment_termination"]["normalized_date"] == "2026-02-03"
    assert dates["commitment_termination"]["derived_from"] == "stated"
    # the current maturity has no prior sibling, so it carries forward marked
    assert dates["maturity"]["normalized_date"] == "2030-02-03"
    assert dates["maturity"]["derived_from"] == "inherited"

    # And the mirror case, which is the other frozenset: with the prior value on
    # `maturity` instead, the current `commitment_termination` has no prior
    # sibling and is the one carried forward. Without `commitment_termination`
    # in `INHERITED_DATE_KINDS` the predecessor simply loses that term.
    mirrored = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=False
                        ),
                        _fact(
                            kind="maturity", normalized_date="2027-02-03", prior=True
                        ),
                        _fact(
                            kind="maturity", normalized_date="2030-02-03", prior=False
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2029-02-03",
                            prior=False,
                        ),
                    ]
                )
            ]
        )
        if row.get("synthesized_by") == "prior_state"
    ]
    mirrored_dates = {
        entry["kind"]: entry for entry in json.loads(mirrored[0]["dates_json"])
    }
    assert mirrored_dates["maturity"]["normalized_date"] == "2027-02-03"
    assert mirrored_dates["maturity"]["derived_from"] == "stated"
    assert mirrored_dates["commitment_termination"]["normalized_date"] == "2029-02-03"
    assert mirrored_dates["commitment_termination"]["derived_from"] == "inherited"


def test_an_expected_date_is_never_inherited_onto_the_predecessor() -> None:
    """A date the filing only projects cannot be a term the earlier state had.

    No fixture carried `expected: True`, so deleting the
    `and not entry.get("expected")` guard left the suite green (#211).
    """
    from cdt.extractor.prior_state import mint_prior_state_rows

    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=True
                        ),
                        _fact(
                            kind="maturity",
                            normalized_date="2030-02-03",
                            prior=False,
                            expected=True,
                        ),
                    ]
                )
            ]
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    kinds = {entry["kind"] for entry in json.loads(minted[0]["dates_json"])}
    assert "maturity" not in kinds
    assert minted[0]["maturity_date"] is None
