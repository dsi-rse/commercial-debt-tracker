"""Tests for 6-K windowing: the XBRL prologue, the debt-vocabulary gate, windows and their expansion."""

from __future__ import annotations

import pytest

from cdt.segmenter.sixk import (
    WINDOW_TOKENS,
    _to_windows,
    count_tokens,
    expand_admitted_windows,
    matched_debt_keywords,
    prepare_filing,
    split_into_windows,
    strip_inline_xbrl_prologue,
)


def test_windows_respect_the_token_budget() -> None:
    """No window exceeds the target size."""
    text = "\n\n".join(f"Paragraph {index} of the filing body." for index in range(40))
    windows = split_into_windows(text, target_tokens=60)
    assert windows
    assert all(window.token_count <= 60 for window in windows)


def test_windows_cover_the_text_without_loss() -> None:
    """Concatenating window spans reproduces every non-space character."""
    text = "First para.\n\nSecond para is longer.\n\nThird para ends it."
    windows = split_into_windows(text, target_tokens=8)
    joined = "".join(window.text for window in windows)
    assert "".join(joined.split()) == "".join(text.split())


def test_keyword_gate_matches_plurals() -> None:
    """The gate lemmatises trailing plurals so it does not miss filings."""
    assert matched_debt_keywords(
        "amended its credit agreements and two term loans"
    ) == (
        "credit agreement",
        "term loan",
    )
    assert matched_debt_keywords("the debentures were issued") == ("debenture",)


def test_xbrl_prologue_stripped_but_prose_untouched() -> None:
    """A tag-dominated prologue goes; ordinary prose is returned unchanged."""
    prose = "The Company entered into a term loan with the bank on that date."
    assert strip_inline_xbrl_prologue(prose) == prose
    prologue = "\n".join(
        ["ifrs-full:PropertyPlantAndEquipmentMember", "0001234567", "2025-06-30"] * 9
    )
    stripped = strip_inline_xbrl_prologue(f"{prologue}\n{prose}")
    assert stripped.strip() == prose


def test_window_indices_are_contiguous_when_a_candidate_is_bisected() -> None:
    """A candidate split into pieces numbers them consecutively, without gaps.

    ``_to_windows`` bisects any candidate still over budget, which is the one
    place more than one window comes out of a single candidate. Reading the
    running length inside the generator handed to ``list.extend`` numbered those
    pieces 0, 2, 4, ... and repeated indices across candidates.
    """
    windows = _to_windows("alpha " * 300, [(0, 1800)], max_tokens=50)

    assert len(windows) > 1
    assert [window.index for window in windows] == list(range(len(windows)))


def test_window_indices_are_unique_across_candidates() -> None:
    """Indices identify a window, so nothing downstream can collide on them."""
    windows = _to_windows("alpha " * 300, [(0, 900), (900, 1800)], max_tokens=50)

    assert [window.index for window in windows] == list(range(len(windows)))


def test_prepare_filing_gates_on_the_whole_document_not_per_window() -> None:
    """One mention anywhere admits every window, which is what was measured.

    Gating per window instead would keep only the windows repeating the keyword,
    silently changing the documented 13.4% pass rate.
    """
    text = "\n\n".join(
        ["The Company entered into an indenture on 3 March.", *["Unrelated prose."] * 8]
    )

    windows = prepare_filing(text, target_tokens=8)

    assert len(windows) > 1
    assert sum("indenture" in window.text for window in windows) == 1


def test_prepare_filing_strips_the_prologue_before_gating() -> None:
    """A prologue's tag names are debt vocabulary and must not pass a filing.

    Gating before stripping would admit this document on `iso4217:USD`-style
    context facts alone, even though its prose never mentions debt.
    """
    pad = ["lzm-20250630", "false", "0001958217", "ifrs-full:BorrowingsMember"] * 8
    body = "The Company declared a quarterly dividend to its shareholders today."

    assert prepare_filing("\n".join([*pad, body])) == []


def test_prepare_filing_rejects_a_filing_with_no_debt_vocabulary() -> None:
    """The gate is what keeps stage 1 off the great majority of filings."""
    assert prepare_filing("The board appointed a new auditor this quarter.") == []


def _series(index: int) -> str:
    """Return a series label, so a table can be any number of rows long."""
    letters = "LMNOPQRSTUVWXYZ"
    suffix = "" if index < len(letters) else str(index // len(letters) + 1)
    return f"{letters[index % len(letters)]}{suffix}"


def _table(rows: int, *, header: str = "LONG-TERM DEBT", cells: bool = False) -> str:
    """Build a borrowings table under a header, in prose rows or in cells.

    ``cells=True`` is the shape extracted 6-K exhibits actually take: one table
    cell per line, so every capitalised cell reads like a heading.
    """
    if cells:
        columns = ["Class", "Amount", "Maturity", "Rate"]
        body = "\n".join(
            "\n".join(
                [_series(index), f"{index}0,820,459", f"2/7/203{index % 10}", "3.5%"]
            )
            for index in range(rows)
        )
        return f"{header}\n" + "\n".join(columns) + "\n" + body
    return f"{header}\n\n" + "\n".join(
        f"Series {_series(index)} pays 3.5% and matures 2/7/2031, principal 40,820,459"
        for index in range(rows)
    )


def test_expansion_gives_back_the_noun_the_crop_cut_away() -> None:
    """The failure #172 describes: numbers kept, the naming header lost.

    A window deep in a borrowings table carries amounts, rates and dates with
    nothing naming the instrument. Expansion walks back to the table's header.
    """
    body = _table(30)
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)
    admitted = windows[1]
    assert "LONG-TERM DEBT" not in admitted.text

    expanded = expand_admitted_windows([admitted])

    assert "LONG-TERM DEBT" in expanded[0].window.text
    assert expanded[0].window.text.endswith(admitted.text)
    assert expanded[0].member_indices == (admitted.index,)


def test_expansion_stops_at_an_isolated_section_header() -> None:
    """A header ends the walk where it is found, short of the minimum."""
    body = "Unrelated narrative about the quarter.\n\nBORROWINGS\n\n" + _table(
        3, header="Details of the facilities appear below."
    )
    windows = split_into_windows(body, target_tokens=30)
    admitted = next(window for window in windows if "Series N" in window.text)

    expanded = expand_admitted_windows([admitted], min_tokens=200, max_tokens=400)

    assert expanded[0].window.text.startswith("BORROWINGS")
    assert "Unrelated narrative" not in expanded[0].window.text


def test_expansion_walks_past_capitalised_table_cells() -> None:
    """A cell in a column of cells is not a header, however it is capitalised.

    Extracted tables put one cell per line, so `Class`, `Amount` and `L` all
    look like headings. Treating them as one stops the walk inside the table
    and leaves the window as unanswerable as it started.
    """
    body = _table(12, cells=True)
    windows = split_into_windows(body, target_tokens=60)
    admitted = windows[-1]

    expanded = expand_admitted_windows([admitted], min_tokens=200, max_tokens=400)

    assert "LONG-TERM DEBT" in expanded[0].window.text
    assert "Maturity" in expanded[0].window.text


def test_expansion_honours_the_cap() -> None:
    """Context added stays inside the cap when no stop is found sooner."""
    body = _table(60, header="Series A pays 1% and matures 2/7/2030, principal 1")
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)
    admitted = windows[-1]

    expanded = expand_admitted_windows([admitted], min_tokens=100, max_tokens=150)

    added = expanded[0].window.token_count - admitted.token_count
    assert 100 <= added <= 150


def test_expansion_reaches_the_document_start_rather_than_stopping_short() -> None:
    """A window near the top takes everything above it."""
    body = _table(3)
    windows = split_into_windows(body, target_tokens=30)

    expanded = expand_admitted_windows(windows[-1:], min_tokens=200, max_tokens=400)

    assert expanded[0].window.start == 0


def test_adjacent_admitted_windows_merge_instead_of_repeating_context() -> None:
    """Two admitted neighbours become one window, and say which they were."""
    body = _table(40)
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)
    pair = windows[-2:]

    expanded = expand_admitted_windows(pair)

    assert len(expanded) == 1
    assert expanded[0].member_indices == (pair[0].index, pair[1].index)
    assert expanded[0].window.end == pair[1].end
    # The shared context is present once, not once per member.
    assert expanded[0].window.text.count(pair[0].text) == 1


def test_merged_windows_stay_near_the_merge_budget() -> None:
    """A long run of admitted windows is cut rather than merged without limit.

    The generalization window has a run of 21 adjacent admitted windows, which
    merges to 8,191 tokens -- four times the largest snippet the 8-K path sends
    the same extractor.
    """
    body = _table(200)
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)

    expanded = expand_admitted_windows(windows, max_merged_tokens=1_000)

    assert len(expanded) > 1
    assert all(window.window.token_count < 1_600 for window in expanded)
    assert [index for window in expanded for index in window.member_indices] == [
        window.index for window in windows
    ]


def test_expansion_leaves_the_windows_stage_1_scored_alone() -> None:
    """Stage 1's input must stay the crop its threshold was calibrated on."""
    body = _table(40)
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)
    before = list(windows)

    expanded = expand_admitted_windows(windows[-1:])

    assert windows == before
    assert expanded[0].window.token_count > windows[-1].token_count


def test_expansion_windows_still_slice_their_source() -> None:
    """The TextWindow invariant survives expansion and merging."""
    body = _table(40)
    windows = split_into_windows(body, target_tokens=WINDOW_TOKENS)

    for expanded in expand_admitted_windows(windows):
        window = expanded.window
        assert window.text == window.source[window.start : window.end]
        assert window.token_count == count_tokens(window.text)


def test_expansion_refuses_windows_from_two_documents() -> None:
    """Offsets only mean something inside the text they index into."""
    first = split_into_windows(_table(3), target_tokens=30)
    second = split_into_windows(_table(3, header="OTHER FILING"), target_tokens=30)

    with pytest.raises(ValueError, match="same document"):
        expand_admitted_windows([first[-1], second[-1]])


def test_expansion_rejects_a_minimum_above_the_cap() -> None:
    """A minimum the cap cannot satisfy is a configuration error."""
    windows = split_into_windows(_table(3), target_tokens=30)

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        expand_admitted_windows(windows, min_tokens=400, max_tokens=100)


def test_expansion_of_nothing_is_nothing() -> None:
    """Most filings have no admitted window at all."""
    assert expand_admitted_windows([]) == []


def test_merged_windows_stay_within_the_ceiling_when_members_are_not_adjacent() -> None:
    """The unadmitted text a merge pulls in counts toward ``MAX_MERGED_TOKENS``.

    With every other window admitted, each later member's backward expansion
    reaches the span through an unadmitted window. Counting only the members'
    own tokens let a merge reach about 1.8 times the ceiling.
    """
    from cdt.segmenter.sixk import MAX_MERGED_TOKENS

    # Lines of about 140 tokens: the backward walk stops at the first line
    # boundary past its 200-token minimum, which is then a whole unadmitted
    # window back, so it reaches the span before it.
    sentence = (
        "Under the agreement the revolving credit facility bears interest at a "
        "floating rate plus an applicable margin. "
    )
    text = "\n".join(f"{index}. " + sentence * 7 for index in range(200))
    windows = split_into_windows(text, target_tokens=281)
    assert len(windows) >= 60

    expanded = expand_admitted_windows(windows[::2])

    assert any(len(window.member_indices) > 1 for window in expanded)
    assert max(window.window.token_count for window in expanded) <= MAX_MERGED_TOKENS
