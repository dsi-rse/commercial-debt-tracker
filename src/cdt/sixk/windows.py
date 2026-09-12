"""Preprocessing for Form 6-K documents ahead of the two-stage triage.

Ported from the ``uchicago-dsi/commercial-debt-tracker-models`` research repo,
where the window size, the keyword gate and the inline-XBRL rule were each
chosen against labelled data. The pieces belong together because they only make
sense as a sequence: strip the XBRL padding, gate on debt vocabulary, then cut
the survivor into the windows the classifier was trained on. That sequence is
:func:`prepare_filing` rather than a comment, because the granularity each step
runs at is not recoverable from the individual functions.

Windows are 400 tokens, not the 2,000 the 8-K path uses. Only 1.2% of positive
windows proved context-dependent at that size, and the smaller crop cut
extraction tokens to 0.34x while still matching 146 of 154 known mentions.

A crop that small does lose the noun naming the instrument often enough to
matter, so :func:`expand_admitted_windows` gives it back -- after stage 1 has
scored the unexpanded window, which is the only place the fix is free. See the
section comment above it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tiktoken import Encoding

#: Encoding used for every token count in this module.
TIKTOKEN_ENCODING_NAME = "o200k_base"

#: Window size the shipped stage-1 model was trained on. Changing this
#: invalidates the model and its calibrated threshold together.
WINDOW_TOKENS = 400

#: Cut points tried in order when a span exceeds the token budget:
#: paragraph, then line, then sentence. A span still too long after all
#: three is bisected on characters.
_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+"),
)

#: Words common enough that a line containing one is almost certainly prose.
_PROSE_MARKERS: frozenset[str] = frozenset(
    {
        "the",
        "of",
        "and",
        "to",
        "in",
        "for",
        "a",
        "is",
        "was",
        "were",
        "that",
        "this",
        "with",
        "on",
        "as",
        "at",
        "by",
        "from",
        "its",
        "our",
        "has",
        "have",
        "had",
        "will",
        "which",
        "under",
        "any",
        "such",
        "shall",
    }
)

#: A line of inline-XBRL context: a namespaced tag, or a bare scalar such as a
#: CIK, a ticker-date stem, a fiscal period, a boolean or a lone number.
_XBRL_CONTEXT_LINE = re.compile(
    r"""(?xi)
    ^(?:
        [A-Za-z][\w-]*:[\w.-]+          # iso4217:USD, ifrs-full:...Member
      | -{0,2}\d[\d,./-]*%?             # 0001865408, --12-31, 2025-06-30, .3333
      | (?:true|false)                   # boolean facts
      | Q[1-4]
      | [A-Za-z]{1,6}-\d{6,8}            # lzm-20250630 document stem
      | [A-Z][a-z]+\s+\d{1,2}            # December 31
    )$
    """
)

#: A namespaced inline-XBRL tag, the signature that a block is context padding
#: rather than a numeric table (whose lines are bare scalars too).
_XBRL_TAG_LINE = re.compile(r"(?i)^[A-Za-z][\w-]*:[\w.-]+$")

#: A prologue must be at least this many lines before stripping is worthwhile.
MIN_XBRL_PROLOGUE_LINES = 20

#: Share of prologue lines that must be namespaced tags. Bare scalars alone are
#: not enough: a borrowings schedule is also mostly bare numbers, and stripping
#: one would delete the table bodies the annotation codebook rules relevant.
MIN_XBRL_TAG_SHARE = 0.10

#: Share of prologue lines that must be context facts of some kind.
MIN_XBRL_CONTEXT_SHARE = 0.8

#: Words a line needs before it can count as prose rather than a context fact.
MIN_PROSE_WORDS = 5


def _is_prose_line(line: str) -> bool:
    """Return whether a line reads as prose rather than an XBRL context fact.

    Args:
        line: A single line of extracted text.

    Returns:
        ``True`` when the line has several words and at least one function word.

    >>> _is_prose_line("The Company entered into a term loan with the bank.")
    True
    >>> _is_prose_line("ifrs-full:PropertyPlantAndEquipmentMember")
    False
    >>> _is_prose_line("Unsecured corporate bonds 2032 3.45 90,000 90,000")
    False
    """
    words = line.split()
    if len(words) < MIN_PROSE_WORDS:
        return False
    return any(word.strip(".,;:()").lower() in _PROSE_MARKERS for word in words)


def strip_inline_xbrl_prologue(text: str) -> str:
    r"""Drop the leading block of inline-XBRL context facts from extracted text.

    Inline-XBRL 6-K documents (``<ticker>-<yyyymmdd>.htm``) extract with a long
    prologue of context facts before any prose -- namespaced tags such as
    ``iso4217:USD`` but also bare scalars such as the CIK, the period end and
    lone numbers. The NER stage must echo its input verbatim, so every prologue
    token is paid for at output prices and adds a chance of failing the identity
    check, and TF-IDF windows of tag soup dilute the classifier signal. These
    documents carry real prose after the prologue, so they are cleaned rather
    than dropped.

    Stripping stops at the first line that reads as prose, and is skipped unless
    the block before it is long enough to matter, is almost entirely context
    facts, and contains namespaced tags. A document with no prose at all is left
    untouched. That last condition is what separates a
    context dump from a numeric table: a borrowings schedule is also mostly bare
    numbers, and stripping one would delete the very rows the annotation
    codebook rules relevant. Text without such a prologue is returned unchanged.

    Args:
        text: Extracted document text.

    Returns:
        The text with any inline-XBRL prologue removed.

    >>> body = "The Company issued senior notes due 2030 under an indenture."
    >>> pad = ["lzm-20250630", "false", "0001958217", "iso4217:USD"] * 8
    >>> strip_inline_xbrl_prologue("\n".join([*pad, body])) == body
    True
    >>> strip_inline_xbrl_prologue(body) == body
    True
    >>> table = ["Unsecured corporate bonds 2032 3.45 90,000"] * 40
    >>> strip_inline_xbrl_prologue("\n".join(table)).count("bonds")
    40
    """
    lines = text.split("\n")
    index = next(
        (offset for offset, line in enumerate(lines) if _is_prose_line(line)), None
    )
    if index is None:
        return text
    # End the block at the last context fact, not at the first prose line, so
    # that title and cover-header lines between the two survive. Losing them
    # would cost the extraction stage the filer's own name.
    end = 0
    for offset in range(index):
        if _XBRL_CONTEXT_LINE.match(lines[offset].strip()):
            end = offset + 1
    prologue = [candidate.strip() for candidate in lines[:end] if candidate.strip()]
    if len(prologue) < MIN_XBRL_PROLOGUE_LINES:
        return text
    tags = sum(1 for candidate in prologue if _XBRL_TAG_LINE.match(candidate))
    if tags / len(prologue) < MIN_XBRL_TAG_SHARE:
        return text
    context = sum(1 for candidate in prologue if _XBRL_CONTEXT_LINE.match(candidate))
    if context / len(prologue) < MIN_XBRL_CONTEXT_SHARE:
        return text
    return "\n".join(lines[end:])


#: Document-level gate vocabulary. Pass ``keywords=`` to
#: :func:`matched_debt_keywords` or :func:`has_debt_keyword` to widen a single
#: run; the shipped tuple is what the 13.4% pass rate in the docs was measured
#: against, so changing it in place invalidates that figure.
DEBT_KEYWORDS: tuple[str, ...] = (
    "credit agreement",
    "indenture",
    "notes due",
    "term loan",
    "revolving credit",
    "notes offering",
    "debenture",
    "syndicated loan",
    "bond issuance",
)

#: Plural suffixes allowed after a keyword lemma.
KEYWORD_PLURAL_SUFFIX = r"(?:s|es)?"


@cache
def _compiled_keywords(
    keywords: tuple[str, ...],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile a keyword vocabulary into word-boundary patterns (cached)."""
    return tuple(
        (
            keyword,
            re.compile(rf"(?i)\b{re.escape(keyword)}{KEYWORD_PLURAL_SUFFIX}\b"),
        )
        for keyword in keywords
    )


@lru_cache(maxsize=1)
def _get_encoding() -> Encoding:
    """Return the tiktoken encoding, loaded once.

    Returns:
        The encoding every token count in this module uses.
    """
    import tiktoken

    return tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count tokens in a text.

    Args:
        text: Text to measure.

    Returns:
        The number of tokens under the workflow's encoding.
    """
    return len(_get_encoding().encode(text))


def matched_debt_keywords(
    text: str,
    *,
    keywords: tuple[str, ...] = DEBT_KEYWORDS,
) -> tuple[str, ...]:
    """Return the debt keywords present in a text.

    Args:
        text: Text to search.
        keywords: Vocabulary to match, in reporting order.

    Returns:
        Matching keywords in ``keywords`` order.

    >>> matched_debt_keywords("entered into a new Term Loan and an indenture")
    ('indenture', 'term loan')
    >>> matched_debt_keywords("amended its credit agreements and two term loans")
    ('credit agreement', 'term loan')
    >>> matched_debt_keywords("the debenture was issued")
    ('debenture',)
    """
    return tuple(
        keyword
        for keyword, pattern in _compiled_keywords(keywords)
        if pattern.search(text)
    )


def has_debt_keyword(
    text: str,
    *,
    keywords: tuple[str, ...] = DEBT_KEYWORDS,
) -> bool:
    """Return whether a text mentions any debt keyword.

    Args:
        text: Text to search.
        keywords: Vocabulary to match.

    Returns:
        ``True`` when at least one debt keyword is present.

    >>> has_debt_keyword("priced its notes offering")
    True
    >>> has_debt_keyword("declared a quarterly dividend")
    False
    """
    return any(pattern.search(text) for _, pattern in _compiled_keywords(keywords))


@dataclass(frozen=True)
class TextWindow:
    """One contiguous window of a document.

    ``text`` always equals ``source[start:end]``, and ``source`` is carried
    rather than left to the caller because the post-admission expansion in
    :func:`expand_admitted_windows` reads text *outside* the window. The text
    those offsets refer to is the gated body, not the original document -- they
    differ by however long an inline-XBRL prologue was stripped -- and a caller
    holding both would eventually pass the wrong one, shifting every expansion
    by that length with nothing to fail on.
    """

    index: int
    text: str
    start: int
    end: int
    token_count: int
    #: Excluded from ``repr`` only because it is a whole document: a window's
    #: repr in a failing assertion should stay readable.
    source: str = field(repr=False)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it excludes surrounding whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_span(
    text: str,
    start: int,
    end: int,
    pattern: re.Pattern[str],
) -> list[tuple[int, int]]:
    """Split a span on a boundary pattern into spans that tile it exactly.

    Each boundary stays with the span it follows, so the spans concatenate back
    to the original slice. Windows are emitted as one slice of the source, so
    dropping the boundaries here would undercount their tokens.
    """
    spans: list[tuple[int, int]] = []
    position = start
    for match in pattern.finditer(text, start, end):
        if match.end() > position:
            spans.append((position, match.end()))
            position = match.end()
    if position < end:
        spans.append((position, end))
    return spans


def _halve_span(
    text: str,
    start: int,
    end: int,
    *,
    max_tokens: int,
) -> list[tuple[int, int]]:
    """Bisect a span by characters until every piece fits the token budget."""
    if end - start <= 1 or count_tokens(text[start:end]) <= max_tokens:
        return [(start, end)]
    middle = (start + end) // 2
    return [
        *_halve_span(text, start, middle, max_tokens=max_tokens),
        *_halve_span(text, middle, end, max_tokens=max_tokens),
    ]


def _leaf_spans(
    text: str,
    start: int,
    end: int,
    *,
    max_tokens: int,
    depth: int = 0,
) -> list[tuple[int, int]]:
    """Split a span on the coarsest boundary that fits the token budget."""
    if count_tokens(text[start:end]) <= max_tokens:
        return [(start, end)]
    if depth >= len(_BOUNDARY_PATTERNS):
        return _halve_span(text, start, end, max_tokens=max_tokens)
    children = _split_span(text, start, end, _BOUNDARY_PATTERNS[depth])
    if len(children) <= 1:
        return _leaf_spans(text, start, end, max_tokens=max_tokens, depth=depth + 1)
    return [
        span
        for child_start, child_end in children
        for span in _leaf_spans(
            text,
            child_start,
            child_end,
            max_tokens=max_tokens,
            depth=depth + 1,
        )
    ]


def _bounded_spans(text: str, *, max_tokens: int) -> list[tuple[int, int, int]]:
    r"""Return token-bounded spans that tile the text, with their token counts.

    Counts include each span's trailing whitespace, so the packing in
    :func:`_pack_spans` accounts for every character it will emit.

    Summing counts is still only an approximation of the joined slice's cost,
    not an upper bound on it: ``o200k_base`` groups digits as ``\p{N}{1,3}``, so
    joining two spans can *raise* the count -- ``count_tokens(", 8,72,6  ")`` is
    8 and ``count_tokens("217")`` is 1, but the concatenation is 10, not 9.
    :func:`_to_windows` re-measures each candidate and bisects anything over
    budget, which is what makes ``max_tokens`` an actual guarantee. That
    bisection branch is load-bearing rather than defensive.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    start, end = _strip_span(text, 0, len(text))
    if start >= end:
        return []
    return [
        (span_start, span_end, count_tokens(text[span_start:span_end]))
        for span_start, span_end in _leaf_spans(text, start, end, max_tokens=max_tokens)
    ]


def _pack_spans(
    spans: Sequence[tuple[int, int, int]],
    *,
    max_tokens: int,
) -> list[list[tuple[int, int, int]]]:
    """Greedily group adjacent spans while staying inside the token budget."""
    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    current_tokens = 0
    for span in spans:
        if current and current_tokens + span[2] > max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(span)
        current_tokens += span[2]
    if current:
        groups.append(current)
    return groups


def _to_windows(
    text: str,
    candidates: Sequence[tuple[int, int]],
    *,
    max_tokens: int,
) -> list[TextWindow]:
    """Strip, verify, and index candidate spans as windows.

    Any candidate that still measures over ``max_tokens`` is bisected, so the
    budget is a guarantee rather than an estimate.
    """
    windows: list[TextWindow] = []
    for candidate_start, candidate_end in candidates:
        start, end = _strip_span(text, candidate_start, candidate_end)
        if start >= end:
            continue
        token_count = count_tokens(text[start:end])
        pieces = (
            [(start, end, token_count)]
            if token_count <= max_tokens
            else [
                (piece_start, piece_end, count_tokens(text[piece_start:piece_end]))
                for piece_start, piece_end in _halve_span(
                    text, start, end, max_tokens=max_tokens
                )
            ]
        )
        # Bind the base before extending: ``list.extend`` consumes the
        # generator incrementally, so reading ``len(windows)`` inside it would
        # see the earlier pieces of this same candidate already appended.
        base = len(windows)
        windows.extend(
            TextWindow(
                index=base + offset,
                text=text[piece_start:piece_end],
                start=piece_start,
                end=piece_end,
                token_count=piece_tokens,
                source=text,
            )
            for offset, (piece_start, piece_end, piece_tokens) in enumerate(pieces)
        )
    return windows


def split_into_windows(
    text: str,
    *,
    target_tokens: int = WINDOW_TOKENS,
) -> list[TextWindow]:
    """Split a text into contiguous, non-overlapping classifier windows.

    Windows are cut on paragraph boundaries where possible, falling back to
    lines, then sentences, then a character bisection, so no window exceeds
    ``target_tokens``.

    Args:
        text: Document text to split.
        target_tokens: Largest allowed window size in tokens. The shipped
            stage-1 model was fitted at :data:`WINDOW_TOKENS`, so moving
            this invalidates its calibrated threshold.

    Returns:
        Windows in document order; empty when the text is blank.

    >>> [window.text for window in split_into_windows("a. b. c.", target_tokens=4)]
    ['a.', 'b.', 'c.']
    >>> split_into_windows("   ", target_tokens=4)
    []
    """
    spans = _bounded_spans(text, max_tokens=target_tokens)
    groups = _pack_spans(spans, max_tokens=target_tokens)
    candidates = [(group[0][0], group[-1][1]) for group in groups]
    return _to_windows(text, candidates, max_tokens=target_tokens)


def prepare_filing(
    text: str,
    *,
    target_tokens: int = WINDOW_TOKENS,
    keywords: tuple[str, ...] = DEBT_KEYWORDS,
) -> list[TextWindow]:
    """Run steps 1-3 of the sequence: strip, gate, then window.

    The order is load-bearing and was previously recorded only in prose. The
    keyword gate applies to the *whole document*, after the prologue is stripped
    and before it is cut up. Applying it per window instead would silently change
    the measured 13.4% pass rate -- a filing that says "indenture" once in its
    introduction would keep only the windows repeating the word, rather than all
    of them -- and nothing would fail while it happened. Stripping after gating
    would likewise let a prologue's tag names, which are made of debt
    vocabulary, pass a filing whose prose never mentions debt.

    Args:
        text: Extracted document text.
        target_tokens: Window size; see :func:`split_into_windows`.
        keywords: Gate vocabulary; see :data:`DEBT_KEYWORDS`.

    These are the windows stage 1 scores. What stage 2 and extraction see is
    :func:`expand_admitted_windows` applied to the ones stage 1 admits.

    Returns:
        Windows for a filing that passes the gate, in document order; empty when
        the gate rejects it or nothing is left to window.

    >>> len(prepare_filing("The Company issued senior notes due 2030 today."))
    1
    >>> prepare_filing("The Company declared a quarterly dividend of $0.10.")
    []
    """
    body = strip_inline_xbrl_prologue(text)
    if not has_debt_keyword(body, keywords=keywords):
        return []
    return split_into_windows(body, target_tokens=target_tokens)


# --- Post-admission expansion -------------------------------------------------
#
# Everything above runs before stage 1. Everything below runs *after* it, on the
# windows stage 1 admitted, and never on the windows it scores: the shipped
# stage-1 model and its calibrated threshold are properties of the 400-token
# crop, so widening the text it sees would invalidate both. Expanding after
# admission buys context for extraction at no cost to the classifier.

#: Tokens of context prepended to an admitted window, unless a section header
#: is reached sooner. A 400-token crop can keep an instrument's amounts, rates
#: and dates while cutting away the noun that names it, which leaves the
#: extractor nothing to anchor on -- the input no longer determines an answer.
#: Chosen by replaying the generalization window's 392 admitted windows: of the
#: kept snippets carrying money, a rate or a date with no instrument noun
#: anywhere, 7 of 7 recover a noun at 200 tokens, against 4 of 7 at 100. Most
#: of the way there by 150; 200 is what also reaches the table header above a
#: page break, which is the shape the reviewers could not read at all.
MIN_EXPANSION_TOKENS = 200

#: Hard cap on the context prepended to one admitted window. Expansion is paid
#: only on the 5.8% of windows stage 1 admits, but the walk backwards needs a
#: stop for the case where no header and no blank line is found: without one,
#: a document of unbroken table rows would prepend itself to every window.
#: Past the minimum this only buys a tidier boundary, so it is one window wide
#: rather than generous.
MAX_EXPANSION_TOKENS = 400

#: Ceiling on one merged window. Adjacent admitted windows merge rather than
#: emit their shared context twice, and a run of them would otherwise merge
#: without limit -- the generalization window has a run of 21, reaching 8,191
#: tokens. 2,000 is what the 8-K path already sends the same extractor, so no
#: 6-K snippet is larger than something that stage already handles. The budget
#: is summed from the parts rather than measured on the join, like the packing
#: in :func:`_bounded_spans`, so it is a target and not a guarantee.
MAX_MERGED_TOKENS = 2_000

#: Longest a line can be and still read as a heading rather than a sentence.
MAX_HEADER_WORDS = 12

#: How far back a paragraph-boundary test looks. Long enough to see a blank
#: line through an indented continuation, short enough that the test stays
#: bounded work per candidate.
_PARAGRAPH_LOOKBACK = 200

#: Characters scanned per token of budget when collecting candidate stops. Only
#: a bound on the search, not on the result: the token budget is what decides
#: how far the walk goes. Generous because table text runs far fewer characters
#: per token than prose does.
_CHARS_PER_TOKEN_BOUND = 24

#: An explicitly numbered heading: ``Item 5.02``, ``NOTE 12 - BORROWINGS``,
#: ``Part II``, ``Schedule 3``. Matched before the casing rules below, because
#: a numbered heading may be sentence-cased and may end in a period.
_NUMBERED_HEADING = re.compile(
    r"""(?xi)
    ^(?:item|note|section|part|exhibit|schedule|annex|appendix|article)
    \s*(?:no\.?\s*)?
    (?:\d|[ivxlc]+\b)
    """
)

#: A blank line immediately before an offset, i.e. a paragraph boundary.
_PARAGRAPH_BREAK_BEFORE = re.compile(r"\n[^\S\n]*\n\s*\Z")


@dataclass(frozen=True)
class ExpandedWindow:
    """One admitted window with its context, and the windows it absorbed.

    ``window`` is the text extraction should see. ``member_indices`` holds the
    indices of every admitted window it covers, in document order: adjacent
    admitted windows merge rather than emit their shared context twice, so a
    caller persisting one row per admitted window still knows which rows this
    text answers for. A merged window is larger than ``WINDOW_TOKENS`` by
    design, bounded by :data:`MAX_MERGED_TOKENS`: its members were separate
    snippets bound for extraction anyway, and merging them sends the text they
    share once instead of twice.
    """

    window: TextWindow
    member_indices: tuple[int, ...]


def _is_section_header(line: str, *, isolated: bool) -> bool:
    """Return whether a line reads as a section header rather than body text.

    Casing alone does not decide it. Extracted 6-K text puts one table cell per
    line, and a cell reads exactly like a heading: ``Currency``, ``Book Value``
    and ``I`` are all short, unpunctuated and capitalised. Worse, the lines that
    survive a page break inside a table -- ``GRUPO SUPERVIELLE S.A.``, ``NOTES
    TO THE CONSOLIDATED FINANCIAL STATEMENTS`` -- look like the strongest
    headings in the document while marking nothing at all. So a casing-based
    header also has to stand alone in its own paragraph, which a cell in a
    column of cells does not. Only an explicitly numbered heading is taken on
    its wording alone.

    Erring towards ``False`` is the cheap direction: the walk backwards then
    carries on to its token minimum and passes over the unrecognised heading on
    the way. A false ``True`` stops the walk early and can leave the instrument
    noun outside the window, which is the failure expansion exists to fix.

    Args:
        line: A single line of extracted text.
        isolated: Whether a blank line precedes the line, i.e. whether it
            stands on its own rather than sitting in a run of table cells.

    Returns:
        ``True`` when the line marks the start of a section.

    >>> _is_section_header("NOTE 12 - LONG-TERM BORROWINGS", isolated=False)
    True
    >>> _is_section_header("Long-Term Debt", isolated=True)
    True
    >>> _is_section_header("Long-Term Debt", isolated=False)
    False
    >>> _is_section_header("Borrowings:", isolated=True)
    True
    >>> _is_section_header("The Company issued notes due 2030.", isolated=True)
    False
    >>> _is_section_header("Series L 50,974,086 2/7/2026", isolated=True)
    False
    """
    text = line.strip()
    words = text.split()
    if not words or len(words) > MAX_HEADER_WORDS:
        return False
    if _NUMBERED_HEADING.match(text):
        return True
    # A digit outside a numbered heading means the line carries data, and a
    # table row is the shape most easily mistaken for a heading. Stopping the
    # walk on a row is the worst outcome available here, because the row above
    # it is where the header actually is.
    if any(character.isdigit() for character in text):
        return False
    if not isolated or not any(character.isalpha() for character in text):
        return False
    if text.endswith((".", ",", ";")):
        return False
    if text.endswith(":") or text.isupper():
        return True
    return all(word[0].isupper() for word in words if word[0].isalpha())


def _line_starts(text: str, *, floor: int, limit: int) -> list[int]:
    r"""Return line-start offsets in ``[floor, limit)``, nearest ``limit`` first.

    ``floor`` bounds the walk in characters so a window near the end of a long
    document does not scan the whole of it; the token budget in
    :func:`_expansion_start` is what actually decides where expansion stops.

    Args:
        text: Document text.
        floor: Earliest offset the walk may reach.
        limit: Offset to walk back from, exclusive.

    Returns:
        Line starts in decreasing offset order.

    >>> _line_starts("alpha\nbeta\ngamma", floor=0, limit=11)
    [6, 0]
    """
    offsets: list[int] = []
    cursor = limit
    while cursor > floor:
        break_at = text.rfind("\n", floor, cursor)
        if break_at < 0:
            if floor == 0 and cursor > 0:
                offsets.append(0)
            break
        if break_at + 1 < limit:
            offsets.append(break_at + 1)
        cursor = break_at
    return offsets


def _is_paragraph_start(text: str, offset: int) -> bool:
    r"""Return whether an offset follows a blank line.

    Args:
        text: Document text.
        offset: A line start.

    Returns:
        ``True`` when a blank line immediately precedes the offset.

    >>> _is_paragraph_start("a\n\nb", 3), _is_paragraph_start("a\nb", 2)
    (True, False)
    """
    if offset <= 0:
        return True
    return bool(
        _PARAGRAPH_BREAK_BEFORE.search(
            text[max(0, offset - _PARAGRAPH_LOOKBACK) : offset]
        )
    )


def _farthest_within_budget(
    text: str,
    candidates: Sequence[int],
    limit: int,
    *,
    budget: int,
) -> int | None:
    """Return the last candidate position whose span to ``limit`` fits a budget.

    Candidates run nearest-first, so the span to ``limit`` only grows along the
    list and a bisection finds the boundary in a handful of token counts rather
    than one per line. Token counts of nested spans are non-decreasing but not
    provably so -- BPE merges across a new boundary -- and an off-by-one token
    at the edge of a heuristic budget does not matter.
    """
    if count_tokens(text[candidates[0] : limit]) > budget:
        return None
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(text[candidates[middle] : limit]) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _first_reaching_minimum(
    text: str,
    candidates: Sequence[int],
    limit: int,
    *,
    minimum: int,
) -> int | None:
    """Return the first candidate position whose span to ``limit`` reaches a minimum.

    Bisects on the same monotonicity as :func:`_farthest_within_budget`.
    """
    if count_tokens(text[candidates[-1] : limit]) < minimum:
        return None
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if count_tokens(text[candidates[middle] : limit]) >= minimum:
            high = middle
        else:
            low = middle + 1
    return low


def _expansion_start(
    text: str,
    start: int,
    *,
    min_tokens: int,
    max_tokens: int,
) -> int:
    r"""Return the offset an admitted window should be expanded back to.

    The stop is chosen in the order the failure demands. A section header ends
    the walk wherever it is found, because the section title is both the
    likeliest place the instrument is named and the point past which the text
    belongs to something else. Otherwise the walk takes at least ``min_tokens``
    and then stops at the nearest blank line, falling back to the line boundary
    where the minimum was met. Nothing crosses ``max_tokens``.

    Args:
        text: Document text the window indexes into.
        start: The admitted window's start offset.
        min_tokens: Context to take unless a header is reached sooner.
        max_tokens: Cap on context taken.

    Returns:
        The new start offset; ``start`` itself when nothing can be added.

    >>> body = "BORROWINGS\nSeries L pays 3.5%\n40,820,459 2/7/2026"
    >>> _expansion_start(body, 30, min_tokens=4, max_tokens=100)
    0
    >>> _expansion_start(body, 30, min_tokens=4, max_tokens=2)
    30
    """
    if start <= 0 or max_tokens <= 0:
        return start
    floor = max(0, start - max_tokens * _CHARS_PER_TOKEN_BOUND)
    candidates = _line_starts(text, floor=floor, limit=start)
    if not candidates:
        return start
    within = _farthest_within_budget(text, candidates, start, budget=max_tokens)
    if within is None:
        return start
    reachable = candidates[: within + 1]
    header = next(
        (
            offset
            for offset in reachable
            if _is_section_header(
                _line_at(text, offset), isolated=_is_paragraph_start(text, offset)
            )
        ),
        None,
    )
    if header is not None:
        return header
    minimum = _first_reaching_minimum(text, reachable, start, minimum=min_tokens)
    if minimum is None:
        # The whole reachable prefix is shorter than the minimum, so there is
        # nothing to choose between: take all of it.
        return reachable[-1]
    paragraph = next(
        (offset for offset in reachable[minimum:] if _is_paragraph_start(text, offset)),
        None,
    )
    return paragraph if paragraph is not None else reachable[minimum]


def _line_at(text: str, offset: int) -> str:
    """Return the line beginning at an offset."""
    end = text.find("\n", offset)
    return text[offset:] if end < 0 else text[offset:end]


@dataclass(frozen=True)
class _MergedSpan:
    """A span under construction, with a running token estimate.

    The estimate sums the parts -- prepended context plus each member's own
    count -- rather than re-measuring the join on every merge, which would make
    a long run quadratic in the text it covers.
    """

    start: int
    end: int
    members: tuple[int, ...]
    tokens: int

    @classmethod
    def of(
        cls: type[_MergedSpan],
        window: TextWindow,
        *,
        start: int,
        context: int,
    ) -> _MergedSpan:
        """Start a span at one window plus the context prepended to it."""
        return cls(
            start=start,
            end=window.end,
            members=(window.index,),
            tokens=context + window.token_count,
        )

    def merged(self: _MergedSpan, window: TextWindow) -> _MergedSpan:
        """Return this span extended over an adjacent or overlapping window."""
        return _MergedSpan(
            start=self.start,
            end=max(self.end, window.end),
            members=(*self.members, window.index),
            tokens=self.tokens + window.token_count,
        )


def expand_admitted_windows(
    windows: Sequence[TextWindow],
    *,
    min_tokens: int = MIN_EXPANSION_TOKENS,
    max_tokens: int = MAX_EXPANSION_TOKENS,
    max_merged_tokens: int = MAX_MERGED_TOKENS,
) -> list[ExpandedWindow]:
    r"""Prepend context to the windows stage 1 admitted, merging what overlaps.

    Run this between stage 1 and stage 2, never before stage 1: see the section
    comment above. Windows must come from one document, since offsets are only
    comparable within the text they index into and merging spans across two
    documents would splice unrelated text together.

    Args:
        windows: Admitted windows from a single document, in any order.
        min_tokens: Context to prepend unless a section header is reached
            sooner; see :data:`MIN_EXPANSION_TOKENS`.
        max_tokens: Cap on context prepended to one window; see
            :data:`MAX_EXPANSION_TOKENS`.
        max_merged_tokens: Ceiling on a merged window; see
            :data:`MAX_MERGED_TOKENS`.

    Returns:
        Expanded windows in document order, one per group of admitted windows
        whose expansions ran into each other.

    Raises:
        ValueError: If the windows do not share a document, or the minimum
            exceeds the cap.

    >>> body = "LONG-TERM DEBT\n" + "\n".join(
    ...     f"Series {letter} 3.5% 2031 40,820,459" for letter in "LMNOPQRST"
    ... )
    >>> windows = split_into_windows(body, target_tokens=40)
    >>> windows[2].text.splitlines()[0]
    'Series P 3.5% 2031 40,820,459'
    >>> reach = {"min_tokens": 8, "max_tokens": 200}
    >>> expanded = expand_admitted_windows(windows[2:3], **reach)
    >>> expanded[0].window.text.splitlines()[0]
    'LONG-TERM DEBT'
    >>> expanded[0].member_indices
    (2,)
    >>> merged = expand_admitted_windows(windows[2:4], **reach)
    >>> [window.member_indices for window in merged]
    [(2, 3)]
    """
    if not windows:
        return []
    if min_tokens > max_tokens:
        raise ValueError(f"min_tokens {min_tokens} exceeds max_tokens {max_tokens}")
    source = windows[0].source
    if any(
        window.source is not source and window.source != source for window in windows
    ):
        raise ValueError("every window must come from the same document")
    spans: list[_MergedSpan] = []
    for window in sorted(windows, key=lambda item: (item.start, item.end)):
        start = _expansion_start(
            source, window.start, min_tokens=min_tokens, max_tokens=max_tokens
        )
        # An expansion that reaches into its predecessor -- or is separated from
        # it by whitespace alone -- emits text the predecessor already carries,
        # so the two become one window instead of two overlapping ones. Slicing
        # with a start before the previous end yields "", which is why the
        # overlap case and the abutting case read the same here.
        adjacent = bool(spans) and not source[spans[-1].end : start].strip()
        if adjacent and spans[-1].tokens + window.token_count <= max_merged_tokens:
            spans[-1] = spans[-1].merged(window)
        else:
            # Either this window's expansion did not reach its predecessor, or
            # the run is longer than one window may be and the budget cut it
            # here. A window opening a cut run still expands, so the text
            # either side of the cut is sent twice; that is the cheaper
            # mistake, because the duplicate is bounded by ``max_tokens``
            # while a window starting cold at the cut is back to the failure
            # this function exists to fix. The run of 21 in the generalization
            # window is exactly that case: its table header sits above a cut.
            spans.append(
                _MergedSpan.of(
                    window,
                    start=start,
                    context=count_tokens(source[start : window.start]),
                )
            )
    return [
        ExpandedWindow(
            window=_window_over(source, span.start, span.end, index=span.members[0]),
            member_indices=tuple(span.members),
        )
        for span in spans
    ]


def _window_over(text: str, start: int, end: int, *, index: int) -> TextWindow:
    """Build a window over a span, stripped of its surrounding whitespace."""
    span_start, span_end = _strip_span(text, start, end)
    return TextWindow(
        index=index,
        text=text[span_start:span_end],
        start=span_start,
        end=span_end,
        token_count=count_tokens(text[span_start:span_end]),
        source=text,
    )
