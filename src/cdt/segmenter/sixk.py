"""Form 6-K segmentation: windows ahead of the two-stage triage, and its stage.

:func:`prepare_filing` strips the inline-XBRL prologue, gates the document on
debt vocabulary, and cuts it into :data:`WINDOW_TOKENS`-token windows; the
segment stage (:func:`segment_pending_sixk_documents`) persists those windows as
spans in ``sixk-windows``. After stage 1, :func:`expand_admitted_windows`
prepends context to the admitted windows and merges adjacent ones. Window size, gate vocabulary and
expansion limits were chosen against labelled data; see
``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cdt.datasets import (
    SIXK_DOCUMENT_DATASET_NAME,
    SIXK_WINDOW_DATASET_NAME,
    resolve_artifact_root,
)
from cdt.ingest.core import DOCUMENT_COLUMNS
from cdt.partition_stage import PartitionOutput, run_partition_stage
from cdt.segmenter.core import document_text_for_record, ensure_s3_client
from cdt.segmenter.text import DOCUMENT_RE, TYPE_RE, normalize_body_lines
from cdt.storage.tables import read_table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tiktoken import Encoding


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

#: Share of prologue lines that must be namespaced tags, so that a borrowings
#: schedule (also mostly bare numbers) is not mistaken for a prologue.
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
    prologue of context facts -- namespaced tags such as ``iso4217:USD`` and bare
    scalars such as the CIK or period end -- before any prose.

    The prologue ends at the last context-fact line before the first prose line,
    so title lines between the two survive. It is removed only when it has at
    least :data:`MIN_XBRL_PROLOGUE_LINES` lines, at least
    :data:`MIN_XBRL_CONTEXT_SHARE` of them context facts and at least
    :data:`MIN_XBRL_TAG_SHARE` namespaced tags; the tag share is what keeps a
    numeric table, also mostly bare scalars, from being stripped. Text with no
    prose line, or no qualifying prologue, is returned unchanged.

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
    # End at the last context fact, not the first prose line: the cover lines
    # between them carry the filer's name.
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


#: Document-level gate vocabulary. Pass ``keywords=`` to widen a single run;
#: the documented gate pass rate was measured against this tuple.
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

    ``text`` always equals ``source[start:end]``. ``source`` is the text the
    offsets index into -- the prologue-stripped body, not the original
    document -- carried so :func:`expand_admitted_windows` can read outside the
    window without the caller supplying a text that might not match.
    """

    index: int
    text: str
    start: int
    end: int
    token_count: int
    #: The whole indexed text; kept out of ``repr`` for readability.
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
    to the original slice and their token counts cover every character.
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

    Counts include each span's trailing whitespace. Summed counts are not an
    upper bound on the joined slice: ``o200k_base`` groups digits as
    ``\p{N}{1,3}``, so ``", 8,72,6  "`` (8) plus ``"217"`` (1) joins to 10.
    :func:`_to_windows` therefore re-measures and bisects, which is what makes
    ``max_tokens`` a guarantee.

    Raises:
        ValueError: If ``max_tokens`` is less than 1.
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
        # ``extend`` consumes the generator incrementally, so ``len(windows)``
        # read inside it would count this candidate's earlier pieces.
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
    """Strip the XBRL prologue, gate the whole document, then window it.

    The keyword gate applies to the whole stripped document, not per window:
    a filing that passes keeps all its windows. These are the windows stage 1
    scores; see docs/sixk-two-stage-triage.md for why the order matters.

    Args:
        text: Extracted document text.
        target_tokens: Window size; see :func:`split_into_windows`.
        keywords: Gate vocabulary; see :data:`DEBT_KEYWORDS`.

    Returns:
        Windows for a filing that passes the gate, in document order; empty when
        the gate rejects it or nothing is left to window.

    >>> len(prepare_filing("The Company issued senior notes due 2030 today."))
    1
    >>> prepare_filing("The Company declared a quarterly dividend of $0.10.")
    []
    """
    body = gated_body(text, keywords=keywords)
    if body is None:
        return []
    return split_into_windows(body, target_tokens=target_tokens)


def gated_body(text: str, *, keywords: tuple[str, ...] = DEBT_KEYWORDS) -> str | None:
    """Return the prologue-stripped text windows index into, or None if gated out.

    >>> gated_body("The Company issued senior notes due 2030 today.")
    'The Company issued senior notes due 2030 today.'
    >>> gated_body("The Company declared a quarterly dividend of $0.10.") is None
    True
    """
    body = strip_inline_xbrl_prologue(text)
    if not has_debt_keyword(body, keywords=keywords):
        return None
    return body


def body_digest(body: str) -> str:
    """Return the SHA-256 hex digest of the text a document's windows index into.

    Stored with each window span, so a reader that rebuilds the text can tell
    whether it rebuilt the same text the spans were cut from.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def window_from_span(
    body: str, *, index: int, start: int, end: int, token_count: int
) -> TextWindow:
    """Rebuild a stored window span over the text it was cut from.

    >>> window_from_span("abc def", index=0, start=4, end=7, token_count=1).text
    'def'
    """
    return TextWindow(
        index=index,
        text=body[start:end],
        start=start,
        end=end,
        token_count=token_count,
        source=body,
    )


# --- Post-admission expansion -------------------------------------------------
#
# Everything below runs on the windows stage 1 admitted, never on the windows it
# scores: the stage-1 model and threshold are calibrated on the unexpanded crop.

#: Tokens of context prepended to an admitted window unless a section header is
#: reached sooner. Chosen empirically; see docs/sixk-two-stage-triage.md.
MIN_EXPANSION_TOKENS = 200

#: Hard cap on the context prepended to one admitted window, for when the walk
#: backwards finds no header or blank line (e.g. unbroken table rows).
MAX_EXPANSION_TOKENS = 400

#: Ceiling on the merged-window estimate, matching the largest snippet the 8-K
#: path sends the extractor. The estimate counts the first member's context,
#: each member's own tokens and the gap text a later member's expansion pulls
#: in to reach the span, so it is the merged text's size up to tokenization at
#: the joins.
MAX_MERGED_TOKENS = 2_000

#: Longest a line can be and still read as a heading rather than a sentence.
MAX_HEADER_WORDS = 12

#: Characters a paragraph-boundary test looks back through.
_PARAGRAPH_LOOKBACK = 200

#: Characters scanned per token of budget when collecting candidate stops. A
#: bound on the search only; the token budget decides how far the walk goes.
#: Generous because table text has far fewer characters per token than prose.
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

    ``window`` is the text stage 2 and extraction see; its ``index`` is the
    first member's. ``member_indices`` holds the indices of every admitted
    window it covers, in document order. A merged window may exceed
    :data:`WINDOW_TOKENS` but not, up to tokenization at the joins,
    :data:`MAX_MERGED_TOKENS`.
    """

    window: TextWindow
    member_indices: tuple[int, ...]


def _is_section_header(line: str, *, isolated: bool) -> bool:
    """Return whether a line reads as a section header rather than body text.

    A header has at most :data:`MAX_HEADER_WORDS` words. An explicitly numbered
    heading (``Item 5.02``, ``Note 12``) qualifies on its wording; otherwise
    the line must be isolated, contain no digits, not end in
    ``.``/``,``/``;``, and be upper case, end in ``:``, or be title case --
    isolation is what rejects table cells, which extract one per line and read
    like headings. Ambiguous lines return ``False``: a missed header only lets
    the walk continue to its token minimum, while a false one stops it early.

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
    # A digit outside a numbered heading marks a table row, whose real header
    # sits above it.
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

    Candidates run nearest-first, so the span grows along the list and the
    boundary is found by bisection. BPE merges make nested-span counts only
    approximately monotone; an off-by-one token at the edge is acceptable.
    ``None`` when even the nearest candidate exceeds the budget.
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

    Bisects like :func:`_farthest_within_budget`; ``None`` when even the
    farthest candidate falls short.
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

    The walk backwards stops at the first of: the nearest section header; the
    nearest blank line once ``min_tokens`` are taken; the line boundary where
    ``min_tokens`` was met. Nothing crosses ``max_tokens``.

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
        # Everything reachable is shorter than the minimum: take all of it.
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

    The estimate sums the first member's prepended context, each member's own
    count and the gap text between the span and each later member, rather than
    re-measuring the whole join, which would make a long run quadratic.
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

    def merged(self: _MergedSpan, window: TextWindow, *, gap: int) -> _MergedSpan:
        """Return this span extended over a window ``gap`` tokens beyond its end."""
        return _MergedSpan(
            start=self.start,
            end=max(self.end, window.end),
            members=(*self.members, window.index),
            tokens=self.tokens + gap + window.token_count,
        )


def expand_admitted_windows(
    windows: Sequence[TextWindow],
    *,
    min_tokens: int = MIN_EXPANSION_TOKENS,
    max_tokens: int = MAX_EXPANSION_TOKENS,
    max_merged_tokens: int = MAX_MERGED_TOKENS,
) -> list[ExpandedWindow]:
    r"""Prepend context to the windows stage 1 admitted, merging what overlaps.

    Runs between stage 1 and stage 2, never before stage 1. Each window is
    expanded backwards (see :func:`_expansion_start`); a window whose expansion
    reaches or abuts the previous span joins it while the merged estimate stays
    within ``max_merged_tokens``, and otherwise starts a new span.

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
        # Overlap slices to "", so overlapping and whitespace-separated
        # expansions both count as adjacent.
        adjacent = bool(spans) and not source[spans[-1].end : start].strip()
        # The unadmitted text between the span and this window joins the span
        # with it, so it counts toward the ceiling.
        gap = (
            count_tokens(source[spans[-1].end : window.start])
            if adjacent and window.start > spans[-1].end
            else 0
        )
        if (
            adjacent
            and spans[-1].tokens + gap + window.token_count <= max_merged_tokens
        ):
            spans[-1] = spans[-1].merged(window, gap=gap)
        else:
            # A window opening a run cut by the budget still expands, so up to
            # ``max_tokens`` of text is sent twice rather than starting cold.
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


KEEP_TYPE_RE = re.compile(
    r"^(6-K(/A)?|EX-99(\.\d+)?|EX-1(\.\d+)?|EX-4(\.\d+)?|EX-10(\.\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SixkDocument:
    """One prose document from a 6-K submission."""

    #: The submission's own ``<TYPE>`` label, e.g. ``6-K`` or ``EX-99.1``.
    document_type: str
    #: Flattened plain text, one line per normalized body line.
    text: str


def prose_documents(submission: str) -> list[SixkDocument]:
    r"""Return the flattened prose documents of one complete submission.

    Keeps documents whose ``<TYPE>`` matches :data:`KEEP_TYPE_RE` and whose
    text is not blank, in submission order, so a document's index is a stable
    part of a snippet's identity. The whole ``<DOCUMENT>`` block is flattened,
    not just its ``<TEXT>``, so the text opens with the type, sequence and
    filename lines; the triage stages were evaluated on text in that form. The
    inline-XBRL prologue is left for :func:`cdt.segmenter.sixk.prepare_filing` to strip.

    >>> submission = (
    ...     "<DOCUMENT><TYPE>6-K\n<TEXT><p>The Company issued notes.</p></TEXT>"
    ...     "</DOCUMENT>"
    ...     "<DOCUMENT><TYPE>GRAPHIC\n<TEXT>begin 644 logo.jpg</TEXT></DOCUMENT>"
    ... )
    >>> documents = prose_documents(submission)
    >>> [document.document_type for document in documents]
    ['6-K']
    >>> documents[0].text
    '6-K\nThe Company issued notes.'
    """
    documents: list[SixkDocument] = []
    for match in DOCUMENT_RE.finditer(submission):
        block = match.group(1)
        type_match = TYPE_RE.search(block)
        document_type = type_match.group(1).strip() if type_match else ""
        if not KEEP_TYPE_RE.match(document_type):
            continue
        text = "\n".join(line.text for line in normalize_body_lines(block))
        if text.strip():
            documents.append(SixkDocument(document_type=document_type, text=text))
    return documents


# --- The segment stage --------------------------------------------------------
#
# Persists each gated document's windows as spans: no text. The classify stage
# rebuilds the text from the source submission, which it reads anyway to expand
# admitted windows, and checks ``source_sha256`` to know it rebuilt the text
# the spans were cut from.

#: Completion-registry and run-manifest name of the 6-K segment stage.
SEGMENT_STAGE_NAME = "sixk-segment"

SIXK_WINDOW_COLUMNS = [
    "accession_number",
    "cik",
    "date",
    # Position among the submission's prose documents (prose_documents order).
    "document_index",
    # The document's own <TYPE>: 6-K, EX-99.1, ...
    "document_type",
    "window",
    # Character offsets into the document's gated body (gated_body).
    "start",
    "end",
    "token_count",
    # body_digest of that gated body.
    "source_sha256",
]
SIXK_WINDOW_INTEGER_COLUMNS = [
    "document_index",
    "window",
    "start",
    "end",
    "token_count",
]


def window_rows_for_submission(
    document: dict[str, object], submission: str
) -> list[dict[str, object]]:
    """Return one span row per window of a submission's gated prose documents.

    ``document`` is the filing's documents row; a document the keyword gate
    rejects contributes no rows.
    """
    rows: list[dict[str, object]] = []
    for document_index, prose in enumerate(prose_documents(submission)):
        body = gated_body(prose.text)
        if body is None:
            continue
        digest = body_digest(body)
        for window in split_into_windows(body):
            rows.append(
                {
                    "accession_number": document["accession_number"],
                    "cik": document.get("cik"),
                    "date": document.get("date"),
                    "document_index": document_index,
                    "document_type": prose.document_type,
                    "window": window.index,
                    "start": window.start,
                    "end": window.end,
                    "token_count": window.token_count,
                    "source_sha256": digest,
                }
            )
    return rows


def window_documents(
    documents: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    s3_client: object | None = None,
) -> pd.DataFrame:
    """Window in-memory 6-K document rows into span rows (SIXK_WINDOW_COLUMNS)."""
    rows: list[dict[str, object]] = []
    for record in documents.to_dict("records"):
        submission = document_text_for_record(
            record, data_dir=data_dir, s3_client=s3_client
        )
        rows.extend(window_rows_for_submission(record, submission))
    table = pd.DataFrame(rows, columns=SIXK_WINDOW_COLUMNS)
    for column in SIXK_WINDOW_INTEGER_COLUMNS:
        table[column] = pd.to_numeric(table[column]).astype("Int64")
    return table


def segment_pending_sixk_documents(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    s3_client: object | None = None,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Window pending 6-K documents partitions into ``sixk-windows`` partitions.

    A documents partition is pending when its fingerprint changed since it was
    last segmented, or always with ``force``, and is recomputed whole. Returns
    the span rows written this run.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    shared_client = s3_client

    def process(source_path: str, partition: dict[str, str]) -> PartitionOutput:
        nonlocal shared_client
        del partition
        documents = read_table(source_path, DOCUMENT_COLUMNS).reindex(
            columns=DOCUMENT_COLUMNS
        )
        shared_client = ensure_s3_client(shared_client, documents.to_dict("records"))
        return PartitionOutput(
            rows=window_documents(
                documents, data_dir=data_dir, s3_client=shared_client
            ),
            source_rows=len(documents),
        )

    return run_partition_stage(
        SEGMENT_STAGE_NAME,
        source_dataset=SIXK_DOCUMENT_DATASET_NAME,
        output_dataset=SIXK_WINDOW_DATASET_NAME,
        output_columns=SIXK_WINDOW_COLUMNS,
        process=process,
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
        renew=renew,
        manifest_extra={"window_tokens": WINDOW_TOKENS},
    ).rows
