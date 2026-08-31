"""Preprocessing for Form 6-K documents ahead of the two-stage triage.

Ported from the ``uchicago-dsi/commercial-debt-tracker-models`` research repo,
where the window size, the keyword gate and the inline-XBRL rule were each
chosen against labelled data. The pieces belong together because they only make
sense as a sequence: strip the XBRL padding, gate on debt vocabulary, then cut
the survivor into the windows the classifier was trained on.

Windows are 400 tokens, not the 2,000 the 8-K path uses. Only 1.2% of positive
windows proved context-dependent at that size, and the smaller crop cut
extraction tokens to 0.34x while still matching 146 of 154 known mentions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tiktoken import Encoding

#: Encoding used for every token count in this module.
TIKTOKEN_ENCODING_NAME = "o200k_base"

#: Window size the shipped stage-1 model was trained on. Changing this
#: invalidates the model and its calibrated threshold together.
CHILD_WINDOW_TOKENS = 400

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

    ``text`` always equals ``source[start:end]`` for the document it came from.
    """

    index: int
    text: str
    start: int
    end: int
    token_count: int


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
            )
            for offset, (piece_start, piece_end, piece_tokens) in enumerate(pieces)
        )
    return windows


def split_into_windows(
    text: str,
    *,
    target_tokens: int = CHILD_WINDOW_TOKENS,
) -> list[TextWindow]:
    """Split a text into contiguous, non-overlapping classifier windows.

    Windows are cut on paragraph boundaries where possible, falling back to
    lines, then sentences, then a character bisection, so no window exceeds
    ``target_tokens``.

    Args:
        text: Document text to split.
        target_tokens: Largest allowed window size in tokens. The shipped
            stage-1 model was fitted at :data:`CHILD_WINDOW_TOKENS`, so moving
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
