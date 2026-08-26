"""Pure Form 8-K item text extraction logic."""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from typing import Any, Self

PREFIX = "ITEM INFORMATION:"
DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TYPE_RE = re.compile(r"<TYPE>\s*([^\n\r<]+)", re.IGNORECASE)
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)
SEC_HEADER_END = "</SEC-HEADER>"
ITEM_NUMBER_RE = re.compile(r"\b(\d)\s*\.\s*(\d)\s*(\d)\b")
SECTION_SIMILARITY_THRESHOLD = 0.95
TERMINAL_RE = re.compile(
    r"^\s*(SIGNATURES?|EXHIBIT\s+INDEX)\s*$",
    re.IGNORECASE,
)
NON_HEADING_PHRASES = (
    "incorporated by reference",
    "shall not be deemed",
    "is not deemed",
    "is being furnished",
    "is being filed",
    "of this current report",
    "of this report",
    "of this form 8-k",
)

ITEM_NAME_TO_NUMBER = {
    "entry into a material definitive agreement": "1.01",
    "termination of a material definitive agreement": "1.02",
    "bankruptcy or receivership": "1.03",
    "mine safety - reporting of shutdowns and patterns of violations": "1.04",
    "material cybersecurity incidents": "1.05",
    "completion of acquisition or disposition of assets": "2.01",
    "results of operations and financial condition": "2.02",
    "creation of a direct financial obligation or an obligation under an off-balance sheet arrangement of a registrant": "2.03",
    "triggering events that accelerate or increase a direct financial obligation or an obligation under an off-balance sheet arrangement": "2.04",
    "costs associated with exit or disposal activities": "2.05",
    "cost associated with exit or disposal activities": "2.05",
    "material impairments": "2.06",
    "notice of delisting or failure to satisfy a continued listing rule or standard; transfer of listing": "3.01",
    "unregistered sales of equity securities": "3.02",
    "material modifications to rights of security holders": "3.03",
    "changes in registrant's certifying accountant": "4.01",
    "non-reliance on previously issued financial statements or a related audit report or completed interim review": "4.02",
    "changes in control of registrant": "5.01",
    "departure of directors or certain officers; election of directors; appointment of certain officers: compensatory arrangements of certain officers": "5.02",
    "amendments to articles of incorporation or bylaws; change in fiscal year": "5.03",
    "temporary suspension of trading under registrant's employee benefit plans": "5.04",
    "amendments to the registrant's code of ethics, or waiver of a provision of the code of ethics": "5.05",
    "change in shell company status": "5.06",
    "submission of matters to a vote of security holders": "5.07",
    "shareholder director nominations": "5.08",
    "shareholder nominations pursuant to exchange act rule 14a-11": "5.08",
    "abs informational and computational material": "6.01",
    "change of servicer or trustee": "6.02",
    "change in credit enhancement or other external support": "6.03",
    "failure to make a required distribution": "6.04",
    "securities act updating disclosure": "6.05",
    "regulation fd disclosure": "7.01",
    "other events": "8.01",
    "financial statements and exhibits": "9.01",
}


@dataclass(frozen=True)
class DocumentText:
    """Complete text and metadata for one SEC document.

    Attributes:
        accession_number: Normalized SEC accession number.
        cik: SEC Central Index Key with no leading zeros.
        company_name: Filing issuer display name from the source manifest.
        url: Source SEC URL for the document.
        text: Complete submission text.
        date: Filing date in ISO ``YYYY-MM-DD`` format.
    """

    accession_number: str
    cik: str
    company_name: str
    url: str
    text: str
    date: str


@dataclass(frozen=True)
class BodyLine:
    """A normalized plain-text body line.

    Attributes:
        line_number: One-based line number in the normalized body text.
        text: Collapsed plain-text content for the line.
    """

    line_number: int
    text: str


@dataclass(frozen=True)
class Heading:
    """A candidate 8-K item heading.

    Attributes:
        line_index: Zero-based index in the normalized body line list.
        line_number: One-based source line number from ``BodyLine``.
        text: Heading text as it appears after normalization.
        item_numbers: Item numbers represented by the heading.
    """

    line_index: int
    line_number: int
    text: str
    item_numbers: tuple[str, ...]


@dataclass(frozen=True)
class SectionCandidate:
    """A candidate section extracted from one heading.

    Attributes:
        heading: Heading used as the section start.
        text: Original normalized section text selected for output.
        normalized_text: Further-normalized text used for comparison.
        start_line: One-based start line for the selected section.
        end_line: One-based end line for the selected section.
    """

    heading: Heading
    text: str
    normalized_text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ItemSection:
    """Extracted item section for one ``ITEM INFORMATION`` value.

    Attributes:
        accession_number: SEC accession number for the source document.
        cik: SEC Central Index Key for the source filer.
        company_name: Filing issuer display name for the source filer.
        date: Filing date in ISO ``YYYY-MM-DD`` format.
        url: Source SEC URL for the document.
        item_information: Normalized ``ITEM INFORMATION`` caption.
        item_number: Form 8-K item number, or an empty string if unmapped.
        heading_count: Number of matching item headings found in the body.
        extraction_status: Extraction status label.
        duplicate_resolution: Duplicate-heading resolution label, if any.
        selected_heading_index: One-based selected heading index, if any.
        section_heading: Heading text selected for the section.
        section_text: Extracted item section text.
        start_line: One-based section start line, if available.
        end_line: One-based section end line, if available.
        section_char_count: Character count for ``section_text``.
    """

    accession_number: str
    cik: str
    company_name: str
    date: str
    url: str
    item_information: str
    item_number: str
    heading_count: int
    extraction_status: str
    duplicate_resolution: str
    selected_heading_index: int | str
    section_heading: str
    section_text: str
    start_line: int | str
    end_line: int | str
    section_char_count: int

    def to_dict(self: Self) -> dict[str, Any]:
        """Return this item section as a plain dictionary.

        Returns:
            Dictionary representation suitable for dataframe construction.
        """
        return asdict(self)


def extract_items_from_document(document: DocumentText) -> list[ItemSection]:
    """Extract item sections from one complete 8-K submission.

    Args:
        document: Complete SEC document text and metadata.

    Returns:
        Item sections corresponding to ``ITEM INFORMATION`` header values.
    """
    lines = normalize_body_lines(primary_8k_body(document.text))
    headings = item_headings(lines)
    rows = []
    # item_id is accession + item_number, so a header repeating an ITEM
    # INFORMATION line (which SEC headers do produce) or two labels mapping to
    # one number would emit duplicate primary keys and corrupt downstream joins
    # (#74). Keep the first occurrence of each key.
    seen_keys: set[str] = set()
    for item_information in iter_item_information_values(document.text):
        item_number = ITEM_NAME_TO_NUMBER.get(item_information)
        key = item_number or item_information
        if key in seen_keys:
            continue
        seen_keys.add(key)
        section = (
            extract_section(lines, headings, item_number)
            if item_number is not None
            else unmapped_item_section()
        )
        rows.append(
            ItemSection(
                accession_number=document.accession_number,
                cik=document.cik,
                company_name=document.company_name,
                date=document.date,
                url=document.url,
                item_information=item_information,
                item_number=item_number or "",
                **section,
            )
        )
    return rows


def iter_item_information_values(text: object) -> list[str]:
    """Return normalized ITEM INFORMATION values from one document text.

    Args:
        text: Complete SEC submission text.

    Returns:
        Case-folded and whitespace-stripped ``ITEM INFORMATION:`` values.
    """
    if not isinstance(text, str):
        return []

    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.casefold().startswith(PREFIX.casefold()):
            values.append(stripped[len(PREFIX) :].strip().casefold())

    return values


def primary_8k_body(text: object) -> str:
    """Return the primary 8-K body text from a complete SEC submission.

    Args:
        text: Complete SEC submission text.

    Returns:
        First ``<DOCUMENT>`` block whose ``<TYPE>`` is ``8-K``. If no
        structured block is found, returns content after ``</SEC-HEADER>`` or
        the original text.
    """
    if not isinstance(text, str):
        return ""

    for document_match in DOCUMENT_RE.finditer(text):
        document = document_match.group(1)
        type_match = TYPE_RE.search(document)
        if type_match and type_match.group(1).strip().casefold() == "8-k":
            text_match = TEXT_RE.search(document)
            return text_match.group(1) if text_match else document

    header_end = text.upper().find(SEC_HEADER_END)
    if header_end != -1:
        return text[header_end + len(SEC_HEADER_END) :]

    return text


def normalize_body_lines(body: str) -> list[BodyLine]:
    """Normalize filing body text while preserving likely item boundaries.

    Args:
        body: Primary 8-K body text, usually still containing HTML tags.

    Returns:
        Non-empty plain-text lines with collapsed whitespace and approximate
        source line numbers.
    """
    body = html.unescape(body)
    body = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", body)
    body = re.sub(r"(?i)</\s*(p|div|tr|td|th|li|h[1-6])\s*>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)

    lines = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        collapsed = re.sub(r"\s+", " ", line).strip()
        if collapsed:
            lines.append(BodyLine(line_number=line_number, text=collapsed))
    return lines


def normalize_item_number(text: str) -> str:
    """Normalize split item number formatting in text.

    Args:
        text: Heading or body line that may contain a split item number.

    Returns:
        Text with split item numbers normalized to ``N.NN``.
    """
    return ITEM_NUMBER_RE.sub(r"\1.\2\3", text)


def leading_item_numbers(line: str) -> tuple[str, ...]:
    """Return item numbers if a line looks like an item heading.

    Args:
        line: Normalized body line to inspect.

    Returns:
        Tuple of item numbers when the line starts like an item heading.
    """
    normalized = normalize_item_number(line)
    normalized_casefold = normalized.casefold()
    if any(phrase in normalized_casefold for phrase in NON_HEADING_PHRASES):
        return ()

    # Every 8-K item number is X.0Y, and a number followed by '%' is a coupon
    # rate, not a heading — '5.25% Senior Notes due 2029' as a body line (HTML
    # table cells become their own lines) otherwise truncated the enclosing
    # item section right at the debt text this pipeline targets (#63).
    item_match = re.match(r"^\s*Item\b(?P<rest>.*)$", normalized, re.IGNORECASE)
    if item_match:
        return tuple(re.findall(r"\b\d\.0\d\b(?!\s*%)", item_match.group("rest")))

    bare_match = re.match(
        r"^\s*(?P<number>\d\.0\d)(?!\s*%)\s*(?:[.:;\-)]|\b)",
        normalized,
        re.IGNORECASE,
    )
    if bare_match:
        return (bare_match.group("number"),)

    return ()


def is_subitem_heading(line: str, item_number: str) -> bool:
    """Return whether a line is a subordinate item heading.

    Args:
        line: Normalized body line to inspect.
        item_number: Parent item number, such as ``4.01``.

    Returns:
        ``True`` if the line starts with the item followed by a parenthetical.
    """
    normalized = normalize_item_number(line)
    return bool(
        re.match(
            rf"^\s*(?:Item\s+)?{re.escape(item_number)}\s*\([a-z]\)",
            normalized,
            re.IGNORECASE,
        )
    )


def item_headings(lines: list[BodyLine]) -> list[Heading]:
    """Return item heading candidates, excluding table-of-contents entries.

    Args:
        lines: Normalized body lines.

    Returns:
        Candidate item headings after table-of-contents filtering.
    """
    headings = []
    for line_index, line in enumerate(lines):
        numbers = leading_item_numbers(line.text)
        if numbers:
            headings.append(
                Heading(
                    line_index=line_index,
                    line_number=line.line_number,
                    text=line.text,
                    item_numbers=numbers,
                )
            )
    return filter_table_of_contents_headings(lines, headings)


def table_of_contents_indices(lines: list[BodyLine]) -> list[int]:
    """Return normalized line indices that mark a table of contents.

    Args:
        lines: Normalized body lines.

    Returns:
        Zero-based indices for lines containing ``table of contents``.
    """
    return [
        line_index
        for line_index, line in enumerate(lines)
        if "table of contents" in line.text.casefold()
    ]


def filter_table_of_contents_headings(
    lines: list[BodyLine],
    headings: list[Heading],
) -> list[Heading]:
    """Remove repeated table-of-contents heading entries.

    Args:
        lines: Normalized body lines.
        headings: Candidate headings before filtering.

    Returns:
        Candidate headings with repeated table-of-contents entries removed.
    """
    toc_indices = table_of_contents_indices(lines)
    if not toc_indices:
        return headings

    skip_heading_positions = set()
    for toc_index in toc_indices:
        first_after_toc: dict[str, int] = {}
        for heading_position, heading in enumerate(headings):
            if heading.line_index <= toc_index:
                continue
            for number in heading.item_numbers:
                first_after_toc.setdefault(number, heading_position)

        for number, heading_position in first_after_toc.items():
            if any(
                number in later_heading.item_numbers
                for later_heading in headings[heading_position + 1 :]
                if later_heading.line_index > toc_index
            ):
                skip_heading_positions.add(heading_position)

    return [
        heading
        for heading_position, heading in enumerate(headings)
        if heading_position not in skip_heading_positions
    ]


def primary_item_headings(headings: list[Heading], item_number: str) -> list[Heading]:
    """Return headings used for item-level extraction.

    Args:
        headings: Candidate headings for a document body.
        item_number: Item number to select.

    Returns:
        Top-level headings for the item, or the first subordinate heading.
    """
    matches = [heading for heading in headings if item_number in heading.item_numbers]
    top_level_matches = [
        heading
        for heading in matches
        if not is_subitem_heading(heading.text, item_number)
    ]
    if top_level_matches:
        return top_level_matches
    return matches[:1]


def normalized_section_text(section_text: str, item_number: str) -> str:
    """Normalize section text for duplicate comparison.

    Args:
        section_text: Candidate section text.
        item_number: Item number whose heading should be ignored.

    Returns:
        Comparison-only normalized payload.
    """
    lines = []
    for line in section_text.splitlines():
        normalized = normalize_item_number(line).casefold()
        normalized = re.sub(r"[^\w.]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized or normalized.isdigit():
            continue
        if re.fullmatch(rf"(?:item\s+)?{re.escape(item_number)}\.?", normalized):
            continue
        lines.append(normalized)
    return " ".join(lines)


def equivalent_sections(left: str, right: str) -> bool:
    """Return whether two normalized section payloads are equivalent.

    Args:
        left: Normalized text for the first section candidate.
        right: Normalized text for the second section candidate.

    Returns:
        ``True`` when payloads contain each other or have high token overlap.
    """
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= SECTION_SIMILARITY_THRESHOLD


def section_candidate(
    lines: list[BodyLine],
    headings: list[Heading],
    heading: Heading,
    item_number: str,
) -> SectionCandidate:
    """Build one section candidate from a heading.

    Args:
        lines: Normalized body lines.
        headings: Candidate item headings in the body.
        heading: Heading to use as the section start.
        item_number: Item number being extracted.

    Returns:
        Candidate section text and metadata for duplicate resolution.
    """
    end_index = section_end_index(lines, headings, heading, item_number)
    section_lines = lines[heading.line_index : end_index]
    section_text = "\n".join(line.text for line in section_lines).strip()
    end_line = section_lines[-1].line_number if section_lines else heading.line_number
    return SectionCandidate(
        heading=heading,
        text=section_text,
        normalized_text=normalized_section_text(section_text, item_number),
        start_line=heading.line_number,
        end_line=end_line,
    )


def duplicate_resolution(
    candidates: list[SectionCandidate],
) -> tuple[bool, str, SectionCandidate | None]:
    """Return whether duplicate section candidates are benign.

    Args:
        candidates: Candidate sections for one item number.

    Returns:
        Tuple of ``(is_resolved, resolution_label, selected_candidate)``.
    """
    if not candidates:
        return False, "", None
    if len(candidates) == 1:
        return True, "single_heading", candidates[0]

    for candidate in candidates:
        if all(
            candidate is other
            or equivalent_sections(candidate.normalized_text, other.normalized_text)
            for other in candidates
        ):
            return True, "benign_equivalent", candidate

    for candidate in candidates:
        if all(
            candidate is other or other.normalized_text in candidate.normalized_text
            for other in candidates
        ):
            return True, "benign_contained", candidate

    return False, "unresolved_duplicate", candidates[0]


def terminal_line_index(lines: list[BodyLine], start_index: int) -> int | None:
    """Return the terminal line index after a section start, if present.

    Args:
        lines: Normalized body lines.
        start_index: Zero-based index of the section heading line.

    Returns:
        Index for ``SIGNATURES`` or ``EXHIBIT INDEX``, or ``None``.
    """
    for line_index in range(start_index + 1, len(lines)):
        if TERMINAL_RE.match(lines[line_index].text):
            return line_index
    return None


def section_end_index(
    lines: list[BodyLine],
    headings: list[Heading],
    start_heading: Heading,
    item_number: str,
) -> int:
    """Return the exclusive line index where an item section ends.

    Args:
        lines: Normalized body lines.
        headings: Candidate item headings in the body.
        start_heading: Heading used as the section start.
        item_number: Item number being extracted.

    Returns:
        Exclusive zero-based line index for the section end.
    """
    terminal_index = terminal_line_index(lines, start_heading.line_index)
    for heading in headings:
        is_next_item = heading.line_index > start_heading.line_index and any(
            number != item_number for number in heading.item_numbers
        )
        if is_next_item:
            if terminal_index is not None:
                return min(heading.line_index, terminal_index)
            return heading.line_index

    return terminal_index if terminal_index is not None else len(lines)


def extract_section(
    lines: list[BodyLine],
    headings: list[Heading],
    item_number: str,
) -> dict[str, Any]:
    """Extract section text for one item number.

    Args:
        lines: Normalized body lines.
        headings: Candidate item headings in the body.
        item_number: Item number to extract.

    Returns:
        Extraction status, selected heading metadata, and section text fields.
    """
    matches = primary_item_headings(headings, item_number)
    if not matches:
        return missing_heading_section()

    candidates = [
        section_candidate(lines, headings, heading, item_number) for heading in matches
    ]
    is_resolved, resolution, selected_candidate = duplicate_resolution(candidates)
    if selected_candidate is None:
        selected_candidate = candidates[0]
    status = "ok" if is_resolved else "duplicate_heading"
    selected_heading_index = candidates.index(selected_candidate) + 1

    return {
        "heading_count": len(matches),
        "extraction_status": status,
        "duplicate_resolution": resolution,
        "selected_heading_index": selected_heading_index,
        "section_heading": selected_candidate.heading.text,
        "section_text": selected_candidate.text,
        "start_line": selected_candidate.start_line,
        "end_line": selected_candidate.end_line,
        "section_char_count": len(selected_candidate.text),
    }


def missing_heading_section() -> dict[str, Any]:
    """Return section metadata for a mapped item with no matching heading.

    Returns:
        Empty section fields with ``missing_heading`` status.
    """
    return empty_section("missing_heading")


def unmapped_item_section() -> dict[str, Any]:
    """Return section metadata for an unmapped ``ITEM INFORMATION`` value.

    Returns:
        Empty section fields with ``unmapped_item_information`` status.
    """
    return empty_section("unmapped_item_information")


def empty_section(status: str) -> dict[str, Any]:
    """Return empty section metadata for a non-extracted item.

    Args:
        status: Extraction status to assign.

    Returns:
        Empty section fields using ``status``.
    """
    return {
        "heading_count": 0,
        "extraction_status": status,
        "duplicate_resolution": "",
        "selected_heading_index": "",
        "section_heading": "",
        "section_text": "",
        "start_line": "",
        "end_line": "",
        "section_char_count": 0,
    }
