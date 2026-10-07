"""SEC submission text shared by both genres: document blocks, type headers and normalized body lines."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

PREFIX = "ITEM INFORMATION:"
DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TYPE_RE = re.compile(r"<TYPE>\s*([^\n\r<]+)", re.IGNORECASE)
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)
SEC_HEADER_END = "</SEC-HEADER>"


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
