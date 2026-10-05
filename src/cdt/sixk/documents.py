"""Split a 6-K complete submission into the documents worth extracting from.

A 6-K has no item structure, so its unit of text is the document: the 6-K
body and the exhibits that carry prose. Documents are flattened with the
itemizer's :func:`cdt.itemizer.extract.normalize_body_lines`, so both genres
reach the extractor through the same normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cdt.itemizer.extract import DOCUMENT_RE, TYPE_RE, normalize_body_lines

#: Document types worth extracting from: the 6-K body and the exhibit families
#: that carry agreement text.
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
    inline-XBRL prologue is left for :func:`cdt.sixk.prepare_filing` to strip.

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
