"""Split a 6-K complete submission into the documents worth extracting from.

A submission is a container: the 6-K body, its exhibits, and a pile of
artifacts that carry no prose — graphics, XBRL instances, cover-page shells.
The 8-K path never needed this because the itemizer works on the whole
submission and finds items by heading; a 6-K has no item structure, so the
genre's unit of text is the document.

Flattening reuses the itemizer's HTML handling
(:func:`cdt.itemizer.extract.normalize_body_lines`) rather than a second
implementation: both genres then reach the extractor through the same
normalization, which is what makes their measured behaviour comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cdt.itemizer.extract import DOCUMENT_RE, TYPE_RE, normalize_body_lines

#: Document types worth extracting from: the 6-K body and the exhibit families
#: that carry agreement text. Everything else in a submission — graphics, XBRL
#: instances, cover shells — has no prose to window.
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

    Documents keep submission order, so a document's index is a stable part of a
    snippet's identity.

    Two fidelity notes, because this is the code path whose output the
    generalization eval scored and small changes here move the measured numbers:

    - The whole ``<DOCUMENT>`` block is flattened, not just its ``<TEXT>``, so a
      document's first lines are its own ``<TYPE>``, sequence, filename and
      description. That is noise, and it is the noise stage 1 and stage 2 were
      measured against.
    - The inline-XBRL prologue is *not* stripped here. It is stripped by
      :func:`cdt.sixk.prepare_filing`, where the documented step order puts it.
      Doing it in both places changes nothing — stripping is idempotent, since
      stripped text begins at prose and a second pass finds no prologue to
      measure — which is why the research harness could do both harmlessly.

    In a raw docstring, so these escapes are the ones doctest evaluates:

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
