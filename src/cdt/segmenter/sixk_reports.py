"""Form 6-K document kind: periodic financial reports versus event documents.

A 6-K carries whatever the issuer furnished that month. Some documents announce
one transaction -- a notes offering, a tender result, a credit facility, an
indenture -- the way an 8-K item does. Others are periodic reports: financial
statements, MD&A, annual and interim reports, results releases and
presentations, shareholder-meeting materials. The extractor reads event
documents well; a periodic report's debt schedules come out as one unmatched
"instrument" per table row and reporting period. The 8-K path has the same
boundary, since it never segments Item 2.02 or a 10-K/10-Q.

:func:`classify_sixk_document` decides the kind from the gated body the
windows index into (:func:`cdt.segmenter.sixk.gated_body`) and the document's
``<TYPE>``. The segment stage windows event documents only. Rules and their
measured accuracy: ``docs/sixk-two-stage-triage.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: Exhibit types that are always transaction documents: underwriting and
#: distribution agreements (EX-1), instruments and indentures (EX-4), material
#: contracts (EX-10).
TRANSACTION_EXHIBIT_TYPE = re.compile(r"(?i)^EX-(?:1|4|10)(?:\.|$)")

#: A line of the 6-K cover page, which precedes the content of a main 6-K
#: document and says nothing about its kind.
_COVER_LINE = re.compile(
    r"""(?ix)
    securities\s+and\s+exchange\s+commission | washington,?\s+d\.?c
    | form\s+6-k | report\s+of\s+foreign\s+private\s+issuer | rule\s+13a-16 | 15d-16
    | commission\s+file\s+number | form\s+20-f | form\s+40-f | regulation\s+s-t
    | indicate\s+by\s+check\s+mark | exact\s+name\s+of\s+registrant
    | translation\s+of\s+registrant | address\s+of\s+principal | for\s+the\s+month\s+of
    | securities\s+exchange\s+act\s+of\s+1934 | 12g3-2 | signature | \(check\s+one\)
    | furnishing\s+(?:the\s+)?information | incorporat(?:ed|ion)\s+by\s+reference
    """
)

#: Characters of non-cover text in which a transaction-document title counts.
AGREEMENT_TITLE_CHARS = 400

#: Characters of non-cover text in which a report title counts.
REPORT_TITLE_CHARS = 1_000

#: A transaction document's title. Underwriting and credit agreements, consent
#: solicitations and offering tables cite the issuer's financial statements, so
#: an agreement title ahead of a report title decides.
_AGREEMENT_TITLE = re.compile(
    r"""(?ix)
    \b(?:credit|loan|facility|facilities|underwriting|purchase|subscription
       |arrangement|merger|amalgamation|cooperation|implementation|sales
       |distribution|dealer|agency|guarantee|guaranty|security|intercreditor
       |amending|amendment|commitment)
      \s+agreement\b
    | \b(?:supplemental\s+)?indenture\b | \bexecution\s+(?:version|copy)\b
    | \bconformed\s+copy\b | \bas\s+borrowers?\b | \bglobal\s+(?:note|security)\b
    | \bofficer['’]?s['’]?\s+certificate\b | \bpricing\s+supplement\b
    | \bfinal\s+terms\b | \bconsent\s+solicitations?\b
    | \birrevocable\s+undertaking\b | \bcapitali[sz]ation\s+and\s+indebtedness\b
    """
)

#: A periodic report's title: statements, MD&A, reports, results releases and
#: presentations, annual disclosure forms, meeting materials.
_REPORT_TITLE = re.compile(
    r"""(?ix)
    (?:condensed\s+)?(?:consolidated\s+)?(?:interim\s+)?(?:unaudited\s+)?
      (?:condensed\s+)?(?:consolidated\s+)?(?:interim\s+)?
      financial\s+(?:statements|information|report)
    | management['’]?s\s+discussion\s+and\s+analysis | \bmd&a\b
    | report\s+of\s+independent\s+registered | independent\s+auditor['’]?s?\s+report
    | report\s+on\s+review\s+of\s+interim
    | \bannual\s+report\b | \binterim\s+report\b | \bquarterly\s+report\b
    | half[-\s]year(?:ly)?\s+(?:report|results|financial)
    | \breports?\b[^.]{0,80}\b(?:quarter|half|year|annual|interim)\b[^.]{0,40}\bresults\b
    | \b(?:first|second|third|fourth)\s+quarter\s+(?:\d{4}\s+)?(?:financial\s+)?results\b
    | \b(?:full[-\s]year|annual|year[-\s]end)\s+(?:\d{4}\s+)?results\b
    | \bresults\s+for\s+the\s+(?:first|second|third|fourth|three|six|nine|twelve)\b
    | \b[1-4]Q\d{2}\b | \b[1-4]Q\s?20\d\d\b | \bearnings\s+release\b
    | \breference\s+form\b | formul[aá]rio\s+de\s+refer[eê]ncia
    | annual\s+information\s+form | management\s+(?:information|proxy)\s+circular
    | notice\s+of\s+(?:annual|special)\s+(?:general\s+)?meeting
    | sustainability\s+report | production\s+report | operating\s+results
    """
)

#: The name of one primary financial statement or of its notes.
_STATEMENT_NAME = re.compile(
    r"""(?ix)
    balance\s+sheets? | statements?\s+of\s+financial\s+position
    | statements?\s+of\s+cash\s+flows? | cash\s+flow\s+statements?
    | income\s+statements?
    | statements?\s+of\s+(?:income|operations|profit\s+or\s+loss|comprehensive\s+income)
    | notes\s+to\s+the\s+(?:\w+\s+){0,4}financial\s+statements
    | statements?\s+of\s+changes\s+in\s+(?:shareholders['’]?\s+)?equity
    """
)

#: Statement names a document needs, without a report title, to be a report.
MIN_STATEMENT_NAMES = 3

#: Characters a document needs, without a report title, to be a report; a
#: short notice can name the statements without containing them.
MIN_STATEMENT_DOCUMENT_CHARS = 30_000

DocumentKind = Literal["event", "report"]


@dataclass(frozen=True)
class SixkDocumentKind:
    """A 6-K document's kind and the rule that decided it."""

    kind: DocumentKind
    #: ``exhibit_type``, ``agreement_title``, ``report_title``,
    #: ``statements`` or ``default``.
    rule: str
    #: The text the rule matched, or the statement-name count.
    evidence: str = ""

    @property
    def is_report(self: SixkDocumentKind) -> bool:
        """Whether the document is a periodic report."""
        return self.kind == "report"


def content_head(body: str, *, chars: int) -> str:
    r"""Return the first ``chars`` characters of a body's non-cover lines, whitespace collapsed.

    >>> content_head("FORM 6-K\nFor the month of May\nVale S.A.\nAnnual  Report", chars=40)
    'Vale S.A. Annual Report'
    """
    kept: list[str] = []
    taken = 0
    for line in body.split("\n"):
        text = line.strip()
        if not text or _COVER_LINE.search(text):
            continue
        kept.append(text)
        taken += len(text) + 1
        if taken >= chars:
            break
    return " ".join(" ".join(kept).split())[:chars]


def classify_sixk_document(body: str, document_type: str) -> SixkDocumentKind:
    """Classify one gated 6-K prose document as an event document or a periodic report.

    Rules, first match wins:

    1. an EX-1, EX-4 or EX-10 exhibit is an event document;
    2. a transaction-document title in the first :data:`AGREEMENT_TITLE_CHARS`
       non-cover characters, or a report title in the first
       :data:`REPORT_TITLE_CHARS`, decides by whichever comes first: an
       agreement makes an event document, a report title a report;
    3. at least :data:`MIN_STATEMENT_NAMES` financial-statement names in a body
       of at least :data:`MIN_STATEMENT_DOCUMENT_CHARS` make a report (financial
       statements whose title did not survive extraction);
    4. anything else is an event document.

    Ambiguity resolves to ``event``: an event document misread as a report
    loses its instruments, while a report misread as an event only keeps noise.

    Args:
        body: The document's gated body (:func:`cdt.segmenter.sixk.gated_body`).
        document_type: The document's own ``<TYPE>``, e.g. ``6-K`` or ``EX-99.1``.

    Returns:
        The kind and the rule that decided it.

    >>> classify_sixk_document("Teck Reports Unaudited Third Quarter Results", "EX-99.1").kind
    'report'
    >>> classify_sixk_document("Underwriting Agreement ... financial statements", "EX-1.1").rule
    'exhibit_type'
    >>> classify_sixk_document("Petrobras announces pricing of global notes", "6-K").kind
    'event'
    """
    if TRANSACTION_EXHIBIT_TYPE.match(document_type.strip()):
        return SixkDocumentKind("event", "exhibit_type", document_type.strip())
    agreement = _AGREEMENT_TITLE.search(content_head(body, chars=AGREEMENT_TITLE_CHARS))
    title = _REPORT_TITLE.search(content_head(body, chars=REPORT_TITLE_CHARS))
    # Both heads start at the same character, so offsets compare.
    if agreement and (title is None or agreement.start() <= title.start()):
        return SixkDocumentKind("event", "agreement_title", agreement.group(0))
    if title:
        return SixkDocumentKind("report", "report_title", title.group(0))
    if len(body) >= MIN_STATEMENT_DOCUMENT_CHARS:
        names = len(_STATEMENT_NAME.findall(body))
        if names >= MIN_STATEMENT_NAMES:
            return SixkDocumentKind("report", "statements", str(names))
    return SixkDocumentKind("event", "default")
