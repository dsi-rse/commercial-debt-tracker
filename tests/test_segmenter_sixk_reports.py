"""Tests for the 6-K document kind: periodic reports versus event documents."""

from __future__ import annotations

import pytest

from cdt.segmenter.sixk_reports import (
    MIN_STATEMENT_DOCUMENT_CHARS,
    classify_sixk_document,
    content_head,
)

COVER = (
    "SECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\n"
    "FORM 6-K\nREPORT OF FOREIGN PRIVATE ISSUER\nFor the month of May, 2026\n"
    "Commission File Number 001-15106\nExample S.A.\n"
)
DEBT_NOTE = (
    "The Company issued US$1,000 million of 5.25% notes due 2031, and its "
    "revolving credit facility remains undrawn."
)
STATEMENTS = (
    "CONSOLIDATED BALANCE SHEET\n...\nCONSOLIDATED STATEMENT OF INCOME\n...\n"
    "CONSOLIDATED STATEMENT OF CASH FLOWS\n...\n"
    "CONSOLIDATED STATEMENT OF CHANGES IN EQUITY\n"
)


@pytest.mark.parametrize(
    "title",
    [
        "Unaudited Condensed Consolidated Interim Financial Statements",
        "Management's Discussion and Analysis for the third quarter",
        "Teck Reports Unaudited Third Quarter Results for 2020",
        "Quarterly Report to Shareholders",
        "Institutional Presentation 3Q18",
        "Reference Form 2025",
        "Notice of Meeting and Management Proxy Circular",
    ],
)
def test_a_report_title_after_the_cover_makes_a_report(title: str) -> None:
    """The cover page is skipped; the first title line decides."""
    kind = classify_sixk_document(f"{COVER}{title}\n{DEBT_NOTE}", "6-K")

    assert kind.kind == "report"
    assert kind.rule == "report_title"


@pytest.mark.parametrize("document_type", ["EX-1.1", "EX-4.2", "EX-10", "ex-10.3"])
def test_transaction_exhibits_are_events_whatever_they_say(document_type: str) -> None:
    """Underwriting agreements and indentures cite the issuer's financial statements."""
    body = f"Underwriting Agreement\nFinancial Statements\n{STATEMENTS * 3}"

    kind = classify_sixk_document(body, document_type)

    assert kind.kind == "event"
    assert kind.rule == "exhibit_type"


def test_an_agreement_title_ahead_of_a_report_title_makes_an_event() -> None:
    """A credit agreement filed as EX-99.1 names the financial statements it requires."""
    body = (
        "EXECUTION VERSION\nCREDIT AGREEMENT\nExample Corp. as Borrower\n"
        "annual audited financial statements\n" + STATEMENTS * 20
    )

    kind = classify_sixk_document(body, "EX-99.1")

    assert kind.kind == "event"
    assert kind.rule == "agreement_title"


@pytest.mark.parametrize(
    "opening",
    [
        "WHEATON PRECIOUS METALS CORP. AND WHEATON PRECIOUS METALS INTERNATIONAL LTD. "
        "as Borrowers - and - BANK OF MONTREAL as Administrative Agent",
        "EXECUTION COPY\nCOOPERATION AGREEMENT",
        "CONSENT SOLICITATIONS: NOTICE OF SEPARATE MEETINGS of the holders of the Notes",
        "IRREVOCABLE UNDERTAKING",
        "CAPITALIZATION AND INDEBTEDNESS as at 31 March 2021",
    ],
)
def test_long_transaction_documents_citing_the_statements_are_events(
    opening: str,
) -> None:
    """A credit agreement, consent solicitation or offering table names the statements it summarises."""
    body = f"{COVER}{opening}\nannual audited financial statements\n{STATEMENTS * 20}"

    kind = classify_sixk_document(body, "EX-99.1")

    assert kind.kind == "event"
    assert kind.rule == "agreement_title"


def test_a_report_title_ahead_of_an_agreement_makes_a_report() -> None:
    """A report whose first paragraph mentions a credit agreement is still a report."""
    body = f"{COVER}Unaudited Interim Financial Statements\nThe credit agreement ..."

    kind = classify_sixk_document(body, "EX-99.2")

    assert kind.kind == "report"
    assert kind.rule == "report_title"


def test_long_untitled_statements_make_a_report() -> None:
    """Financial statements whose title did not survive extraction are still a report."""
    body = STATEMENTS + "1,234\n" * (MIN_STATEMENT_DOCUMENT_CHARS // 6)

    kind = classify_sixk_document(body, "6-K")

    assert kind.kind == "report"
    assert kind.rule == "statements"


def test_a_short_notice_naming_the_statements_stays_an_event() -> None:
    """Below the size floor, statement names are references rather than statements."""
    body = f"Example announces redemption of notes\n{STATEMENTS}\n{DEBT_NOTE}"

    assert classify_sixk_document(body, "6-K").kind == "event"


def test_a_debt_press_release_is_an_event() -> None:
    """The kind the extractor was built for."""
    body = (
        f"{COVER}PETROBRAS ANNOUNCES EARLY TENDER RESULTS\nRio de Janeiro, "
        f"July 12, 2022 - Petrobras announced the results of its tender offers. "
        f"{DEBT_NOTE}"
    )

    kind = classify_sixk_document(body, "6-K")

    assert kind.kind == "event"
    assert kind.rule == "default"


def test_a_report_title_deep_in_the_body_does_not_count() -> None:
    """A press release that cites its annual report further down stays an event."""
    body = (
        f"Example prices US$500 million of notes\n{DEBT_NOTE * 10}\n"
        "See the annual report for details."
    )

    assert classify_sixk_document(body, "EX-99.1").kind == "event"


def test_content_head_skips_cover_lines_and_collapses_whitespace() -> None:
    """Only the text after the cover page is read for titles."""
    assert content_head(f"{COVER}Annual   Report\n2025", chars=600) == (
        "Example S.A. Annual Report 2025"
    )
