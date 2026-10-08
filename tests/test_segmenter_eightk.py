"""Tests for pure 8-K item text extraction."""

from __future__ import annotations

from cdt.segmenter.eightk import extract_items_from_document, primary_8k_body
from cdt.segmenter.text import DocumentText, normalize_body_lines


def test_extract_items_from_complete_submission() -> None:
    """Extraction maps ITEM INFORMATION and slices the matching body section."""
    document = DocumentText(
        accession_number="0001",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="""
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<p>Item 8.01 Other Events.</p>
<p>The company entered a material update.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p>
<p>Exhibit text.</p>
</body></html>
</TEXT>
</DOCUMENT>
""",
    )

    sections = extract_items_from_document(document)

    assert len(sections) == 1
    assert sections[0].item_number == "8.01"
    assert sections[0].extraction_status == "ok"
    assert "material update" in sections[0].section_text
    assert "Exhibit text" not in sections[0].section_text


def test_extract_items_marks_unmapped_item_information() -> None:
    """Unknown ITEM INFORMATION captions produce an unmapped status row."""
    document = DocumentText(
        accession_number="0001",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="ITEM INFORMATION: Not A Real Item\nItem 8.01 Other Events",
    )

    sections = extract_items_from_document(document)

    assert sections[0].item_number == ""
    assert sections[0].extraction_status == "unmapped_item_information"


def test_extract_items_marks_missing_heading() -> None:
    """Mapped ITEM INFORMATION without a body heading is reported."""
    document = DocumentText(
        accession_number="0001",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="ITEM INFORMATION: Other Events\nNo matching heading here.",
    )

    sections = extract_items_from_document(document)

    assert sections[0].item_number == "8.01"
    assert sections[0].extraction_status == "missing_heading"


def test_primary_8k_body_prefers_8k_document_block() -> None:
    """Only the primary 8-K document block is selected from submissions."""
    text = """
<DOCUMENT>
<TYPE>EX-99
<TEXT>wrong</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>8-K
<TEXT>right</TEXT>
</DOCUMENT>
"""

    assert primary_8k_body(text).strip() == "right"


def test_duplicate_item_information_lines_emit_one_row() -> None:
    """SEC headers repeat ITEM INFORMATION lines; item_id is a primary key (#74)."""
    document = DocumentText(
        accession_number="0002",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="""
ITEM INFORMATION: Other Events
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<p>Item 8.01 Other Events.</p>
<p>The company entered a material update.</p>
</body></html>
</TEXT>
</DOCUMENT>
""",
    )

    sections = extract_items_from_document(document)

    assert len(sections) == 1
    assert sections[0].item_number == "8.01"


def test_duplicate_unmapped_item_information_lines_emit_one_row() -> None:
    """Unmapped duplicates would collide on the same accession-'' item_id (#74)."""
    document = DocumentText(
        accession_number="0003",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text=(
            "ITEM INFORMATION: Not A Real Item\n"
            "ITEM INFORMATION: Not A Real Item\n"
            "Item 8.01 Other Events"
        ),
    )

    sections = extract_items_from_document(document)

    assert len(sections) == 1


def test_coupon_rate_lines_do_not_end_item_sections() -> None:
    """'5.25% Senior Notes due 2029' is content, not a heading (#63)."""
    document = DocumentText(
        accession_number="0004",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="""
ITEM INFORMATION: Entry into a Material Definitive Agreement
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<p>Item 1.01 Entry into a Material Definitive Agreement.</p>
<p>The company issued its</p>
<p>5.25% Senior Notes due 2029</p>
<p>under the indenture described below.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p>
</body></html>
</TEXT>
</DOCUMENT>
""",
    )

    sections = extract_items_from_document(document)

    assert len(sections) == 1
    assert sections[0].item_number == "1.01"
    assert "5.25% Senior Notes due 2029" in sections[0].section_text
    assert "indenture described below" in sections[0].section_text


def test_rate_cells_with_percent_in_next_cell_do_not_end_item_sections() -> None:
    """A bare 'X.0Y' rate cell whose '%' sits in the adjacent cell is content (#63)."""
    document = DocumentText(
        accession_number="0005",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text="""
ITEM INFORMATION: Entry into a Material Definitive Agreement
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<p>Item 1.01 Entry into a Material Definitive Agreement.</p>
<p>The notes bear interest at a rate of</p>
<table><tr><td>6.00</td><td>%</td></tr></table>
<p>per annum as described in the indenture.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p>
</body></html>
</TEXT>
</DOCUMENT>
""",
    )

    sections = extract_items_from_document(document)

    assert len(sections) == 1
    assert sections[0].item_number == "1.01"
    assert "per annum as described in the indenture" in sections[0].section_text


def _eightk(item_informations: list[str], body: str) -> DocumentText:
    header = "".join(f"ITEM INFORMATION: {value}\n" for value in item_informations)
    return DocumentText(
        accession_number="0006",
        cik="320193",
        company_name="Example Inc.",
        url="https://sec.example/full.txt",
        date="2024-01-02",
        text=f"{header}<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n<html><body>\n{body}\n"
        "</body></html>\n</TEXT>\n</DOCUMENT>\n",
    )


def test_a_bare_coupon_cell_naming_an_undeclared_item_does_not_end_the_section() -> (
    None
):
    """A ``5.06`` coupon cell in a bond table is content, not Item 5.06.

    Shaped on an FHLB consolidated-obligation filing, whose 2.03 section lists
    each bond's coupon in its own cell.
    """
    document = _eightk(
        [
            "Creation of a Direct Financial Obligation or an Obligation under an "
            "Off-Balance Sheet Arrangement of a Registrant"
        ],
        """
<p>Item 2.03 Creation of a Direct Financial Obligation.</p>
<p>The Bank issued the following consolidated obligation bonds:</p>
<table>
<tr><td>3130BC2E2</td><td>9/18/2029</td><td>4.80</td><td>10,000,000</td></tr>
<tr><td>3130BC2Q5</td><td>9/18/2031</td><td>5.06</td><td>10,000,000</td></tr>
</table>
<p>(1) Call Type Description: callable bonds may be redeemed by the Bank.</p>
<p>Signatures</p>
""",
    )

    (section,) = extract_items_from_document(document)

    assert section.extraction_status == "ok"
    assert "Call Type Description" in section.section_text


def test_a_bare_heading_with_its_title_still_ends_the_section() -> None:
    """An undeclared item still ends the section when its title follows the number."""
    document = _eightk(
        ["Other Events"],
        """
<p>8.01. Other Events.</p>
<p>The Board declared a dividend.</p>
<p>Item</p>
<p>9.01 Financial Statements and Exhibits.</p>
<p>(d) Exhibits</p>
""",
    )

    (section,) = extract_items_from_document(document)

    assert "declared a dividend" in section.section_text
    assert "Exhibits" not in section.section_text


def test_dollar_amounts_and_non_item_numbers_are_not_headings() -> None:
    """Money figures and numbers outside the 8-K item set never register as headings."""
    from cdt.segmenter.eightk import leading_item_numbers

    assert leading_item_numbers(
        "Item 8.01 Other Events. The Company issued $1.05 billion of notes."
    ) == ("8.01",)
    assert leading_item_numbers("6.00") == ()
    assert leading_item_numbers("Item 7.03 Not A Real Item.") == ()
    assert leading_item_numbers("Item 1.05 Material Cybersecurity Incidents.") == (
        "1.05",
    )


def test_escaped_angle_brackets_in_text_survive_the_tag_strip() -> None:
    """``&lt;`` and ``&gt;`` are text; unescaping them first made them a tag."""
    lines = normalize_body_lines(
        "<p>multiplier &gt; 1 = leveraged; multiplier &lt; 1</p>"
    )

    assert [line.text for line in lines] == [
        "multiplier > 1 = leveraged; multiplier < 1"
    ]
    assert [
        line.text for line in normalize_body_lines("<td>x &lt; 5 and y &gt; 3</td>")
    ] == ["x < 5 and y > 3"]
