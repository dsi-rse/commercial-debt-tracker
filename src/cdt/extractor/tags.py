"""Handle the NER stage's tagged XML: tags, offsets, and evidence payloads built from cited tags."""

from __future__ import annotations

import re
from typing import cast
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as DefusedET

# An ampersand that starts no entity, and a `<` that cannot start a tag
# (`multiplier < 1`, `p<0.05`). The NER response has to be well-formed XML
# while reproducing text that may carry either.
UNESCAPED_AMPERSAND_PATTERN = re.compile(
    r"&(?!(?:amp|lt|gt|quot|apos);|#(?:\d+|x[0-9A-Fa-f]+);)"
)
UNESCAPED_LESS_THAN_PATTERN = re.compile(r"<(?![A-Za-z_/])")


def parse_tag_details(
    xml_text: str,
) -> tuple[ET.Element, str, dict[str, dict[str, object]]]:
    """Parse tagged XML and return root, plain text, and tag metadata."""
    root = DefusedET.fromstring(xml_text)
    plain_parts: list[str] = []
    tag_details: dict[str, dict[str, object]] = {}

    def walk(element: ET.Element) -> None:
        if element.text:
            plain_parts.append(element.text)
        for child in list(element):
            start = sum(len(part) for part in plain_parts)
            walk(child)
            end = sum(len(part) for part in plain_parts)
            tag_id = child.attrib.get("id")
            if tag_id:
                tag_details[tag_id] = {
                    "type": child.tag,
                    "text": "".join(plain_parts)[start:end],
                    "char_start": start,
                    "char_end": end,
                }
            if child.tail:
                plain_parts.append(child.tail)

    walk(root)
    return root, "".join(plain_parts), tag_details


def realign_tag_details(
    tag_details: dict[str, dict[str, object]],
    roundtrip_text: str,
    original_text: str,
) -> dict[str, dict[str, object]]:
    """Rewrite tag offsets from the NER round-trip text onto the original text.

    Evidence `char_start`/`char_end` must index the item's own `text` exactly.
    The NER stage only validates whitespace-collapsed equality, so the model
    may add or drop whitespace anywhere; the non-whitespace characters are
    identical in order, and each span is snapped to the original text along
    that alignment.

    Returns the input unchanged when the texts already match, and unchanged
    (round-trip offsets) rather than raising when they cannot be aligned.
    """
    if roundtrip_text == original_text or not tag_details:
        return tag_details
    nonws_map: dict[int, int] = {}
    target_index = 0
    target_length = len(original_text)
    for source_index, char in enumerate(roundtrip_text):
        if char.isspace():
            continue
        while target_index < target_length and original_text[target_index].isspace():
            target_index += 1
        if target_index >= target_length or original_text[target_index] != char:
            return tag_details
        nonws_map[source_index] = target_index
        target_index += 1
    realigned: dict[str, dict[str, object]] = {}
    for tag_id, detail in tag_details.items():
        start = cast(int, detail["char_start"])
        end = cast(int, detail["char_end"])
        while start < end and roundtrip_text[start].isspace():
            start += 1
        last = end - 1
        while last >= start and roundtrip_text[last].isspace():
            last -= 1
        if last < start or start not in nonws_map or last not in nonws_map:
            realigned[tag_id] = detail
            continue
        new_start = nonws_map[start]
        new_end = nonws_map[last] + 1
        realigned[tag_id] = {
            **detail,
            "char_start": new_start,
            "char_end": new_end,
            "text": original_text[new_start:new_end],
        }
    return realigned


def repair_unescaped_text(text: str) -> str:
    """Escape the `&` and `<` the NER response left bare, so it can be parsed.

    The item text reaches the model unescaped (`ner_input_body`), so an item
    containing `A&R` or `multiplier < 1` yields a response that reproduces the
    text but is not well-formed XML. Repaired: an `&` not already starting an
    entity, and a `<` not followed by a letter, `_` or `/`. A `<` that could
    open or close a tag is a real malformation and is left to fail parsing.
    """
    text = UNESCAPED_AMPERSAND_PATTERN.sub("&amp;", text)
    return UNESCAPED_LESS_THAN_PATTERN.sub("&lt;", text)


def assign_tag_ids(xml_text: str) -> str:
    """Assign sequential tag IDs to non-body tags."""
    root = DefusedET.fromstring(xml_text)
    counter = 0
    for element in root.iter():
        if element.tag == "body":
            continue
        counter += 1
        element.attrib = {"id": f"tag-{counter}"}
    return ET.tostring(root, encoding="unicode")


def collapse_whitespace(value: str) -> str:
    """Collapse all whitespace in a string for comparison."""
    return re.sub(r"\s+", "", value)


def normalize_span_whitespace(value: str) -> str:
    """Collapse whitespace runs in a span promoted to a canonical text field.

    Filings wrap instrument names across lines, so a verbatim span can carry a
    newline, tab, or non-breaking space. Canonical fields are display and
    comparison surfaces; the verbatim text stays in the `*_json` payloads, where
    the character offsets make it meaningful as provenance.
    """
    return re.sub(r"\s+", " ", value).strip()


def single_value_evidence_tag_ids(value: object) -> object:
    """Return evidence tag IDs from one single-value extractor payload."""
    if isinstance(value, dict):
        return value.get("evidence", [])
    return value


def cluster_span_texts(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Return the normalized texts of one coreference cluster's spans."""
    if not isinstance(tag_ids, list) or not tag_ids:
        return []
    values = [
        normalize_span_whitespace(str(tag_details[tag_id]["text"]))
        for tag_id in tag_ids
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    return [value for value in values if value]


def canonical_value(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the longest textual member of one coreference cluster."""
    values = cluster_span_texts(tag_ids, tag_details)
    if not values:
        return None
    return max(values, key=len)


def cluster_payload(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return the evidence spans for one cluster.

    ``char_start``/``char_end`` index the item's own ``text`` exactly.
    ``tag_id`` is retained for the relation stage's tag-to-mention mapping and
    for audit debugging; downstream consumers need only the offsets and text.
    """
    if not isinstance(tag_ids, list):
        return {"spans": []}
    spans = [
        {
            "tag_id": tag_id,
            "char_start": tag_details[tag_id]["char_start"],
            "char_end": tag_details[tag_id]["char_end"],
            "text": tag_details[tag_id]["text"],
        }
        for tag_id in tag_ids
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    return {"spans": spans}


def payload_tag_ids(payload: object) -> list[str]:
    """Return the tag ids recorded in one evidence payload."""
    if not isinstance(payload, dict):
        return []
    spans = payload.get("spans")
    if not isinstance(spans, list):
        return []
    return [
        str(span["tag_id"])
        for span in spans
        if isinstance(span, dict) and span.get("tag_id")
    ]


def escape_xml_attribute(value: str) -> str:
    """Escape one string for use inside an XML attribute value."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
