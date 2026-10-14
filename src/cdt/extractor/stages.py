"""The NER, instrument IE and relation stages, and salvage of a failed IE response."""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

import json
import re
from typing import Any, cast
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as DefusedET

from cdt.extractor.llm import load_prompt
from cdt.extractor.normalize.amounts import (
    currency_candidates_from_text,
    select_principal_amount,
    standardized_amounts_payloads,
    standardized_interest_rate_payload,
)
from cdt.extractor.normalize.dates import (
    derived_status_payload,
    mark_post_filing_events_expected,
    select_date_payload,
    standardized_dates_payloads,
)
from cdt.extractor.normalize.parties import (
    canonical_instrument_name,
    party_payloads_and_disclosure,
)
from cdt.extractor.schema import (
    INSTRUMENT_RELATION_TYPES,
    INSTRUMENT_TYPES,
    TERMINAL_DATE_KINDS,
    debt_instrument_mention_id_for,
    raw_id_for,
)
from cdt.extractor.state import ABORTED_ATTEMPT_STATUS, ExtractionRowState, StageSpec
from cdt.extractor.tags import (
    assign_tag_ids,
    cluster_payload,
    collapse_whitespace,
    escape_xml_attribute,
    parse_tag_details,
    payload_tag_ids,
    realign_tag_details,
    repair_unescaped_text,
)
from cdt.extractor.validate import is_one_of, validate_instrument_entry
from cdt.storage.columns import coerce_dataset_text

# A regex rather than a parse: the high-water check counts tags in earlier,
# failed attempts, which may be truncated or not well-formed XML. Tolerates an
# attribute NER output should never carry.
DEBT_INSTRUMENT_OPEN_TAG_RE = re.compile(r"<debt_instrument(?:\s[^>]*)?>")


# Every tag `NERStage.validate` accepts, and the entity subset of it. `body` is
# the wrapper the stage supplies itself, so it is not evidence the model tagged
# anything -- an untagged echo carries it.
NER_ALLOWED_TAGS = frozenset(
    {
        "body",
        "person",
        "organization",
        "debt_instrument",
        "agreement",
        "date",
        "duration",
        "amount",
        "interest_rate",
    }
)
NER_ENTITY_TAGS = NER_ALLOWED_TAGS - {"body"}
NER_ENTITY_OPEN_TAG_RE = re.compile(
    r"<(?:" + "|".join(sorted(NER_ENTITY_TAGS)) + r")(?:\s[^>]*)?>"
)


def count_ner_entity_tags(response: str | None) -> int:
    """Count opening entity tags in one raw, possibly malformed NER response.

    Returns 0 for None or an empty response.
    """
    if not response:
        return 0
    return len(NER_ENTITY_OPEN_TAG_RE.findall(response))


def prior_attempt_tagged(row_state: ExtractionRowState, stage_name: str) -> bool:
    """Whether any earlier, non-aborted attempt of this stage tagged any entity.

    Any entity tag counts, not just `debt_instrument`. Provider-aborted
    attempts are excluded: their partial text is not the model's work, and the
    model cannot see it. See docs/decisions/extraction.md.
    """
    return any(
        count_ner_entity_tags(attempt.response)
        for attempt in row_state.all_attempts
        if attempt.stage_name == stage_name and attempt.status != ABORTED_ATTEMPT_STATUS
    )


def ner_input_body(row_state: ExtractionRowState) -> str:
    """Return the exact `<body>`-wrapped text the NER stage sends the model.

    The text is wrapped unescaped, deliberately: an item containing a bare `&`
    or `<` produces a response that only parses after `repair_unescaped_text`.
    """
    return f"<body>{row_state.text}</body>"


def count_debt_instrument_tags(response: str | None) -> int:
    """Count `<debt_instrument>` opening tags in one raw NER response.

    A regex, not `parse_tag_details`, so a truncated or malformed response
    still counts. Returns 0 for None or an empty response.
    """
    if not response:
        return 0
    return len(DEBT_INSTRUMENT_OPEN_TAG_RE.findall(response))


def prior_debt_instrument_high_water(
    row_state: ExtractionRowState, stage_name: str
) -> int:
    """Most `debt_instrument` tags any earlier attempt of this stage produced.

    Reads only completed attempts in `all_attempts`, never the response being
    validated, and excludes provider-aborted ones (the retry message tells the
    model to keep tags it must be able to see). Returns 0 when there are none.
    """
    return max(
        (
            count_debt_instrument_tags(attempt.response)
            for attempt in row_state.all_attempts
            if attempt.stage_name == stage_name
            and attempt.status != ABORTED_ATTEMPT_STATUS
        ),
        default=0,
    )


class NERStage:
    """NER stage using XML-tagged output."""

    name = "ner"

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        prompt = load_prompt("ner")
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ner_input_body(row_state)},
        ]

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        """Return the failures for one NER response; empty means it passed.

        Structural checks: well-formed XML rooted at `<body>`, only
        `NER_ALLOWED_TAGS`, bare non-empty tags, and stripped text equal to the
        input up to whitespace. Two cross-attempt checks reject a give-up that
        passes all of those (an untagged echo of the input):

        * no entity tags at all, when an earlier attempt of this row tagged any;
        * no `debt_instrument` tags, when an earlier attempt tagged some
          (the high-water mark).

        On an item never tagged before, an untagged response is accepted as a
        genuine zero. See docs/decisions/extraction.md for the motivating case
        and corpus measurements.
        """
        if not response or not isinstance(response, str):
            return [
                "Model returned empty or non-text output. Even if no entities are present, return the input text."
            ]

        # Gated on earlier tagged work, not the attempt number: an item with
        # nothing to tag may honestly echo its input after an unrelated failure.
        # A truncated earlier attempt still carries tags, so it counts.
        if prior_attempt_tagged(row_state, self.name) and not count_ner_entity_tags(
            response
        ):
            return [
                "Response contains no tags at all, but an earlier attempt on this item "
                "tagged entities, so this response drops every one of them. Re-emit "
                "your previous tagged output with the text corrected."
            ]

        response = repair_unescaped_text(response)
        try:
            root = DefusedET.fromstring(response)
        except ET.ParseError as exc:
            return [f"Response is not valid XML: {exc}"]
        if root.tag != "body":
            return ["Response root must be <body>."]

        failures: list[str] = []
        for element in root.iter():
            if element.tag not in NER_ALLOWED_TAGS:
                failures.append(f"Disallowed tag found: {element.tag}")
            if element.tag != "body" and element.attrib:
                failures.append("Tags contain attributes; only bare tags are allowed.")
            if element.tag != "body" and not "".join(element.itertext()).strip():
                failures.append("Tags must contain non-whitespace text.")

        _, plain_text, _ = parse_tag_details(response)
        if collapse_whitespace(plain_text) != collapse_whitespace(row_state.text):
            failures.append(
                "Response text with tags stripped must match the input text exactly."
            )

        high_water = prior_debt_instrument_high_water(row_state, self.name)
        if high_water and not count_debt_instrument_tags(response):
            failures.append(
                f"Response contains no <debt_instrument> tags, but an earlier attempt "
                f"on this item tagged {high_water}. Keep every tag you found and "
                f"correct only the text."
            )
        return failures

    def postprocess(self, row_state: ExtractionRowState) -> None:
        response = row_state.stage_responses.get(self.name)
        if not response:
            return
        row_state.ner_tagged_xml = assign_tag_ids(repair_unescaped_text(response))

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        if not row_state.ner_tagged_xml:
            return False
        _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        return not any(
            detail["type"] == "debt_instrument" for detail in tag_details.values()
        )

    def build_retry_message(self, failures: list[str]) -> str:
        """Build the NER retry turn, asking for a repair rather than a redo.

        The keep-every-tag clause comes first so that returning the input
        untagged is never the cheapest compliant answer.
        """
        return (
            "Your previous NER output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
            "- Keep every tag from your previous output. Fix only the text so it "
            "matches the input exactly.\n"
            "- Returning the input untagged is not a valid fix; it will be rejected.\n"
            "- Return the original input text exactly, wrapped in <body>...</body>.\n"
            "- Only add the allowed bare tags.\n"
            "- Do not add attributes, comments, or extra text.\n"
            "- The stripped text must match the original input exactly."
        )


class InstrumentIEStage:
    """Instrument-mention extraction stage."""

    name = "instrument_ie"

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        if not row_state.ner_tagged_xml:
            raise ValueError("ner_tagged_xml is required for instrument_ie.")
        return [
            {"role": "system", "content": load_prompt("instrument_ie")},
            {"role": "user", "content": row_state.ner_tagged_xml},
        ]

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        if not row_state.ner_tagged_xml:
            return ["ner_tagged_xml is required for instrument_ie validation."]
        try:
            data = instrument_entries_from_response(response)
        except json.JSONDecodeError as exc:
            return [f"Output is not valid JSON: {exc}"]
        if not isinstance(data, list):
            return ["Output must be a JSON array of objects."]

        _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        failures: list[str] = []
        for index, obj in enumerate(data):
            failures.extend(validate_instrument_entry(index, obj, tag_details))
        return failures

    def postprocess(self, row_state: ExtractionRowState) -> None:
        if not row_state.ner_tagged_xml:
            return
        response = row_state.stage_responses.get(self.name)
        if not response:
            return
        try:
            data = instrument_entries_from_response(response)
        except json.JSONDecodeError:
            return
        if not isinstance(data, list):
            return
        _, roundtrip_text, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        # Published evidence offsets index the item's own text, not the model's
        # whitespace-drifted echo of it.
        tag_details = realign_tag_details(tag_details, roundtrip_text, row_state.text)
        document_currencies = frozenset(currency_candidates_from_text(row_state.text))
        mention_entries = iter_instrument_entries(
            cast(list[dict[str, Any]], data), tag_details
        )
        mentions: list[dict[str, object]] = []
        seen_mention_ids: set[str] = set()
        for index, obj in mention_entries:
            raw_id = raw_id_for(index)
            name_text = canonical_instrument_name(obj.get("name", []), tag_details)
            amount_payloads = standardized_amounts_payloads(
                obj,
                tag_details,
                name_text=name_text,
                document_currencies=document_currencies,
            )
            principal = select_principal_amount(amount_payloads)
            date_payloads = standardized_dates_payloads(
                obj,
                tag_details,
                name_text=name_text,
            )
            mark_post_filing_events_expected(
                date_payloads, str(row_state.item_row.get("date") or "")
            )
            start_date_payload = select_date_payload(date_payloads, "closing")
            if start_date_payload.get("normalized_date") is None:
                # A facility whose only stated date is its `dated as of` date
                # started then.
                start_date_payload = select_date_payload(date_payloads, "agreement")
            maturity_payload = select_date_payload(date_payloads, "maturity")
            commitment_termination_payload = select_date_payload(
                date_payloads, "commitment_termination"
            )
            status_payload = (
                derived_status_payload(date_payloads)
                if "dates" in obj
                else {"status": None, "status_date": None}
            )
            interest_rate_payload = standardized_interest_rate_payload(
                obj.get("interest_rate"),
                tag_details,
                name_text=name_text,
            )
            party_clusters, lender_disclosure = party_payloads_and_disclosure(
                obj, tag_details
            )
            mention_row: dict[str, object] = {
                "item_id": row_state.item_id,
                "accession_number": row_state.item_row.get("accession_number"),
                "cik": row_state.item_row.get("cik"),
                # Preserve filer display metadata for downstream instrument pages.
                "company_name": row_state.item_row.get("company_name"),
                "date": row_state.item_row.get("date"),
                "raw_id": raw_id,
                "name": name_text,
                "instrument_type": (
                    obj["instrument_type"]
                    if obj.get("instrument_type") in INSTRUMENT_TYPES
                    else None
                ),
                "start_date": start_date_payload["normalized_date"],
                "maturity_date": maturity_payload["normalized_date"],
                "commitment_termination_date": commitment_termination_payload[
                    "normalized_date"
                ],
                "principal_amount": principal.get("normalized_amount"),
                "principal_currency": principal.get("currency"),
                "principal_amount_kind": principal.get("kind"),
                "interest_rate_kind": interest_rate_payload["kind"],
                "interest_rate_pct": interest_rate_payload["rate_pct"],
                "status": status_payload["status"],
                "status_date": (
                    cast(dict[str, object], status_payload["status_date"]).get(
                        "normalized_date"
                    )
                    if isinstance(status_payload["status_date"], dict)
                    else None
                ),
                "amendment_of": None,
                "retired_by_json": "[]",
                "split_of": None,
                "parties_json": json.dumps(party_clusters, sort_keys=True),
                "lender_disclosure": lender_disclosure,
                "name_json": json.dumps(
                    cluster_payload(obj.get("name", []), tag_details),
                    sort_keys=True,
                ),
                "start_date_json": json.dumps(start_date_payload, sort_keys=True),
                "maturity_date_json": json.dumps(maturity_payload, sort_keys=True),
                "commitment_termination_date_json": json.dumps(
                    commitment_termination_payload, sort_keys=True
                ),
                "amounts_json": json.dumps(amount_payloads, sort_keys=True),
                "status_json": json.dumps(status_payload, sort_keys=True),
                "interest_rate_json": json.dumps(interest_rate_payload, sort_keys=True),
                "dates_json": json.dumps(date_payloads, sort_keys=True),
            }
            mention_id = debt_instrument_mention_id_for(
                row_state.item_id,
                mention_row,
            )
            if mention_id in seen_mention_ids:
                # Objects that differ in no extracted property are the same mention.
                # One name span covering several note classes produces these, and they
                # would otherwise write duplicate primary keys.
                continue
            seen_mention_ids.add(mention_id)
            mention_row["debt_instrument_mention_id"] = mention_id
            mentions.append(mention_row)
        row_state.debt_instrument_mentions = mentions

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        return False

    def build_retry_message(self, failures: list[str]) -> str:
        return (
            "Your previous instrument extraction output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
            "- Return a JSON array with one object per concrete debt instrument described as its own obligation: `[ { ... } ]` even for a single instrument, `[]` for none.\n"
            "- Ignore collective labels or contextual references that should not become standalone debt instruments.\n"
            "- One object has one current closing date and one current commitment or principal; a term stated before a change is a `prior: true` entry, and two unrelated values are two objects.\n"
            "- Shared evidence tags may appear in more than one object when the text supports that.\n"
            "- Do not return agreements as output objects.\n"
            "- Return only valid JSON."
        )


LINEAGE_SUCCESSOR_FIRST_TYPES = {"amendment_of"}
LINEAGE_PREDECESSOR_FIRST_TYPES = {"retired_by"}


def oriented_lineage_pair(
    source_id: str,
    target_id: str,
    relation_type: str,
    by_raw_id: dict[str, dict[str, object]],
) -> tuple[str, str]:
    """Return one lineage pair oriented the way its type reads.

    `amendment_of` runs from the instrument as amended to the predecessor, so
    the source is the later of the two; `retired_by` runs from the retired
    obligation to the instrument that retired it, so the source is the earlier.
    When both sides carry a start date and the source sits on the wrong side of
    that order, the model has named the pair the wrong way round and the
    pointer is flipped. Any other type, a missing side, or a missing or equal
    start date returns the pair unchanged.
    """
    if (
        relation_type not in LINEAGE_SUCCESSOR_FIRST_TYPES
        and relation_type not in LINEAGE_PREDECESSOR_FIRST_TYPES
    ):
        return source_id, target_id
    source = by_raw_id.get(source_id)
    target = by_raw_id.get(target_id)
    if source is None or target is None:
        return source_id, target_id
    source_start = coerce_dataset_text(source.get("start_date"))
    target_start = coerce_dataset_text(target.get("start_date"))
    if not source_start or not target_start:
        return source_id, target_id
    if relation_type in LINEAGE_SUCCESSOR_FIRST_TYPES and source_start < target_start:
        return target_id, source_id
    if relation_type in LINEAGE_PREDECESSOR_FIRST_TYPES and source_start > target_start:
        return target_id, source_id
    return source_id, target_id


class InstrumentRelationStage:
    """Mention-level lineage relation stage."""

    name = "instrument_relation"

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": load_prompt("instrument_relation")},
            {"role": "user", "content": relation_prompt_xml(row_state)},
        ]

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            return [f"Output is not valid JSON: {exc}"]
        if not isinstance(data, list):
            return ["Output must be a JSON array of objects."]
        instrument_ids = {
            str(mention["raw_id"]) for mention in row_state.debt_instrument_mentions
        }
        failures: list[str] = []
        for relation in data:
            if not isinstance(relation, dict):
                failures.append("Relation entry is not an object.")
                continue
            if set(relation) != {"from", "to", "type"}:
                failures.append(
                    "Relation entry must have exactly 'from', 'to', and 'type' keys."
                )
            rel_from = relation.get("from")
            rel_to = relation.get("to")
            rel_type = relation.get("type")
            if not isinstance(rel_from, str) or not isinstance(rel_to, str):
                failures.append("'from' and 'to' must be strings.")
                continue
            if not is_one_of(rel_type, INSTRUMENT_RELATION_TYPES):
                failures.append(
                    f"Invalid relation type: {rel_type}. Must be amendment_of, retired_by, or split_of."
                )
            if rel_from not in instrument_ids or rel_to not in instrument_ids:
                failures.append(
                    "Instrument relations must link valid mention raw IDs only."
                )
            if rel_from == rel_to:
                failures.append("Instrument relations cannot link a mention to itself.")
        return failures

    def postprocess(self, row_state: ExtractionRowState) -> None:
        response = row_state.stage_responses.get(self.name)
        if not response:
            return
        data = json.loads(response)
        by_raw_id = {
            str(mention["raw_id"]): mention
            for mention in row_state.debt_instrument_mentions
        }
        raw_to_global = {
            str(mention["raw_id"]): str(mention["debt_instrument_mention_id"])
            for mention in row_state.debt_instrument_mentions
        }
        for relation in data:
            source_id, target_id = oriented_lineage_pair(
                str(relation["from"]),
                str(relation["to"]),
                str(relation["type"]),
                by_raw_id,
            )
            mention = by_raw_id.get(source_id)
            if mention is None:
                continue
            target = raw_to_global.get(target_id)
            if str(relation["type"]) == "retired_by":
                # A list, not a scalar: one obligation may be retired jointly by
                # several instruments (a dual-tranche offering funding one
                # redemption).
                retirers = json.loads(str(mention.get("retired_by_json") or "[]"))
                if target and target not in retirers:
                    retirers.append(target)
                mention["retired_by_json"] = json.dumps(retirers)
            else:
                mention[str(relation["type"])] = target

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        return False

    def build_retry_message(self, failures: list[str]) -> str:
        return (
            "Your previous instrument relation output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
            "- Return a JSON array.\n"
            "- Each relation must have from, to, and type.\n"
            "- Use only amendment_of, retired_by, or split_of.\n"
            "- Use only instrument IDs from the input."
        )


# The ordered extraction stages. They are stateless singletons; the resumable
# state machine and the synchronous workflow both drive this same list so that
# audit semantics stay identical across the live and batch backends.
EXTRACTOR_STAGES: list[StageSpec] = [
    NERStage(),
    InstrumentIEStage(),
    InstrumentRelationStage(),
]
STAGE_BY_NAME: dict[str, StageSpec] = {stage.name: stage for stage in EXTRACTOR_STAGES}
STAGE_INDEX: dict[str, int] = {
    stage.name: index for index, stage in enumerate(EXTRACTOR_STAGES)
}


def instrument_entries_from_response(response: str) -> object:
    """Parse an instrument_ie response, accepting a bare object as a one-entry list.

    Raises ``json.JSONDecodeError`` on invalid JSON; any other JSON value is
    returned as parsed for the caller to reject.
    """
    data = json.loads(response)
    if isinstance(data, dict):
        return [data]
    return data


def salvage_instrument_ie_entries(row_state: ExtractionRowState) -> int | None:
    """Filter the final instrument_ie response down to its valid entries.

    Returns the number of dropped entries, or None when nothing is salvageable
    (unparseable JSON, a non-list response, or no individually valid entry).
    On success the stored stage response is replaced with the surviving
    entries, so postprocess and the audit log see exactly what was kept.
    """
    if not row_state.ner_tagged_xml:
        return None
    response = row_state.stage_responses.get(InstrumentIEStage.name)
    if not response:
        return None
    try:
        data = instrument_entries_from_response(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
    kept = [obj for obj in data if not validate_instrument_entry(0, obj, tag_details)]
    if not kept:
        return None
    row_state.stage_responses[InstrumentIEStage.name] = json.dumps(kept)
    return len(data) - len(kept)


def iter_instrument_entries(
    data: list[dict[str, Any]],
    tag_details: dict[str, dict[str, object]],
) -> list[tuple[int, dict[str, Any]]]:
    """Return valid instrument entries in sequential order."""
    entries: list[tuple[int, dict[str, Any]]] = []
    counter = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        name_tags = obj.get("name")
        if not isinstance(name_tags, list) or not name_tags:
            continue
        tag_types = {
            str(tag_details[tag_id]["type"])
            for tag_id in name_tags
            if tag_id in tag_details
        }
        if tag_types == {"debt_instrument"}:
            counter += 1
            entries.append((counter, obj))
    return entries


def relation_prompt_xml(row_state: ExtractionRowState) -> str:
    """Build relation-stage XML with instrument-id attributes."""
    if not row_state.ner_tagged_xml:
        raise ValueError("ner_tagged_xml is required for instrument_relation.")
    root, _, _ = parse_tag_details(row_state.ner_tagged_xml)
    tag_to_raw_id: dict[str, str] = {}
    for mention in row_state.debt_instrument_mentions:
        payload = json.loads(str(mention["name_json"]))
        for tag_id in payload_tag_ids(payload):
            key = str(tag_id)
            raw_id = str(mention["raw_id"])
            if key not in tag_to_raw_id:
                tag_to_raw_id[key] = raw_id
            else:
                tag_to_raw_id[key] = f"{tag_to_raw_id[key]}||{raw_id}"
    body = render_relation_body(root, tag_to_raw_id)
    return f"{relation_instrument_manifest(row_state)}<body>{body}</body>"


def relation_instrument_manifest(row_state: ExtractionRowState) -> str:
    """List each instrument id with the terms already extracted for it.

    Returns an `<instruments>` block (empty string when there are none) giving
    each id its name, amount, dates, status and `expected_retirement` flag, so
    the relation stage can tell apart objects built from one name span, which
    render identically in the body.
    """
    lines: list[str] = []
    for mention in row_state.debt_instrument_mentions:
        attributes = [f'id="{escape_xml_attribute(str(mention["raw_id"]))}"']
        # The manifest keeps the attribute name `amount`: the relation prompt
        # speaks in the filing's own vocabulary, not the storage schema's.
        manifest_fields = (
            ("name", "name"),
            ("amount", "principal_amount"),
            ("start_date", "start_date"),
            ("maturity_date", "maturity_date"),
            ("status", "status"),
        )
        for attribute_name, field_name in manifest_fields:
            value = coerce_dataset_text(mention.get(field_name))
            if value is not None:
                attributes.append(f'{attribute_name}="{escape_xml_attribute(value)}"')
        # A planned retirement is invisible in the body's tags; without it the
        # relation stage cannot tell a use-of-proceeds target from a note
        # merely mentioned, and it is exactly the `retired_by` case.
        try:
            facts = json.loads(str(mention.get("dates_json") or "[]"))
        except json.JSONDecodeError:
            facts = []
        if any(
            isinstance(fact, dict)
            and fact.get("kind") in TERMINAL_DATE_KINDS
            and fact.get("expected") is True
            for fact in facts
        ):
            attributes.append('expected_retirement="true"')
        lines.append(f"  <instrument {' '.join(attributes)}/>")
    if not lines:
        return ""
    joined = "\n".join(lines)
    return f"<instruments>\n{joined}\n</instruments>\n"


def render_relation_body(root: ET.Element, tag_to_raw_id: dict[str, str]) -> str:
    """Render only debt instrument tags needed for relation extraction."""

    def render_element(element: ET.Element) -> str:
        parts: list[str] = [element.text or ""]
        for child in list(element):
            rendered = render_element(child)
            tag_id = child.attrib.get("id")
            if child.tag == "debt_instrument" and tag_id in tag_to_raw_id:
                for instrument_id in tag_to_raw_id[tag_id].split("||"):
                    parts.append(
                        f'<debt_instrument instrument-id="{instrument_id}">{rendered}</debt_instrument>'
                    )
            else:
                parts.append(rendered)
            parts.append(child.tail or "")
        return "".join(parts)

    return render_element(root)
