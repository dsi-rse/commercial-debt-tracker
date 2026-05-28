"""LLM-backed extractor stage for relevant SEC 8-K items."""

# ruff: noqa: ANN101, D102, D105, D107

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, cast
from xml.etree import ElementTree as ET

import pandas as pd
from defusedxml import ElementTree as DefusedET

from cdt import settings
from cdt.classifier.core import load_pending_item_batches
from cdt.database import (
    cdt_db_path,
    connect_cdt_db,
    mark_items_extracted,
    mark_items_extraction_failed,
    read_items,
    replace_instrument_mentions,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MODEL = "openai/gpt-5.4"
DEFAULT_REASONING_EFFORT = "none"
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
INSTRUMENT_ENTITY_TAG_TYPES = {"debt_instrument"}
LENDER_TAG_TYPES = {"person", "organization"}
INSTRUMENT_SINGLE_VALUE_PROPERTIES = {
    "start_date": {"date"},
    "end_date": {"date"},
    "amount": {"amount"},
    "name": {"debt_instrument"},
}
INSTRUMENT_RELATION_TYPES = {"amendment_of", "split_of"}
INSTRUMENT_MENTION_COLUMNS = [
    "instrument_mention_id",
    "item_id",
    "raw_id",
    "name",
    "start_date",
    "end_date",
    "amount",
    "amendment_of",
    "split_of",
    "lenders_json",
    "other_interested_parties_json",
    "mention_corefs_json",
    "start_date_corefs_json",
    "end_date_corefs_json",
    "amount_corefs_json",
    "instrument_mention_json",
]


class SupportsChatCompletion(Protocol):
    """Protocol for chat-capable extractor clients."""

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Return one chat completion as plain text."""


@dataclass
class AttemptRecord:
    """Recorded data for one stage attempt."""

    stage_name: str
    attempt_index: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    response: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    status: str = "incomplete"

    def to_dict(self) -> dict[str, object]:
        """Convert one attempt to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ExtractionRowState:
    """Mutable row-level extractor state."""

    item_row: dict[str, object]
    stage_name: str
    all_attempts: list[AttemptRecord] = field(default_factory=list)
    stage_responses: dict[str, str] = field(default_factory=dict)
    instrument_mentions: list[dict[str, object]] = field(default_factory=list)
    ner_tagged_xml: str | None = None
    state: str | None = None
    current_attempt: AttemptRecord = field(init=False)

    def __post_init__(self) -> None:
        self.current_attempt = AttemptRecord(stage_name=self.stage_name)

    @property
    def item_id(self) -> str:
        """Return the item identifier."""
        return str(self.item_row["item_id"])

    @property
    def text(self) -> str:
        """Return the item text."""
        return str(self.item_row.get("text", ""))

    def add_messages(self, messages: list[dict[str, str]]) -> None:
        """Append prompt messages to the current attempt."""
        self.current_attempt.messages.extend(messages)

    def add_response(self, response: str) -> None:
        """Record the latest model response."""
        self.current_attempt.response = response
        self.current_attempt.attempt_index += 1
        self.stage_responses[self.current_attempt.stage_name] = response

    def add_validation(self, failures: list[str]) -> None:
        """Record validation output for the current attempt."""
        self.current_attempt.validation_errors = failures
        self.current_attempt.status = "FAILED" if failures else "SUCCESS"

    def retry(self, retry_message: str) -> None:
        """Prepare a retry attempt for the current stage."""
        self.all_attempts.append(self.current_attempt)
        new_messages = list(self.current_attempt.messages)
        new_messages.append({"role": "user", "content": retry_message})
        self.current_attempt = AttemptRecord(
            stage_name=self.current_attempt.stage_name,
            attempt_index=self.current_attempt.attempt_index,
            messages=new_messages,
        )

    def finish(self, state: str) -> None:
        """Finish processing for this row."""
        self.all_attempts.append(self.current_attempt)
        self.state = state

    def next_stage(self, stage_name: str) -> None:
        """Advance to the next stage."""
        self.all_attempts.append(self.current_attempt)
        self.current_attempt = AttemptRecord(stage_name=stage_name)

    def to_audit_dict(self) -> dict[str, object]:
        """Return one audit record for full.jsonl."""
        attempts = [attempt.to_dict() for attempt in self.all_attempts]
        return {
            "item_id": self.item_id,
            "accession_number": self.item_row.get("accession_number"),
            "item": self.item_row.get("item"),
            "stage_responses": self.stage_responses,
            "instrument_mentions": self.instrument_mentions,
            "state": self.state,
            "attempts": attempts,
        }


class StageSpec(Protocol):
    """Minimal stage interface for the local extractor workflow."""

    name: str

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        """Build messages for the LLM."""

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        """Validate raw stage output."""

    def postprocess(self, row_state: ExtractionRowState) -> None:
        """Mutate row state after validation succeeds."""

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        """Return whether the row should finish early."""

    def build_retry_message(self, failures: list[str]) -> str:
        """Build retry guidance after a validation failure."""


class OpenRouterChatClient:
    """Native OpenRouter client wrapper used by the extractor."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for cdt extractor.")

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Run one OpenRouter chat completion."""
        from openrouter import OpenRouter

        request_kwargs: dict[str, object] = {
            "messages": messages,
            "model": model,
            "stream": False,
            "temperature": 0.0,
        }
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        async with OpenRouter(
            api_key=self.api_key,
            x_open_router_title="commercial-debt-tracker",
            x_open_router_categories="cli-agent",
        ) as client:
            response = await client.chat.send_async(**request_kwargs)
        return extract_response_text(response)


class NERStage:
    """NER stage using XML-tagged output."""

    name = "ner"

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        prompt = load_prompt("ner")
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"<body>{row_state.text}</body>"},
        ]

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        if not response or not isinstance(response, str):
            return [
                "Model returned empty or non-text output. Even if no entities are present, return the input text."
            ]

        try:
            root = DefusedET.fromstring(response)
        except ET.ParseError as exc:
            return [f"Response is not valid XML: {exc}"]
        if root.tag != "body":
            return ["Response root must be <body>."]

        failures: list[str] = []
        allowed_tags = {
            "body",
            "person",
            "organization",
            "debt_instrument",
            "agreement",
            "date",
            "duration",
            "amount",
        }
        for element in root.iter():
            if element.tag not in allowed_tags:
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
        return failures

    def postprocess(self, row_state: ExtractionRowState) -> None:
        response = row_state.stage_responses.get(self.name)
        if not response:
            return
        row_state.ner_tagged_xml = assign_tag_ids(response)

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        if not row_state.ner_tagged_xml:
            return False
        _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        return not any(
            detail["type"] == "debt_instrument" for detail in tag_details.values()
        )

    def build_retry_message(self, failures: list[str]) -> str:
        return (
            "Your previous NER output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
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
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            return [f"Output is not valid JSON: {exc}"]
        if not isinstance(data, list):
            return ["Output must be a JSON array of objects."]

        _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        failures: list[str] = []
        name_tags_used: set[str] = set()
        for index, obj in enumerate(data):
            if not isinstance(obj, dict):
                failures.append(f"Entry {index} is not a JSON object.")
                continue
            for (
                property_name,
                expected_types,
            ) in INSTRUMENT_SINGLE_VALUE_PROPERTIES.items():
                if property_name not in obj:
                    continue
                tag_ids = obj[property_name]
                if not isinstance(tag_ids, list):
                    failures.append(
                        f"Entry {index}: '{property_name}' must be a list of tag IDs."
                    )
                    continue
                if not all(isinstance(tag_id, str) for tag_id in tag_ids):
                    failures.append(
                        f"Entry {index}: '{property_name}' must contain string tag IDs only."
                    )
                    continue
                for tag_id in tag_ids:
                    tag_info = tag_details.get(tag_id)
                    if tag_info is None:
                        failures.append(
                            f"Entry {index}: '{property_name}' contains unknown tag ID {tag_id}."
                        )
                        continue
                    if tag_info["type"] not in expected_types:
                        expected = ", ".join(sorted(expected_types))
                        failures.append(
                            f"Entry {index}: '{property_name}' tag {tag_id} is type '{tag_info['type']}', expected {expected}."
                        )
                if property_name == "name":
                    name_tags_used.update(tag_ids)

            for property_name in ("lenders", "other_interested_parties"):
                if property_name not in obj:
                    continue
                values = obj[property_name]
                if not isinstance(values, list):
                    failures.append(
                        f"Entry {index}: '{property_name}' must be a list of lists."
                    )
                    continue
                for cluster_index, cluster in enumerate(values):
                    if not isinstance(cluster, list):
                        failures.append(
                            f"Entry {index}: '{property_name}'[{cluster_index}] must be a list of tag IDs."
                        )
                        continue
                    if not all(isinstance(tag_id, str) for tag_id in cluster):
                        failures.append(
                            f"Entry {index}: '{property_name}'[{cluster_index}] must contain string tag IDs only."
                        )
                        continue
                    for tag_id in cluster:
                        tag_info = tag_details.get(tag_id)
                        if tag_info is None:
                            failures.append(
                                f"Entry {index}: '{property_name}'[{cluster_index}] contains unknown tag ID {tag_id}."
                            )
                            continue
                        if tag_info["type"] not in LENDER_TAG_TYPES:
                            failures.append(
                                f"Entry {index}: '{property_name}'[{cluster_index}] tag {tag_id} must be person or organization."
                            )

        instrument_tag_ids = {
            tag_id
            for tag_id, info in tag_details.items()
            if info["type"] in INSTRUMENT_ENTITY_TAG_TYPES
        }
        missing = instrument_tag_ids.difference(name_tags_used)
        if missing:
            failures.append(
                "Missing debt_instrument tags in 'name': " + ", ".join(sorted(missing))
            )

        counts: dict[str, int] = {}
        for obj in data:
            if isinstance(obj, dict):
                name_value = obj.get("name", [])
                if not isinstance(name_value, list):
                    continue
                if not all(isinstance(tag_id, str) for tag_id in name_value):
                    continue
                for tag_id in name_value:
                    counts[tag_id] = counts.get(tag_id, 0) + 1
        duplicates = [
            tag_id
            for tag_id, count in counts.items()
            if tag_id in instrument_tag_ids and count > 1
        ]
        if duplicates:
            failures.append(
                "Debt instrument tags appear in multiple 'name' clusters: "
                + ", ".join(sorted(duplicates))
            )
        return failures

    def postprocess(self, row_state: ExtractionRowState) -> None:
        if not row_state.ner_tagged_xml:
            return
        response = row_state.stage_responses.get(self.name)
        if not response:
            return
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return
        _, _, tag_details = parse_tag_details(row_state.ner_tagged_xml)
        mention_entries = iter_instrument_entries(
            cast(list[dict[str, Any]], data), tag_details
        )
        raw_to_global = {
            raw_id_for(index): instrument_mention_id_for(row_state.item_id, index)
            for index, _ in mention_entries
        }
        mentions: list[dict[str, object]] = []
        for index, obj in mention_entries:
            raw_id = raw_id_for(index)
            mention_row: dict[str, object] = {
                "instrument_mention_id": raw_to_global[raw_id],
                "item_id": row_state.item_id,
                "raw_id": raw_id,
                "name": canonical_value(obj.get("name", []), tag_details),
                "start_date": canonical_value(obj.get("start_date", []), tag_details),
                "end_date": canonical_value(obj.get("end_date", []), tag_details),
                "amount": canonical_value(obj.get("amount", []), tag_details),
                "amendment_of": None,
                "split_of": None,
                "lenders_json": json.dumps(
                    cluster_payload_list(obj.get("lenders", []), tag_details),
                    sort_keys=True,
                ),
                "other_interested_parties_json": json.dumps(
                    cluster_payload_list(
                        obj.get("other_interested_parties", []),
                        tag_details,
                    ),
                    sort_keys=True,
                ),
                "mention_corefs_json": json.dumps(
                    cluster_payload(obj.get("name", []), tag_details),
                    sort_keys=True,
                ),
                "start_date_corefs_json": json.dumps(
                    cluster_payload(obj.get("start_date", []), tag_details),
                    sort_keys=True,
                ),
                "end_date_corefs_json": json.dumps(
                    cluster_payload(obj.get("end_date", []), tag_details),
                    sort_keys=True,
                ),
                "amount_corefs_json": json.dumps(
                    cluster_payload(obj.get("amount", []), tag_details),
                    sort_keys=True,
                ),
            }
            mention_payload = {
                "instrument_mention_id": mention_row["instrument_mention_id"],
                "raw_id": raw_id,
                "name": cluster_payload(obj.get("name", []), tag_details),
                "start_date": cluster_payload(obj.get("start_date", []), tag_details),
                "end_date": cluster_payload(obj.get("end_date", []), tag_details),
                "amount": cluster_payload(obj.get("amount", []), tag_details),
                "lenders": cluster_payload_list(obj.get("lenders", []), tag_details),
                "other_interested_parties": cluster_payload_list(
                    obj.get("other_interested_parties", []),
                    tag_details,
                ),
            }
            mention_row["instrument_mention_json"] = json.dumps(
                mention_payload,
                sort_keys=True,
            )
            mentions.append(mention_row)
        row_state.instrument_mentions = mentions

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        return False

    def build_retry_message(self, failures: list[str]) -> str:
        return (
            "Your previous instrument extraction output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
            "- Return one JSON object per distinct debt instrument mention cluster.\n"
            "- Every debt_instrument tag must appear exactly once in a name cluster.\n"
            "- Do not return agreements as output objects.\n"
            "- Return only valid JSON."
        )


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
            str(mention["raw_id"]) for mention in row_state.instrument_mentions
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
            if rel_type not in INSTRUMENT_RELATION_TYPES:
                failures.append(
                    f"Invalid relation type: {rel_type}. Must be amendment_of or split_of."
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
            str(mention["raw_id"]): mention for mention in row_state.instrument_mentions
        }
        raw_to_global = {
            str(mention["raw_id"]): str(mention["instrument_mention_id"])
            for mention in row_state.instrument_mentions
        }
        for relation in data:
            mention = by_raw_id.get(relation["from"])
            if mention is None:
                continue
            mention[str(relation["type"])] = raw_to_global.get(relation["to"])
            mention_payload = json.loads(str(mention["instrument_mention_json"]))
            mention_payload[str(relation["type"])] = raw_to_global.get(relation["to"])
            mention["instrument_mention_json"] = json.dumps(
                mention_payload, sort_keys=True
            )

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        return False

    def build_retry_message(self, failures: list[str]) -> str:
        return (
            "Your previous instrument relation output failed validation.\n"
            f"Validation errors: {failures}\n"
            "Retry requirements:\n"
            "- Return a JSON array.\n"
            "- Each relation must have from, to, and type.\n"
            "- Use only amendment_of or split_of.\n"
            "- Use only instrument IDs from the input."
        )


def extracted_tables_path(data_dir: Path | None = None) -> Path:
    """Return the directory that stores extractor run artifacts."""
    return (data_dir or settings.DATA_DIR) / "extractor_runs"


def extract_pending_items(
    *,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client: SupportsChatCompletion | None = None,
) -> pd.DataFrame:
    """Extract instrument mentions for pending classified relevant items."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    resolved_model = model or settings.EXTRACTOR_MODEL or DEFAULT_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_client = client or OpenRouterChatClient()
    run_dir = create_run_dir(data_dir)
    full_jsonl_path = run_dir / "full.jsonl"
    conn = connect_cdt_db(cdt_db_path(data_dir))
    processed_item_ids: set[str] = set()
    processed_frames: list[pd.DataFrame] = []
    successful_item_ids: list[str] = []
    failed_rows: list[dict[str, str]] = []
    try:
        while True:
            index_rows = read_items(
                conn,
                statuses=("classified",)
                if not force
                else ("classified", "extracted", "extraction_failed"),
                exclude_item_ids=processed_item_ids,
                limit=batch_size,
            )
            if not index_rows:
                break
            processed_item_ids.update(str(row["item_id"]) for row in index_rows)
            relevant_rows = [row for row in index_rows if bool(row.get("relevance"))]
            if not relevant_rows:
                continue
            batch_items = load_pending_item_batches(relevant_rows, data_dir=data_dir)
            for item_row in batch_items.to_dict("records"):
                row_state = asyncio.run(
                    run_extraction_workflow(
                        item_row=item_row,
                        model=resolved_model,
                        reasoning_effort=resolved_reasoning,
                        max_attempts=max_attempts,
                        client=resolved_client,
                    )
                )
                append_audit_record(full_jsonl_path, row_state.to_audit_dict())
                replace_instrument_mentions(
                    conn,
                    row_state.item_id,
                    row_state.instrument_mentions,
                )
                if row_state.state == "SUCCESS":
                    successful_item_ids.append(row_state.item_id)
                    if row_state.instrument_mentions:
                        processed_frames.append(
                            pd.DataFrame(
                                row_state.instrument_mentions,
                                columns=INSTRUMENT_MENTION_COLUMNS,
                            )
                        )
                else:
                    failed_rows.append(
                        {
                            "item_id": row_state.item_id,
                            "extractor_error": summarize_failure(row_state),
                        }
                    )
    finally:
        if successful_item_ids:
            mark_items_extracted(
                conn,
                successful_item_ids,
                extractor_model=resolved_model,
                extractor_reasoning=resolved_reasoning,
                extractor_run_path=str(run_dir),
            )
        if failed_rows:
            mark_items_extraction_failed(
                conn,
                failed_rows,
                extractor_model=resolved_model,
                extractor_reasoning=resolved_reasoning,
                extractor_run_path=str(run_dir),
            )
        conn.close()

    LOGGER.info(
        "Extractor complete: successes=%s failures=%s mentions=%s run_dir=%s",
        len(successful_item_ids),
        len(failed_rows),
        sum(len(frame) for frame in processed_frames),
        run_dir,
    )
    if not processed_frames:
        return pd.DataFrame(columns=INSTRUMENT_MENTION_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=INSTRUMENT_MENTION_COLUMNS
    )


def extract_tables(
    classified_items: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    force: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client: SupportsChatCompletion | None = None,
) -> dict[str, pd.DataFrame]:
    """Run in-memory extraction and return instrument mention tables."""
    del force
    if classified_items.empty:
        return {"instrument_mentions": pd.DataFrame(columns=INSTRUMENT_MENTION_COLUMNS)}
    relevant_items = (
        classified_items.loc[classified_items["relevance"].fillna(False)]
        if "relevance" in classified_items
        else classified_items
    )
    if relevant_items.empty:
        return {"instrument_mentions": pd.DataFrame(columns=INSTRUMENT_MENTION_COLUMNS)}
    resolved_model = model or settings.EXTRACTOR_MODEL or DEFAULT_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_client = client or OpenRouterChatClient()
    run_dir = create_run_dir(data_dir)
    full_jsonl_path = run_dir / "full.jsonl"
    rows: list[dict[str, object]] = []
    for item_row in relevant_items.to_dict("records"):
        row_state = asyncio.run(
            run_extraction_workflow(
                item_row=item_row,
                model=resolved_model,
                reasoning_effort=resolved_reasoning,
                max_attempts=max_attempts,
                client=resolved_client,
            )
        )
        append_audit_record(full_jsonl_path, row_state.to_audit_dict())
        if row_state.state == "SUCCESS":
            rows.extend(row_state.instrument_mentions)
        else:
            LOGGER.warning(
                "In-memory extractor failed for item %s: %s",
                row_state.item_id,
                summarize_failure(row_state),
            )
    return {
        "instrument_mentions": pd.DataFrame(rows, columns=INSTRUMENT_MENTION_COLUMNS)
    }


async def run_extraction_workflow(
    *,
    item_row: dict[str, object],
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    client: SupportsChatCompletion,
) -> ExtractionRowState:
    """Run the three-stage extraction workflow for one item row."""
    stages: list[StageSpec] = [
        NERStage(),
        InstrumentIEStage(),
        InstrumentRelationStage(),
    ]
    row_state = ExtractionRowState(item_row=item_row, stage_name=stages[0].name)
    for stage_index, stage in enumerate(stages):
        row_state.current_attempt.stage_name = stage.name
        row_state.current_attempt.messages = []
        try:
            row_state.add_messages(stage.preprocess(row_state))
        except Exception as exc:  # noqa: BLE001
            row_state.current_attempt.validation_errors = [
                f"{type(exc).__name__}: {exc}"
            ]
            row_state.current_attempt.status = "ERROR"
            row_state.finish("ERROR")
            return row_state

        while True:
            try:
                response = await client.complete(
                    messages=row_state.current_attempt.messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:  # noqa: BLE001
                row_state.current_attempt.validation_errors = [
                    f"{type(exc).__name__}: {exc}"
                ]
                row_state.current_attempt.status = "ERROR"
                row_state.finish("ERROR")
                return row_state

            row_state.add_response(response)
            failures = stage.validate(row_state, response)
            row_state.add_validation(failures)
            if not failures:
                stage.postprocess(row_state)
                if stage.early_stop(row_state):
                    row_state.finish("SUCCESS")
                    return row_state
                if stage_index == len(stages) - 1:
                    row_state.finish("SUCCESS")
                    return row_state
                if (
                    stages[stage_index + 1].name == "instrument_relation"
                    and len(row_state.instrument_mentions) <= 1
                ):
                    row_state.finish("SUCCESS")
                    return row_state
                row_state.next_stage(stages[stage_index + 1].name)
                break

            if row_state.current_attempt.attempt_index >= max_attempts:
                row_state.finish("FAILED")
                return row_state
            row_state.retry(stage.build_retry_message(failures))
    row_state.finish("SUCCESS")
    return row_state


def load_prompt(name: str) -> str:
    """Load one extractor prompt from the local package."""
    return (
        resources.files("cdt.extractor.prompts")
        .joinpath(f"{name}.md")
        .read_text(encoding="utf-8")
    )


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


def extract_response_text(response: object) -> str:
    """Extract the assistant text from an OpenRouter SDK response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("OpenRouter response did not include choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("OpenRouter response did not include a message.")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    raise RuntimeError("OpenRouter response content was not text.")


def collapse_whitespace(value: str) -> str:
    """Collapse all whitespace in a string for comparison."""
    return re.sub(r"\s+", "", value)


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


def raw_id_for(index: int) -> str:
    """Return the stage-local raw instrument-mention ID."""
    return f"i-{index}"


def instrument_mention_id_for(item_id: str, index: int) -> str:
    """Return a stable persisted instrument-mention ID."""
    return f"{item_id}--i-{index}"


def canonical_value(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the longest textual member of one coreference cluster."""
    if not isinstance(tag_ids, list) or not tag_ids:
        return None
    values = [
        str(tag_details[tag_id]["text"])
        for tag_id in tag_ids
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    if not values:
        return None
    return max(values, key=len)


def cluster_payload(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return rich details for one cluster."""
    if not isinstance(tag_ids, list):
        return {"tag_ids": [], "mentions": []}
    mentions = [
        {
            "tag_id": tag_id,
            "type": tag_details[tag_id]["type"],
            "text": tag_details[tag_id]["text"],
            "char_start": tag_details[tag_id]["char_start"],
            "char_end": tag_details[tag_id]["char_end"],
        }
        for tag_id in tag_ids
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    return {
        "tag_ids": [mention["tag_id"] for mention in mentions],
        "mentions": mentions,
    }


def cluster_payload_list(
    clusters: object,
    tag_details: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return rich details for a list of clusters."""
    if not isinstance(clusters, list):
        return []
    return [cluster_payload(cluster, tag_details) for cluster in clusters]


def relation_prompt_xml(row_state: ExtractionRowState) -> str:
    """Build relation-stage XML with instrument-id attributes."""
    if not row_state.ner_tagged_xml:
        raise ValueError("ner_tagged_xml is required for instrument_relation.")
    root, _, _ = parse_tag_details(row_state.ner_tagged_xml)
    tag_to_raw_id: dict[str, str] = {}
    for mention in row_state.instrument_mentions:
        payload = json.loads(str(mention["mention_corefs_json"]))
        for tag_id in payload.get("tag_ids", []):
            tag_to_raw_id[str(tag_id)] = str(mention["raw_id"])
    body = render_relation_body(root, tag_to_raw_id)
    return f"<body>{body}</body>"


def render_relation_body(root: ET.Element, tag_to_raw_id: dict[str, str]) -> str:
    """Render only debt instrument tags needed for relation extraction."""

    def render_element(element: ET.Element) -> str:
        parts: list[str] = [element.text or ""]
        for child in list(element):
            rendered = render_element(child)
            tag_id = child.attrib.get("id")
            if child.tag == "debt_instrument" and tag_id in tag_to_raw_id:
                parts.append(
                    f'<debt_instrument instrument-id="{tag_to_raw_id[tag_id]}">{rendered}</debt_instrument>'
                )
            else:
                parts.append(rendered)
            parts.append(child.tail or "")
        return "".join(parts)

    return render_element(root)


def create_run_dir(data_dir: Path | None = None) -> Path:
    """Create and return one extractor run directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = extracted_tables_path(data_dir) / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_audit_record(path: Path, record: dict[str, object]) -> None:
    """Append one JSONL audit record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        json.dump(record, file_obj, sort_keys=True)
        file_obj.write("\n")


def summarize_failure(row_state: ExtractionRowState) -> str:
    """Summarize the last failure message for one row."""
    failures = row_state.current_attempt.validation_errors
    if failures:
        return "; ".join(failures)
    if row_state.current_attempt.response:
        return f"Unexpected response at stage {row_state.current_attempt.stage_name}"
    return f"Extractor failed at stage {row_state.current_attempt.stage_name}"


def normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    """Resolve and validate configured reasoning effort."""
    resolved = (
        reasoning_effort or settings.EXTRACTOR_REASONING or DEFAULT_REASONING_EFFORT
    )
    if resolved not in REASONING_EFFORTS:
        allowed = ", ".join(sorted(REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported reasoning effort {resolved!r}; expected one of {allowed}"
        )
    return resolved
