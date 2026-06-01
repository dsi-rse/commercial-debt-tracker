"""LLM-backed extractor stage for relevant SEC 8-K items."""

# ruff: noqa: ANN101, D102, D105, D107

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from importlib import resources
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast
from xml.etree import ElementTree as ET

import pandas as pd
from defusedxml import ElementTree as DefusedET

from cdt import settings
from cdt.classifier.core import CLASSIFICATION_DATASET_NAME, CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    completion_registry_path,
    dataset_root,
    date_shard_partition_path,
    extractor_run_path,
    iter_date_shard_partitions,
    load_completed_partitions,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_completed_partitions,
)
from cdt.shared import get_logger
from cdt.storage import (
    artifact_exists,
    read_table,
    write_json_artifact,
    write_partition_table,
    write_text_artifact,
)

LOGGER = get_logger(__name__)
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
STANDARDIZED_SINGLE_VALUE_PROPERTIES = {"start_date", "end_date", "amount"}
INSTRUMENT_RELATION_TYPES = {"amendment_of", "split_of"}
NUMERIC_STRING_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}
AMOUNT_MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}
COMMON_CURRENCY_CODES = {
    "AED",
    "AUD",
    "BRL",
    "CAD",
    "CHF",
    "CNY",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "INR",
    "JPY",
    "KRW",
    "MXN",
    "NOK",
    "NZD",
    "SAR",
    "SEK",
    "SGD",
    "TRY",
    "USD",
    "ZAR",
}
CURRENCY_CODE_LENGTH = 3
EXTRACTOR_PROGRESS_LOG_INTERVAL = 10
_SUPPORTED_CURRENCY_CODES: set[str] | None = None
DEBT_INSTRUMENT_MENTION_COLUMNS = [
    "debt_instrument_mention_id",
    "item_id",
    "accession_number",
    "cik",
    "date",
    "raw_id",
    "name",
    "start_date",
    "end_date",
    "amount",
    "amendment_of",
    "split_of",
    "lenders_json",
    "other_interested_parties_json",
    "name_json",
    "start_date_json",
    "end_date_json",
    "amount_json",
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
    debt_instrument_mentions: list[dict[str, object]] = field(default_factory=list)
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
            "debt_instrument_mentions": self.debt_instrument_mentions,
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
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for cdt extractor. "
                "OPENROUTER_API_TOKEN is also accepted as a compatibility alias."
            )

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
                tag_ids = single_value_evidence_tag_ids(obj[property_name])
                if property_name in STANDARDIZED_SINGLE_VALUE_PROPERTIES:
                    failures.extend(
                        validate_standardized_single_value_shape(
                            index=index,
                            property_name=property_name,
                            value=obj[property_name],
                        )
                    )
                if not isinstance(tag_ids, list):
                    failures.append(
                        f"Entry {index}: '{property_name}' evidence must be a list of tag IDs."
                    )
                    continue
                if not all(isinstance(tag_id, str) for tag_id in tag_ids):
                    failures.append(
                        f"Entry {index}: '{property_name}' evidence must contain string tag IDs only."
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
        mentions: list[dict[str, object]] = []
        for index, obj in mention_entries:
            raw_id = raw_id_for(index)
            amount_payload = standardized_amount_payload(
                obj.get("amount"),
                tag_details,
            )
            start_date_payload = standardized_date_payload(
                obj.get("start_date"),
                tag_details,
            )
            end_date_payload = standardized_date_payload(
                obj.get("end_date"),
                tag_details,
            )
            mention_row: dict[str, object] = {
                "item_id": row_state.item_id,
                "accession_number": row_state.item_row.get("accession_number"),
                "cik": row_state.item_row.get("cik"),
                "date": row_state.item_row.get("date"),
                "raw_id": raw_id,
                "name": canonical_value(obj.get("name", []), tag_details),
                "start_date": start_date_payload["normalized_date"],
                "end_date": end_date_payload["normalized_date"],
                "amount": amount_payload["normalized_amount"],
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
                "name_json": json.dumps(
                    cluster_payload(obj.get("name", []), tag_details),
                    sort_keys=True,
                ),
                "start_date_json": json.dumps(start_date_payload, sort_keys=True),
                "end_date_json": json.dumps(end_date_payload, sort_keys=True),
                "amount_json": json.dumps(amount_payload, sort_keys=True),
            }
            mention_row["debt_instrument_mention_id"] = debt_instrument_mention_id_for(
                row_state.item_id,
                mention_row,
            )
            mentions.append(mention_row)
        row_state.debt_instrument_mentions = mentions

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
            str(mention["raw_id"]): mention
            for mention in row_state.debt_instrument_mentions
        }
        raw_to_global = {
            str(mention["raw_id"]): str(mention["debt_instrument_mention_id"])
            for mention in row_state.debt_instrument_mentions
        }
        for relation in data:
            mention = by_raw_id.get(relation["from"])
            if mention is None:
                continue
            mention[str(relation["type"])] = raw_to_global.get(relation["to"])

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


MENTIONS_DATASET_NAME = "mentions"


def mentions_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical mentions dataset root."""
    return dataset_root(
        MENTIONS_DATASET_NAME, artifact_root=artifact_root, data_dir=data_dir
    )


def extracted_tables_path(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the root that stores extractor audit artifacts."""
    return dataset_root(
        "extractor-runs", artifact_root=artifact_root, data_dir=data_dir
    )


def extract_pending_items(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client: SupportsChatCompletion | None = None,
) -> pd.DataFrame:
    """Extract instrument mentions for classified item partitions."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    resolved_model = model or settings.EXTRACTOR_MODEL or DEFAULT_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    processed_frames: list[pd.DataFrame] = []
    failed_rows: list[dict[str, str]] = []
    audit_records: list[str] = []
    partitions_written: list[str] = []
    completed_classification_paths = (
        set()
        if force
        else load_completed_partitions(
            "extract", artifact_root=resolved_root, data_dir=data_dir
        )
    )
    visited_classification_paths: set[str] = set()
    empty_partitions = 0
    pending_classification_paths: list[str] = []

    for classification_path in iter_date_shard_partitions(
        CLASSIFICATION_DATASET_NAME,
        artifact_root=resolved_root,
        data_dir=data_dir,
    ):
        partition = parse_date_shard_partition(classification_path)
        target_path = date_shard_partition_path(
            MENTIONS_DATASET_NAME,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        if not force and artifact_exists(target_path):
            continue
        if not force and classification_path in completed_classification_paths:
            continue
        pending_classification_paths.append(classification_path)

    total_partitions = len(pending_classification_paths)
    for chunk_start in range(0, total_partitions, batch_size):
        chunk_paths = pending_classification_paths[
            chunk_start : chunk_start + batch_size
        ]
        for partition_index, classification_path in enumerate(
            chunk_paths, start=chunk_start + 1
        ):
            partition = parse_date_shard_partition(classification_path)
            partition_label = f"date={partition['date']} shard={partition['shard']}"
            partition_start = perf_counter()
            visited_classification_paths.add(classification_path)
            batch_items = read_table(
                classification_path,
                CLASSIFIED_ITEM_COLUMNS,
            ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
            relevant_items = batch_items.loc[batch_items["relevance"].fillna(False)]
            mention_rows: list[dict[str, object]] = []
            partition_failures = 0
            relevant_records = relevant_items.to_dict("records")
            total_relevant_items = len(relevant_records)
            for item_index, item_row in enumerate(relevant_records, start=1):
                row_state = asyncio.run(
                    run_extraction_workflow(
                        item_row=item_row,
                        model=resolved_model,
                        reasoning_effort=resolved_reasoning,
                        max_attempts=max_attempts,
                        client=client,
                    )
                )
                audit_records.append(
                    json.dumps(row_state.to_audit_dict(), sort_keys=True)
                )
                if row_state.state == "SUCCESS":
                    mention_rows.extend(row_state.debt_instrument_mentions)
                else:
                    failed_rows.append(
                        {
                            "item_id": row_state.item_id,
                            "extractor_error": summarize_failure(row_state),
                        }
                    )
                    partition_failures += 1
                if (
                    item_index == total_relevant_items
                    or item_index % EXTRACTOR_PROGRESS_LOG_INTERVAL == 0
                ):
                    LOGGER.info(
                        "Extractor item progress: %s partition=%s/%s items=%s/%s mentions=%s failures=%s elapsed=%.1fs",
                        partition_label,
                        partition_index,
                        total_partitions,
                        item_index,
                        total_relevant_items,
                        len(mention_rows),
                        partition_failures,
                        perf_counter() - partition_start,
                    )
            mentions = pd.DataFrame(
                mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
            if mentions.empty:
                empty_partitions += 1
            else:
                write_partition_table(
                    mentions_root(resolved_root, data_dir=data_dir),
                    partition={"date": partition["date"], "shard": partition["shard"]},
                    table=mentions.reindex(columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
                )
                processed_frames.append(mentions)
                partitions_written.append(
                    date_shard_partition_path(
                        MENTIONS_DATASET_NAME,
                        partition_date=partition["date"],
                        shard=partition["shard"],
                        artifact_root=resolved_root,
                        data_dir=data_dir,
                    )
                )
            LOGGER.info(
                "Extraction partition complete: %s progress=%s/%s classified_items=%s relevant_items=%s mentions=%s wrote_output=%s elapsed=%.1fs",
                partition_label,
                partition_index,
                total_partitions,
                len(batch_items),
                total_relevant_items,
                len(mentions),
                not mentions.empty,
                perf_counter() - partition_start,
            )

    updated_completed_paths = (
        completed_classification_paths | visited_classification_paths
    )
    save_completed_partitions(
        "extract",
        updated_completed_paths,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    if audit_records:
        write_text_artifact(full_jsonl_path, "\n".join(audit_records) + "\n")
    else:
        write_text_artifact(full_jsonl_path, "")
    write_json_artifact(
        run_manifest_path(
            "extract",
            run_id,
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        {
            "artifact_root": resolved_root,
            "stage": "extract",
            "batch_size": batch_size,
            "force": force,
            "model": resolved_model,
            "reasoning_effort": resolved_reasoning,
            "max_attempts": max_attempts,
            "partitions_visited": sorted(visited_classification_paths),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "failure_count": len(failed_rows),
            "audit_path": full_jsonl_path,
            "completion_registry": completion_registry_path(
                "extract", artifact_root=resolved_root, data_dir=data_dir
            ),
        },
    )

    LOGGER.info(
        "Extractor complete: successes=%s failures=%s mentions=%s run_dir=%s",
        sum(
            len(frame["item_id"].unique())
            for frame in processed_frames
            if not frame.empty
        ),
        len(failed_rows),
        sum(len(frame) for frame in processed_frames),
        full_jsonl_path,
    )
    if not processed_frames:
        return pd.DataFrame(columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=DEBT_INSTRUMENT_MENTION_COLUMNS
    )


def extract_tables(
    classified_items: pd.DataFrame,
    *,
    artifact_root: str | Path | None = None,
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
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
        }
    relevant_items = (
        classified_items.loc[classified_items["relevance"].fillna(False)]
        if "relevance" in classified_items
        else classified_items
    )
    if relevant_items.empty:
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
        }
    resolved_model = model or settings.EXTRACTOR_MODEL or DEFAULT_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    rows: list[dict[str, object]] = []
    audit_records: list[str] = []
    for item_row in relevant_items.to_dict("records"):
        row_state = asyncio.run(
            run_extraction_workflow(
                item_row=item_row,
                model=resolved_model,
                reasoning_effort=resolved_reasoning,
                max_attempts=max_attempts,
                client=client,
            )
        )
        audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
        if row_state.state == "SUCCESS":
            rows.extend(row_state.debt_instrument_mentions)
        else:
            LOGGER.warning(
                "In-memory extractor failed for item %s: %s",
                row_state.item_id,
                summarize_failure(row_state),
            )
    write_text_artifact(
        full_jsonl_path,
        ("\n".join(audit_records) + "\n") if audit_records else "",
    )
    return {
        "debt_instrument_mentions": pd.DataFrame(
            rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
        )
    }


async def run_extraction_workflow(
    *,
    item_row: dict[str, object],
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    client: SupportsChatCompletion | None = None,
) -> ExtractionRowState:
    """Run the three-stage extraction workflow for one item row."""
    resolved_client = client or OpenRouterChatClient()
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
                response = await resolved_client.complete(
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
                    and len(row_state.debt_instrument_mentions) <= 1
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


def single_value_evidence_tag_ids(value: object) -> object:
    """Return evidence tag IDs from one single-value extractor payload."""
    if isinstance(value, dict):
        return value.get("evidence", [])
    return value


def validate_standardized_single_value_shape(
    *,
    index: int,
    property_name: str,
    value: object,
) -> list[str]:
    """Validate one standardized single-value object shape."""
    if not isinstance(value, dict):
        return [f"Entry {index}: '{property_name}' must be an object."]
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return [f"Entry {index}: '{property_name}.evidence' must be a list of tag IDs."]

    failures: list[str] = []
    if property_name == "amount":
        normalized_amount = value.get("normalized_amount")
        currency = value.get("currency")
        if normalized_amount is not None and (
            not isinstance(normalized_amount, str)
            or not NUMERIC_STRING_PATTERN.fullmatch(normalized_amount)
        ):
            failures.append(
                f"Entry {index}: 'amount.normalized_amount' must be a numeric string or null."
            )
        if currency is not None and (
            not isinstance(currency, str)
            or len(currency) != CURRENCY_CODE_LENGTH
            or currency != currency.upper()
        ):
            failures.append(
                f"Entry {index}: 'amount.currency' must be an uppercase 3-letter code or null."
            )
    else:
        normalized_date = value.get("normalized_date")
        if normalized_date is not None and (
            not isinstance(normalized_date, str)
            or not ISO_DATE_PATTERN.fullmatch(normalized_date)
            or not is_valid_iso_date(normalized_date)
        ):
            failures.append(
                f"Entry {index}: '{property_name}.normalized_date' must be YYYY-MM-DD or null."
            )
    return failures


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


def debt_instrument_mention_id_for(
    item_id: str,
    mention_row: dict[str, object],
) -> str:
    """Return a stable persisted debt-instrument-mention ID."""
    payload = {
        "amount": mention_row.get("amount"),
        "amount_json": normalize_json_text(mention_row.get("amount_json")),
        "end_date": mention_row.get("end_date"),
        "end_date_json": normalize_json_text(mention_row.get("end_date_json")),
        "item_id": item_id,
        "lenders_json": normalize_json_text(mention_row.get("lenders_json")),
        "name_json": normalize_json_text(mention_row.get("name_json")),
        "name": mention_row.get("name"),
        "other_interested_parties_json": normalize_json_text(
            mention_row.get("other_interested_parties_json")
        ),
        "start_date": mention_row.get("start_date"),
        "start_date_json": normalize_json_text(mention_row.get("start_date_json")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"dim::{digest}"


def normalize_json_text(value: object) -> str:
    """Normalize one JSON-encoded payload for deterministic hashing."""
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return str(value)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


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


def is_valid_iso_date(value: str) -> bool:
    """Return whether one ISO date string is valid."""
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def supported_currency_codes() -> set[str]:
    """Return supported ISO 4217 codes."""
    global _SUPPORTED_CURRENCY_CODES
    if _SUPPORTED_CURRENCY_CODES is not None:
        return _SUPPORTED_CURRENCY_CODES

    codes: set[str] = set(COMMON_CURRENCY_CODES)
    try:
        import pycountry

        codes.update(
            currency.alpha_3
            for currency in pycountry.currencies
            if getattr(currency, "alpha_3", None)
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug("pycountry unavailable; falling back to bundled currency set.")
    _SUPPORTED_CURRENCY_CODES = codes
    return _SUPPORTED_CURRENCY_CODES


def normalize_numeric_string(value: float) -> str:
    """Return one deterministic numeric string."""
    if value.is_integer():
        return str(int(value))
    return f"{value:.12f}".rstrip("0").rstrip(".")


def normalized_amount_from_text(text: str | None) -> str | None:
    """Parse one amount mention into a normalized numeric string."""
    if not text:
        return None
    lowered = text.lower().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return None
    amount = float(match.group(0))
    for word, multiplier in AMOUNT_MULTIPLIERS.items():
        if re.search(rf"\b{word}\b", lowered):
            amount *= multiplier
            break
    return normalize_numeric_string(amount)


def currency_candidates_from_text(text: str | None) -> set[str]:
    """Infer plausible ISO currency codes from one amount mention."""
    if not text:
        return set()
    lowered = text.lower()
    candidates: set[str] = set()
    if "$" in text or "u.s. dollar" in lowered or "us dollar" in lowered:
        candidates.add("USD")
    if "€" in text or " euro" in lowered:
        candidates.add("EUR")
    if "£" in text or " pound sterling" in lowered or " british pound" in lowered:
        candidates.add("GBP")
    if "¥" in text or " yen" in lowered:
        candidates.add("JPY")
    for match in re.findall(r"\b[A-Z]{3}\b", text):
        if match in supported_currency_codes():
            candidates.add(match)
    return candidates


def normalized_date_from_text(text: str | None) -> str | None:
    """Parse one date mention into ISO format."""
    if not text:
        return None
    stripped = text.strip()
    if ISO_DATE_PATTERN.fullmatch(stripped) and is_valid_iso_date(stripped):
        return stripped
    match = re.search(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})",
        stripped,
    )
    if not match:
        return None
    month = MONTH_MAP.get(match.group("month").lower())
    if month is None:
        return None
    normalized = f"{match.group('year')}-{month}-{int(match.group('day')):02d}"
    return normalized if is_valid_iso_date(normalized) else None


def standardized_amount_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return evidence payload plus validated normalized amount fields."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_value(evidence_tag_ids, tag_details)
    parsed_amount = normalized_amount_from_text(evidence_text)
    parsed_currency_candidates = currency_candidates_from_text(evidence_text)
    model_amount = value.get("normalized_amount") if isinstance(value, dict) else None
    model_currency = value.get("currency") if isinstance(value, dict) else None

    payload["normalized_amount"] = (
        model_amount
        if isinstance(model_amount, str) and parsed_amount == model_amount
        else None
    )
    payload["currency"] = (
        model_currency
        if isinstance(model_currency, str)
        and model_currency in supported_currency_codes()
        and model_currency in parsed_currency_candidates
        else None
    )
    return payload


def standardized_date_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return evidence payload plus validated normalized date field."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_value(evidence_tag_ids, tag_details)
    parsed_date = normalized_date_from_text(evidence_text)
    model_date = value.get("normalized_date") if isinstance(value, dict) else None
    payload["normalized_date"] = (
        model_date
        if isinstance(model_date, str) and model_date == parsed_date
        else None
    )
    return payload


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
    for mention in row_state.debt_instrument_mentions:
        payload = json.loads(str(mention["name_json"]))
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
