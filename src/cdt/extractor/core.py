"""LLM-backed extractor stage for relevant SEC 8-K items."""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
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
    PARTITION_PATTERN,
    CompletedPartition,
    completion_registry_path,
    dataset_root,
    date_shard_partition_path,
    existing_date_shard_partition_ids,
    extractor_run_path,
    load_completion_registry,
    load_row_failures,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_completion_registry,
    save_row_failures,
)
from cdt.shared import get_logger
from cdt.storage import (
    artifact_exists,
    list_artifacts_with_versions,
    read_table,
    write_json_artifact,
    write_partition_table,
    write_text_artifact,
)

LOGGER = get_logger(__name__)
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MODEL = settings.DEFAULT_EXTRACTOR_MODEL
DEFAULT_REASONING_EFFORT = "none"
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
EXTRACTOR_TEMPERATURE = 0.0
# Reasoning models take a reasoning_effort and reject temperature != 1, so both
# backends must decide sampling params the same way or the same model produces
# different output live versus in batch. Prefixes are matched against the native
# id, so both "gpt-5.4" and "openai/gpt-5.4" resolve identically.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
INSTRUMENT_ENTITY_TAG_TYPES = {"debt_instrument"}
LENDER_TAG_TYPES = {"person", "organization"}
LENDER_CLUSTER_KINDS = {"named", "collective"}
DEFAULT_LENDER_CLUSTER_KIND = "named"
OTHER_PARTY_ROLES = {
    "agent",
    "trustee",
    "underwriter",
    "guarantor",
    "borrower",
    "other",
}
DEFAULT_OTHER_PARTY_ROLE = "other"
BORROWER_PARTY_ROLE = "borrower"
COLLECTIVE_LENDER_KIND = "collective"
# The model labels every party cluster so collective lenders and the borrower can
# be dropped here. The labels are extraction-time signals and are not persisted.
PARTY_PROPERTY_ANNOTATIONS = {
    "lenders": ("kind", LENDER_CLUSTER_KINDS, DEFAULT_LENDER_CLUSTER_KIND),
    "other_interested_parties": (
        "role",
        OTHER_PARTY_ROLES,
        DEFAULT_OTHER_PARTY_ROLE,
    ),
}
INSTRUMENT_SINGLE_VALUE_PROPERTIES = {
    "start_date": {"date"},
    # NER tags maturity phrases like "notes due 2028" inside the instrument name,
    # so end_date evidence may cite that name span instead of a standalone date.
    "end_date": {"date", "debt_instrument"},
    "amount": {"amount"},
    "name": {"debt_instrument"},
}
MATURITY_EVIDENCE_TAG_TYPES = {"debt_instrument"}
STANDARDIZED_SINGLE_VALUE_PROPERTIES = {"start_date", "end_date", "amount"}
INSTRUMENT_RELATION_TYPES = {"amendment_of", "retired_of", "split_of"}
NUMERIC_STRING_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
# One `due` can carry a list of maturities: `due 2028 and 2030`,
# `due October 1, 2028 and 2030`, `due October 1, 2028 and October 1, 2030`. Each
# is two maturities, and matching only the first silently invented one (#104).
MATURITY_COORDINATED_YEARS = (
    r"(?:\s*(?:,|/|&|and(?:/or)?|or)\s*(?:[A-Za-z]+\s+\d{1,2},?\s+)?\d{4})*"
)
MATURITY_FULL_DATE_PATTERN = re.compile(
    r"\bdue\s+(?:on\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    rf"(?P<more>{MATURITY_COORDINATED_YEARS})",
    re.IGNORECASE,
)
MATURITY_YEAR_PATTERN = re.compile(
    rf"\bdue\s+(?:in\s+)?(?P<years>\d{{4}}{MATURITY_COORDINATED_YEARS})\b",
    re.IGNORECASE,
)
FOUR_DIGIT_YEAR_PATTERN = re.compile(r"\d{4}")
YEAR_ONLY_MATURITY_SUFFIX = "-12-31"
# A rate marker counts only where it sits on a number, so the value the parser
# would read is the rate itself rather than a percentage of something else (#103).
AMOUNT_VALUE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
RATE_SUFFIX_PATTERN = re.compile(r"\s*(?:%|basis\s+points?\b)", re.IGNORECASE)
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
QUALIFIED_DOLLAR_CODES = {
    "A": "AUD",
    "C": "CAD",
    "CA": "CAD",
    "HK": "HKD",
    "NZ": "NZD",
    "R": "BRL",
    "S": "SGD",
}
QUALIFIED_DOLLAR_PATTERN = re.compile(
    rf"\b({'|'.join(sorted(QUALIFIED_DOLLAR_CODES, key=len, reverse=True))})\$",
    re.IGNORECASE,
)
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
    "company_name",
    "date",
    "raw_id",
    "name",
    "start_date",
    "end_date",
    "amount",
    "amendment_of",
    "retired_of",
    "split_of",
    "lenders_json",
    "other_interested_parties_json",
    "name_json",
    "start_date_json",
    "end_date_json",
    "amount_json",
    "lenders_known_incomplete",
]


class InfrastructureError(RuntimeError):
    """A provider/transport failure that says nothing about the row's content.

    Billing (402), throttling (429), timeouts, connection resets, and 5xx are
    properties of the run environment, not of the filing being extracted: one
    occurrence predicts thousands more, so the live driver aborts the run at the
    first one instead of burning retries across the corpus and terminating rows
    that never got a real verdict (#49).
    """


# HTTP statuses that indicate the provider, not the content.
_INFRASTRUCTURE_STATUSES = frozenset({402, 408, 429, 500, 502, 503, 504})


def is_infrastructure_status(status: object) -> bool:
    """Classify an HTTP status from a batch result line as infrastructure."""
    return isinstance(status, int) and status in _INFRASTRUCTURE_STATUSES


def is_infrastructure_error(exc: BaseException) -> bool:
    """Classify an exception from a chat call as infrastructure vs content."""
    if isinstance(exc, InfrastructureError):
        return True
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    for attribute in ("status_code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and value in _INFRASTRUCTURE_STATUSES:
            return True
    # Provider SDKs name their billing/rate errors without exposing a status.
    name = type(exc).__name__.casefold()
    return any(
        marker in name
        for marker in ("paymentrequired", "ratelimit", "serviceunavailable")
    )


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
        if self.current_attempt.response is not None:
            new_messages.append(
                {"role": "assistant", "content": self.current_attempt.response}
            )
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

    def to_state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for resumable batch extraction.

        Unlike ``to_audit_dict`` this preserves everything the resumable state
        machine needs to continue after a process exit: the in-flight
        ``current_attempt`` (which ``__post_init__`` rebuilds blank), the current
        stage, ``ner_tagged_xml``, partial mentions, and a native-typed copy of
        the item fields the stages consume.
        """
        return {
            "item_row": {
                key: coerce_native(self.item_row.get(key))
                for key in STATE_ITEM_ROW_FIELDS
            },
            "all_attempts": [attempt.to_dict() for attempt in self.all_attempts],
            "stage_responses": dict(self.stage_responses),
            "debt_instrument_mentions": self.debt_instrument_mentions,
            "ner_tagged_xml": self.ner_tagged_xml,
            "state": self.state,
            "current_attempt": self.current_attempt.to_dict(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> ExtractionRowState:
        """Rebuild a row state previously produced by ``to_state_dict``."""
        current_attempt = AttemptRecord(
            **cast(dict[str, Any], payload["current_attempt"])
        )
        row_state = cls(
            item_row=cast(dict[str, object], payload["item_row"]),
            stage_name=current_attempt.stage_name,
        )
        row_state.all_attempts = [
            AttemptRecord(**cast(dict[str, Any], attempt))
            for attempt in cast(list[dict[str, object]], payload["all_attempts"])
        ]
        row_state.stage_responses = cast(dict[str, str], payload["stage_responses"])
        row_state.debt_instrument_mentions = cast(
            list[dict[str, object]], payload["debt_instrument_mentions"]
        )
        row_state.ner_tagged_xml = cast(str | None, payload["ner_tagged_xml"])
        row_state.state = cast(str | None, payload["state"])
        row_state.current_attempt = current_attempt
        return row_state


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


# Bounds every live chat call: without a client timeout, one non-responsive
# provider socket wedges the whole synchronous run — and the ECS task hosting
# it — indefinitely (#93). Generous because reasoning models legitimately take
# minutes per response.
LIVE_REQUEST_TIMEOUT_SECONDS = 600


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
            **sampling_params(model),
        }
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        async with OpenRouter(
            api_key=self.api_key,
            x_open_router_title="commercial-debt-tracker",
            x_open_router_categories="cli-agent",
            timeout_ms=LIVE_REQUEST_TIMEOUT_SECONDS * 1000,
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
                    if property_name in STANDARDIZED_SINGLE_VALUE_PROPERTIES:
                        failures.extend(
                            validate_standardized_single_value_cardinality(
                                index=index,
                                property_name=property_name,
                                value=obj[property_name],
                                tag_details=tag_details,
                            )
                        )
                    if property_name == "amount":
                        failures.extend(
                            validate_amount_is_not_rate(
                                index=index,
                                value=obj[property_name],
                                tag_details=tag_details,
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
            for property_name in PARTY_PROPERTY_ANNOTATIONS:
                failures.extend(
                    validate_party_property(
                        index=index,
                        property_name=property_name,
                        obj=obj,
                        tag_details=tag_details,
                    )
                )
            failures.extend(
                validate_lenders_known_incomplete(
                    index=index,
                    obj=obj,
                )
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
        seen_mention_ids: set[str] = set()
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
            name_text = canonical_value(obj.get("name", []), tag_details)
            end_date_payload = standardized_end_date_payload(
                obj.get("end_date"),
                tag_details,
                name_text=name_text,
            )
            lender_clusters, lenders_known_incomplete = (
                lender_payloads_and_incompleteness(obj, tag_details)
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
                "start_date": start_date_payload["normalized_date"],
                "end_date": end_date_payload["normalized_date"],
                "amount": amount_payload["normalized_amount"],
                "amendment_of": None,
                "retired_of": None,
                "split_of": None,
                "lenders_json": json.dumps(lender_clusters, sort_keys=True),
                "lenders_known_incomplete": lenders_known_incomplete,
                "other_interested_parties_json": json.dumps(
                    disclosed_party_payloads(obj, tag_details),
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
            "- Return one JSON object per concrete debt instrument described as its own obligation.\n"
            "- Ignore collective labels or contextual references that should not become standalone debt instruments.\n"
            "- If one object would have multiple distinct start dates or amounts, split it into separate debt instrument objects.\n"
            "- Shared evidence tags may appear in more than one object when the text supports that.\n"
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
                    f"Invalid relation type: {rel_type}. Must be amendment_of, retired_of, or split_of."
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
            "- Use only amendment_of, retired_of, or split_of.\n"
            "- Use only instrument IDs from the input."
        )


MENTIONS_DATASET_NAME = "mentions"

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
# Only the item fields the stages actually read are persisted in batch job state.
STATE_ITEM_ROW_FIELDS = (
    "item_id",
    "text",
    "accession_number",
    "cik",
    "company_name",
    "date",
    "item",
)


def coerce_native(value: object) -> str | None:
    """Coerce one pandas/numpy scalar to a JSON-safe native string or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


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


@dataclass
class PendingExtractPartition:
    """One classification partition with extraction work outstanding."""

    classification_path: str
    date: str
    shard: str
    fingerprint: str | None
    done_item_ids: frozenset[str]


def pending_extract_partitions(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    exclude_paths: set[str] | None = None,
) -> tuple[list[PendingExtractPartition], dict[str, CompletedPartition]]:
    """Select partitions with unextracted rows, keyed on outcomes and versions.

    A partition is pending when it has no completion entry, its entry is marked
    incomplete (an aborted pass), or its source fingerprint changed (ingest
    merged late-arriving rows into it, #62). ``done_item_ids`` are rows that
    already reached a terminal state and must not be re-paid (#49).

    Also returns the loaded registry so the caller can update and persist it —
    including the opportunistic stamping this function does: legacy entries
    (v1 lists, or partitions whose mentions predate the registry) get the
    current fingerprint so future source growth is detectable.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    registry = (
        {}
        if force
        else load_completion_registry(
            "extract", artifact_root=resolved_root, data_dir=data_dir
        )
    )
    fingerprints = {
        path: version
        for path, version in list_artifacts_with_versions(
            dataset_root(
                CLASSIFICATION_DATASET_NAME,
                artifact_root=resolved_root,
                data_dir=data_dir,
            ),
            suffix=".parquet",
        ).items()
        if PARTITION_PATTERN.search(path)
    }
    existing_mention_ids = (
        set()
        if force
        else existing_date_shard_partition_ids(
            MENTIONS_DATASET_NAME, artifact_root=resolved_root, data_dir=data_dir
        )
    )
    pending: list[PendingExtractPartition] = []
    for classification_path in sorted(fingerprints):
        if exclude_paths and classification_path in exclude_paths:
            continue
        partition = parse_date_shard_partition(classification_path)
        fingerprint = fingerprints[classification_path]
        entry = registry.get(classification_path)
        if force or entry is None:
            if (
                not force
                and (partition["date"], partition["shard"]) in existing_mention_ids
            ):
                # Mentions predate the registry: complete as of this version.
                registry[classification_path] = CompletedPartition(
                    fingerprint=fingerprint
                )
                continue
            pending.append(
                PendingExtractPartition(
                    classification_path=classification_path,
                    date=partition["date"],
                    shard=partition["shard"],
                    fingerprint=fingerprint,
                    done_item_ids=frozenset(),
                )
            )
            continue
        if entry.complete and entry.fingerprint == fingerprint:
            continue
        if entry.complete and entry.fingerprint is None:
            # v1 entry: complete as recorded; stamp so future growth registers.
            # Reassign rather than mutate so the change lands in the registry's
            # dirty set and survives the compare-and-swap merge on save (#88).
            registry[classification_path] = replace(entry, fingerprint=fingerprint)
            continue
        pending.append(
            PendingExtractPartition(
                classification_path=classification_path,
                date=partition["date"],
                shard=partition["shard"],
                fingerprint=fingerprint,
                done_item_ids=entry.item_ids,
            )
        )
    return pending, registry


def _merge_mentions_partition(
    resolved_root: str,
    *,
    data_dir: Path | None,
    partition: dict[str, str],
    new_mentions: pd.DataFrame,
    replaced_item_ids: set[str],
) -> str:
    """Merge newly extracted mentions into a partition, replacing per item.

    Row-level re-processing means a target partition can already hold mentions
    from earlier passes; overwriting it wholesale would drop them.
    """
    target_path = date_shard_partition_path(
        MENTIONS_DATASET_NAME,
        partition_date=partition["date"],
        shard=partition["shard"],
        artifact_root=resolved_root,
        data_dir=data_dir,
    )
    table = new_mentions
    if artifact_exists(target_path):
        existing = read_table(target_path, DEBT_INSTRUMENT_MENTION_COLUMNS)
        kept = existing.loc[~existing["item_id"].astype(str).isin(replaced_item_ids)]
        table = pd.concat([kept, new_mentions], ignore_index=True)
    write_partition_table(
        mentions_root(resolved_root, data_dir=data_dir),
        partition=partition,
        table=table.reindex(columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
    )
    return target_path


def collect_pending_extract_items(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    max_rows: int | None = None,
) -> tuple[list[tuple[dict[str, str | None], str, str]], dict[str, dict[str, object]]]:
    """Collect relevant items awaiting extraction across pending partitions.

    Returns ``(entries, claimed)`` where each entry is a native-typed item row
    plus its originating ``(date, shard)``, and ``claimed`` maps each claimed
    classification partition to its source fingerprint and the item_ids that
    were already terminal before this job — the state finalize needs to record
    row-outcome-keyed completion (#49) and to detect source growth (#62). Uses
    the same selection as ``extract_pending_items`` so both backends claim the
    same work, row by row.

    ``max_rows`` stops claiming partitions once the collected row count reaches
    it (whole partitions stay the atomic claim unit, so the last claimed
    partition may overshoot). Unclaimed partitions are simply left pending: a
    post-backfill job holding every pending item's full text OOMs the poll
    tick (#92), and the next job picks up the remainder.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    pending, _registry = pending_extract_partitions(
        artifact_root=resolved_root, data_dir=data_dir, force=force
    )
    entries: list[tuple[dict[str, str | None], str, str]] = []
    claimed: dict[str, dict[str, object]] = {}
    deferred_partitions = 0
    for pending_partition in pending:
        if max_rows is not None and len(entries) >= max_rows:
            deferred_partitions += 1
            continue
        claimed[pending_partition.classification_path] = {
            "fingerprint": pending_partition.fingerprint,
            "prior_item_ids": sorted(pending_partition.done_item_ids),
        }
        batch_items = read_table(
            pending_partition.classification_path, CLASSIFIED_ITEM_COLUMNS
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        relevant_items = batch_items.loc[batch_items["relevance"].fillna(False)]
        for item_row in relevant_items.to_dict("records"):
            if str(item_row["item_id"]) in pending_partition.done_item_ids:
                continue
            coerced = {
                key: coerce_native(item_row.get(key)) for key in STATE_ITEM_ROW_FIELDS
            }
            entries.append((coerced, pending_partition.date, pending_partition.shard))
    if deferred_partitions:
        LOGGER.info(
            "Deferred %s pending partition(s) beyond the %s-row job cap; the "
            "next job claims them once this one completes.",
            deferred_partitions,
            max_rows,
        )
    return entries, claimed


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
    resolved_model = model or settings.EXTRACTOR_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    processed_frames: list[pd.DataFrame] = []
    failed_rows: dict[str, dict[str, object]] = {}
    succeeded_item_ids: set[str] = set()
    audit_records: list[str] = []
    partitions_written: list[str] = []
    visited_classification_paths: set[str] = set()
    empty_partitions = 0
    # Partitions the active batch job claimed are its to finish: extracting them
    # live too would pay for every row twice and let the job's later finalize
    # overwrite the newer live mentions with stale results. Imported lazily —
    # batch.py imports from this module.
    from cdt.extractor.batch import active_job_claimed_partition_paths

    claimed_by_batch_job = (
        set()
        if force
        else active_job_claimed_partition_paths(resolved_root, data_dir=data_dir)
    )
    if claimed_by_batch_job:
        LOGGER.info(
            "Skipping %s classification partition(s) claimed by the active batch "
            "extract job; a poll tick will finish them.",
            len(claimed_by_batch_job),
        )

    pending_partitions, registry = pending_extract_partitions(
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
        exclude_paths=claimed_by_batch_job,
    )
    aborted: str | None = None
    total_partitions = len(pending_partitions)
    for partition_index, pending in enumerate(pending_partitions, start=1):
        partition = {"date": pending.date, "shard": pending.shard}
        partition_label = f"date={pending.date} shard={pending.shard}"
        partition_start = perf_counter()
        visited_classification_paths.add(pending.classification_path)
        batch_items = read_table(
            pending.classification_path,
            CLASSIFIED_ITEM_COLUMNS,
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        relevant_items = batch_items.loc[batch_items["relevance"].fillna(False)]
        # Row-level work list: rows that already reached a terminal state in an
        # earlier pass are never re-paid (#49); rows ingest merged in later are
        # exactly the ones missing from done_item_ids (#62).
        relevant_records = [
            record
            for record in relevant_items.to_dict("records")
            if str(record["item_id"]) not in pending.done_item_ids
        ]
        relevant_item_ids = {
            str(value) for value in relevant_items["item_id"].astype(str)
        }
        terminal_ids = set(pending.done_item_ids)
        mention_rows: list[dict[str, object]] = []
        replaced_item_ids: set[str] = set()
        partition_failures = 0
        total_relevant_items = len(relevant_records)
        for item_index, item_row in enumerate(relevant_records, start=1):
            try:
                row_state = asyncio.run(
                    run_extraction_workflow(
                        item_row=item_row,
                        model=resolved_model,
                        reasoning_effort=resolved_reasoning,
                        max_attempts=max_attempts,
                        client=client,
                    )
                )
            except InfrastructureError as exc:
                # A provider failure predicts thousands more: stop the run now.
                # Everything terminal so far in this partition is persisted, so
                # the retry pays only for what never got a verdict.
                aborted = str(exc)
                break
            audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
            terminal_ids.add(row_state.item_id)
            replaced_item_ids.add(row_state.item_id)
            if row_state.state == "SUCCESS":
                succeeded_item_ids.add(row_state.item_id)
                mention_rows.extend(row_state.debt_instrument_mentions)
            else:
                failed_rows[row_state.item_id] = _failure_record(
                    row_state,
                    partition_date=pending.date,
                    shard=pending.shard,
                    run_id=run_id,
                    backend="live",
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

        mentions = pd.DataFrame(mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
        if mentions.empty and not replaced_item_ids:
            empty_partitions += 1
        elif not mentions.empty or replaced_item_ids & pending.done_item_ids:
            partitions_written.append(
                _merge_mentions_partition(
                    resolved_root,
                    data_dir=data_dir,
                    partition=partition,
                    new_mentions=mentions,
                    replaced_item_ids=replaced_item_ids,
                )
            )
            if not mentions.empty:
                processed_frames.append(mentions)
        else:
            empty_partitions += 1
        registry[pending.classification_path] = CompletedPartition(
            fingerprint=pending.fingerprint,
            item_ids=frozenset(terminal_ids),
            complete=relevant_item_ids <= terminal_ids,
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
        if aborted is not None:
            break

    save_completion_registry(
        "extract",
        registry,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )
    failure_registry, total_known_failures = _merge_row_failures(
        failed_rows,
        succeeded_item_ids,
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
            "failure_registry": failure_registry,
            "aborted_on_infrastructure_error": aborted,
        },
    )
    if aborted is not None:
        # Persisted everything first (registry, mentions, audit, failures), so
        # the retry resumes from exactly the rows that never got a verdict.
        raise InfrastructureError(aborted)

    LOGGER.info(
        "Extractor complete: successes=%s failures=%s mentions=%s run_dir=%s "
        "failure_registry=%s (%s total)",
        sum(
            len(frame["item_id"].unique())
            for frame in processed_frames
            if not frame.empty
        ),
        len(failed_rows),
        sum(len(frame) for frame in processed_frames),
        full_jsonl_path,
        failure_registry,
        total_known_failures,
    )
    if not processed_frames:
        return pd.DataFrame(columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=DEBT_INSTRUMENT_MENTION_COLUMNS
    )


def finalize_extract_outputs(
    row_entries: list[tuple[ExtractionRowState, str, str]],
    *,
    claimed: dict[str, dict[str, object]],
    run_id: str,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Write mention partitions, audit log, and manifests for a completed job.

    This is the batch backend's analogue of the tail of ``extract_pending_items``.
    Every row in ``row_entries`` must already be terminal. ``claimed`` carries each
    claimed classification partition's fingerprint and prior terminal item_ids
    (from ``collect_pending_extract_items``), so completion is recorded per row
    outcome rather than per visit. Mentions are regrouped by their originating
    ``(date, shard)`` partition and merged into existing targets per item.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    mentions_by_partition: dict[tuple[str, str], list[dict[str, object]]] = {}
    audit_records: list[str] = []
    failed_rows: dict[str, dict[str, object]] = {}
    succeeded_item_ids: set[str] = set()
    for row_state, partition_date, shard in row_entries:
        audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
        if row_state.state == "SUCCESS":
            succeeded_item_ids.add(row_state.item_id)
            mentions_by_partition.setdefault((partition_date, shard), []).extend(
                row_state.debt_instrument_mentions
            )
        else:
            failed_rows[row_state.item_id] = _failure_record(
                row_state,
                partition_date=partition_date,
                shard=shard,
                run_id=run_id,
                backend="batch",
            )
        # Ensure a visited-but-empty partition still exists as a key so we do not
        # lose track of which partitions the job covered.
        mentions_by_partition.setdefault((partition_date, shard), [])

    terminal_by_partition: dict[tuple[str, str], set[str]] = {}
    for row_state, partition_date, shard in row_entries:
        if row_state.state is not None:
            terminal_by_partition.setdefault((partition_date, shard), set()).add(
                row_state.item_id
            )

    processed_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    empty_partitions = 0
    for (partition_date, shard), mention_rows in sorted(mentions_by_partition.items()):
        mentions = pd.DataFrame(mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
        replaced = terminal_by_partition.get((partition_date, shard), set())
        if mentions.empty and not replaced:
            empty_partitions += 1
            continue
        if mentions.empty:
            empty_partitions += 1
            continue
        partitions_written.append(
            _merge_mentions_partition(
                resolved_root,
                data_dir=data_dir,
                partition={"date": partition_date, "shard": shard},
                new_mentions=mentions,
                replaced_item_ids=replaced,
            )
        )
        processed_frames.append(mentions)

    registry = load_completion_registry(
        "extract", artifact_root=resolved_root, data_dir=data_dir
    )
    for classification_path, claim in claimed.items():
        partition = parse_date_shard_partition(classification_path)
        prior = {
            str(item) for item in cast(list[object], claim.get("prior_item_ids") or [])
        }
        terminal = prior | terminal_by_partition.get(
            (partition["date"], partition["shard"]), set()
        )
        relevant = read_table(classification_path, CLASSIFIED_ITEM_COLUMNS).reindex(
            columns=CLASSIFIED_ITEM_COLUMNS
        )
        relevant = relevant.loc[relevant["relevance"].fillna(False)]
        relevant_ids = {str(value) for value in relevant["item_id"].astype(str)}
        fingerprint = claim.get("fingerprint")
        registry[classification_path] = CompletedPartition(
            fingerprint=str(fingerprint) if fingerprint else None,
            item_ids=frozenset(terminal),
            complete=relevant_ids <= terminal,
        )
    save_completion_registry(
        "extract", registry, artifact_root=resolved_root, data_dir=data_dir
    )
    # Claiming the partitions above marks these rows done for good, so record the
    # ones that produced nothing before that fact is only visible in the audit log.
    failure_registry, total_known_failures = _merge_row_failures(
        failed_rows,
        succeeded_item_ids,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    write_text_artifact(
        full_jsonl_path,
        ("\n".join(audit_records) + "\n") if audit_records else "",
    )
    write_json_artifact(
        run_manifest_path(
            "extract", run_id, artifact_root=resolved_root, data_dir=data_dir
        ),
        {
            "artifact_root": resolved_root,
            "stage": "extract",
            "backend": "batch",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "partitions_completed": sorted(claimed),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "failure_count": len(failed_rows),
            "audit_path": full_jsonl_path,
            "completion_registry": completion_registry_path(
                "extract", artifact_root=resolved_root, data_dir=data_dir
            ),
            "failure_registry": failure_registry,
        },
    )
    LOGGER.info(
        "Batch extractor finalize complete: rows=%s successes=%s failures=%s "
        "mentions=%s audit=%s failure_registry=%s (%s total)",
        len(row_entries),
        len(row_entries) - len(failed_rows),
        len(failed_rows),
        sum(len(frame) for frame in processed_frames),
        full_jsonl_path,
        failure_registry,
        total_known_failures,
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
    resolved_model = model or settings.EXTRACTOR_MODEL
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


def record_stage_error(row_state: ExtractionRowState, message: str) -> None:
    """Mark the current attempt as a terminal ERROR for one row."""
    row_state.current_attempt.validation_errors = [message]
    row_state.current_attempt.status = "ERROR"
    row_state.finish("ERROR")


def _begin_stage(row_state: ExtractionRowState, stage: StageSpec) -> bool:
    """Populate the current attempt's messages via ``stage.preprocess``.

    Returns True on success. On a preprocess exception the row is finished as a
    terminal ERROR (matching the synchronous workflow) and False is returned.
    """
    row_state.current_attempt.stage_name = stage.name
    row_state.current_attempt.messages = []
    try:
        row_state.add_messages(stage.preprocess(row_state))
    except Exception as exc:  # noqa: BLE001
        record_stage_error(row_state, f"{type(exc).__name__}: {exc}")
        return False
    return True


def initial_messages(row_state: ExtractionRowState) -> list[dict[str, str]] | None:
    """Prepare the first request for a fresh row.

    Returns the messages for the first LLM call, or None if the row terminates
    before any call (a preprocess failure on the first stage).
    """
    if not _begin_stage(row_state, EXTRACTOR_STAGES[0]):
        return None
    return list(row_state.current_attempt.messages)


def handle_response(
    row_state: ExtractionRowState,
    response: str,
    *,
    max_attempts: int,
) -> list[dict[str, str]] | None:
    """Advance one row given the response to its outstanding request.

    Applies the current stage's validate/postprocess, then either advances to the
    next stage, schedules a retry, or terminates the row. Returns the messages for
    the next LLM call, or None when the row has reached a terminal state.
    """
    stage = STAGE_BY_NAME[row_state.current_attempt.stage_name]
    stage_index = STAGE_INDEX[stage.name]
    row_state.add_response(response)
    failures = stage.validate(row_state, response)
    row_state.add_validation(failures)
    if not failures:
        stage.postprocess(row_state)
        if stage.early_stop(row_state):
            row_state.finish("SUCCESS")
            return None
        if stage_index == len(EXTRACTOR_STAGES) - 1:
            row_state.finish("SUCCESS")
            return None
        next_stage = EXTRACTOR_STAGES[stage_index + 1]
        if (
            next_stage.name == "instrument_relation"
            and len(row_state.debt_instrument_mentions) <= 1
        ):
            row_state.finish("SUCCESS")
            return None
        row_state.next_stage(next_stage.name)
        if not _begin_stage(row_state, next_stage):
            return None
        return list(row_state.current_attempt.messages)

    if row_state.current_attempt.attempt_index >= max_attempts:
        row_state.finish("FAILED")
        return None
    row_state.retry(stage.build_retry_message(failures))
    return list(row_state.current_attempt.messages)


async def run_extraction_workflow(
    *,
    item_row: dict[str, object],
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    client: SupportsChatCompletion | None = None,
) -> ExtractionRowState:
    """Run the three-stage extraction workflow for one item row (live backend)."""
    resolved_client = client or OpenRouterChatClient()
    row_state = ExtractionRowState(
        item_row=item_row, stage_name=EXTRACTOR_STAGES[0].name
    )
    messages = initial_messages(row_state)
    while messages is not None:
        try:
            response = await resolved_client.complete(
                messages=messages,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001
            if is_infrastructure_error(exc):
                # Not a verdict on this row: leave it non-terminal and let the
                # driver abort the run. The row stays pending via the registry.
                raise InfrastructureError(f"{type(exc).__name__}: {exc}") from exc
            record_stage_error(row_state, f"{type(exc).__name__}: {exc}")
            return row_state
        messages = handle_response(row_state, response, max_attempts=max_attempts)
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


def extract_batch_response_text(line: dict[str, object]) -> str:
    """Extract the assistant text from one OpenAI Batch output JSONL line.

    Batch results are plain JSON dicts (not SDK objects). Per-request failures
    surface either as a top-level ``error`` or a non-200 ``response.status_code``;
    both raise so the caller can mark the row terminal ERROR.
    """
    error = line.get("error")
    if error:
        raise RuntimeError(f"Batch request error: {error}")
    response = cast(dict[str, object], line.get("response") or {})
    status_code = response.get("status_code")
    if status_code != 200:  # noqa: PLR2004
        raise RuntimeError(
            f"Batch request returned status {status_code}: {response.get('body')}"
        )
    body = cast(dict[str, object], response.get("body") or {})
    choices = cast(list[dict[str, object]], body.get("choices") or [])
    if not choices:
        raise RuntimeError("Batch response did not include choices.")
    message = cast(dict[str, object], choices[0].get("message") or {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    raise RuntimeError("Batch response content was not text.")


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


def validate_standardized_single_value_cardinality(
    *,
    index: int,
    property_name: str,
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate that one standardized single-value payload does not encode conflicts."""
    if not isinstance(value, dict):
        return []
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return []
    evidence_texts = [
        str(tag_details[tag_id]["text"])
        for tag_id in evidence
        if isinstance(tag_id, str)
        and tag_id in tag_details
        # Name spans carrying an embedded maturity are not comparable date mentions.
        and tag_details[tag_id]["type"] not in MATURITY_EVIDENCE_TAG_TYPES
    ]
    if property_name == "amount":
        normalized_values = {
            parsed
            for parsed in (normalized_amount_from_text(text) for text in evidence_texts)
            if parsed is not None
        }
    else:
        normalized_values = {
            parsed
            for parsed in (normalized_date_from_text(text) for text in evidence_texts)
            if parsed is not None
        }
    if len(normalized_values) <= 1:
        return []
    return [
        (
            f"Entry {index}: '{property_name}' contains multiple distinct normalized "
            "values. Split this into separate debt instrument objects instead of "
            "combining them."
        )
    ]


def validate_party_property(
    *,
    index: int,
    property_name: str,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate one party property against the annotated cluster shape."""
    if property_name not in obj:
        return []
    value = obj[property_name]
    annotation_key, allowed_annotations, _default = PARTY_PROPERTY_ANNOTATIONS[
        property_name
    ]
    expected_annotations = ", ".join(sorted(allowed_annotations))
    if not isinstance(value, list):
        return [
            f"Entry {index}: '{property_name}' must be a list of cluster objects "
            f'shaped like {{"tag_ids": ["tag-..."], "{annotation_key}": "..."}}.'
        ]
    failures: list[str] = []
    for cluster_index, cluster in enumerate(value):
        location = f"Entry {index}: '{property_name}'[{cluster_index}]"
        if not isinstance(cluster, dict):
            failures.append(
                f"{location} must be an object with 'tag_ids' and "
                f"'{annotation_key}' keys."
            )
            continue
        annotation = cluster.get(annotation_key)
        if annotation not in allowed_annotations:
            failures.append(
                f"{location} '{annotation_key}' must be one of {expected_annotations}."
            )
        tag_ids = cluster.get("tag_ids")
        if not isinstance(tag_ids, list):
            failures.append(f"{location} 'tag_ids' must be a list of tag IDs.")
            continue
        if not all(isinstance(tag_id, str) for tag_id in tag_ids):
            failures.append(f"{location} 'tag_ids' must contain string tag IDs only.")
            continue
        for tag_id in tag_ids:
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(f"{location} contains unknown tag ID {tag_id}.")
                continue
            if tag_info["type"] not in LENDER_TAG_TYPES:
                failures.append(
                    f"{location} tag {tag_id} must be person or organization."
                )
    return failures


def validate_lenders_known_incomplete(*, index: int, obj: dict[str, Any]) -> list[str]:
    """Validate the optional lenders_known_incomplete flag."""
    if "lenders_known_incomplete" not in obj:
        return []
    if isinstance(obj["lenders_known_incomplete"], bool):
        return []
    return [f"Entry {index}: 'lenders_known_incomplete' must be true or false."]


def validate_amount_is_not_rate(
    *,
    index: int,
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Reject an amount whose evidence only describes a rate, margin, or fee."""
    if not isinstance(value, dict):
        return []
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return []
    # The same normalization `canonical_value` applies, so the validator and
    # `standardized_amount_payload` judge one text. Stripping whitespace instead
    # hid the `basis point` marker and both amount guards from the predicate
    # (#102), and it also made the quoted text in the retry message unreadable.
    evidence_texts = [
        normalize_span_whitespace(str(tag_details[tag_id]["text"]))
        for tag_id in evidence
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    if not evidence_texts or not all(
        is_rate_like_amount_text(text) for text in evidence_texts
    ):
        return []
    quoted = ", ".join(f"'{text}'" for text in evidence_texts)
    return [
        (
            f"Entry {index}: 'amount' evidence {quoted} describes an interest rate, "
            "margin, or fee rather than a principal or commitment amount. Cite the "
            "principal or commitment amount instead, or omit 'amount'."
        )
    ]


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
        "lenders_known_incomplete": mention_row.get("lenders_known_incomplete"),
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


def canonical_amount_value(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the amount cluster's canonical text, preferring a parseable span.

    An amount cluster often pairs the figure with the label that names it, and
    the label is the longer span: `['$2,000,000', 'Principal Amount']` resolved
    to `Principal Amount`, which parses to nothing, so the amount published as
    null (#120). The rate guard reads this same text, so judging it on the span
    the parser actually reads keeps #102/#103 pointed at the right words.
    """
    values = cluster_span_texts(tag_ids, tag_details)
    if not values:
        return None
    parseable = [
        value for value in values if normalized_amount_from_text(value) is not None
    ]
    return max(parseable or values, key=len)


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


def normalize_numeric_string(value: Decimal) -> str:
    """Return one deterministic numeric string.

    Decimal rather than float: `float("372246148.11")` is not that number, and
    `f"{value:.12f}"` renders the difference as `372246148.110000014305`. That
    string is what the model's `normalized_amount` was compared against, so
    every amount carrying cents failed the agreement check in
    `standardized_amount_payload` and published as null (#119).
    """
    quantized = value.normalize()
    if quantized == quantized.to_integral_value():
        quantized = quantized.to_integral_value()
    return f"{quantized:f}"


def decimal_from_amount_string(value: str | None) -> Decimal | None:
    """Return one amount string as a Decimal, or None when it is not numeric."""
    if not isinstance(value, str):
        return None
    stripped = value.strip().replace(",", "")
    if not stripped:
        return None
    try:
        parsed = Decimal(stripped)
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def amounts_agree(model_amount: object, parsed_amount: str | None) -> bool:
    """Return whether the model's amount is the same value the parser read.

    Compared numerically, so the model reporting `500000.00` against a parsed
    `500000` counts as agreement rather than losing the amount (#119).
    """
    if not isinstance(model_amount, str) or parsed_amount is None:
        return False
    model_value = decimal_from_amount_string(model_amount)
    parsed_value = decimal_from_amount_string(parsed_amount)
    if model_value is None or parsed_value is None:
        return False
    return model_value == parsed_value


def normalized_amount_from_text(text: str | None) -> str | None:
    """Parse one amount mention into a normalized numeric string."""
    if not text:
        return None
    lowered = text.lower().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return None
    amount = decimal_from_amount_string(match.group(0))
    if amount is None:
        return None
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
    # A qualified dollar sign is a different currency. Reading `C$300 million`
    # as USD both mislabelled the amount and kept CAD out of the candidate set,
    # so the model's correct currency was rejected and published as null (#121).
    qualified = QUALIFIED_DOLLAR_PATTERN.findall(text)
    candidates.update(QUALIFIED_DOLLAR_CODES[prefix.upper()] for prefix in qualified)
    if (
        text.count("$") > len(qualified)
        or "u.s. dollar" in lowered
        or "us dollar" in lowered
    ):
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


def is_rate_like_amount_text(text: str | None) -> bool:
    """Return whether one amount evidence string reads as a rate, margin, or fee.

    Every number in the span has to carry a rate marker. A marker appearing
    somewhere is not enough: `500,000,000 (100% of principal)` states a
    principal and then a percentage of it, and `normalized_amount_from_text`
    reads the first number, so treating the whole span as a rate would discard a
    real amount (#103).
    """
    if not text:
        return False
    lowered = text.lower()
    if currency_candidates_from_text(text):
        return False
    if any(re.search(rf"\b{word}\b", lowered) for word in AMOUNT_MULTIPLIERS):
        return False
    values = list(AMOUNT_VALUE_PATTERN.finditer(text))
    if not values:
        return False
    return all(RATE_SUFFIX_PATTERN.match(text, value.end()) for value in values)


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
    return iso_date_from_parts(
        match.group("year"), match.group("month"), match.group("day")
    )


def normalized_maturity_from_text(text: str | None) -> str | None:
    """Parse one maturity phrase such as 'notes due 2028' into ISO format.

    A phrase stating more than one maturity names no single instrument, so it
    parses to None rather than to whichever maturity comes first.
    """
    if not text:
        return None
    full_dates: set[str] = set()
    years: set[str] = set()
    for match in MATURITY_FULL_DATE_PATTERN.finditer(text):
        normalized = iso_date_from_parts(
            match.group("year"), match.group("month"), match.group("day")
        )
        if normalized is not None:
            full_dates.add(normalized)
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("more")))
    for match in MATURITY_YEAR_PATTERN.finditer(text):
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("years")))
    if full_dates:
        # A bare alternate year alongside a full date states a second maturity too.
        if len(full_dates) != 1 or years - {value[:4] for value in full_dates}:
            return None
        return full_dates.pop()
    if len(years) != 1:
        return None
    return f"{years.pop()}{YEAR_ONLY_MATURITY_SUFFIX}"


def iso_date_from_parts(year: str, month_name: str, day: str) -> str | None:
    """Return one ISO date built from year, month name, and day parts."""
    month = MONTH_MAP.get(month_name.lower())
    if month is None:
        return None
    normalized = f"{year}-{month}-{int(day):02d}"
    return normalized if is_valid_iso_date(normalized) else None


def standardized_amount_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return evidence payload plus validated normalized amount fields."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_amount_value(evidence_tag_ids, tag_details)
    parsed_amount = normalized_amount_from_text(evidence_text)
    parsed_currency_candidates = currency_candidates_from_text(evidence_text)
    model_amount = value.get("normalized_amount") if isinstance(value, dict) else None
    model_currency = value.get("currency") if isinstance(value, dict) else None
    if is_rate_like_amount_text(evidence_text):
        # Rates, margins, and fees are not principal amounts.
        parsed_amount = None
        parsed_currency_candidates = set()

    # The parser's own string is published, so a model reporting the same value
    # with different formatting keeps its amount rather than losing it (#119).
    payload["normalized_amount"] = (
        parsed_amount if amounts_agree(model_amount, parsed_amount) else None
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
    *,
    allow_maturity_phrase: bool = False,
) -> dict[str, object]:
    """Return evidence payload plus validated normalized date field."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_value(evidence_tag_ids, tag_details)
    parsed_date = normalized_date_from_text(evidence_text)
    if parsed_date is None and allow_maturity_phrase:
        parsed_date = normalized_maturity_from_text(evidence_text)
    model_date = value.get("normalized_date") if isinstance(value, dict) else None
    payload["normalized_date"] = (
        model_date
        if isinstance(model_date, str) and model_date == parsed_date
        else None
    )
    return payload


def standardized_end_date_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> dict[str, object]:
    """Return the end-date payload, falling back to the maturity in the name."""
    payload = standardized_date_payload(
        value,
        tag_details,
        allow_maturity_phrase=True,
    )
    if payload["normalized_date"] is None:
        derived_date = normalized_maturity_from_text(name_text)
        if derived_date is not None:
            payload["normalized_date"] = derived_date
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


def annotated_party_clusters(
    clusters: object,
    tag_details: dict[str, dict[str, object]],
    *,
    property_name: str,
) -> list[tuple[dict[str, object], str]]:
    """Return (cluster payload, annotation) pairs for one party property.

    Each payload keeps the persisted cluster shape. The annotation is the model's
    extraction-time label, used only to decide which clusters to persist.
    """
    annotation_key, allowed_annotations, default_annotation = (
        PARTY_PROPERTY_ANNOTATIONS[property_name]
    )
    if not isinstance(clusters, list):
        return []
    pairs: list[tuple[dict[str, object], str]] = []
    for cluster in clusters:
        if isinstance(cluster, dict):
            tag_ids = cluster.get("tag_ids")
            annotation = cluster.get(annotation_key)
        else:
            # Tolerate the legacy bare tag-id list shape when replaying old responses.
            tag_ids = cluster
            annotation = None
        payload = cluster_payload(tag_ids, tag_details)
        if not payload["tag_ids"]:
            continue
        resolved = (
            str(annotation) if annotation in allowed_annotations else default_annotation
        )
        pairs.append((payload, resolved))
    return pairs


def lender_payloads_and_incompleteness(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    """Return named lender clusters plus whether lenders are known to be missing."""
    pairs = annotated_party_clusters(
        obj.get("lenders", []),
        tag_details,
        property_name="lenders",
    )
    named = [payload for payload, kind in pairs if kind != COLLECTIVE_LENDER_KIND]
    has_collective = len(named) < len(pairs)
    declared_incomplete = obj.get("lenders_known_incomplete") is True
    return named, has_collective or declared_incomplete


def disclosed_party_payloads(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return other-interested-party clusters, excluding the borrower itself."""
    pairs = annotated_party_clusters(
        obj.get("other_interested_parties", []),
        tag_details,
        property_name="other_interested_parties",
    )
    return [payload for payload, role in pairs if role != BORROWER_PARTY_ROLE]


def relation_prompt_xml(row_state: ExtractionRowState) -> str:
    """Build relation-stage XML with instrument-id attributes."""
    if not row_state.ner_tagged_xml:
        raise ValueError("ner_tagged_xml is required for instrument_relation.")
    root, _, _ = parse_tag_details(row_state.ner_tagged_xml)
    tag_to_raw_id: dict[str, str] = {}
    for mention in row_state.debt_instrument_mentions:
        payload = json.loads(str(mention["name_json"]))
        for tag_id in payload.get("tag_ids", []):
            key = str(tag_id)
            raw_id = str(mention["raw_id"])
            if key not in tag_to_raw_id:
                tag_to_raw_id[key] = raw_id
            else:
                tag_to_raw_id[key] = f"{tag_to_raw_id[key]}||{raw_id}"
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
                for instrument_id in tag_to_raw_id[tag_id].split("||"):
                    parts.append(
                        f'<debt_instrument instrument-id="{instrument_id}">{rendered}</debt_instrument>'
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


def _failure_record(
    row_state: ExtractionRowState,
    *,
    partition_date: str,
    shard: str,
    run_id: str,
    backend: str,
) -> dict[str, object]:
    """Build one failure-registry entry for a terminal non-SUCCESS row."""
    return {
        "item_id": row_state.item_id,
        "accession_number": row_state.item_row.get("accession_number"),
        "cik": row_state.item_row.get("cik"),
        "date": partition_date,
        "shard": shard,
        "state": row_state.state,
        "stage": row_state.current_attempt.stage_name,
        "run_id": run_id,
        "backend": backend,
        "error": summarize_failure(row_state),
    }


def _merge_row_failures(
    failures: dict[str, dict[str, object]],
    succeeded_item_ids: set[str],
    *,
    artifact_root: str,
    data_dir: Path | None,
) -> tuple[str, int]:
    """Merge this run's row outcomes into the extract failure registry.

    Failures are added or refreshed; rows that succeeded this run clear any
    earlier entry, so a re-extract that fixes a row does not leave a stale
    failure behind. Returns the registry path and its total entry count.
    """
    registry = load_row_failures(
        "extract", artifact_root=artifact_root, data_dir=data_dir
    )
    for item_id in succeeded_item_ids:
        registry.pop(item_id, None)
    registry.update(failures)
    path = save_row_failures(
        "extract", registry, artifact_root=artifact_root, data_dir=data_dir
    )
    return path, len(registry)


def native_model_id(model: str) -> str:
    """Strip any provider prefix so an OpenRouter slug becomes a native id."""
    return model.split("/", 1)[1] if "/" in model else model


def is_reasoning_model(model: str) -> bool:
    """Return True for model families that take reasoning_effort over temperature."""
    return native_model_id(model).lower().startswith(REASONING_MODEL_PREFIXES)


def sampling_params(model: str) -> dict[str, object]:
    """Return the sampling params to send with one extract call.

    Shared by both backends. Reasoning models reject ``temperature != 1``, so
    temperature is only sent to models that can honor it; for those, pinning it
    to 0 keeps extraction as reproducible as the provider allows.
    """
    if is_reasoning_model(model):
        return {}
    return {"temperature": EXTRACTOR_TEMPERATURE}
