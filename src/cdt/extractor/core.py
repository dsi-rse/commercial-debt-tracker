"""LLM-backed extractor stage for relevant SEC 8-K items."""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
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
    coerce_dataset_text,
    list_artifacts_with_versions,
    read_table,
    write_json_artifact,
    write_partition_table,
    write_text_artifact,
)

LOGGER = get_logger(__name__)
DEFAULT_MAX_ATTEMPTS = 3
# PARTIAL rows publish their mentions like SUCCESS but also keep a failure
# registry entry recording what salvage dropped (#152).
PUBLISHABLE_ROW_STATES = frozenset({"SUCCESS", "PARTIAL"})
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
    # so maturity evidence may cite that name span instead of a standalone date.
    "maturity_date": {"date", "debt_instrument", "duration"},
    # When the lender's obligation to lend ends — the draw/availability window
    # closes (#158). Distinct from the maturity so draw-period dates stop
    # publishing as maturities.
    "commitment_termination_date": {"date"},
    # Pre-#158 name for maturity_date, still accepted so stored batch
    # responses replay.
    "end_date": {"date", "debt_instrument"},
    # The same holds for a principal stated inside the name, as in
    # `$183.36 million term loan`: there is no separate `amount` span to cite,
    # and rejecting the name span lost the amount outright (#129). The bare
    # `amount` property is the pre-#140 shape, still accepted so stored batch
    # responses replay; the prompt now teaches `amounts`.
    "amount": {"amount", "debt_instrument"},
    "name": {"debt_instrument"},
}
# One mention can state several money facts about one instrument — a $2.5B
# commitment and a $270.5M outstanding balance — so amounts are a kind-typed
# list (#140). The matcher keys only on commitment/principal; the other kinds
# are observations.
AMOUNT_KINDS = {
    "commitment",
    "principal",
    "outstanding_balance",
    "draw",
    "repayment",
    "proceeds",
}
PRINCIPAL_AMOUNT_KINDS = ("commitment", "principal")
# Kind-typed date facts (stage 1 of the dates overhaul). The old single-value
# `start_date` / `maturity_date` / `commitment_termination_date` slots made the
# model arbitrate between competing dates; a list of facts records each stated
# date once and lets the post-processor choose the published columns.
DATE_KINDS = {
    "agreement",  # the instrument's own `dated as of` date
    "announcement",  # pricing, launch, or commitment-letter date
    "expected_closing",  # stage-1 shape for `closing` + `expected: true`; replays only
    "closing",  # closing, issuance, funding, or effective date: the start
    "maturity",  # when the borrowed money must be repaid
    "commitment_termination",  # when the lender's obligation to lend ends
    # Stage 2: events are dated facts too, and the mention's status is derived.
    "amendment",  # terms modified; the amendment's effective or signing date
    "repayment",  # a payment that leaves the obligation outstanding
    "retirement",  # repaid in full, redeemed in whole, defeased, discharged
    "termination",  # the agreement or facility ended before its scheduled end
    "exchange",  # satisfied by delivering other securities or equity
    "default",  # default, event of default, or acceleration
}
EVENT_DATE_KINDS = {
    "announcement",
    "closing",
    "amendment",
    "repayment",
    "retirement",
    "termination",
    "exchange",
    "default",
}
TERMINAL_DATE_KINDS = {"retirement", "termination", "exchange", "default"}
# The kinds an instrument has at most one current value of, which
# `instrument_ie.md` advertises as "(validated)". `closing` is an event kind but
# still singular: two current closings describe two instruments, exactly as two
# maturities do. Keying the cap off `not in EVENT_DATE_KINDS` silently exempted
# it, so one of the two dates published and the other was dropped.
SINGLE_CURRENT_DATE_KINDS = {
    "agreement",
    "closing",
    "expected_closing",
    "maturity",
    "commitment_termination",
}
# Kinds that are only meaningful with a value: a stated maturity without a
# date is nothing, whereas `the notes were redeemed` with no date is still an
# event the filing states.
DATE_KINDS_REQUIRING_EVIDENCE = {"maturity", "commitment_termination", "agreement"}
# The mention-level `status` column derived from the newest completed event.
STATUS_FOR_DATE_KIND = {
    "announcement": "announced",
    "closing": "entered_into",
    "amendment": "amended",
    "retirement": "repaid",
    "termination": "terminated",
    "exchange": "exchanged",
    "default": "defaulted",
}
# Ties among undated events resolve by how much the event says about the
# obligation's life: its end beats a change beats its start.
EVENT_KIND_PRECEDENCE = {
    "default": 6,
    "exchange": 5,
    "retirement": 4,
    "termination": 4,
    "amendment": 3,
    "closing": 2,
    "announcement": 1,
}
DATE_KIND_EVIDENCE_TAG_TYPES = {
    "maturity": {"date", "debt_instrument", "duration"},
}
DEFAULT_DATE_EVIDENCE_TAG_TYPES = {"date"}
# Stage 2: one `parties` list with a role per cluster replaces `lenders` +
# `other_interested_parties`; `lender_disclosure` is derived from it.
PARTY_ROLES = {
    "lender",
    "agent",
    "trustee",
    "underwriter",
    "guarantor",
    "borrower",
    "other",
}
PARTY_KINDS = {"named", "collective"}
# How completely the document identifies who holds the debt. The boolean this
# replaces was true for two different reasons — a collective lender phrase, or
# no named lender at all — and 395 of 517 true values on the 2026-09 window were
# the second case, so a consumer reading the flag could not tell "something is
# undisclosed" from "nothing was disclosed here".
LENDER_DISCLOSURE_COMPLETE = "complete"
LENDER_DISCLOSURE_COLLECTIVE_PRESENT = "collective_present"
LENDER_DISCLOSURE_NONE_NAMED = "none_named"
LENDER_DISCLOSURE_VALUES = {
    LENDER_DISCLOSURE_COMPLETE,
    LENDER_DISCLOSURE_COLLECTIVE_PRESENT,
    LENDER_DISCLOSURE_NONE_NAMED,
}
# Precedence for rolling several mentions of one instrument into one answer.
# `collective_present` wins outright: one filing showing `the other lenders
# party thereto` means holders are hidden however many other filings name some.
# `complete` beats `none_named` because a filing that named every lender
# supersedes one that named none — the reverse would let a passing reference
# erase a full syndicate list.
LENDER_DISCLOSURE_PRECEDENCE = {
    LENDER_DISCLOSURE_NONE_NAMED: 0,
    LENDER_DISCLOSURE_COMPLETE: 1,
    LENDER_DISCLOSURE_COLLECTIVE_PRESENT: 2,
}
# Published flat columns and the fact kind each one reads.
DATE_COLUMN_KINDS = {
    "start_date": "closing",
    "maturity_date": "maturity",
    "commitment_termination_date": "commitment_termination",
}
# Pre-dates[] responses replay through the same path with their kind implied.
LEGACY_DATE_PROPERTY_KINDS = {
    "start_date": "closing",
    "maturity_date": "maturity",
    "end_date": "maturity",
    "commitment_termination_date": "commitment_termination",
}
DATE_PRECISIONS = ("day", "month", "year")
AMOUNT_EVIDENCE_TAG_TYPES = {"amount", "debt_instrument"}
# What one mention says happened to its instrument (#141). `matured` is
# deliberately absent: filings almost never say it, and the matcher derives it
# from maturity_date instead.
INTEREST_RATE_KINDS = {"fixed", "floating"}
INTEREST_RATE_EVIDENCE_TAG_TYPES = {"interest_rate", "debt_instrument"}
# `6.5 percent senior notes` spells the marker out; it is still a rate, not an amount.
RATE_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
# The four instrument categories the site facets on (#156); anything else
# stays null rather than stretching a bucket.
INSTRUMENT_TYPES = {
    "term_loan",
    "revolving_credit",
    "credit_line",
    "note_bond",
}
STATUS_EVENT_VALUES = {
    "announced",
    "entered_into",
    "amended",
    "terminated",
    "repaid",
    "exchanged",
    "defaulted",
}
MATURITY_EVIDENCE_TAG_TYPES = {"debt_instrument"}
NAME_EMBEDDED_AMOUNT_TAG_TYPES = {"debt_instrument"}
STANDARDIZED_SINGLE_VALUE_PROPERTIES = {
    "start_date",
    "maturity_date",
    "commitment_termination_date",
    "end_date",
    "amount",
}
INSTRUMENT_RELATION_TYPES = {"amendment_of", "retired_by", "split_of"}
NUMERIC_STRING_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
# One `due` can carry a list of maturities: `due 2028 and 2030`,
# `due October 1, 2028 and 2030`, `due October 1, 2028 and October 1, 2030`. Each
# is two maturities, and matching only the first silently invented one (#104).
MATURITY_COORDINATED_YEARS = (
    r"(?:\s*(?:,|/|&|and(?:/or)?|or)\s*(?:[A-Za-z]+\s+\d{1,2},?\s+)?\d{4})*"
)
# Like MATURITY_COORDINATED_YEARS, but each further year may carry a bare month
# with no day, so `due October 1, 2028 and April 2030` and `due April 2033 and
# June 2035` both read as two maturities (#164).
MATURITY_MONTH_YEAR_COORDINATION = (
    r"(?:\s*(?:,|/|&|and(?:/or)?|or)\s*(?:[A-Za-z]+\s+(?:\d{1,2},?\s+)?)?\d{4})*"
)
MATURITY_FULL_DATE_PATTERN = re.compile(
    r"\bdue\s+(?:on\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    rf"(?P<more>{MATURITY_MONTH_YEAR_COORDINATION})",
    re.IGNORECASE,
)
MATURITY_YEAR_PATTERN = re.compile(
    rf"\bdue\s+(?:in\s+)?(?P<years>\d{{4}}{MATURITY_COORDINATED_YEARS})\b",
    re.IGNORECASE,
)
# A facility tenor such as `five-year` or `364-day` (#166). Only a duration
# span stating exactly one tenor anchors computed-maturity arithmetic.
TENOR_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
}
TENOR_PATTERN = re.compile(
    r"\b(?P<num>\d{1,3}|"
    + "|".join(TENOR_WORD_NUMBERS)
    + r")[-\s](?P<unit>year|month|day)s?\b",
    re.IGNORECASE,
)
# `due April 2033` states a month-resolution maturity (#164); it normalizes to
# the month's last day.
MATURITY_MONTH_YEAR_PATTERN = re.compile(
    r"\bdue\s+(?:in\s+)?(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})"
    rf"(?P<more>{MATURITY_MONTH_YEAR_COORDINATION})",
    re.IGNORECASE,
)
FOUR_DIGIT_YEAR_PATTERN = re.compile(r"\d{4}")
YEAR_ONLY_MATURITY_SUFFIX = "-12-31"
# A rate marker counts only where it sits on a number, so the value the parser
# would read is the rate itself rather than a percentage of something else (#103).
AMOUNT_VALUE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
RATE_SUFFIX_PATTERN = re.compile(r"\s*(?:%|percent\b|basis\s+points?\b)", re.IGNORECASE)
# A principal stated inside an instrument name: `$183.36 million term loan`,
# `C$300 million notes due 2033`. The currency marker is required, so a coupon
# rate or a maturity year in the same name cannot be read as the principal.
# Magnitude words, spelled out and abbreviated (#182). The abbreviations matter
# because #129's name-derived principal reads the instrument's own name, and
# names use them: `Citibank $382.5 mil. Revolving Credit Facility`, `Syndicated
# $850.0 mil. Facility` (Costamare's 6-K facility schedules). Without them
# `$382.5 mil.` parsed as 382.5 — six orders out. On a cited span that merely
# published null, because `amounts_agree` rejected the mismatch; on the
# name-derived path there is no model value to disagree with, so it published.
AMOUNT_MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "mil": 1_000_000,
    "mils": 1_000_000,
    "mm": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "bil": 1_000_000_000,
    "bln": 1_000_000_000,
    "bn": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}
# Built from the table above so a magnitude this pattern recognizes is always one
# the parser can apply. The two drifted before: `trillion` matched here and
# multiplied nowhere.
AMOUNT_SCALE_ALTERNATION = "|".join(sorted(AMOUNT_MULTIPLIERS, key=len, reverse=True))
NAME_EMBEDDED_AMOUNT_PATTERN = re.compile(
    r"(?P<currency>[A-Z]{0,2}\$|€|£|¥)\s?"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)"
    rf"(?:\s*(?P<scale>{AMOUNT_SCALE_ALTERNATION})\b\.?)?",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# An ampersand that starts no entity. The NER response has to be well-formed XML
# while reproducing text that may carry a bare `&` (#127).
UNESCAPED_AMPERSAND_PATTERN = re.compile(
    r"&(?!(?:amp|lt|gt|quot|apos);|#(?:\d+|x[0-9A-Fa-f]+);)"
)
# Zero-padding is optional on the way in, so a model writing `2026-7-28` is read
# as the day it means rather than dropped for its shape (#133).
LENIENT_ISO_DATE_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$"
)
# Filing dates come in three further spellings. The four-digit year is required:
# `7/28/26` needs a century guessed, and a null beats a wrong decade (#133).
NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})(?!\d)"
)
# The comma is optional, so `July 28 2026` reads the same as `July 28, 2026`.
MONTH_FIRST_DATE_PATTERN = re.compile(
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})(?!\d)"
)
# `28 July 2026`, as non-US issuers write it.
DAY_FIRST_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})(?!\d)"
)
# `June 2016`, `in March 2056`: a month-resolution date outside an instrument
# name. #164 only read these after `due`, so `matures in June 2016` and `legal
# final maturity date is in March 2056` published nothing. Read for maturities
# only (`normalized_month_year_from_text`), to the month's last day.
MONTH_YEAR_DATE_PATTERN = re.compile(
    r"(?<![A-Za-z\d])(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})(?!\d)"
)
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
    "instrument_type",
    "start_date",
    "maturity_date",
    "commitment_termination_date",
    "principal_amount",
    "principal_currency",
    "principal_amount_kind",
    "status",
    "status_date",
    "interest_rate_kind",
    "interest_rate_pct",
    "amendment_of",
    "retired_by_json",
    "split_of",
    "parties_json",
    "name_json",
    "start_date_json",
    "maturity_date_json",
    "commitment_termination_date_json",
    "amounts_json",
    "status_json",
    "interest_rate_json",
    "dates_json",
    "lender_disclosure",
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
    ) -> CompletionResult:
        """Return one chat completion with its provider metadata."""


@dataclass
class AttemptRecord:
    """Recorded data for one stage attempt."""

    stage_name: str
    attempt_index: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    response: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    status: str = "incomplete"
    # Provider metadata. Without `finish_reason` a response that the provider
    # aborted is indistinguishable in the audit log from one the model chose to
    # end, and those need opposite remedies: four items lost to NER in one
    # held-out window turned out to be `content_filter` aborts rather than the
    # length cap or the model stopping that they were twice diagnosed as (#135).
    finish_reason: str | None = None
    refusal: str | None = None
    usage: dict[str, object] | None = None
    response_id: str | None = None
    served_model: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert one attempt to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CompletionResult:
    """One model response plus the provider metadata that explains it."""

    text: str
    finish_reason: str | None = None
    refusal: str | None = None
    usage: dict[str, object] | None = None
    response_id: str | None = None
    served_model: str | None = None


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
    salvage_notes: list[str] = field(default_factory=list)
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

    def add_response(
        self, response: str, completion: CompletionResult | None = None
    ) -> None:
        """Record the latest model response and the provider metadata for it."""
        self.current_attempt.response = response
        self.current_attempt.attempt_index += 1
        self.stage_responses[self.current_attempt.stage_name] = response
        if completion is not None:
            self.current_attempt.finish_reason = completion.finish_reason
            self.current_attempt.refusal = completion.refusal
            self.current_attempt.usage = completion.usage
            self.current_attempt.response_id = completion.response_id
            self.current_attempt.served_model = completion.served_model

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
        """Finish processing for this row.

        A row that reached the end only because a terminal failure was salvaged
        (#152) finishes PARTIAL rather than SUCCESS: its mentions publish, but
        the failure registry keeps a record of what was lost.
        """
        self.all_attempts.append(self.current_attempt)
        if state == "SUCCESS" and self.salvage_notes:
            state = "PARTIAL"
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
            "salvage_notes": self.salvage_notes,
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
            "salvage_notes": self.salvage_notes,
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
        # Absent in job state written before salvage existed (#152).
        row_state.salvage_notes = cast(list[str], payload.get("salvage_notes", []))
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
    ) -> CompletionResult:
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
        return completion_result_from_response(response)


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

        response = repair_unescaped_ampersands(response)
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
            "interest_rate",
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
        row_state.ner_tagged_xml = assign_tag_ids(repair_unescaped_ampersands(response))

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


def validate_instrument_entry(
    index: int,
    obj: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate one instrument entry on its own.

    Factored out of ``InstrumentIEStage.validate`` so terminal salvage (#152)
    can keep the individually-valid entries of a response whose other entries
    failed.
    """
    if not isinstance(obj, dict):
        return [f"Entry {index} is not a JSON object."]
    failures: list[str] = []
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
    if "instrument_type" in obj and obj["instrument_type"] not in INSTRUMENT_TYPES:
        allowed = ", ".join(sorted(INSTRUMENT_TYPES))
        failures.append(
            f"Entry {index}: 'instrument_type' must be one of {allowed}, or omitted."
        )
    failures.extend(
        validate_amounts_property(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_dates_property(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_status_event(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_interest_rate(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
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
        validate_parties_property(index=index, obj=obj, tag_details=tag_details)
    )
    failures.extend(
        validate_lenders_known_incomplete(
            index=index,
            obj=obj,
        )
    )
    failures.extend(validate_cross_field_semantics(index=index, obj=obj))
    return failures


# Which amount kinds fit which instrument types: a facility has a commitment, a
# security has a principal. Balances, draws, repayments and proceeds fit any.
FACILITY_INSTRUMENT_TYPES = {"revolving_credit", "credit_line"}
AMOUNT_KIND_TYPE_CONFLICTS = {
    ("commitment", "note_bond"),
    ("principal", "revolving_credit"),
    ("principal", "credit_line"),
}


# Properties of the pre-facts schema. Stored responses replay through
# postprocess with their old semantics; a live response that uses them is a
# model reverting to a shape the current prompt no longer describes.
LEGACY_INSTRUMENT_PROPERTIES = {
    "status_event": "events are `dates` entries (kinds closing, amendment, retirement, ...)",
    "lenders": "parties are one `parties` list with role lender",
    "other_interested_parties": "parties are one `parties` list with a role per cluster",
    "lenders_known_incomplete": "derived from the lender clusters; do not return it",
    "start_date": "a `dates` entry of kind closing (or agreement)",
    "maturity_date": "a `dates` entry of kind maturity",
    "end_date": "a `dates` entry of kind maturity",
    "commitment_termination_date": "a `dates` entry of kind commitment_termination",
    "amount": "an `amounts` entry with a kind",
}


def validate_no_legacy_properties(index: int, obj: object) -> list[str]:
    """Reject pre-facts-schema properties on a freshly generated response."""
    if not isinstance(obj, dict):
        return []
    return [
        f"Entry {index}: '{name}' is not a property of this schema; use {replacement}."
        for name, replacement in LEGACY_INSTRUMENT_PROPERTIES.items()
        if name in obj
    ]


def validate_cross_field_semantics(*, index: int, obj: dict[str, Any]) -> list[str]:
    """Reject shapes the schema forbids but no single-field check can see.

    Each of these is a rule the prompt states; before this check the model's
    violation was accepted and silently rewritten downstream.
    """
    failures: list[str] = []
    dates = obj.get("dates") if isinstance(obj.get("dates"), list) else []
    amounts = obj.get("amounts") if isinstance(obj.get("amounts"), list) else []
    date_kinds = {entry.get("kind") for entry in dates if isinstance(entry, dict)}
    amount_kinds = {entry.get("kind") for entry in amounts if isinstance(entry, dict)}
    for entry in amounts:
        if (
            isinstance(entry, dict)
            and entry.get("prior") is True
            and entry.get("kind") not in PRINCIPAL_AMOUNT_KINDS
        ):
            failures.append(
                f"Entry {index}: 'amounts' entry of kind '{entry.get('kind')}' cannot "
                "be `prior`. Only a commitment or principal has a before-the-change "
                "figure; a balance, draw, repayment or proceeds is dated, not prior."
            )
    if "repayment" in date_kinds and "repayment" not in amount_kinds:
        failures.append(
            f"Entry {index}: a `repayment` date entry needs the repaid figure as a "
            "`repayment` entry in 'amounts' when the document states one; if it states "
            "no figure, the payment is a `retirement` (in full) or nothing."
        )
    if (
        "repayment" in amount_kinds
        and dates
        and not date_kinds & {"repayment", *TERMINAL_DATE_KINDS}
    ):
        # A repayment figure is dated by the payment event that produced it: a
        # partial paydown (`repayment`) or the retirement/exchange that paid the
        # rest. Requiring a separate `repayment` date beside a `retirement` made
        # the model manufacture undated placeholder events (HASI, Smith Micro).
        failures.append(
            f"Entry {index}: a `repayment` amount is an event; add a `repayment` entry to "
            "'dates' (with `evidence` [] and `normalized_date` null when no date is stated), "
            "unless a retirement, termination or exchange entry already dates the payment."
        )
    instrument_type = obj.get("instrument_type")
    for kind in sorted(k for k in amount_kinds if isinstance(k, str)):
        if (kind, instrument_type) in AMOUNT_KIND_TYPE_CONFLICTS:
            failures.append(
                f"Entry {index}: 'amounts' kind '{kind}' does not fit instrument_type "
                f"'{instrument_type}'. A facility's size is a `commitment`; a security's "
                "face amount is a `principal`. Fix the kind or the type."
            )
    return failures


def validate_parties_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the unified `parties` list: role required, kind optional."""
    if "parties" not in obj:
        return []
    value = obj["parties"]
    if not isinstance(value, list):
        return [
            f"Entry {index}: 'parties' must be a list of cluster objects shaped like "
            '{"tag_ids": ["tag-..."], "role": "...", "kind": "named" | "collective"}.'
        ]
    failures: list[str] = []
    roles = ", ".join(sorted(PARTY_ROLES))
    for cluster_index, cluster in enumerate(value):
        location = f"Entry {index}: 'parties'[{cluster_index}]"
        if not isinstance(cluster, dict):
            failures.append(f"{location} must be an object with 'tag_ids' and 'role'.")
            continue
        if cluster.get("role") not in PARTY_ROLES:
            failures.append(f"{location} 'role' must be one of {roles}.")
        if "kind" in cluster and cluster["kind"] not in PARTY_KINDS:
            failures.append(f"{location} 'kind' must be named or collective.")
        tag_ids = cluster.get("tag_ids")
        if not isinstance(tag_ids, list):
            failures.append(f"{location} 'tag_ids' must be a list of tag IDs.")
            continue
        for tag_id in tag_ids:
            if not isinstance(tag_id, str):
                failures.append(
                    f"{location} 'tag_ids' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(f"{location} contains unknown tag ID {tag_id}.")
            elif tag_info["type"] not in LENDER_TAG_TYPES:
                failures.append(
                    f"{location} tag {tag_id} must be person or organization."
                )
    return failures


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
            failures.extend(validate_no_legacy_properties(index, obj))
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
        # whitespace-drifted echo of it (#154).
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
                # started then; the old single slot read it that way too.
                start_date_payload = select_date_payload(date_payloads, "agreement")
            maturity_payload = select_date_payload(date_payloads, "maturity")
            commitment_termination_payload = select_date_payload(
                date_payloads, "commitment_termination"
            )
            status_payload = (
                # Stage-2 responses carry events as date facts; anything older
                # (a status_event, or no dates list at all) replays verbatim.
                derived_status_payload(date_payloads)
                if "dates" in obj and "status_event" not in obj
                else standardized_status_payload(obj.get("status_event"), tag_details)
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
    pointer is flipped.

    Alclear's revolver is the confirmed case: a Credit Agreement dated as of
    2020-03-31, amended 2026-06-23 to cut commitments from $100,000,000 and
    extend maturity from 2026-06-28 to 2031-06-23. Every figure on the
    `$100,000,000` object is pre-amendment, yet it was the object carrying
    `amendment_of` (#138).

    The dates have to disagree for this to fire. While the prompt gave the
    predecessor and the amended instrument the same start date there was nothing
    to orient on, which is why the direction went unchecked.
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
            if rel_type not in INSTRUMENT_RELATION_TYPES:
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
                # redemption), and a scalar overwrote all but the last edge.
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
            if row_state.state in PUBLISHABLE_ROW_STATES:
                mention_rows.extend(row_state.debt_instrument_mentions)
            if row_state.state == "SUCCESS":
                succeeded_item_ids.add(row_state.item_id)
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
        if row_state.state in PUBLISHABLE_ROW_STATES:
            mentions_by_partition.setdefault((partition_date, shard), []).extend(
                row_state.debt_instrument_mentions
            )
        if row_state.state == "SUCCESS":
            succeeded_item_ids.add(row_state.item_id)
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
        if row_state.state in PUBLISHABLE_ROW_STATES:
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
    completion: CompletionResult | None = None,
) -> list[dict[str, str]] | None:
    """Advance one row given the response to its outstanding request.

    Applies the current stage's validate/postprocess, then either advances to the
    next stage, schedules a retry, or terminates the row. Returns the messages for
    the next LLM call, or None when the row has reached a terminal state.
    """
    stage = STAGE_BY_NAME[row_state.current_attempt.stage_name]
    stage_index = STAGE_INDEX[stage.name]
    row_state.add_response(response, completion)
    failures = stage.validate(row_state, response)
    row_state.add_validation(failures)
    if not failures:
        stage.postprocess(row_state)
        return _advance_after_stage(row_state, stage, stage_index)

    if row_state.current_attempt.attempt_index >= max_attempts:
        return _salvage_or_fail(row_state, stage, stage_index, max_attempts)
    row_state.retry(stage.build_retry_message(failures))
    return list(row_state.current_attempt.messages)


def _advance_after_stage(
    row_state: ExtractionRowState,
    stage: StageSpec,
    stage_index: int,
) -> list[dict[str, str]] | None:
    """Move one row past a completed stage: finish it or start the next stage."""
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


def _salvage_or_fail(
    row_state: ExtractionRowState,
    stage: StageSpec,
    stage_index: int,
    max_attempts: int,
) -> list[dict[str, str]] | None:
    """Keep what the row's final failed attempt still supports (#152).

    A whole-item drop used to be the only terminal outcome, so one invalid
    entry cost every valid one, and a relation-stage failure discarded mentions
    that had already passed `instrument_ie` validation. Both salvages finish the
    row PARTIAL: mentions publish, and the failure registry records the loss.
    NER has nothing to salvage — without tags no downstream stage can run.
    """
    if stage.name == InstrumentIEStage.name:
        dropped = salvage_instrument_ie_entries(row_state)
        if dropped is not None:
            row_state.salvage_notes.append(
                f"instrument_ie kept the valid entries and dropped {dropped} "
                f"invalid ones after {max_attempts} failed attempts"
                if dropped
                else "instrument_ie published every entry after "
                f"{max_attempts} failed attempts; the response was rejected as a "
                "whole but each entry validated on its own"
            )
            stage.postprocess(row_state)
            return _advance_after_stage(row_state, stage, stage_index)
    if (
        stage.name == InstrumentRelationStage.name
        and row_state.debt_instrument_mentions
    ):
        row_state.salvage_notes.append(
            f"instrument_relation failed after {max_attempts} attempts; "
            "mentions published without lineage relations"
        )
        row_state.finish("PARTIAL")
        return None
    row_state.finish("FAILED")
    return None


def instrument_entries_from_response(response: str) -> object:
    """Parse an instrument_ie response, accepting a bare object as a one-entry list.

    The model returns one object instead of a one-element array on most
    single-instrument items (73 of 300 on the 2026-09 window, every one of them
    recovered on retry); the object is the same entry, so it is read as such
    rather than paid for twice.
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
            completion = await resolved_client.complete(
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
        messages = handle_response(
            row_state,
            completion.text,
            max_attempts=max_attempts,
            completion=completion,
        )
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


def realign_tag_details(
    tag_details: dict[str, dict[str, object]],
    roundtrip_text: str,
    original_text: str,
) -> dict[str, dict[str, object]]:
    """Rewrite tag offsets from the NER round-trip text onto the original text.

    Evidence `char_start`/`char_end` must index the item's own `text` exactly —
    that is the published contract span highlighting rests on (#154). The NER
    stage only validates whitespace-collapsed equality, so the model may add or
    drop whitespace anywhere; the non-whitespace characters are identical in
    order, and each span is snapped to the original text along that alignment.

    Returns the input unchanged when the texts already match, and unchanged
    when they cannot be aligned (which collapse-equality validation rules out;
    tolerated here rather than raised so one pathological row degrades to the
    old offsets instead of failing the item).
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


def repair_unescaped_ampersands(text: str) -> str:
    """Escape ampersands the NER response left bare, so it can be parsed.

    `NERStage.preprocess` wraps the item text in `<body>` without escaping it, so
    an item containing `A&R Registration Rights Agreement` is handed to the model
    as invalid XML. The response then has to satisfy two requirements at once:
    reproduce the text exactly, and be well-formed XML. For text carrying a bare
    ampersand those conflict unless the model escapes on its own initiative,
    which the prompt never asks for.

    Bare ampersands appear in 44 of 342 relevant items in one held-out window
    across 37 issuers, and in 13% to 17% of relevant items in each of three
    windows, so this is a standing tax rather than one filer's quirk. Repairing
    the response rather than escaping the input keeps what the model sees
    unchanged, so nothing about its tagging behaviour moves (#127).

    Only `&` is repaired. A stray `<` or `>` never occurs in the source text, so
    one in a response is a real malformation and should still be rejected.
    """
    return UNESCAPED_AMPERSAND_PATTERN.sub("&amp;", text)


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


def as_plain_dict(value: object) -> dict[str, object] | None:
    """Return one SDK model or mapping as a plain JSON-serializable dict.

    Falls back to the instance attributes rather than returning None, so a
    provider SDK that stops using pydantic models does not silently start
    logging no usage at all.
    """
    if value is None:
        return None
    candidate: object = value
    if not isinstance(value, dict):
        dump = getattr(value, "model_dump", None)
        candidate = dump() if callable(dump) else getattr(value, "__dict__", None)
    if not isinstance(candidate, dict):
        return None
    return cast("dict[str, object]", json.loads(json.dumps(candidate, default=str)))


def completion_result_from_response(response: object) -> CompletionResult:
    """Build one completion result from an OpenRouter SDK response."""
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    return CompletionResult(
        text=extract_response_text(response),
        finish_reason=getattr(choice, "finish_reason", None),
        refusal=getattr(message, "refusal", None),
        usage=as_plain_dict(getattr(response, "usage", None)),
        response_id=getattr(response, "id", None),
        served_model=getattr(response, "model", None),
    )


def completion_result_from_batch_line(line: dict[str, object]) -> CompletionResult:
    """Build one completion result from an OpenAI Batch output JSONL line.

    The batch route reaches OpenAI directly rather than through OpenRouter, so
    the usage block carries token counts but no `cost`; spend has to be derived
    from the counts. A filtered response arrives here as a normal `200` with a
    body, so it never reaches the infrastructure-error path and today is
    indistinguishable from ordinary bad output (#135).
    """
    text = extract_batch_response_text(line)
    response = cast(dict[str, object], line.get("response") or {})
    body = cast(dict[str, object], response.get("body") or {})
    choices = cast(list[dict[str, object]], body.get("choices") or [])
    choice = choices[0] if choices else {}
    message = cast(dict[str, object], choice.get("message") or {})
    refusal = message.get("refusal")
    finish_reason = choice.get("finish_reason")
    return CompletionResult(
        text=text,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        refusal=refusal if isinstance(refusal, str) else None,
        usage=as_plain_dict(body.get("usage")),
        response_id=body.get("id") if isinstance(body.get("id"), str) else None,
        served_model=body.get("model") if isinstance(body.get("model"), str) else None,
    )


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
    property_label: str = "amount",
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
            f"Entry {index}: '{property_label}' evidence {quoted} describes an interest rate, "
            "margin, or fee rather than a money amount. Cite the money amount "
            f"instead, or omit '{property_label}'."
        )
    ]


def validate_dates_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the kind-typed dates list on one instrument entry.

    Every entry needs a known kind and date-typed evidence (a maturity may also
    cite the instrument's name span or a duration span, as before). At most one
    non-prior entry per kind: two current maturities describe two instruments,
    exactly as two principals do.
    """
    if "dates" not in obj:
        return []
    entries = obj["dates"]
    if not isinstance(entries, list):
        return [f"Entry {index}: 'dates' must be a list of objects."]
    failures: list[str] = []
    current_kinds: dict[str, int] = {}
    for position, entry in enumerate(entries):
        label = f"dates[{position}]"
        if not isinstance(entry, dict):
            failures.append(f"Entry {index}: '{label}' must be an object.")
            continue
        kind = entry.get("kind")
        if kind not in DATE_KINDS:
            allowed = ", ".join(sorted(DATE_KINDS))
            failures.append(f"Entry {index}: '{label}.kind' must be one of {allowed}.")
            kind = None
        prior = entry.get("prior", False)
        if not isinstance(prior, bool):
            failures.append(f"Entry {index}: '{label}.prior' must be true or false.")
        elif (
            kind is not None
            and not prior
            # `expected` is not current either: `select_date_payload` publishes
            # the entry that is neither prior nor expected, so a real closing
            # beside a planned one is one current closing, not two. Counting the
            # planned one rejected Costamare's 6-K, which states a 2026-04-30
            # closing and a 2026-06-30 expected closing for one facility.
            and entry.get("expected") is not True
            and kind in SINGLE_CURRENT_DATE_KINDS
        ):
            current_kinds[kind] = current_kinds.get(kind, 0) + 1
        if not isinstance(entry.get("expected", False), bool):
            failures.append(f"Entry {index}: '{label}.expected' must be true or false.")
        elif (
            entry.get("expected") is True
            and kind is not None
            and kind not in EVENT_DATE_KINDS
        ):
            failures.append(
                f"Entry {index}: '{label}' of kind '{kind}' cannot be `expected`. Only "
                "events (announcement, closing, amendment, repayment, retirement, "
                "termination, exchange, default) are planned or completed; a term such "
                "as a maturity is simply stated."
            )
        if prior is True and kind is not None and kind in EVENT_DATE_KINDS:
            failures.append(
                f"Entry {index}: '{label}' of kind '{kind}' cannot be `prior`. `prior` marks "
                "a term (maturity, commitment_termination, agreement) as it stood before "
                "a change; an event either happened or is `expected`."
            )
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            failures.append(
                f"Entry {index}: '{label}.evidence' must be a list of tag IDs."
            )
            continue
        if not evidence:
            if kind in DATE_KINDS_REQUIRING_EVIDENCE:
                failures.append(
                    f"Entry {index}: '{label}' of kind '{kind}' must cite a date span."
                )
            if entry.get("normalized_date") is not None:
                failures.append(
                    f"Entry {index}: '{label}.normalized_date' must be null when no "
                    "span is cited."
                )
        expected_types = DATE_KIND_EVIDENCE_TAG_TYPES.get(
            kind or "", DEFAULT_DATE_EVIDENCE_TAG_TYPES
        )
        for tag_id in evidence:
            if not isinstance(tag_id, str):
                failures.append(
                    f"Entry {index}: '{label}.evidence' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(
                    f"Entry {index}: '{label}' contains unknown tag ID {tag_id}."
                )
            elif tag_info["type"] not in expected_types:
                expected = ", ".join(sorted(expected_types))
                failures.append(
                    f"Entry {index}: '{label}' tag {tag_id} is type "
                    f"'{tag_info['type']}', expected {expected}."
                )
        normalized_date = entry.get("normalized_date")
        if normalized_date is not None and (
            not isinstance(normalized_date, str)
            or not ISO_DATE_PATTERN.fullmatch(normalized_date)
            or not is_valid_iso_date(normalized_date)
        ):
            failures.append(
                f"Entry {index}: '{label}.normalized_date' must be YYYY-MM-DD or null."
            )
        failures.extend(
            validate_standardized_single_value_cardinality(
                index=index,
                property_name=label,
                value=entry,
                tag_details=tag_details,
            )
        )
    for kind, count in sorted(current_kinds.items()):
        if count > 1:
            failures.append(
                f"Entry {index}: 'dates' has {count} current entries of kind "
                f"'{kind}'. One instrument has one {kind} date; a term stated "
                "before and after a change marks the earlier one `prior: true`, "
                "and two unrelated values are two instruments."
            )
    return failures


def validate_amounts_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the kind-typed amounts list on one instrument entry (#140)."""
    if "amounts" not in obj:
        return []
    entries = obj["amounts"]
    if not isinstance(entries, list):
        return [f"Entry {index}: 'amounts' must be a list of objects."]
    failures: list[str] = []
    for position, entry in enumerate(entries):
        label = f"amounts[{position}]"
        if not isinstance(entry, dict):
            failures.append(f"Entry {index}: '{label}' must be an object.")
            continue
        kind = entry.get("kind")
        if kind not in AMOUNT_KINDS:
            allowed = ", ".join(sorted(AMOUNT_KINDS))
            failures.append(f"Entry {index}: '{label}.kind' must be one of {allowed}.")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            failures.append(
                f"Entry {index}: '{label}.evidence' must be a list of tag IDs."
            )
            continue
        for tag_id in evidence:
            if not isinstance(tag_id, str):
                failures.append(
                    f"Entry {index}: '{label}.evidence' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(
                    f"Entry {index}: '{label}' contains unknown tag ID {tag_id}."
                )
            elif tag_info["type"] not in AMOUNT_EVIDENCE_TAG_TYPES:
                expected = ", ".join(sorted(AMOUNT_EVIDENCE_TAG_TYPES))
                failures.append(
                    f"Entry {index}: '{label}' tag {tag_id} is type "
                    f"'{tag_info['type']}', expected {expected}."
                )
        normalized_amount = entry.get("normalized_amount")
        if normalized_amount is not None and (
            not isinstance(normalized_amount, str)
            or not NUMERIC_STRING_PATTERN.fullmatch(normalized_amount)
        ):
            failures.append(
                f"Entry {index}: '{label}.normalized_amount' must be a numeric "
                "string or null."
            )
        currency = entry.get("currency")
        if currency is not None and (
            not isinstance(currency, str)
            or len(currency) != CURRENCY_CODE_LENGTH
            or currency != currency.upper()
        ):
            failures.append(
                f"Entry {index}: '{label}.currency' must be an uppercase "
                "3-letter code or null."
            )
        as_of = entry.get("as_of_date")
        if as_of is not None and (
            not isinstance(as_of, str)
            or not ISO_DATE_PATTERN.fullmatch(as_of)
            or not is_valid_iso_date(as_of)
        ):
            failures.append(
                f"Entry {index}: '{label}.as_of_date' must be YYYY-MM-DD or null."
            )
        if not isinstance(entry.get("prior", False), bool):
            failures.append(f"Entry {index}: '{label}.prior' must be true or false.")
        failures.extend(
            validate_amount_is_not_rate(
                index=index,
                value=entry,
                tag_details=tag_details,
                property_label=label,
            )
        )
    return failures


def validate_status_event(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the optional status_event on one instrument entry (#141)."""
    if "status_event" not in obj:
        return []
    event = obj["status_event"]
    if not isinstance(event, dict):
        return [f"Entry {index}: 'status_event' must be an object."]
    failures: list[str] = []
    status = event.get("status")
    if status not in STATUS_EVENT_VALUES:
        allowed = ", ".join(sorted(STATUS_EVENT_VALUES))
        failures.append(
            f"Entry {index}: 'status_event.status' must be one of {allowed}."
        )
    status_date = event.get("status_date")
    if status_date is None:
        return failures
    if not isinstance(status_date, dict):
        failures.append(
            f"Entry {index}: 'status_event.status_date' must be an object or null."
        )
        return failures
    evidence = status_date.get("evidence")
    if not isinstance(evidence, list):
        failures.append(
            f"Entry {index}: 'status_event.status_date.evidence' must be a list of tag IDs."
        )
        return failures
    for tag_id in evidence:
        if not isinstance(tag_id, str):
            failures.append(
                f"Entry {index}: 'status_event.status_date.evidence' must contain string tag IDs only."
            )
            continue
        tag_info = tag_details.get(tag_id)
        if tag_info is None:
            failures.append(
                f"Entry {index}: 'status_event.status_date' contains unknown tag ID {tag_id}."
            )
        elif tag_info["type"] != "date":
            failures.append(
                f"Entry {index}: 'status_event.status_date' tag {tag_id} is type "
                f"'{tag_info['type']}', expected date."
            )
    normalized_date = status_date.get("normalized_date")
    if normalized_date is not None and (
        not isinstance(normalized_date, str)
        or not ISO_DATE_PATTERN.fullmatch(normalized_date)
        or not is_valid_iso_date(normalized_date)
    ):
        failures.append(
            f"Entry {index}: 'status_event.status_date.normalized_date' must be YYYY-MM-DD or null."
        )
    return failures


def validate_interest_rate(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the optional interest_rate on one instrument entry (#157)."""
    if "interest_rate" not in obj:
        return []
    value = obj["interest_rate"]
    if not isinstance(value, dict):
        return [f"Entry {index}: 'interest_rate' must be an object."]
    failures: list[str] = []
    kind = value.get("kind")
    if kind not in INTEREST_RATE_KINDS:
        allowed = ", ".join(sorted(INTEREST_RATE_KINDS))
        failures.append(
            f"Entry {index}: 'interest_rate.kind' must be one of {allowed}."
        )
    rate_pct = value.get("rate_pct")
    if rate_pct is not None and (
        not isinstance(rate_pct, str) or not NUMERIC_STRING_PATTERN.fullmatch(rate_pct)
    ):
        failures.append(
            f"Entry {index}: 'interest_rate.rate_pct' must be a numeric string or null."
        )
    # An absent `evidence` key is an empty citation list, not a malformed
    # response: the prompt's own `3.875% senior notes due 2028` example omits it
    # and postprocess verifies the rate off the instrument name instead.
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        failures.append(
            f"Entry {index}: 'interest_rate.evidence' must be a list of tag IDs."
        )
        return failures
    for tag_id in evidence:
        if not isinstance(tag_id, str):
            failures.append(
                f"Entry {index}: 'interest_rate.evidence' must contain string tag IDs only."
            )
            continue
        tag_info = tag_details.get(tag_id)
        if tag_info is None:
            failures.append(
                f"Entry {index}: 'interest_rate' contains unknown tag ID {tag_id}."
            )
        elif tag_info["type"] not in INTEREST_RATE_EVIDENCE_TAG_TYPES:
            expected = ", ".join(sorted(INTEREST_RATE_EVIDENCE_TAG_TYPES))
            failures.append(
                f"Entry {index}: 'interest_rate' tag {tag_id} is type "
                f"'{tag_info['type']}', expected {expected}."
            )
    return failures


# `6 1/2%`, `5 7/8 %`: coupons written as vulgar fractions, common in older
# indentures and in Targa's note names. Read as a decimal rate.
FRACTION_RATE_PATTERN = re.compile(
    r"(\d+)\s+(\d+)\s*/\s*(\d+)\s*(?:%|percent\b)", re.IGNORECASE
)


def rate_tokens(text: str) -> list[str]:
    """Return every rate figure in a span as a decimal string, fractions included."""
    tokens: list[str] = []
    for whole, numerator, denominator in FRACTION_RATE_PATTERN.findall(text):
        if int(denominator):
            value = Decimal(whole) + Decimal(numerator) / Decimal(denominator)
            tokens.append(format(value.normalize(), "f"))
    stripped = FRACTION_RATE_PATTERN.sub(" ", text)
    tokens.extend(RATE_PCT_PATTERN.findall(stripped))
    return tokens


def rate_tokens_in_rate_span(text: str) -> list[str]:
    """Return the rate figures in a span the tagger already typed `interest_rate`.

    A prose span carries its own marker (`4.125%`, `6.5 percent`, `6 1/2%`). A
    table cell under a `COUPON PCT` header is the bare number `4.125`: the
    tagger's type is the marker, so a span that is nothing but a number is that
    rate. FHLB consolidated-obligation schedules lost every coupon this way (47
    mentions with kind fixed and no pct on the 2026-09 window).
    """
    marked = rate_tokens(text)
    if marked:
        return marked
    bare = text.strip()
    return [bare] if NUMERIC_STRING_PATTERN.fullmatch(bare) else []


def standardized_interest_rate_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> dict[str, object]:
    """Return the persisted interest-rate payload for one entry (#157).

    The model's ``rate_pct`` publishes only when a rate token in the cited
    evidence — or, failing that, in the instrument's name — parses to the same
    number, mirroring how amounts and dates are parser-verified.
    """
    if not isinstance(value, dict):
        return {"kind": None, "rate_pct": None, "spans": [], "derived_from": None}
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    kind = value.get("kind")
    payload["kind"] = kind if kind in INTEREST_RATE_KINDS else None
    tag_ids = evidence_tag_ids if isinstance(evidence_tag_ids, list) else []
    stated_rates = {
        rate
        for tag_id in tag_ids
        if isinstance(tag_id, str)
        and tag_details.get(tag_id, {}).get("type") == "interest_rate"
        for rate in rate_tokens_in_rate_span(str(tag_details[tag_id]["text"]))
    }
    # A rate read off the instrument's own name span is name-derived, exactly
    # like a maturity embedded in `notes due 2028`.
    name_rates = {
        rate
        for tag_id in tag_ids
        if isinstance(tag_id, str)
        and tag_details.get(tag_id, {}).get("type") == "debt_instrument"
        for rate in rate_tokens(str(tag_details[tag_id]["text"]))
    }
    if not stated_rates and not name_rates and name_text:
        name_rates = set(rate_tokens(name_text))
    evidence_rates = stated_rates or name_rates
    derived_from = (
        DERIVED_FROM_STATED
        if stated_rates
        else DERIVED_FROM_NAME
        if name_rates
        else None
    )
    model_rate = value.get("rate_pct")
    verified_rate = None
    if isinstance(model_rate, str) and NUMERIC_STRING_PATTERN.fullmatch(model_rate):
        for candidate in evidence_rates:
            try:
                if Decimal(candidate) == Decimal(model_rate):
                    verified_rate = model_rate
                    break
            except InvalidOperation:
                continue
    payload["rate_pct"] = verified_rate
    payload["derived_from"] = derived_from if verified_rate is not None else None
    return payload


def standardized_status_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return the persisted status payload for one instrument entry (#141)."""
    if not isinstance(value, dict):
        return {"status": None, "status_date": None}
    status = value.get("status")
    date_payload = standardized_date_payload(value.get("status_date"), tag_details)
    return {
        "status": status if status in STATUS_EVENT_VALUES else None,
        "status_date": date_payload,
    }


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
        "amounts_json": normalize_json_text(mention_row.get("amounts_json")),
        "dates_json": normalize_json_text(mention_row.get("dates_json")),
        "commitment_termination_date": mention_row.get("commitment_termination_date"),
        "commitment_termination_date_json": normalize_json_text(
            mention_row.get("commitment_termination_date_json")
        ),
        "maturity_date": mention_row.get("maturity_date"),
        "maturity_date_json": normalize_json_text(
            mention_row.get("maturity_date_json")
        ),
        "instrument_type": mention_row.get("instrument_type"),
        "interest_rate_json": normalize_json_text(
            mention_row.get("interest_rate_json")
        ),
        "item_id": item_id,
        "lender_disclosure": mention_row.get("lender_disclosure"),
        "name_json": normalize_json_text(mention_row.get("name_json")),
        "name": mention_row.get("name"),
        "parties_json": normalize_json_text(mention_row.get("parties_json")),
        "principal_amount": mention_row.get("principal_amount"),
        "status_json": normalize_json_text(mention_row.get("status_json")),
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


# The title of the contract, as opposed to a description of the obligation it
# creates: `Amended and Restated Credit Agreement`, `Indenture`, `Note Purchase
# Agreement`. NER now tags both for one facility (ner.md rule 11), and the
# agreement title is usually the longer string.
AGREEMENT_NAME_PATTERN = re.compile(
    r"\b(?:agreement|indenture|supplemental\s+indenture)\b", re.IGNORECASE
)


def canonical_instrument_name(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the name that describes the obligation, not the contract.

    `ner.md` rule 11 has NER tag both the facility phrase and its agreement
    name, and says the descriptive phrase "must never be dropped in favour of
    the agreement name". Plain longest-span selection did exactly that on 9 of
    the 476 multi-span names in the 2026-09 window — publishing `Second Amended
    and Restated Credit Agreement` over `term loan B facility`, and the
    `Super-Priority Senior Secured Priming ...` title over `DIP Facility`.

    That is not only a display problem: the published name feeds the matcher's
    `normalize_name_fingerprint`, and an amendment title is the generic,
    near-duplicate string that `NAME_CLASS_GATE` and the identifying-name guard
    then have to defend against. Preferring the obligation's own description
    keeps the individuating name where those heuristics can use it.

    Falls back to the longest span when every span names the contract, which is
    the right answer for an instrument the filing only ever calls by its
    agreement.
    """
    values = cluster_span_texts(tag_ids, tag_details)
    if not values:
        return None
    described = [value for value in values if not AGREEMENT_NAME_PATTERN.search(value)]
    return max(described or values, key=len)


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


def computed_sum_amount(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
    model_amount: object,
) -> str | None:
    """Return the model's amount when it equals the sum of the cited spans (#165).

    An increase-by amendment states a prior total and an increment but often
    never the result; the sum is deterministic arithmetic anchored to the cited
    evidence, and it is the only arithmetic accepted. Requires at least two
    parseable spans, and refuses when any single span already equals the
    model's value — that is agreement, not computation — or when any cited
    span reads as a rate.
    """
    if not isinstance(model_amount, str):
        return None
    model_value = decimal_from_amount_string(model_amount)
    if model_value is None:
        return None
    texts = cluster_span_texts(tag_ids, tag_details)
    if any(is_rate_like_amount_text(text) for text in texts):
        return None
    parsed = [normalized_amount_from_text(text) for text in texts]
    values = [
        decimal_from_amount_string(value) for value in parsed if value is not None
    ]
    if len(values) < MINIMUM_COMPUTED_SUM_SPANS or None in values:
        return None
    if any(value == model_value for value in values):
        return None
    if sum(values) != model_value:
        return None
    return normalized_amount_from_text(model_amount) or model_amount


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
    for word in sorted(AMOUNT_MULTIPLIERS, key=len, reverse=True):
        # `(?<![a-z])` rather than `\b` on the left so `$500mm` reads as well as
        # `$500 mm`, while `million` still cannot match inside a longer word.
        if re.search(rf"(?<![a-z]){word}\b", lowered):
            amount *= AMOUNT_MULTIPLIERS[word]
            break
    return normalize_numeric_string(amount)


def normalized_amount_from_name(text: str | None) -> str | None:
    """Parse a principal stated inside an instrument name into a numeric string.

    `$183.36 million term loan` carries its own principal, and NER tags the whole
    phrase as one `debt_instrument`, so there is no `amount` span to cite and the
    amount was lost (#129). A name stating more than one figure names no single
    principal, so it parses to None rather than to whichever comes first — the
    same rule `normalized_maturity_from_text` applies to maturities (#104).
    """
    if not text:
        return None
    values = {
        normalized_amount_from_text(match.group(0))
        for match in NAME_EMBEDDED_AMOUNT_PATTERN.finditer(text)
    }
    values.discard(None)
    if len(values) != 1:
        return None
    return values.pop()


def currency_from_name(text: str | None) -> str | None:
    """Return the currency of a principal stated inside an instrument name."""
    if not text:
        return None
    matches = list(NAME_EMBEDDED_AMOUNT_PATTERN.finditer(text))
    if len(matches) != 1:
        return None
    candidates = currency_candidates_from_text(matches[0].group(0))
    if len(candidates) != 1:
        return None
    return candidates.pop()


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
    """Parse one date mention into ISO format.

    The parser has to read every spelling a filing uses, because
    `standardized_date_payload` keeps the model's date only when it matches what
    the parser reads. A format the parser cannot read discards a date the model
    got right: `M/D/YYYY` alone cost 67 start dates and 67 end dates on one
    held-out window, all of them from tabular schedules (#133).

    Two-digit years stay unparsed. `7/28/26` cannot be resolved without guessing
    a century, and a null is better than a wrong decade.
    """
    if not text:
        return None
    # `March 5 , 2026` — a stray space before the comma is a formatting artefact.
    stripped = re.sub(r"\s+,", ",", text.strip())
    iso = LENIENT_ISO_DATE_PATTERN.fullmatch(stripped)
    if iso is not None:
        candidate = iso_date_from_numeric_parts(
            iso.group("year"), iso.group("month"), iso.group("day")
        )
        if candidate is not None:
            return candidate
    numeric = NUMERIC_DATE_PATTERN.search(stripped)
    if numeric is not None:
        candidate = iso_date_from_numeric_parts(
            numeric.group("year"), numeric.group("month"), numeric.group("day")
        )
        if candidate is not None:
            return candidate
    for pattern in (MONTH_FIRST_DATE_PATTERN, DAY_FIRST_DATE_PATTERN):
        match = pattern.search(stripped)
        if match is None:
            continue
        candidate = iso_date_from_parts(
            match.group("year"), match.group("month"), match.group("day")
        )
        if candidate is not None:
            return candidate
    return None


def normalized_month_year_from_text(text: str | None) -> str | None:
    """Parse a month-resolution date such as `in March 2056` to the month's last day.

    Only maturities accept month resolution (#164): `matures in June 2016` and
    `legal final maturity date is in March 2056` state the maturity as precisely
    as the filing ever will, while a start or status date at month resolution
    would be a guess. A span naming two months names no single date.
    """
    if not text:
        return None
    found: set[str] = set()
    for match in MONTH_YEAR_DATE_PATTERN.finditer(text):
        normalized = iso_month_end_from_parts(match.group("year"), match.group("month"))
        if normalized is not None:
            found.add(normalized)
    return next(iter(found)) if len(found) == 1 else None


def dates_agree(model_date: object, parsed_date: str | None) -> bool:
    """Return whether the model's date is the same day the parser read.

    Compared as dates rather than as strings, so a model writing `2026-7-28`
    against a parsed `2026-07-28` keeps its value (#133, same shape as #119).
    """
    if not isinstance(model_date, str) or parsed_date is None:
        return False
    normalized = normalized_date_from_text(model_date)
    return normalized is not None and normalized == parsed_date


def normalized_maturity_from_text(text: str | None) -> str | None:
    """Parse one maturity phrase such as 'notes due 2028' into ISO format.

    A phrase stating more than one maturity names no single instrument, so it
    parses to None rather than to whichever maturity comes first.
    """
    if not text:
        return None
    full_dates: set[str] = set()
    month_dates: set[str] = set()
    years: set[str] = set()
    for match in MATURITY_FULL_DATE_PATTERN.finditer(text):
        normalized = iso_date_from_parts(
            match.group("year"), match.group("month"), match.group("day")
        )
        if normalized is not None:
            full_dates.add(normalized)
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("more")))
    for match in MATURITY_MONTH_YEAR_PATTERN.finditer(text):
        normalized = iso_month_end_from_parts(match.group("year"), match.group("month"))
        if normalized is not None:
            month_dates.add(normalized)
            years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("more")))
    for match in MATURITY_YEAR_PATTERN.finditer(text):
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("years")))
    if full_dates:
        # A bare alternate year alongside a full date states a second maturity
        # too, and so does a month-year phrase for a different month (#164).
        if len(full_dates) != 1 or years - {value[:4] for value in full_dates}:
            return None
        full_date = full_dates.pop()
        if any(value[:7] != full_date[:7] for value in month_dates):
            return None
        return full_date
    if month_dates:
        if len(month_dates) != 1:
            return None
        month_date = month_dates.pop()
        if years - {month_date[:4]}:
            return None
        return month_date
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


def iso_month_end_from_parts(year: str, month_name: str) -> str | None:
    """Return the last day of one month-year maturity such as `due April 2033`.

    Month resolution is strictly better than the year-end synthetic the same
    name would produce without the month (#164); the matcher compares
    name-derived values at their true resolution either way.
    """
    month = MONTH_MAP.get(month_name.lower())
    if month is None:
        return None
    last_day = calendar.monthrange(int(year), int(month))[1]
    normalized = f"{year}-{month}-{last_day:02d}"
    return normalized if is_valid_iso_date(normalized) else None


def tenor_from_text(text: str | None) -> tuple[int, str] | None:
    """Parse one duration span such as `five-year` or `364-day` (#166).

    A span stating more than one distinct tenor anchors nothing, mirroring how
    coordinated maturities parse to None (#104).
    """
    if not text:
        return None
    tenors: set[tuple[int, str]] = set()
    for match in TENOR_PATTERN.finditer(text):
        num_text = match.group("num").lower()
        number = (
            TENOR_WORD_NUMBERS[num_text]
            if num_text in TENOR_WORD_NUMBERS
            else int(num_text)
        )
        tenors.add((number, match.group("unit").lower()))
    if len(tenors) != 1:
        return None
    return tenors.pop()


def date_plus_tenor(start: str, tenor: tuple[int, str], *, sign: int = 1) -> str | None:
    """Return start moved by one tenor (back when ``sign`` is -1), clamping to month ends."""
    try:
        anchor = date.fromisoformat(start)
    except ValueError:
        return None
    number, unit = tenor
    number *= sign
    # A tenor that lands outside the representable calendar is not an answer.
    # Raising here would unwind out of postprocess and past the driver, which
    # catches only InfrastructureError, killing the run before the failure
    # registry, the mentions and the audit log were written.
    try:
        if unit == "day":
            return (anchor + timedelta(days=number)).isoformat()
        months = number * 12 if unit == "year" else number
        total = anchor.month - 1 + months
        year = anchor.year + total // 12
        month = total % 12 + 1
        day = min(anchor.day, calendar.monthrange(year, month)[1])
        return date(year, month, day).isoformat()
    except (ValueError, OverflowError):
        return None


def computed_maturity_date(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
    model_date: object,
) -> str | None:
    """Return the model's maturity when it equals a cited date plus or minus a tenor (#166).

    A filing that states a facility's closing date and its tenor but never the
    maturity supports exactly one arithmetic answer. The model must cite both
    the `date` span and the `duration` span; its normalized date publishes only
    when some cited date plus some cited tenor lands on it, and it carries
    ``derived_from: "computed"``.
    """
    if (
        not isinstance(model_date, str)
        or not ISO_DATE_PATTERN.fullmatch(model_date)
        or not is_valid_iso_date(model_date)
    ):
        return None
    if not isinstance(tag_ids, list):
        return None
    starts: set[str] = set()
    tenors: set[tuple[int, str]] = set()
    for tag_id in tag_ids:
        detail = tag_details.get(tag_id) if isinstance(tag_id, str) else None
        if detail is None:
            continue
        text = str(detail["text"])
        if detail["type"] == "date":
            parsed = normalized_date_from_text(text)
            if parsed is not None:
                starts.add(parsed)
        elif detail["type"] == "duration":
            tenor = tenor_from_text(text)
            if tenor is not None:
                tenors.add(tenor)
    # `extended six months to September 3, 2027` states the prior maturity as
    # the new one minus the tenor, so the arithmetic runs both ways; each
    # direction still lands only on a date the cited spans determine exactly.
    for start in starts:
        for tenor in tenors:
            if model_date in (
                date_plus_tenor(start, tenor),
                date_plus_tenor(start, tenor, sign=-1),
            ):
                return model_date
    return None


def iso_date_from_numeric_parts(year: str, month: str, day: str) -> str | None:
    """Return one ISO date built from numeric month-first parts.

    Month-first, as SEC filings write it. An out-of-range month is rejected
    rather than swapped with the day: a span that means `28/07/2026` is more
    likely a format this parser should not be guessing at than a transposition.
    """
    normalized = f"{year}-{int(month):02d}-{int(day):02d}"
    return normalized if is_valid_iso_date(normalized) else None


def standardized_amount_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None = None,
    document_currencies: frozenset[str] | None = None,
) -> dict[str, object]:
    """Return evidence payload plus validated normalized amount fields.

    ``document_currencies`` are the currencies the whole item text evidences.
    A table cell (`35,000,000` under a `BANK PAR ($)` header) carries no marker
    of its own, so the model's `USD` used to be rejected and published as null
    (59 of 901 mentions on the 2026-09 window, all FHLB schedules and one
    private-credit filer). When the cited span shows no currency and the
    document as a whole shows exactly one, that one is accepted.
    """
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_amount_value(evidence_tag_ids, tag_details)
    parsed_amount = normalized_amount_from_text(evidence_text)
    parsed_currency_candidates = currency_candidates_from_text(evidence_text)
    derived_from = DERIVED_FROM_STATED if parsed_amount is not None else None
    if parsed_amount is None:
        # A principal stated inside the name has no `amount` span to cite, so the
        # name is the only evidence there is (#129).
        name_amount = normalized_amount_from_name(name_text)
        if name_amount is not None:
            parsed_amount = name_amount
            derived_from = DERIVED_FROM_NAME
            name_currency = currency_from_name(name_text)
            parsed_currency_candidates = (
                {name_currency} if name_currency else parsed_currency_candidates
            )
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
    if payload["normalized_amount"] is None:
        computed = computed_sum_amount(evidence_tag_ids, tag_details, model_amount)
        if computed is not None:
            payload["normalized_amount"] = computed
            derived_from = DERIVED_FROM_COMPUTED
            parsed_currency_candidates = {
                currency
                for text in cluster_span_texts(evidence_tag_ids, tag_details)
                for currency in currency_candidates_from_text(text)
            }
    if (
        not parsed_currency_candidates
        and document_currencies is not None
        and len(document_currencies) == 1
        and not is_rate_like_amount_text(evidence_text)
    ):
        parsed_currency_candidates = set(document_currencies)
    payload["currency"] = (
        model_currency
        if isinstance(model_currency, str)
        and model_currency in supported_currency_codes()
        and model_currency in parsed_currency_candidates
        else None
    )
    payload["derived_from"] = (
        derived_from if payload["normalized_amount"] is not None else None
    )
    return payload


def standardized_amounts_payloads(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
    document_currencies: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    """Return the kind-typed amount payloads for one instrument entry (#140).

    Reads the ``amounts`` list, falling back to the pre-#140 single ``amount``
    shape (kind unknown → null) so stored batch responses replay. When neither
    yields a value but the instrument's name embeds a principal, one
    name-derived principal entry is synthesized, preserving #129.
    """
    payloads: list[dict[str, object]] = []
    entries = obj.get("amounts")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            payload = standardized_amount_payload(
                entry,
                tag_details,
                name_text=name_text,
                document_currencies=document_currencies,
            )
            kind = entry.get("kind")
            payload["kind"] = kind if kind in AMOUNT_KINDS else None
            as_of = entry.get("as_of_date")
            payload["as_of_date"] = (
                as_of
                if isinstance(as_of, str)
                and ISO_DATE_PATTERN.fullmatch(as_of)
                and is_valid_iso_date(as_of)
                else None
            )
            # A term stated as it stood before a change is history, not the
            # instrument's current figure.
            payload["prior"] = entry.get("prior") is True
            payloads.append(payload)
    elif "amount" in obj:
        payload = standardized_amount_payload(
            obj.get("amount"),
            tag_details,
            name_text=name_text,
            document_currencies=document_currencies,
        )
        payload["kind"] = None
        payload["as_of_date"] = None
        payload["prior"] = False
        payloads.append(payload)
    if not select_principal_amount(payloads):
        synthesized = name_derived_principal_payload(name_text)
        if synthesized is not None:
            payloads.append(synthesized)
    return payloads


def name_derived_principal_payload(name_text: str | None) -> dict[str, object] | None:
    """Return a principal payload read off the instrument's own name (#129).

    `$183.36 million term loan` states its principal in its name and cites no
    `amount` span, so when the entry supplies no principal-bearing amount the
    name is the only evidence there is. There is no model value to agree with
    here — the parser's own reading is the value — so this cannot route through
    `standardized_amount_payload`, whose `amounts_agree` gate rejects a null
    model amount and made the previous version of this fallback unreachable.
    """
    parsed_amount = normalized_amount_from_name(name_text)
    if parsed_amount is None:
        return None
    return {
        "spans": [],
        "normalized_amount": parsed_amount,
        # `currency_from_name` already resolves through `currency_candidates_from_text`,
        # so it is either a supported code or None.
        "currency": currency_from_name(name_text),
        "derived_from": DERIVED_FROM_NAME,
        "kind": "principal",
        "as_of_date": None,
        "prior": False,
    }


def select_principal_amount(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Return the payload that supplies the flat principal columns.

    Commitment and principal are the identity-bearing kinds the matcher keys
    on (#140); a kind-less payload is the legacy single-amount shape, whose
    value carried the same meaning. Balances, draws, repayments, and proceeds
    never become the headline amount.
    """
    current = [payload for payload in payloads if not payload.get("prior")]
    for payload in current:
        if (
            payload.get("normalized_amount") is not None
            and payload.get("kind") in PRINCIPAL_AMOUNT_KINDS
        ):
            return payload
    for payload in current:
        if payload.get("normalized_amount") is not None and payload.get("kind") is None:
            return payload
    return {}


def standardized_date_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    allow_maturity_phrase: bool = False,
) -> dict[str, object]:
    """Return evidence payload plus validated normalized date field."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    # Every cited span is a candidate reading. Checking only the longest span
    # threw away a correct `March 2, 2026` whenever the model also cited the
    # defined term `Redemption Date` (the longer string) as evidence for the
    # same date; the multiple-distinct-values validator already guarantees the
    # parseable spans agree.
    span_texts = cluster_span_texts(evidence_tag_ids, tag_details)
    parsed_date = parsed_date_from_spans(span_texts, normalized_date_from_text)
    derived_from = DERIVED_FROM_STATED if parsed_date is not None else None
    if parsed_date is None and allow_maturity_phrase:
        # A stated month-resolution maturity (`in March 2056`) is still stated.
        parsed_date = parsed_date_from_spans(
            span_texts, normalized_month_year_from_text
        )
        derived_from = DERIVED_FROM_STATED if parsed_date is not None else None
    if parsed_date is None and allow_maturity_phrase:
        # A maturity phrase lives inside the instrument's own name span, so a
        # value parsed this way is name-derived even though evidence is cited.
        parsed_date = parsed_date_from_spans(span_texts, normalized_maturity_from_text)
        derived_from = DERIVED_FROM_NAME if parsed_date is not None else None
    model_date = value.get("normalized_date") if isinstance(value, dict) else None
    # The parser's own string is published, so a model writing the same day in a
    # different shape keeps its value rather than losing it (#133).
    payload["normalized_date"] = (
        parsed_date if dates_agree(model_date, parsed_date) else None
    )
    payload["derived_from"] = (
        derived_from if payload["normalized_date"] is not None else None
    )
    return payload


def parsed_date_from_spans(
    span_texts: list[str],
    parser: Callable[[str | None], str | None],
) -> str | None:
    """Return the one date the cited spans state, or None when they disagree."""
    parsed = {value for value in (parser(text) for text in span_texts) if value}
    return next(iter(parsed)) if len(parsed) == 1 else None


def instrument_date_entries(obj: dict[str, Any]) -> list[tuple[str, object, bool]]:
    """Return (kind, value, prior) date entries for one instrument object.

    Reads the ``dates`` list, falling back to the pre-dates[] single-value
    properties (kind implied by the property name) so stored responses replay.
    """
    entries: list[tuple[str, object, bool]] = []
    dates = obj.get("dates")
    if isinstance(dates, list):
        for entry in dates:
            if not isinstance(entry, dict) or entry.get("kind") not in DATE_KINDS:
                continue
            entries.append((str(entry["kind"]), entry, entry.get("prior") is True))
        return entries
    for property_name, kind in LEGACY_DATE_PROPERTY_KINDS.items():
        if property_name in obj:
            if property_name == "end_date" and "maturity_date" in obj:
                continue
            entries.append((kind, obj[property_name], False))
    return entries


LEGACY_STATUS_DATE_KINDS = {
    status: kind for kind, status in STATUS_FOR_DATE_KIND.items()
}


def date_fact_is_expected(kind: str, entry: object) -> bool:
    """Return whether a fact states a planned date rather than one that occurred."""
    if kind == "expected_closing":
        return True
    return isinstance(entry, dict) and entry.get("expected") is True


def date_precision(
    normalized_date: str | None,
    span_texts: list[str],
    derived_from: str | None,
) -> str | None:
    """Return day / month / year for how precisely the cited text states the date."""
    if normalized_date is None:
        return None
    if derived_from == DERIVED_FROM_COMPUTED:
        return "day"
    if any(normalized_date_from_text(text) == normalized_date for text in span_texts):
        return "day"
    if any(
        normalized_month_year_from_text(text) == normalized_date for text in span_texts
    ) or any(
        MATURITY_MONTH_YEAR_PATTERN.search(text) is not None for text in span_texts
    ):
        return "month"
    if normalized_date.endswith(YEAR_ONLY_MATURITY_SUFFIX):
        return "year"
    return "day"


def standardized_dates_payloads(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> list[dict[str, object]]:
    """Return the kind-typed date payloads for one instrument entry.

    Each payload carries ``kind``, ``prior``, ``precision`` and the evidence
    fields of the single-value payloads it replaces. A maturity keeps the
    name-derived and computed fallbacks (#128, #166); when the object states
    no maturity at all but its name embeds one, that name-derived maturity is
    synthesized, as before.
    """
    payloads: list[dict[str, object]] = []
    saw_maturity = False
    for kind, value, prior in instrument_date_entries(obj):
        if kind == "maturity":
            saw_maturity = True
            payload = standardized_end_date_payload(
                value, tag_details, name_text=name_text
            )
        else:
            payload = standardized_date_payload(value, tag_details)
        payload["expected"] = date_fact_is_expected(kind, value)
        payload["kind"] = "closing" if kind == "expected_closing" else kind
        payload["prior"] = prior
        payload["precision"] = date_precision(
            cast(str | None, payload.get("normalized_date")),
            cluster_span_texts(single_value_evidence_tag_ids(value), tag_details)
            + ([name_text] if name_text else []),
            cast(str | None, payload.get("derived_from")),
        )
        payloads.append(payload)
    if not saw_maturity:
        synthesized = standardized_end_date_payload(
            None, tag_details, name_text=name_text
        )
        if synthesized.get("normalized_date") is not None:
            synthesized["kind"] = "maturity"
            synthesized["prior"] = False
            synthesized["expected"] = False
            synthesized["precision"] = date_precision(
                cast(str | None, synthesized.get("normalized_date")),
                [name_text] if name_text else [],
                cast(str | None, synthesized.get("derived_from")),
            )
            payloads.append(synthesized)
    return payloads


def select_date_payload(
    payloads: list[dict[str, object]], kind: str
) -> dict[str, object]:
    """Return the current (non-prior, not merely expected) payload of one kind."""
    for payload in payloads:
        if (
            payload.get("kind") == kind
            and not payload.get("prior")
            and not payload.get("expected")
        ):
            return payload
    return {"spans": [], "normalized_date": None, "derived_from": None}


def mark_post_filing_events_expected(
    date_payloads: list[dict[str, object]], filing_date: str
) -> None:
    """An event dated after its own filing has not happened yet.

    `Interest on the Notes will accrue from April 6, 2022` in a March 25 pricing
    8-K and a bond table's settlement dates a week after filing both read as
    completed closings to the model; the calendar says otherwise.
    """
    if not filing_date:
        return
    for payload in date_payloads:
        value = payload.get("normalized_date")
        if (
            payload.get("kind") in EVENT_DATE_KINDS
            and payload.get("kind") != "announcement"
            and isinstance(value, str)
            and value > filing_date
        ):
            payload["expected"] = True


def derived_status_payload(
    date_payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Derive what this mention says happened from its dated event facts.

    The newest completed (not `expected`, not `prior`) event wins; undated
    events sort with the newest dated one and ties resolve by how much the
    event says about the obligation's life. A planned retirement or a planned
    closing decides nothing here — an instrument whose only closing is expected
    is `announced`, and a pending retirement is left to the matcher, which reads
    the expected facts off `dates_json`.
    """
    completed = [
        payload
        for payload in date_payloads
        if payload.get("kind") in EVENT_DATE_KINDS
        and not payload.get("prior")
        and not payload.get("expected")
        and payload.get("kind") != "repayment"
    ]
    if not completed:
        expected_closing = [
            payload
            for payload in date_payloads
            if payload.get("kind") == "closing" and payload.get("expected")
        ]
        if expected_closing:
            return {
                "status": "announced",
                "status_date": {
                    "spans": [],
                    "normalized_date": None,
                    "derived_from": None,
                },
                "derived_from_kind": "closing:expected",
            }
        agreement = [
            payload
            for payload in date_payloads
            if payload.get("kind") == "agreement" and not payload.get("prior")
        ]
        if agreement:
            # A dated agreement with no other event is an instrument that was
            # entered into on that date.
            return {
                "status": "entered_into",
                "status_date": {
                    key: agreement[0].get(key)
                    for key in ("spans", "normalized_date", "derived_from")
                },
                "derived_from_kind": "agreement",
            }
        return {"status": None, "status_date": None, "derived_from_kind": None}
    newest_dated = max(
        (
            str(payload["normalized_date"])
            for payload in completed
            if payload.get("normalized_date")
        ),
        default=None,
    )

    def rank(payload: dict[str, object]) -> tuple[str, int]:
        value = payload.get("normalized_date")
        return (
            str(value) if value else (newest_dated or ""),
            EVENT_KIND_PRECEDENCE.get(str(payload.get("kind")), 0),
        )

    winner = max(completed, key=rank)
    kind = str(winner["kind"])
    return {
        "status": STATUS_FOR_DATE_KIND[kind],
        "status_date": {
            key: winner.get(key) for key in ("spans", "normalized_date", "derived_from")
        },
        "derived_from_kind": kind,
    }


def expected_retirement_in_payloads(date_payloads: list[dict[str, object]]) -> bool:
    """Return whether the mention states a planned retirement of the obligation."""
    return any(
        payload.get("kind") in TERMINAL_DATE_KINDS and payload.get("expected")
        for payload in date_payloads
    )


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
        model_date = value.get("normalized_date") if isinstance(value, dict) else None
        computed = computed_maturity_date(
            single_value_evidence_tag_ids(value),
            tag_details,
            model_date,
        )
        if computed is not None:
            payload["normalized_date"] = computed
            payload["derived_from"] = DERIVED_FROM_COMPUTED
    if payload["normalized_date"] is None:
        derived_date = normalized_maturity_from_text(name_text)
        if derived_date is not None:
            payload["normalized_date"] = derived_date
            payload["derived_from"] = DERIVED_FROM_NAME
    return payload


# Where a normalized value came from: cited evidence spans, the instrument's own
# name, or nowhere (no value). Downstream consumers key on this — the matcher
# treats a name-synthesized YYYY-12-31 maturity as year-resolution only (#128),
# and the dashboard can explain a value whose evidence list is empty.
DERIVED_FROM_STATED = "stated"
DERIVED_FROM_NAME = "name"
DERIVED_FROM_COMPUTED = "computed"
# A sum needs at least two addends; one parsed span is agreement, not arithmetic.
MINIMUM_COMPUTED_SUM_SPANS = 2


def cluster_payload(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return the evidence spans for one cluster.

    ``char_start``/``char_end`` index the item's own ``text`` exactly (#154).
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
        if not payload["spans"]:
            continue
        resolved = (
            str(annotation) if annotation in allowed_annotations else default_annotation
        )
        pairs.append((payload, resolved))
    return pairs


LENDER_PARTY_ROLE = "lender"


def party_payloads_and_disclosure(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """Return every party cluster with its role and kind, plus lender disclosure.

    The model already labels every cluster; the labels persist rather than only
    steering what to drop (#150). Lender clusters carry ``role: "lender"`` and
    the model's ``kind``; other clusters carry the model's role — including the
    borrower, whose identity matters exactly when a subsidiary is the obligor
    under the parent filer's 8-K — and ``kind: "named"``, since the collective
    distinction is only elicited for lenders. ``canonical_name`` is the longest
    span, same as instrument names.
    """
    parties: list[dict[str, object]] = []
    lender_pairs = annotated_party_clusters(
        obj.get("lenders", []),
        tag_details,
        property_name="lenders",
    )
    for payload, kind in lender_pairs:
        parties.append(
            {
                "canonical_name": canonical_value(
                    payload_tag_ids(payload), tag_details
                ),
                "role": LENDER_PARTY_ROLE,
                "kind": kind,
                "spans": payload["spans"],
            }
        )
    for payload, role in annotated_party_clusters(
        obj.get("other_interested_parties", []),
        tag_details,
        property_name="other_interested_parties",
    ):
        parties.append(
            {
                "canonical_name": canonical_value(
                    payload_tag_ids(payload), tag_details
                ),
                "role": role,
                "kind": DEFAULT_LENDER_CLUSTER_KIND,
                "spans": payload["spans"],
            }
        )
    # Detect the *legacy* shape positively. Inferring the current shape from the
    # presence of `parties` or `dates` misread a perfectly ordinary current-schema
    # entry that omitted both — the prompt tells the model to omit anything the
    # document does not mention — and sent it down the legacy path, where it
    # published "holders fully disclosed" beside an empty party list.
    legacy_shape = any(
        key in obj
        for key in ("lenders", "other_interested_parties", "lenders_known_incomplete")
    )
    if not legacy_shape:
        # Current shape: one list, one role per cluster, disclosure derived.
        parties = []
        raw_parties = obj.get("parties")
        for cluster in raw_parties if isinstance(raw_parties, list) else []:
            if not isinstance(cluster, dict):
                continue
            payload = cluster_payload(cluster.get("tag_ids"), tag_details)
            if not payload["spans"]:
                continue
            role = cluster.get("role")
            kind = cluster.get("kind")
            parties.append(
                {
                    "canonical_name": canonical_value(
                        payload_tag_ids(payload), tag_details
                    ),
                    "role": role if role in PARTY_ROLES else "other",
                    "kind": kind
                    if kind in PARTY_KINDS
                    else DEFAULT_LENDER_CLUSTER_KIND,
                    "spans": payload["spans"],
                }
            )
        return parties, lender_disclosure_for(
            [p["kind"] for p in parties if p["role"] == LENDER_PARTY_ROLE]
        )
    # Legacy replay: the model declared the flag itself. A declared `true`
    # alongside named lenders is the collective case by another name.
    lender_kinds = [kind for payload, kind in lender_pairs]
    if obj.get("lenders_known_incomplete") is True and lender_kinds:
        return parties, LENDER_DISCLOSURE_COLLECTIVE_PRESENT
    return parties, lender_disclosure_for(lender_kinds)


def lender_disclosure_for(lender_kinds: list[object]) -> str:
    """Return how completely the lender clusters identify who holds the debt.

    No lender cluster at all is `none_named` — a public-market series, a
    redemption notice, a syndicate where only the agent is named. A collective
    cluster (`the other lenders party thereto`) is `collective_present`. Only
    when every lender cluster is named is the list `complete`.
    """
    if not lender_kinds:
        return LENDER_DISCLOSURE_NONE_NAMED
    if COLLECTIVE_LENDER_KIND in lender_kinds:
        return LENDER_DISCLOSURE_COLLECTIVE_PRESENT
    return LENDER_DISCLOSURE_COMPLETE


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

    Two objects built from one name span render as the same tagged text twice,
    once per instrument id, so `Third Amended and Restated Loan Agreement` and
    its predecessor are indistinguishable in the body. The amount and dates that
    tell them apart live in the `instrument_ie` output, which this stage never
    saw, and it was being asked to decide which one carries "the newer terms"
    from the ids alone. Every inverted lineage pair on the held-out run was a
    same-name pair (#138).
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


def escape_xml_attribute(value: str) -> str:
    """Escape one string for use inside an XML attribute value."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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
    """Summarize what this row lost, for its failure-registry entry.

    A salvaged row (#152) is terminal-but-publishable: its last attempt often
    succeeded, so the attempt carries no validation errors and the generic
    "unexpected response" summary below would describe a stage that worked.
    The salvage notes are the only record of what was actually dropped, so they
    are what the registry reports.
    """
    if row_state.salvage_notes:
        return "; ".join(row_state.salvage_notes)
    failures = row_state.current_attempt.validation_errors
    if failures:
        return "; ".join(failures)
    if row_state.current_attempt.response:
        return f"Unexpected response at stage {row_state.current_attempt.stage_name}"
    return f"Extractor failed at stage {row_state.current_attempt.stage_name}"


def failed_stage_name(row_state: ExtractionRowState) -> str:
    """Return the stage whose failure this row is registered for.

    For a salvaged row that is the stage salvage fired in, not the last stage
    the row ran — an operator retrying the row needs the former.
    """
    for note in row_state.salvage_notes:
        stage_name, _, _ = note.partition(" ")
        if stage_name in {stage.name for stage in EXTRACTOR_STAGES}:
            return stage_name
    return row_state.current_attempt.stage_name


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
        "stage": failed_stage_name(row_state),
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
