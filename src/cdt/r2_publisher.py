"""Publish dashboard snapshot JSON into Cloudflare R2."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import boto3
import pandas as pd
from botocore.client import BaseClient

from cdt.extractor import mentions_root
from cdt.itemizer import items_root
from cdt.itemizer.core import ITEM_COLUMNS
from cdt.matcher import debt_instruments_root, mention_matches_root
from cdt.shared import get_logger
from cdt.storage import ArtifactPath, read_dataset

LOGGER = get_logger(__name__)
R2_REGION_NAME = "auto"
INDEX_KEY = "index.json"
COMPANIES_PREFIX = "companies"
INSTRUMENTS_PREFIX = "debt-instruments"
HIGHLIGHT_PROPERTY_MAP = {
    "name_json": "name",
    "amount_json": "amount",
    "start_date_json": "startDate",
    "end_date_json": "endDate",
    "lenders_json": "lenders",
    "other_interested_parties_json": "otherInterestedParties",
}
PROPERTY_LABELS = {
    "name": "Name",
    "amount": "Amount",
    "startDate": "Start Date",
    "endDate": "End Date",
    "lenders": "Lenders",
    "otherInterestedParties": "Other Interested Parties",
}


@dataclass(frozen=True)
class R2PublishConfig:
    """Configuration for publishing dashboard snapshot JSON to R2."""

    account_id: str
    bucket_name: str
    access_key_id: str
    secret_access_key: str
    object_prefix: str = "generated"

    @property
    def endpoint_url(self: R2PublishConfig) -> str:
        """Return the S3-compatible endpoint URL for this R2 account."""
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


@dataclass(frozen=True)
class R2PublishResult:
    """Summary of one publish attempt."""

    object_count: int
    uploaded_count: int
    skipped_count: int
    deleted_count: int


def publish_config_from_env() -> R2PublishConfig | None:
    """Return R2 publish config from env vars, or None when not configured."""
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    bucket_name = os.environ.get("R2_BUCKET_NAME", "").strip()
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    object_prefix = (
        os.environ.get("R2_OBJECT_PREFIX", "generated").strip() or "generated"
    )
    values = (account_id, bucket_name, access_key_id, secret_access_key)
    if not any(values):
        return None
    if not all(values):
        msg = (
            "Incomplete R2 configuration; expected R2_ACCOUNT_ID, R2_BUCKET_NAME, "
            "R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY."
        )
        raise RuntimeError(msg)
    return R2PublishConfig(
        account_id=account_id,
        bucket_name=bucket_name,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        object_prefix=object_prefix.strip("/"),
    )


def publish_dashboard_snapshot(
    *,
    artifact_root: ArtifactPath,
    data_dir: Path | None = None,
    config: R2PublishConfig,
    r2_client: BaseClient | None = None,
) -> R2PublishResult:
    """Build and publish dashboard snapshot JSON into the configured R2 bucket."""
    snapshot = build_dashboard_snapshot(artifact_root=artifact_root, data_dir=data_dir)
    client = r2_client or make_r2_client(config)
    result = sync_snapshot(snapshot=snapshot, config=config, r2_client=client)
    LOGGER.info(
        "R2 publish complete: objects=%s uploaded=%s skipped=%s deleted=%s bucket=%s prefix=%s",
        result.object_count,
        result.uploaded_count,
        result.skipped_count,
        result.deleted_count,
        config.bucket_name,
        config.object_prefix,
    )
    return result


def make_r2_client(config: R2PublishConfig) -> BaseClient:
    """Construct an S3-compatible boto3 client for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name=R2_REGION_NAME,
    )


def build_dashboard_snapshot(
    *,
    artifact_root: ArtifactPath,
    data_dir: Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, bytes]:
    """Build the full dashboard snapshot payload keyed by relative R2 object path."""
    instruments = read_dataset(
        debt_instruments_root(artifact_root=artifact_root, data_dir=data_dir)
    )
    mention_matches = read_dataset(
        mention_matches_root(artifact_root=artifact_root, data_dir=data_dir)
    )
    mentions = read_dataset(
        mentions_root(artifact_root=artifact_root, data_dir=data_dir)
    )
    items = read_dataset(
        items_root(artifact_root=artifact_root, data_dir=data_dir),
        columns=ITEM_COLUMNS,
    ).reindex(columns=ITEM_COLUMNS)

    generated_at_value = generated_at or datetime.now(UTC)
    index_rows = build_index_rows(instruments, mention_matches)
    company_map = build_company_payloads(index_rows)
    instrument_map = build_instrument_payloads(
        instruments=instruments,
        mention_matches=mention_matches,
        mentions=mentions,
        items=items,
    )
    snapshot: dict[str, bytes] = {}
    snapshot[INDEX_KEY] = serialize_json(
        {
            "generatedAt": generated_at_value.isoformat(),
            "instruments": index_rows,
        }
    )
    for cik, payload in company_map.items():
        snapshot[f"{COMPANIES_PREFIX}/{quote(cik, safe='')}.json"] = serialize_json(
            payload
        )
    for instrument_id, payload in instrument_map.items():
        encoded_id = quote(instrument_id, safe="")
        snapshot[f"{INSTRUMENTS_PREFIX}/{encoded_id}.json"] = serialize_json(payload)
    return snapshot


def build_index_rows(
    instruments: pd.DataFrame,
    mention_matches: pd.DataFrame,
) -> list[dict[str, object]]:
    """Return the landing-page instrument rows."""
    mention_counts = (
        mention_matches["debt_instrument_id"].value_counts().to_dict()
        if not mention_matches.empty and "debt_instrument_id" in mention_matches
        else {}
    )
    rows = [
        {
            "id": str(row["debt_instrument_id"]),
            "cik": coerce_text(row.get("cik")) or "",
            "name": coerce_text(row.get("name")),
            "amount": coerce_text(row.get("amount")),
            "startDate": coerce_text(row.get("start_date")),
            "endDate": coerce_text(row.get("end_date")),
            "mentionCount": int(mention_counts.get(str(row["debt_instrument_id"]), 0)),
        }
        for row in instruments.to_dict("records")
        if coerce_text(row.get("debt_instrument_id"))
    ]
    return sorted(rows, key=lambda row: (str(row["cik"]), str(row["id"])))


def build_company_payloads(
    index_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Group index rows into company detail payloads."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in index_rows:
        grouped[str(row["cik"])].append(row)
    return {
        cik: {
            "cik": cik,
            "debtInstruments": sorted(rows, key=lambda row: str(row["id"])),
        }
        for cik, rows in sorted(grouped.items())
    }


def build_instrument_payloads(
    *,
    instruments: pd.DataFrame,
    mention_matches: pd.DataFrame,
    mentions: pd.DataFrame,
    items: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Build detail payloads for every debt instrument."""
    mention_lookup = {
        str(row["debt_instrument_mention_id"]): row
        for row in mentions.to_dict("records")
    }
    item_lookup = {str(row["item_id"]): row for row in items.to_dict("records")}
    instrument_lookup = {
        str(row["debt_instrument_id"]): row for row in instruments.to_dict("records")
    }
    mention_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mention_matches.to_dict("records"):
        mention_id = coerce_text(row.get("debt_instrument_mention_id"))
        instrument_id = coerce_text(row.get("debt_instrument_id"))
        if mention_id and instrument_id:
            mention_groups[instrument_id].append(
                {
                    "mention_id": mention_id,
                    "matcher_status": coerce_text(row.get("matcher_status")) or "",
                }
            )
    child_links = build_child_links(instrument_lookup)

    payloads: dict[str, dict[str, object]] = {}
    for instrument_id, row in sorted(instrument_lookup.items()):
        joined_mentions = build_joined_mentions(
            mention_matches=mention_groups.get(instrument_id, []),
            mention_lookup=mention_lookup,
            item_lookup=item_lookup,
        )
        payloads[instrument_id] = {
            "instrument": {
                "id": instrument_id,
                "cik": coerce_text(row.get("cik")) or "",
                "name": coerce_text(row.get("name")),
                "amount": coerce_text(row.get("amount")),
                "startDate": coerce_text(row.get("start_date")),
                "endDate": coerce_text(row.get("end_date")),
                "seedMentionId": coerce_text(
                    row.get("seed_debt_instrument_mention_id")
                ),
                "lenders": extract_party_names(row.get("lenders_json")),
                "otherInterestedParties": extract_party_names(
                    row.get("other_interested_parties_json")
                ),
            },
            "mentions": joined_mentions,
            "properties": build_properties(row, joined_mentions),
            "relatedInstruments": build_related_instruments(
                instrument_id=instrument_id,
                instrument_row=row,
                instrument_lookup=instrument_lookup,
                child_links=child_links,
            ),
        }
    return payloads


def build_joined_mentions(
    *,
    mention_matches: list[dict[str, str]],
    mention_lookup: dict[str, dict[str, object]],
    item_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return contract-shaped mention payloads for one instrument."""
    joined_rows: list[dict[str, object]] = []
    unique_matches = {
        (row["mention_id"], row["matcher_status"]): row for row in mention_matches
    }
    for match_row in sorted(
        unique_matches.values(),
        key=lambda row: (row["mention_id"], row["matcher_status"]),
    ):
        mention_id = match_row["mention_id"]
        mention_row = mention_lookup.get(mention_id)
        if mention_row is None:
            continue
        item_row = item_lookup.get(str(mention_row.get("item_id")), {})
        joined_rows.append(
            {
                "id": mention_id,
                "itemId": coerce_text(mention_row.get("item_id")) or "",
                "rawId": coerce_text(mention_row.get("raw_id")) or "",
                "itemNumber": coerce_text(item_row.get("item")),
                "accessionNumber": coerce_text(mention_row.get("accession_number")),
                "filedAt": coerce_text(mention_row.get("date")),
                "secUrl": coerce_text(item_row.get("url")),
                "resourceUri": coerce_text(item_row.get("resource_uri")),
                "matcherStatus": match_row["matcher_status"] or None,
                "itemText": coerce_text(item_row.get("text")) or "",
                "summary": {
                    "name": coerce_text(mention_row.get("name")),
                    "amount": coerce_text(mention_row.get("amount")),
                    "startDate": coerce_text(mention_row.get("start_date")),
                    "endDate": coerce_text(mention_row.get("end_date")),
                },
                "highlightMap": build_highlight_map(mention_row),
            }
        )
    joined_rows.sort(
        key=lambda row: (
            str(row.get("filedAt") or ""),
            str(row.get("accessionNumber") or ""),
            str(row.get("itemId") or ""),
            str(row.get("id") or ""),
        )
    )
    return joined_rows


def build_highlight_map(
    mention_row: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Translate extractor evidence JSON into dashboard highlight entries."""
    highlight_map: dict[str, list[dict[str, object]]] = {}
    for source_key, property_name in HIGHLIGHT_PROPERTY_MAP.items():
        highlights = extract_highlights(mention_row.get(source_key), property_name)
        if highlights:
            highlight_map[property_name] = highlights
    return highlight_map


def extract_highlights(
    value: object,
    property_name: str,
) -> list[dict[str, object]]:
    """Return highlight entries from one extractor JSON payload."""
    payload = parse_json(
        value,
        default=[] if property_name in {"lenders", "otherInterestedParties"} else {},
    )
    if isinstance(payload, dict):
        payloads = [payload]
    elif isinstance(payload, list):
        payloads = [entry for entry in payload if isinstance(entry, dict)]
    else:
        return []
    highlights: list[dict[str, object]] = []
    for entry in payloads:
        mentions = entry.get("mentions", [])
        if not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            start = mention.get("char_start")
            end = mention.get("char_end")
            text = coerce_text(mention.get("text"))
            if not isinstance(start, int) or not isinstance(end, int) or text is None:
                continue
            highlights.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "property": property_name,
                }
            )
    highlights.sort(
        key=lambda item: (int(item["start"]), int(item["end"]), str(item["text"]))
    )
    return highlights


def build_properties(
    instrument_row: dict[str, object],
    joined_mentions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build the dashboard property summary for one instrument."""
    properties = [
        property_payload(
            key="name",
            value=coerce_text(instrument_row.get("name")),
            joined_mentions=joined_mentions,
        ),
        property_payload(
            key="amount",
            value=coerce_text(instrument_row.get("amount")),
            joined_mentions=joined_mentions,
        ),
        property_payload(
            key="startDate",
            value=coerce_text(instrument_row.get("start_date")),
            joined_mentions=joined_mentions,
        ),
        property_payload(
            key="endDate",
            value=coerce_text(instrument_row.get("end_date")),
            joined_mentions=joined_mentions,
        ),
        property_payload(
            key="lenders",
            value=extract_party_names(instrument_row.get("lenders_json")),
            joined_mentions=joined_mentions,
        ),
        property_payload(
            key="otherInterestedParties",
            value=extract_party_names(
                instrument_row.get("other_interested_parties_json")
            ),
            joined_mentions=joined_mentions,
        ),
    ]
    return properties


def property_payload(
    *,
    key: str,
    value: object,
    joined_mentions: list[dict[str, object]],
) -> dict[str, object]:
    """Build one property payload."""
    mention_id = property_mention_id(
        key=key, value=value, joined_mentions=joined_mentions
    )
    has_highlight = False
    if mention_id is not None:
        mention = next(
            item for item in joined_mentions if str(item.get("id")) == mention_id
        )
        has_highlight = bool(mention.get("highlightMap", {}).get(key))
    normalized_value = normalize_property_value(value)
    return {
        "key": key,
        "label": PROPERTY_LABELS[key],
        "value": normalized_value,
        "mentionId": mention_id,
        "hasHighlight": has_highlight,
    }


def property_mention_id(
    *,
    key: str,
    value: object,
    joined_mentions: list[dict[str, object]],
) -> str | None:
    """Return the newest mention carrying the canonical property value."""
    if value is None:
        return None
    normalized = normalize_property_value(value)
    for mention in reversed(joined_mentions):
        highlight_map = mention.get("highlightMap", {})
        if isinstance(highlight_map, dict) and highlight_map.get(key):
            summary = mention.get("summary", {})
            if isinstance(normalized, list):
                return str(mention["id"])
            if isinstance(summary, dict) and summary.get(key) == normalized:
                return str(mention["id"])
    return None


def normalize_property_value(value: object) -> str | list[str] | None:
    """Normalize property values into contract-compatible scalars or string arrays."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    text = coerce_text(value)
    return text


def build_related_instruments(
    *,
    instrument_id: str,
    instrument_row: dict[str, object],
    instrument_lookup: dict[str, dict[str, object]],
    child_links: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Build related-instrument payloads from lineage and matcher hints."""
    related: list[dict[str, object]] = []
    amendment_parent = coerce_text(
        instrument_row.get("amendment_of_debt_instrument_id")
    )
    if amendment_parent and amendment_parent in instrument_lookup:
        related.append(
            related_payload(
                target_id=amendment_parent,
                relationship="Amends",
                instrument_lookup=instrument_lookup,
            )
        )
    split_parent = coerce_text(instrument_row.get("split_of_debt_instrument_id"))
    if split_parent and split_parent in instrument_lookup:
        related.append(
            related_payload(
                target_id=split_parent,
                relationship="Split from",
                instrument_lookup=instrument_lookup,
            )
        )
    related.extend(child_links.get(instrument_id, []))

    for related_id in parse_related_ids(instrument_row.get("possibly_related_json")):
        if related_id == instrument_id or related_id not in instrument_lookup:
            continue
        related.append(
            related_payload(
                target_id=related_id,
                relationship="Possibly related",
                instrument_lookup=instrument_lookup,
            )
        )
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for row in related:
        deduped[(str(row["id"]), str(row["relationship"]))] = row
    return sorted(
        deduped.values(), key=lambda row: (str(row["relationship"]), str(row["id"]))
    )


def build_child_links(
    instrument_lookup: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, str]]]:
    """Return reverse lineage payloads keyed by parent instrument id."""
    child_links: dict[str, list[dict[str, str]]] = defaultdict(list)
    for instrument_id, row in instrument_lookup.items():
        amendment_parent = coerce_text(row.get("amendment_of_debt_instrument_id"))
        if amendment_parent and amendment_parent in instrument_lookup:
            child_links[amendment_parent].append(
                related_payload(
                    target_id=instrument_id,
                    relationship="Amended by",
                    instrument_lookup=instrument_lookup,
                )
            )
        split_parent = coerce_text(row.get("split_of_debt_instrument_id"))
        if split_parent and split_parent in instrument_lookup:
            child_links[split_parent].append(
                related_payload(
                    target_id=instrument_id,
                    relationship="Split into",
                    instrument_lookup=instrument_lookup,
                )
            )
    return child_links


def related_payload(
    *,
    target_id: str,
    relationship: str,
    instrument_lookup: dict[str, dict[str, object]],
) -> dict[str, str | None]:
    """Build one related instrument entry."""
    row = instrument_lookup[target_id]
    return {
        "id": target_id,
        "name": coerce_text(row.get("name")),
        "relationship": relationship,
    }


def parse_related_ids(value: object) -> list[str]:
    """Parse advisory related instrument ids from matcher payload."""
    payload = parse_json(value, default=[])
    if not isinstance(payload, list):
        return []
    return sorted({str(item) for item in payload if str(item).strip()})


def extract_party_names(value: object) -> list[str]:
    """Extract canonical display names from cluster JSON."""
    payload = parse_json(value, default=[])
    if not isinstance(payload, list):
        return []
    names: list[str] = []
    for cluster in payload:
        if not isinstance(cluster, dict):
            continue
        mentions = cluster.get("mentions", [])
        if not isinstance(mentions, list):
            continue
        texts = [
            str(mention.get("text"))
            for mention in mentions
            if isinstance(mention, dict) and coerce_text(mention.get("text"))
        ]
        if texts:
            names.append(max(texts, key=len))
    return sorted(dict.fromkeys(names))


def sync_snapshot(
    *,
    snapshot: dict[str, bytes],
    config: R2PublishConfig,
    r2_client: BaseClient,
) -> R2PublishResult:
    """Synchronize snapshot objects into R2 with change detection."""
    existing_keys = list_existing_keys(config=config, r2_client=r2_client)
    desired_keys = {prefix_key(config.object_prefix, key) for key in snapshot}
    uploaded_count = 0
    skipped_count = 0
    deleted_count = 0

    for relative_key in ordered_snapshot_keys(snapshot):
        absolute_key = prefix_key(config.object_prefix, relative_key)
        body = snapshot[relative_key]
        current = read_existing_body(
            bucket_name=config.bucket_name,
            key=absolute_key,
            r2_client=r2_client,
        )
        if current == body:
            skipped_count += 1
            continue
        r2_client.put_object(Bucket=config.bucket_name, Key=absolute_key, Body=body)
        uploaded_count += 1

    for stale_key in sorted(existing_keys - desired_keys):
        r2_client.delete_object(Bucket=config.bucket_name, Key=stale_key)
        deleted_count += 1

    return R2PublishResult(
        object_count=len(snapshot),
        uploaded_count=uploaded_count,
        skipped_count=skipped_count,
        deleted_count=deleted_count,
    )


def ordered_snapshot_keys(snapshot: dict[str, bytes]) -> list[str]:
    """Return snapshot keys in dashboard-safe publish order."""
    instrument_keys = sorted(
        key for key in snapshot if key.startswith(f"{INSTRUMENTS_PREFIX}/")
    )
    company_keys = sorted(
        key for key in snapshot if key.startswith(f"{COMPANIES_PREFIX}/")
    )
    index_keys = [INDEX_KEY] if INDEX_KEY in snapshot else []
    return [*instrument_keys, *company_keys, *index_keys]


def list_existing_keys(
    *,
    config: R2PublishConfig,
    r2_client: BaseClient,
) -> set[str]:
    """List keys currently visible under the configured prefix."""
    paginator = r2_client.get_paginator("list_objects_v2")
    keys: set[str] = set()
    prefix = f"{config.object_prefix.rstrip('/')}/"
    for page in paginator.paginate(Bucket=config.bucket_name, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def read_existing_body(
    *,
    bucket_name: str,
    key: str,
    r2_client: BaseClient,
) -> bytes | None:
    """Read an existing object body or return None if the object is missing."""
    try:
        response = r2_client.get_object(Bucket=bucket_name, Key=key)
    except Exception as exc:  # noqa: BLE001
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return response["Body"].read()


def prefix_key(prefix: str, key: str) -> str:
    """Return one object key beneath the configured prefix."""
    normalized_prefix = prefix.strip("/")
    normalized_key = key.lstrip("/")
    return (
        f"{normalized_prefix}/{normalized_key}" if normalized_prefix else normalized_key
    )


def serialize_json(payload: dict[str, object]) -> bytes:
    """Serialize JSON deterministically for change detection."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parse_json(
    value: object,
    *,
    default: dict[str, object] | list[object],
) -> dict[str, object] | list[object]:
    """Parse a JSON-encoded string or return the provided default."""
    if isinstance(value, dict | list):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def coerce_text(value: object) -> str | None:
    """Normalize one optional text value."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
