"""Acquire Form 6-K filings directly from EDGAR.

No longer the default. The scraper now carries 6-K over its whole history, so
:mod:`cdt.sixk.scraper` reads them the way the 8-K path reads its forms, and
this module is the opt-in source (``ingest-sixk --source edgar``) for the cases
the scraper cannot serve: a filing it has not scraped yet, or a range predating
its coverage.

What it produces is deliberately indistinguishable downstream from what the
scraper path produces: each filing's submission is mirrored under CDT's own
prefix and the document row points at the mirror through ``resource_uri``,
exactly as an 8-K row points at the scraper's copy. Bodies stay out of the
parquet partitions, so a partition read costs the same as it does for 8-K, and
switching sources needs no migration — ingest dedups on accession and both
sources write one mirror, so a filing acquired here is neither re-fetched nor
re-ingested by the scraper path.

SEC's fair-access policy requires a declared contact in the User-Agent; without
one sec.gov answers 403 with an "Undeclared Automated Tool" page rather than the
file. That is a configuration error, not a transient one, so it aborts the run
(see :class:`UndeclaredUserAgentError`) instead of being retried per filing or
persisted as a filing body.
"""

from __future__ import annotations

import gzip
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, Self

import pandas as pd

from cdt import settings
from cdt.ingest import (
    SIXK_FORM_TYPES,
    DocumentCandidate,
    DocumentSource,
    IngestConfig,
    IngestFailureType,
    IngestRunResult,
    default_output_root,
    normalize_accession_number,
    run_ingest_pipeline,
)
from cdt.shared import FailureRegistry, get_logger
from cdt.sixk.mirror import mirror_path, mirror_root
from cdt.storage import artifact_exists, write_bytes_artifact

LOGGER = get_logger(__name__)

__all__ = [
    "EdgarDocumentSource",
    "SIXK_FORM_TYPES",
    "SecFetcher",
    "SecNotFoundError",
    "UndeclaredUserAgentError",
    "acquire_sixk_documents",
    "daily_index_url",
    "mirror_path",
    "mirror_root",
    "parse_form_index",
    "quarterly_index_url",
    "read_index_file",
    "submission_url",
]

# ~4.5 requests/second, inside SEC's published 10/s ceiling for all traffic from
# one source. The ceiling is per requester, not per process, so leave headroom.
REQUEST_INTERVAL_SECONDS = 0.22
DEFAULT_MAX_ATTEMPTS = 4
HTTP_OK = 200
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
# The title SEC serves with the 403 it returns for an undeclared User-Agent.
# Matched because SEC uses 403 for both that and rate limiting, and only one of
# the two is worth retrying.
UNDECLARED_TOOL_MARKER = "Undeclared Automated Tool"
# form.idx columns are not aligned, so anchor from the right, where CIK, date
# and path all have unambiguous shapes. The date is matched in both spellings
# EDGAR uses: the quarterly full-index writes 2026-04-29 and the daily index
# writes 20260908, and a regex accepting only the first silently matches nothing
# in a daily index — 147 of 147 6-K rows dropped, with no error.
FILING_RE = re.compile(
    r"^(?P<form>\S+(?:\s\S+)*?)\s{2,}"
    r"(?P<company_name>.+?)\s{2,}"
    r"(?P<cik>\d+)\s+"
    r"(?P<filing_date>\d{4}-\d{2}-\d{2}|\d{8})\s+"
    r"(?P<file_name>\S+)\s*$"
)
COMPACT_DATE_LENGTH = 8
MONTHS_PER_QUARTER = 3


class UndeclaredUserAgentError(RuntimeError):
    """SEC rejected the request because the User-Agent declares no contact.

    Fatal for the whole run rather than for one filing: every subsequent request
    would be refused the same way, and retrying would be exactly the abusive
    traffic pattern the policy exists to stop.
    """


class SecNotFoundError(RuntimeError):
    """EDGAR has no object at that URL (a withdrawn filing, or a stale index)."""


class SecTransport(Protocol):
    """One HTTP GET against sec.gov, reduced to what the fetcher needs."""

    def __call__(self: Self, url: str, headers: dict[str, str]) -> tuple[int, bytes]:
        """Return the response status and body, without raising for status."""


@dataclass(frozen=True)
class IndexRow:
    """One filing row from an EDGAR ``form.idx``."""

    form: str
    company_name: str
    cik: str
    filing_date: str
    file_name: str
    accession_number: str


def urllib_transport(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    """GET a URL, returning error responses instead of raising for status.

    Error bodies matter here: SEC's 403 carries the reason in the page, and the
    retry decision depends on which 403 it is.
    """
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - https only, built below
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


class SecFetcher:
    """Throttled, retrying GETs against sec.gov with a declared contact."""

    def __init__(
        self: Self,
        *,
        user_agent: str | None = None,
        interval_seconds: float = REQUEST_INTERVAL_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        transport: SecTransport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the fetcher; the contact is resolved on first use."""
        self._user_agent = user_agent
        self._interval_seconds = interval_seconds
        self._max_attempts = max_attempts
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_request_at = 0.0

    @property
    def user_agent(self: Self) -> str:
        """Return the declared contact, or explain that it is required.

        Read from settings on use rather than bound at construction, so an
        override applied after import is honoured — the same pattern the
        extractor and stage-2 triage use for their model ids.
        """
        user_agent = self._user_agent or settings.SEC_USER_AGENT
        if not user_agent.strip():
            msg = (
                "SEC_USER_AGENT is required to fetch from EDGAR. SEC's "
                "fair-access policy wants an organization and a contact "
                'address, e.g. SEC_USER_AGENT="Example University '
                'contact@example.edu". Without it sec.gov returns an '
                '"Undeclared Automated Tool" page instead of the file.'
            )
            raise UndeclaredUserAgentError(msg)
        return user_agent

    def get(self: Self, url: str) -> bytes:
        """Fetch one URL, throttled and retried; raise on permanent failures."""
        headers = {"User-Agent": self.user_agent}
        last_status = 0
        for attempt in range(self._max_attempts):
            self._throttle()
            status, body = self._transport(url, headers)
            if status == HTTP_OK:
                return body
            if status == HTTP_NOT_FOUND:
                raise SecNotFoundError(url)
            if status == HTTP_FORBIDDEN and UNDECLARED_TOOL_MARKER in body.decode(
                "utf-8", errors="replace"
            ):
                msg = (
                    f"sec.gov refused {url} as an undeclared automated tool. Set "
                    "SEC_USER_AGENT to an organization and contact address; "
                    "retrying this would be the traffic pattern the policy "
                    "exists to stop."
                )
                raise UndeclaredUserAgentError(msg)
            last_status = status
            if not self._is_retryable(status):
                break
            self._sleep(2**attempt)
        msg = f"sec.gov returned {last_status} for {url}"
        raise RuntimeError(msg)

    @staticmethod
    def _is_retryable(status: int) -> bool:
        # A 403 reaching here is the rate-limit kind; the undeclared-tool kind
        # already raised.
        return status in {HTTP_FORBIDDEN, HTTP_TOO_MANY_REQUESTS} or (
            status >= HTTP_SERVER_ERROR
        )

    def _throttle(self: Self) -> None:
        now = self._monotonic()
        if now < self._next_request_at:
            self._sleep(self._next_request_at - now)
        self._next_request_at = self._monotonic() + self._interval_seconds


def daily_index_url(day: date) -> str:
    """Return the EDGAR daily form index URL for one filing day."""
    return (
        "https://www.sec.gov/Archives/edgar/daily-index/"
        f"{day.year}/QTR{quarter_of(day)}/form.{day.strftime('%Y%m%d')}.idx"
    )


def quarterly_index_url(day: date) -> str:
    """Return the EDGAR full-index form index URL for one day's quarter.

    Not used by default — a day's filings come from that day's index. It is here
    for wide backfills, where one quarterly index replaces ~63 daily ones: fetch
    it out of band and pass the file to :func:`acquire_sixk_documents` as
    ``index_file``.
    """
    return (
        "https://www.sec.gov/Archives/edgar/full-index/"
        f"{day.year}/QTR{quarter_of(day)}/form.idx"
    )


def quarter_of(day: date) -> int:
    """Return the 1-based calendar quarter containing a date."""
    return (day.month - 1) // MONTHS_PER_QUARTER + 1


def submission_url(cik: str, accession_number: str) -> str:
    """Return the complete-submission text file URL for one filing.

    ``accession_number`` is the dashed form from the index; the directory
    segment is the same digits without dashes.
    """
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
        f"{accession_number.replace('-', '')}/{accession_number}.txt"
    )


def parse_form_index(
    raw: str, *, form_types: Sequence[str] = SIXK_FORM_TYPES
) -> list[IndexRow]:
    """Return the index rows whose form is one of ``form_types``.

    Filing dates are normalized to ISO, whichever spelling the index used.

        >>> rows = parse_form_index(
        ...     "6-K         Barclays PLC        312069      2026-04-29  "
        ...     "edgar/data/312069/0001654954-26-004070.txt"
        ... )
        >>> rows[0].accession_number, rows[0].cik
        ('0001654954-26-004070', '312069')
        >>> parse_form_index(
        ...     "6-K         AIR Global PLC      2097725     20260908    "
        ...     "edgar/data/2097725/0001193125-26-384297.txt"
        ... )[0].filing_date
        '2026-09-08'
    """
    wanted = set(form_types)
    # form.idx rows start with the form type, so this skips the banner and the
    # ~99% of a quarterly index that is some other form without running the
    # regex over it.
    prefixes = tuple(wanted)
    rows: list[IndexRow] = []
    for line in raw.splitlines():
        if not line.startswith(prefixes):
            continue
        match = FILING_RE.match(line)
        if match is None or match.group("form") not in wanted:
            continue
        fields = match.groupdict()
        rows.append(
            IndexRow(
                form=fields["form"],
                company_name=fields["company_name"].strip(),
                cik=fields["cik"],
                filing_date=_iso_filing_date(fields["filing_date"]),
                file_name=fields["file_name"],
                accession_number=Path(fields["file_name"]).stem,
            )
        )
    return rows


def _iso_filing_date(value: str) -> str:
    """Return one index date as ISO, accepting the compact daily spelling."""
    if len(value) == COMPACT_DATE_LENGTH and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def read_index_file(path: Path) -> str:
    """Read a form index from a plain ``.idx`` or a zip containing one."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(
                (item for item in archive.namelist() if item.endswith("form.idx")),
                None,
            )
            if name is None:
                msg = f"{path} contains no form.idx"
                raise ValueError(msg)
            return archive.read(name).decode("latin-1")
    # latin-1 never fails and EDGAR indexes are not UTF-8: company names carry
    # stray high bytes, and a decode error here would drop a whole day.
    return path.read_bytes().decode("latin-1")


@dataclass
class EdgarDocumentSource:
    """Document candidates fetched from EDGAR and mirrored under CDT's root.

    Acquisition happens during iteration: each yielded candidate has already
    been mirrored, so a candidate exists only if its body does. Filings whose
    mirror is already present are yielded without a request, which makes the
    mirror the resume ledger — re-running a range costs one existence check per
    indexed filing and no EDGAR traffic.
    """

    config: IngestConfig
    failure_registry: FailureRegistry | None = None
    ciks: set[str] | None = None
    fetcher: SecFetcher = field(default_factory=SecFetcher)
    index_file: Path | None = None
    _failures: int = field(default=0, init=False)

    @property
    def failures(self: Self) -> int:
        """Return the number of indexed filings that could not be acquired."""
        return self._failures

    def __iter__(self: Self) -> Iterator[DocumentCandidate]:
        """Fetch, mirror and yield every matching filing in the date range."""
        artifact_root = self.config.output_root or default_output_root(
            self.config.data_dir
        )
        indexed = 0
        mirrored = 0
        reused = 0
        seen: set[str] = set()
        for row in self._index_rows():
            # An accession can appear in more than one index — a re-published
            # daily index, or a quarterly one overlapping the dailies. Ingest
            # dedups too, but only once a candidate exists, and building one
            # here means a storage round-trip per duplicate.
            if row.accession_number in seen:
                continue
            seen.add(row.accession_number)
            indexed += 1
            # Namespaced like the scraper path's (bucket, key): the pair names
            # the source object, so one failures.json can hold both genres.
            key = ("sec.gov", submission_url(row.cik, row.accession_number))
            if not self.config.force and self._is_registered_failure(key):
                LOGGER.info(
                    "Skipping known EDGAR failure: cik=%s accession=%s",
                    row.cik,
                    row.accession_number,
                )
                continue
            target = mirror_path(
                artifact_root,
                filing_date=row.filing_date,
                accession_number=row.accession_number,
            )
            if artifact_exists(target) and not self.config.force:
                reused += 1
            else:
                if not self._mirror(row, target, key):
                    continue
                mirrored += 1
            yield self._candidate(row, target)
        LOGGER.info(
            "EDGAR acquisition complete: indexed=%s fetched=%s already_mirrored=%s failures=%s",
            indexed,
            mirrored,
            reused,
            self._failures,
        )

    def _mirror(self: Self, row: IndexRow, target: str, key: tuple[str, str]) -> bool:
        """Fetch one submission into the mirror; False when it could not be."""
        url = submission_url(row.cik, row.accession_number)
        try:
            body = self.fetcher.get(url)
        except SecNotFoundError:
            # Permanent: the filing is gone or the index row is stale, and no
            # number of retries produces it.
            LOGGER.warning("EDGAR has no submission at %s", url)
            self._record(key, IngestFailureType.DOCUMENT_NOT_FOUND)
            return False
        except UndeclaredUserAgentError:
            # Not this filing's problem, and every later request would fail the
            # same way.
            raise
        except Exception:
            LOGGER.exception("Failed to fetch EDGAR submission: %s", url)
            self._record(key, IngestFailureType.DOCUMENT_DOWNLOAD_FAILED)
            return False
        # EDGAR's bytes verbatim: a submission's encoding is not reliably
        # anything, and re-encoding here would change the text the
        # extractor later quotes as evidence.
        write_bytes_artifact(target, gzip.compress(body))
        if self.config.force and self.failure_registry is not None:
            # The registered failure did not reproduce, so stop skipping it.
            self.failure_registry.discard(key)
        return True

    def _candidate(self: Self, row: IndexRow, target: str) -> DocumentCandidate:
        return DocumentCandidate(
            accession_number=normalize_accession_number(row.accession_number),
            # Zero-stripped to match the scraper path, which stores the CIK the
            # same way; the matcher shards on this value.
            cik=row.cik.lstrip("0"),
            company_name=row.company_name,
            url=submission_url(row.cik, row.accession_number),
            resource_uri=target,
            date=row.filing_date,
            form_type=row.form,
            source=DocumentSource.EDGAR,
        )

    def _index_rows(self: Self) -> Iterator[IndexRow]:
        """Yield matching index rows from the indexes this run reads.

        The date range filters a *supplied* index, which covers a whole quarter,
        and deliberately does not filter the daily ones. A daily index is that
        day's dissemination feed, and it lists filings dated earlier — a filing
        dated the 4th shows up in the 8th's feed. Dropping those would lose them
        for good: the run for the 4th already happened, and its own index did
        not list them yet (the shape of #90). Each row is written to the
        partition for its own filing date, so a late one lands where it belongs
        rather than where it was found.
        """
        wanted_ciks = (
            None if self.ciks is None else {str(cik).lstrip("0") for cik in self.ciks}
        )
        filter_dates = self.index_file is not None
        for raw in self._raw_indexes():
            for row in parse_form_index(raw, form_types=self.config.form_types):
                if filter_dates and not (
                    self.config.start_date
                    <= date.fromisoformat(row.filing_date)
                    <= self.config.end_date
                ):
                    continue
                if wanted_ciks is not None and row.cik.lstrip("0") not in wanted_ciks:
                    continue
                yield row

    def _raw_indexes(self: Self) -> Iterator[str]:
        """Yield the index text to scan: one supplied file, or a day at a time."""
        if self.index_file is not None:
            yield read_index_file(self.index_file)
            return
        day = self.config.start_date
        while day <= self.config.end_date:
            try:
                yield self.fetcher.get(daily_index_url(day)).decode("latin-1")
            except SecNotFoundError:
                # Weekends, holidays, and today before EDGAR publishes: a day
                # with no index is a day with no filings, not an error.
                LOGGER.info("No EDGAR daily index for %s", day.isoformat())
            except UndeclaredUserAgentError:
                raise
            except Exception as error:
                # Deliberately fatal. Skipping the day would leave a hole no
                # later run notices: the range has moved on, the partitions for
                # that date look complete, and the filings are simply absent
                # (#90). A failed run is re-runnable; a silent gap is not.
                msg = (
                    f"Could not read the EDGAR daily index for {day.isoformat()}: "
                    f"{error}. Re-run the range; skipping the day would leave a "
                    "coverage gap nothing downstream can detect."
                )
                raise RuntimeError(msg) from error
            day += timedelta(days=1)

    def _is_registered_failure(self: Self, key: tuple[str, str]) -> bool:
        return self.failure_registry is not None and key in self.failure_registry

    def _record(self: Self, key: tuple[str, str], failure: IngestFailureType) -> None:
        self._failures += 1
        if self.failure_registry is not None:
            # Retryable types are dropped by the registry's classifier, so a
            # throttled fetch is retried next run while a missing filing is not.
            self.failure_registry.add(key, failure)


def acquire_sixk_documents(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    fetcher: SecFetcher | None = None,
    index_file: Path | None = None,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Acquire 6-K filings from EDGAR into the config's documents dataset.

    Shares every stage of ingest except where filings come from, so the run
    manifest, accession dedup and partition layout are the scraper path's.
    """
    if config.download:
        msg = (
            "download=True is not supported for EDGAR acquisition: the body is "
            "mirrored under the artifact root and resolved from resource_uri by "
            "the stage that reads it, exactly as an 8-K body is read from the "
            "scraper's copy."
        )
        raise ValueError(msg)
    resolved_fetcher = fetcher or SecFetcher()
    return run_ingest_pipeline(
        config,
        ciks=ciks,
        candidate_source=lambda registry: EdgarDocumentSource(
            config=config,
            failure_registry=registry,
            ciks=ciks,
            fetcher=resolved_fetcher,
            index_file=index_file,
        ),
    )
