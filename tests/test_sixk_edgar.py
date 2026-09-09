"""Tests for direct-EDGAR acquisition of Form 6-K filings."""

from __future__ import annotations

import gzip
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Self

import pytest

from cdt import settings
from cdt.ingest import DOCUMENT_COLUMNS, SIXK_DOCUMENT_DATASET_NAME, IngestConfig
from cdt.ingest import documents_root as ingest_documents_root
from cdt.sixk.edgar import (
    SIXK_FORM_TYPES,
    EdgarDocumentSource,
    SecFetcher,
    SecNotFoundError,
    UndeclaredUserAgentError,
    acquire_sixk_documents,
    daily_index_url,
    mirror_path,
    parse_form_index,
    quarterly_index_url,
    read_index_file,
    submission_url,
)
from cdt.storage import read_dataset, read_json_artifact

USER_AGENT = "Example University contact@example.edu"
UNDECLARED_BODY = (
    b"<html><h1>Your Request Originates from an Undeclared Automated Tool</h1>"
    b"<p>Please declare your traffic.</p></html>"
)
THROTTLED_BODY = b"<html><h1>Request Rate Threshold Exceeded</h1></html>"
# Real form.idx spacing: columns are not aligned to a fixed width, and the
# banner rows do not start with a form type.
INDEX_TEXT = """Form Type   Company Name                              CIK        Date Filed  File Name
---------------------------------------------------------------------------------------
6-K         BARCLAYS PLC                              312069     2026-04-29  edgar/data/312069/0001654954-26-004070.txt
6-K/A       Vale S.A.                          1292814    2026-04-30  edgar/data/1292814/0001292814-26-002379.txt
8-K         SOME DOMESTIC CO                          320193     2026-04-29  edgar/data/320193/0000320193-26-000001.txt
"""
INDEX_BANNER = """Form Type   Company Name                              CIK        Date Filed  File Name
---------------------------------------------------------------------------------------
"""
BARCLAYS_ROW = (
    "6-K         BARCLAYS PLC                              312069     2026-04-29  "
    "edgar/data/312069/0001654954-26-004070.txt\n"
)
VALE_ROW = (
    "6-K/A       Vale S.A.                          1292814    2026-04-30  "
    "edgar/data/1292814/0001292814-26-002379.txt\n"
)
DAY_ONE_INDEX = (
    INDEX_BANNER
    + BARCLAYS_ROW
    + (
        "8-K         SOME DOMESTIC CO                          320193     2026-04-29  "
        "edgar/data/320193/0000320193-26-000001.txt\n"
    )
)
DAY_TWO_INDEX = INDEX_BANNER + VALE_ROW
# The daily index as EDGAR actually serves it: a four-line preamble, the header
# split across two lines, compact dates, and rows dated before the index day.
DAILY_INDEX_TEXT = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    Sep 8, 2026
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/

Form Type   Company Name                                                  CIK
      Date Filed  File Name
---------------------------------------------------------------------------------------
6-K              AIR Global PLC                                                2097725     20260908    edgar/data/2097725/0001193125-26-384297.txt
6-K              LATE FILER PLC                                                2097726     20260904    edgar/data/2097726/0001193125-26-384298.txt
8-K              SOME DOMESTIC CO                                              320193      20260908    edgar/data/320193/0000320193-26-000002.txt
"""
BARCLAYS_ACCESSION = "0001654954-26-004070"
VALE_ACCESSION = "0001292814-26-002379"
EXPECTED_SIXK_ROWS = 2


class FakeTransport:
    """A transport serving canned responses per URL."""

    def __init__(self: Self, responses: dict[str, list[tuple[int, bytes]]]) -> None:
        """Initialize with a queue of responses for each URL."""
        self.responses = {url: list(queue) for url, queue in responses.items()}
        self.calls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self: Self, url: str, headers: dict[str, str]) -> tuple[int, bytes]:
        """Return the next canned response for the URL."""
        self.calls.append(url)
        self.headers.append(headers)
        queue = self.responses.get(url)
        if not queue:
            return 404, b"not found"
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _fetcher(transport: FakeTransport, **kwargs: object) -> SecFetcher:
    """Build a fetcher that neither sleeps nor waits between requests."""
    return SecFetcher(
        user_agent=USER_AGENT,
        transport=transport,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
        **kwargs,  # type: ignore[arg-type]
    )


def _config(tmp_path: Path, **overrides: object) -> IngestConfig:
    defaults: dict[str, object] = {
        "mode": "historical",
        "bucket": "unused-for-edgar",
        "cik_file": Path(),
        "start_date": date(2026, 4, 29),
        "end_date": date(2026, 4, 30),
        "data_dir": tmp_path,
        "output_root": str(tmp_path),
        "form_types": SIXK_FORM_TYPES,
        "dataset_name": SIXK_DOCUMENT_DATASET_NAME,
    }
    defaults.update(overrides)
    return IngestConfig(**defaults)  # type: ignore[arg-type]


def _index_and_submissions(
    *, submissions: dict[str, tuple[int, bytes]] | None = None
) -> dict[str, list[tuple[int, bytes]]]:
    """Canned responses for both filing days plus each submission."""
    responses: dict[str, list[tuple[int, bytes]]] = {
        daily_index_url(date(2026, 4, 29)): [(200, DAY_ONE_INDEX.encode())],
        daily_index_url(date(2026, 4, 30)): [(200, DAY_TWO_INDEX.encode())],
    }
    bodies = submissions or {
        BARCLAYS_ACCESSION: (200, b"BARCLAYS 6-K body"),
        VALE_ACCESSION: (200, b"VALE 6-K/A body"),
    }
    for accession, response in bodies.items():
        cik = "312069" if accession == BARCLAYS_ACCESSION else "1292814"
        responses[submission_url(cik, accession)] = [response]
    return responses


def test_parse_form_index_selects_requested_forms_from_unaligned_columns() -> None:
    """Rows are read by anchoring from the right, and other forms are dropped."""
    rows = parse_form_index(INDEX_TEXT)

    assert [row.form for row in rows] == ["6-K", "6-K/A"]
    assert [row.accession_number for row in rows] == [
        BARCLAYS_ACCESSION,
        VALE_ACCESSION,
    ]
    assert [row.cik for row in rows] == ["312069", "1292814"]
    assert [row.company_name for row in rows] == ["BARCLAYS PLC", "Vale S.A."]
    assert [row.filing_date for row in rows] == ["2026-04-29", "2026-04-30"]


def test_parse_form_index_reads_the_daily_indexs_compact_dates() -> None:
    """EDGAR spells the date two ways, and the daily one has no dashes.

    A regex accepting only the quarterly spelling matches nothing at all in a
    daily index — every 6-K row dropped, no error raised.
    """
    rows = parse_form_index(DAILY_INDEX_TEXT)

    assert [row.filing_date for row in rows] == ["2026-09-08", "2026-09-04"]
    assert [row.cik for row in rows] == ["2097725", "2097726"]
    assert [row.company_name for row in rows] == ["AIR Global PLC", "LATE FILER PLC"]


def test_parse_form_index_does_not_confuse_an_amendment_for_its_base_form() -> None:
    """Asking for 6-K alone must not pick up 6-K/A."""
    rows = parse_form_index(INDEX_TEXT, form_types=("6-K",))

    assert [row.form for row in rows] == ["6-K"]


def test_read_index_file_reads_plain_and_zipped_indexes(tmp_path: Path) -> None:
    """Backfills come as a zip; a daily index comes as plain text."""
    plain = tmp_path / "form.idx"
    plain.write_bytes(INDEX_TEXT.encode("latin-1"))
    archive = tmp_path / "form-2026-QTR2.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("form.idx", INDEX_TEXT)

    assert parse_form_index(read_index_file(plain)) == parse_form_index(
        read_index_file(archive)
    )


def test_index_urls_carry_the_filing_day_and_its_quarter() -> None:
    """Both index URL shapes are built from one date."""
    assert daily_index_url(date(2026, 4, 29)) == (
        "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR2/form.20260429.idx"
    )
    assert quarterly_index_url(date(2026, 4, 29)) == (
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR2/form.idx"
    )


def test_submission_url_uses_the_dashless_directory_and_dashed_file() -> None:
    """EDGAR's archive path spells the accession both ways."""
    assert submission_url("0000312069", BARCLAYS_ACCESSION) == (
        "https://www.sec.gov/Archives/edgar/data/312069/000165495426004070/"
        "0001654954-26-004070.txt"
    )


def test_fetcher_requires_a_declared_contact_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset SEC_USER_AGENT fails before a request leaves the process."""
    monkeypatch.setattr(settings, "SEC_USER_AGENT", "")
    transport = FakeTransport({})
    fetcher = SecFetcher(transport=transport, sleep=lambda _seconds: None)

    with pytest.raises(UndeclaredUserAgentError, match="SEC_USER_AGENT is required"):
        fetcher.get("https://www.sec.gov/Archives/edgar/data/1/2/3.txt")

    assert transport.calls == []


def test_fetcher_sends_the_declared_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured contact rides on every request, read at call time."""
    monkeypatch.setattr(settings, "SEC_USER_AGENT", USER_AGENT)
    url = "https://www.sec.gov/Archives/edgar/data/1/2/3.txt"
    transport = FakeTransport({url: [(200, b"body")]})
    fetcher = SecFetcher(transport=transport, sleep=lambda _seconds: None)

    assert fetcher.get(url) == b"body"
    assert transport.headers == [{"User-Agent": USER_AGENT}]


def test_fetcher_does_not_retry_an_undeclared_tool_refusal() -> None:
    """SEC uses 403 for two things; only the throttle kind is worth retrying.

    Retrying a refusal caused by the User-Agent is both futile and precisely the
    traffic pattern the fair-access policy exists to stop, so it raises on the
    first response instead of backing off four times per filing.
    """
    url = "https://www.sec.gov/Archives/edgar/data/1/2/3.txt"
    transport = FakeTransport({url: [(403, UNDECLARED_BODY)]})

    with pytest.raises(UndeclaredUserAgentError, match="undeclared automated tool"):
        _fetcher(transport).get(url)

    assert transport.calls == [url]


def test_fetcher_retries_a_throttle_then_succeeds() -> None:
    """A rate-limit 403 is transient and is retried."""
    url = "https://www.sec.gov/Archives/edgar/data/1/2/3.txt"
    transport = FakeTransport(
        {url: [(403, THROTTLED_BODY), (503, b"unavailable"), (200, b"body")]}
    )

    assert _fetcher(transport).get(url) == b"body"
    assert transport.calls == [url, url, url]


def test_fetcher_raises_not_found_without_retrying() -> None:
    """A missing object is permanent; the caller decides what that means."""
    url = "https://www.sec.gov/Archives/edgar/data/1/2/3.txt"
    transport = FakeTransport({url: [(404, b"not found")]})

    with pytest.raises(SecNotFoundError):
        _fetcher(transport).get(url)

    assert transport.calls == [url]


def test_fetcher_spaces_requests_by_the_configured_interval() -> None:
    """Requests are throttled to stay inside SEC's rate ceiling."""
    url = "https://www.sec.gov/Archives/edgar/data/1/2/3.txt"
    transport = FakeTransport({url: [(200, b"body")]})
    slept: list[float] = []
    clock = iter([0.0, 0.0, 0.05, 0.05])
    fetcher = SecFetcher(
        user_agent=USER_AGENT,
        interval_seconds=0.22,
        transport=transport,
        sleep=slept.append,
        monotonic=lambda: next(clock),
    )

    fetcher.get(url)
    fetcher.get(url)

    # The second call starts 0.05s after the first finished, so it waits out the
    # remaining 0.17s of the interval.
    assert slept == [pytest.approx(0.17)]


def test_acquire_writes_six_k_documents_pointing_at_mirrored_bodies(
    tmp_path: Path,
) -> None:
    """A run indexes EDGAR, mirrors each body, and writes 6-K document rows."""
    transport = FakeTransport(_index_and_submissions())

    table, result = acquire_sixk_documents(
        _config(tmp_path),
        fetcher=_fetcher(transport),
    )

    assert table["accession_number"].to_list() == [
        "000165495426004070",
        "000129281426002379",
    ]
    assert table["form_type"].to_list() == ["6-K", "6-K/A"]
    assert table["source"].to_list() == ["edgar", "edgar"]
    assert table["cik"].to_list() == ["312069", "1292814"]
    # Bodies stay out of the partition: the row points at the mirror, exactly as
    # an 8-K row points at the scraper's copy.
    assert table["text"].to_list() == ["", ""]
    assert result.failures == 0
    assert result.dataset_name == SIXK_DOCUMENT_DATASET_NAME

    mirror = mirror_path(
        str(tmp_path),
        filing_date="2026-04-29",
        accession_number=BARCLAYS_ACCESSION,
    )
    assert table.loc[0, "resource_uri"] == mirror
    assert gzip.decompress(Path(mirror).read_bytes()) == b"BARCLAYS 6-K body"
    assert (
        len(
            read_dataset(
                ingest_documents_root(
                    str(tmp_path), dataset_name=SIXK_DOCUMENT_DATASET_NAME
                ),
                columns=DOCUMENT_COLUMNS,
            )
        )
        == EXPECTED_SIXK_ROWS
    )


def test_acquire_reuses_the_mirror_instead_of_refetching(tmp_path: Path) -> None:
    """The mirror is the resume ledger, so a re-run makes no body requests.

    Iteration acquires bodies, and ingest's accession dedup happens after the
    candidate exists — so without this check a repeated range would re-download
    every filing it already has.
    """
    transport = FakeTransport(_index_and_submissions())
    acquire_sixk_documents(_config(tmp_path), fetcher=_fetcher(transport))
    first_pass_calls = list(transport.calls)

    table, _ = acquire_sixk_documents(_config(tmp_path), fetcher=_fetcher(transport))

    submission_calls = [
        url for url in transport.calls[len(first_pass_calls) :] if "/data/" in url
    ]
    assert submission_calls == []
    assert len(table) == EXPECTED_SIXK_ROWS


def test_acquire_records_a_missing_filing_as_permanent_and_skips_it_next_run(
    tmp_path: Path,
) -> None:
    """A 404 body is a withdrawn filing: counted, registered, not retried."""
    transport = FakeTransport(
        _index_and_submissions(
            submissions={
                BARCLAYS_ACCESSION: (200, b"BARCLAYS 6-K body"),
                VALE_ACCESSION: (404, b"not found"),
            }
        )
    )
    config = _config(tmp_path)

    table, result = acquire_sixk_documents(config, fetcher=_fetcher(transport))

    assert table["accession_number"].to_list() == ["000165495426004070"]
    assert result.failures == 1
    registry = read_json_artifact(result.failure_file)
    assert json.dumps(registry).count(VALE_ACCESSION) >= 1

    first_pass_calls = len(transport.calls)
    acquire_sixk_documents(config, fetcher=_fetcher(transport))
    assert [
        url
        for url in transport.calls[first_pass_calls:]
        if VALE_ACCESSION.replace("-", "") in url
    ] == []


def test_acquire_leaves_a_transient_body_failure_retryable(tmp_path: Path) -> None:
    """Exhausted retries are counted but not registered as permanent."""
    vale_url = submission_url("1292814", VALE_ACCESSION)
    responses = _index_and_submissions(
        submissions={BARCLAYS_ACCESSION: (200, b"BARCLAYS 6-K body")}
    )
    responses[vale_url] = [(500, b"boom")]
    transport = FakeTransport(responses)

    table, result = acquire_sixk_documents(
        _config(tmp_path), fetcher=_fetcher(transport, max_attempts=2)
    )

    assert table["accession_number"].to_list() == ["000165495426004070"]
    assert result.failures == 1
    # Retryable failures are not persisted, so the next run tries again.
    registry = read_json_artifact(result.failure_file)
    assert VALE_ACCESSION not in json.dumps(registry)


def test_acquire_filters_a_supplied_index_by_date_range_and_ciks(
    tmp_path: Path,
) -> None:
    """A quarterly index covers more than the run does; the range decides."""
    index_file = tmp_path / "form.idx"
    index_file.write_bytes(INDEX_TEXT.encode("latin-1"))
    transport = FakeTransport(_index_and_submissions())

    table, _ = acquire_sixk_documents(
        _config(tmp_path, end_date=date(2026, 4, 29)),
        fetcher=_fetcher(transport),
        index_file=index_file,
        ciks={"0000312069"},
    )

    assert table["accession_number"].to_list() == ["000165495426004070"]
    # A supplied index replaces the daily fetches entirely.
    assert [url for url in transport.calls if "daily-index" in url] == []


def test_source_treats_a_missing_daily_index_as_a_day_without_filings(
    tmp_path: Path,
) -> None:
    """Weekends and holidays have no index; that is not a failure."""
    responses = _index_and_submissions()
    del responses[daily_index_url(date(2026, 4, 30))]
    transport = FakeTransport(responses)

    table, result = acquire_sixk_documents(
        _config(tmp_path), fetcher=_fetcher(transport)
    )

    # The 29th's filing arrives; the 30th contributes nothing and is not a
    # failure. A day whose index is *unreadable* is a different case entirely
    # (see the coverage-gap test below).
    assert table["accession_number"].to_list() == ["000165495426004070"]
    assert result.failures == 0


def test_source_refuses_to_skip_a_day_whose_index_cannot_be_read(
    tmp_path: Path,
) -> None:
    """An unreadable index aborts: a skipped day is an undetectable gap.

    The range moves on, the partitions for that date look complete, and the
    filings are simply absent — the shape of #90. A failed run is re-runnable.
    """
    responses = _index_and_submissions()
    responses[daily_index_url(date(2026, 4, 29))] = [(503, b"unavailable")]
    transport = FakeTransport(responses)

    with pytest.raises(RuntimeError, match="coverage gap"):
        acquire_sixk_documents(
            _config(tmp_path), fetcher=_fetcher(transport, max_attempts=2)
        )


def test_acquire_rejects_inlining_bodies_into_the_partitions(tmp_path: Path) -> None:
    """download=True belongs to the scraper path, where bodies live in S3."""
    with pytest.raises(ValueError, match="download=True is not supported"):
        acquire_sixk_documents(_config(tmp_path, download=True))


def test_source_keeps_a_filing_the_daily_feed_reports_late(
    tmp_path: Path,
) -> None:
    """A daily index lists filings dated earlier; those must not be dropped.

    The run for the earlier date has already happened and its own index did not
    list the filing yet, so a date filter here would lose it permanently — the
    shape of #90. It is written to the partition for its own filing date.
    """
    late_url = submission_url("2097726", "0001193125-26-384298")
    transport = FakeTransport(
        {
            daily_index_url(date(2026, 9, 8)): [(200, DAILY_INDEX_TEXT.encode())],
            submission_url("2097725", "0001193125-26-384297"): [(200, b"on time")],
            late_url: [(200, b"late")],
        }
    )

    table, _ = acquire_sixk_documents(
        _config(tmp_path, start_date=date(2026, 9, 8), end_date=date(2026, 9, 8)),
        fetcher=_fetcher(transport),
    )

    assert late_url in transport.calls
    late_row = table.loc[table["accession_number"] == "000119312526384298"].iloc[0]
    assert late_row["date"] == "2026-09-04"
    assert late_row["resource_uri"] == mirror_path(
        str(tmp_path),
        filing_date="2026-09-04",
        accession_number="0001193125-26-384298",
    )


def test_source_yields_a_filing_listed_in_two_indexes_once(
    tmp_path: Path,
) -> None:
    """A duplicated index row must not cost a second storage round-trip."""
    responses = _index_and_submissions()
    responses[daily_index_url(date(2026, 4, 30))] = [(200, DAY_ONE_INDEX.encode())]
    source = EdgarDocumentSource(
        config=_config(tmp_path),
        fetcher=_fetcher(FakeTransport(responses)),
    )

    assert [candidate.accession_number for candidate in source] == [
        "000165495426004070"
    ]


def test_source_reports_the_configured_form_types(tmp_path: Path) -> None:
    """The source reads the run's form types rather than assuming 6-K."""
    source = EdgarDocumentSource(
        config=_config(tmp_path, form_types=("6-K",)),
        fetcher=_fetcher(FakeTransport(_index_and_submissions())),
    )

    assert [candidate.form_type for candidate in source] == ["6-K"]
