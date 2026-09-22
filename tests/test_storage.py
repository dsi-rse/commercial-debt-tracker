"""Tests for the storage read path and its S3 client factory."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Self

import pandas as pd
import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
import pytest

from cdt import ingest, storage


class FakeFrozenCredentials(NamedTuple):
    """The three fields ``arrow_filesystem`` reads off a frozen credential."""

    access_key: str
    secret_key: str
    token: str


class FakeCredentials:
    """Credentials whose values name the profile they came from.

    Keyed on the profile so a test can tell *which* profile the S3 filesystem
    resolved, not merely that it resolved one.
    """

    def __init__(self: Self, profile_name: str | None) -> None:
        """Remember the profile these belong to."""
        self.profile_name = profile_name or "default"

    def get_frozen_credentials(self: Self) -> FakeFrozenCredentials:
        """Return a key triple tagged with the profile."""
        suffix = self.profile_name
        return FakeFrozenCredentials(f"AK-{suffix}", f"SK-{suffix}", f"TOK-{suffix}")


class FakeSession:
    """Records the profile it was constructed with and hands back a marker."""

    created: list[str | None] = []
    region_name = "us-east-2"

    def __init__(self: Self, profile_name: str | None = None) -> None:
        """Record the requested profile."""
        self.profile_name = profile_name
        FakeSession.created.append(profile_name)

    def client(self: Self, name: str, config: object = None) -> str:
        """Return a marker standing in for a boto3 S3 client."""
        del config
        return f"{name}:{self.profile_name!r}"

    def get_credentials(self: Self) -> FakeCredentials:
        """Return credentials tagged with this session's profile."""
        return FakeCredentials(self.profile_name)


@pytest.fixture(autouse=True)
def _fake_boto3_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub boto3 so this module structurally cannot reach a real account.

    The client-cache and configured-profile resets live in the root
    ``conftest.py`` instead: that state leaks *across* files, so resetting it
    here protected only this one. Not duplicated back, deliberately — these
    tests depend on the shared fixture, so if it ever stops resetting, the
    memoization test fails rather than the leak going unnoticed.
    """
    monkeypatch.setattr(storage.boto3, "Session", FakeSession)
    FakeSession.created = []


def test_storage_clients_use_the_configured_aws_profile() -> None:
    """--aws-profile must reach this module's client, not only ingest's (#71).

    ``_s3_client`` was a singleton built as ``boto3.client("s3", ...)`` with no
    profile at all, so a run pointed at a non-default account read its manifests
    through the right credentials and then did every artifact read, every
    list_objects_v2 behind a partition scan and every write through whatever the
    ambient chain resolved.
    """
    storage.configure_s3_profile("analysis")

    assert storage.configured_s3_profile() == "analysis"
    assert storage._s3_client() == "s3:'analysis'"
    assert FakeSession.created == ["analysis"]


def test_default_s3_client_without_a_profile_resolves_the_configured_one() -> None:
    """``ensure_s3_client()`` passes no profile, and used to get the empty one (#71).

    That call is how itemize and extract resolve document bodies from S3, so its
    default silently decided the credentials for two whole stages.
    """
    storage.configure_s3_profile("analysis")

    assert ingest.default_s3_client() == "s3:'analysis'"


def test_an_explicit_profile_still_wins_over_the_configured_one() -> None:
    """Ingest and the 6-K scraper keep passing ``config.aws_profile``."""
    storage.configure_s3_profile("analysis")

    assert ingest.default_s3_client("other") == "s3:'other'"
    assert storage.s3_client("other") == "s3:'other'"


def test_the_empty_profile_means_the_ambient_credential_chain() -> None:
    """The CLI default is "", which must stay "no profile" (the old behaviour)."""
    storage.configure_s3_profile(None)

    assert storage._s3_client() == "s3:None"
    assert FakeSession.created == [None]


def test_clients_are_memoized_per_profile() -> None:
    """One credentialed object per profile: construction is the #83 cost.

    ``ingest.default_s3_client`` used to build a fresh Session per call, which
    is the only reason ``itemizer.core.ensure_s3_client`` memoizes by hand.
    """
    storage.configure_s3_profile("analysis")

    first = storage._s3_client()
    second = ingest.default_s3_client()
    third = storage.s3_client("analysis")
    other = storage.s3_client("second-account")

    assert first is second is third
    assert other != first
    assert FakeSession.created == ["analysis", "second-account"]


def test_a_configured_profile_is_set_for_the_next_test_to_find() -> None:
    """First half of a pair; see the test immediately below.

    These two are deliberately order-coupled and must stay adjacent and in
    this order. Nothing else in the suite would notice the leak they detect.
    """
    storage.configure_s3_profile("leaks-into-the-next-test")

    assert storage.configured_s3_profile() == "leaks-into-the-next-test"


def test_a_profile_set_by_an_earlier_test_does_not_leak_into_this_one() -> None:
    """Process-global profile state must not survive a test (#71).

    ``configure_s3_profile`` writes a module global and is called
    unconditionally by ``cli.main`` and ``orchestrator.main``, which the suite
    invokes for real dozens of times — and the orchestrator's flag defaults to
    ``os.environ.get("AWS_PROFILE", "")``. Without the reset in the root
    conftest, a developer or CI runner with AWS_PROFILE exported would hand
    real credentials to every later test. The reset is infrastructure, so the
    only way to test it is to pollute deliberately and check the next test is
    clean.
    """
    assert storage.configured_s3_profile() == ""


def test_cli_main_configures_the_profile_from_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag is honored at the one point it enters the process (#71)."""
    from cdt import cli

    seen: list[str | None] = []
    monkeypatch.setattr(cli, "configure_s3_profile", seen.append)
    monkeypatch.setattr(
        cli, "build_parser", _parser_returning(aws_profile="analysis", exit_code=0)
    )

    assert cli.main([]) == 0
    assert seen == ["analysis"]


def test_cli_main_tolerates_a_subcommand_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most subcommands never grew an --aws-profile; they must still run."""
    from cdt import cli

    seen: list[str | None] = []
    monkeypatch.setattr(cli, "configure_s3_profile", seen.append)
    monkeypatch.setattr(cli, "build_parser", _parser_returning(exit_code=0))

    assert cli.main([]) == 0
    assert seen == [None]


def _parser_returning(*, exit_code: int = 0, **attrs: object):  # noqa: ANN202
    import argparse

    namespace = argparse.Namespace(func=lambda _args: exit_code, **attrs)

    class _Parser:
        def parse_args(self: Self, argv: object) -> argparse.Namespace:
            del argv
            return namespace

    return lambda: _Parser()


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    storage.write_table(str(path), pd.DataFrame(rows))


def _write_raw(path: Path, table: pa.Table) -> None:
    """Write a parquet file bypassing apply_declared_column_types.

    That is exactly what every partition written before #187 is: the physical
    type is whatever the frame inferred in that run, not the declared one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_read_dataset_unifies_schemas_instead_of_trusting_the_first_file(
    tmp_path: Path,
) -> None:
    """A column absent from the first partition must not vanish from the read (#190).

    ``pyarrow.dataset`` infers its schema from the *first* fragment alone, so
    the obvious implementation silently drops every value of a column the first
    file happens to predate — and this pipeline has that shape on purpose:
    ``form_type`` and ``source`` were added to ``documents`` after the fact, and
    ``read_table``'s reindex fallback exists for the same reason. Verified
    against the real failure: with inferred schemas this dataset reads as
    ``['k']``.
    """
    root = tmp_path / "items"
    _write(root / "a.parquet", [{"k": "1"}])
    _write(root / "b.parquet", [{"k": "2", "later": "kept"}])

    table = storage.read_dataset(str(root)).sort_values("k").reset_index(drop=True)

    assert list(table.columns) == ["k", "later"]
    assert table["later"].isna().to_list() == [True, False]
    assert table["later"].to_list()[1] == "kept"


def test_the_footer_reads_behind_schema_unification_are_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unifying schemas must not serialise one round trip per partition (#110).

    Every ``fragment.physical_schema`` opens the file and reads its footer, and
    those reads happen before the dataset scanner exists, so its threads do not
    cover them. Locally that is 0.523s against 0.184s threaded on 1,554 files —
    small enough to hide — but on S3 each one is a sequential round trip, so a
    serial pass over the publish's 21,214 objects is ~25 minutes before a byte
    of data is read. That is the whole of #110's cost model, and #110's own
    option 1 is to parallelise exactly this.
    """
    pools: list[int | None] = []
    real_pool = storage.ThreadPoolExecutor

    class RecordingPool(real_pool):  # type: ignore[misc, valid-type]
        def __init__(
            self: Self, max_workers: int | None = None, **kwargs: object
        ) -> None:
            pools.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(storage, "ThreadPoolExecutor", RecordingPool)
    root = tmp_path / "items"
    for index in range(5):
        _write(root / f"p{index}.parquet", [{"k": str(index)}])

    table = storage.read_dataset(str(root))

    assert len(table) == 5
    assert pools == [5], "footers were read one at a time"


def test_read_dataset_falls_back_when_partitions_cannot_be_unified(
    tmp_path: Path,
) -> None:
    """A pre-#187 root still reads, through the per-file path (#190).

    #187 is what makes an Arrow dataset scan possible at all: before it, a
    column's physical type varied by partition. Partitions written *before* that
    fix keep their old types and the production dev root has not been rebuilt
    (#107), so the fallback is the live case. Reproduced here with the same
    conflict the real root has — ``double`` in one partition and ``string`` in
    another, which is what ``data/lineage-probe/debt-instruments`` carries for
    ``amendment_inferred_by`` across 26 of its 41 columns.
    """
    root = tmp_path / "debt-instruments"
    _write_raw(
        root / "cik_shard=0001" / "part-0000.parquet",
        pa.table({"debt_instrument_id": ["a"], "amendment_inferred_by": [1.0]}),
    )
    _write_raw(
        root / "cik_shard=0002" / "part-0000.parquet",
        pa.table(
            {"debt_instrument_id": ["b"], "amendment_inferred_by": ["dated_reference"]}
        ),
    )

    # The Arrow path must genuinely be unable to read this, or the test proves
    # nothing about the fallback — and it must say why, since the log names the
    # cause and a corrupt partition takes the same branch.
    table, error = storage._read_dataset_with_arrow(
        sorted(str(p) for p in root.rglob("*.parquet")), None
    )
    assert table is None
    assert isinstance(error, pa.ArrowException)

    table = (
        storage.read_dataset(str(root))
        .sort_values("debt_instrument_id")
        .reset_index(drop=True)
    )

    assert table["debt_instrument_id"].to_list() == ["a", "b"]
    assert [str(value) for value in table["amendment_inferred_by"]] == [
        "1.0",
        "dated_reference",
    ]


def test_the_fallback_log_names_the_error_that_caused_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every fallback used to be reported as "pre-#187 per-partition types".

    That is a confident diagnosis the code has not made: a corrupt partition, a
    truncated read or a vanished file take the same branch. Telling an operator
    the wrong cause — after silently paying a full re-read — is how an hour
    disappears, so the log carries the actual exception.
    """
    root = tmp_path / "items"
    _write(root / "a.parquet", [{"k": "1"}])
    (root / "b.parquet").write_bytes(b"not a parquet file at all")

    with caplog.at_level("INFO"), pytest.raises(pa.ArrowInvalid):
        storage.read_dataset(str(root))

    assert "ArrowInvalid" in caplog.text
    assert "Parquet magic bytes not found" in caplog.text


def test_one_filesystem_is_built_for_a_whole_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving a path list must not build a client per partition (#190).

    ``arrow_filesystem`` constructs an ``S3FileSystem`` — a full AWS SDK client
    with its own connection pool — and freezes credentials. Calling it once per
    path to get the stripped string built and discarded one per partition,
    which on the publish's 21,214 objects is 21,213 wasted clients and, worse,
    a fresh TLS handshake per file on the fallback path.
    """
    root = tmp_path / "items"
    for index in range(6):
        _write(root / f"p{index}.parquet", [{"k": str(index)}])

    calls: list[object] = []
    original = storage.arrow_filesystem

    def counting_arrow_filesystem(path: object) -> tuple[object | None, str]:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(storage, "arrow_filesystem", counting_arrow_filesystem)

    assert len(storage.read_dataset(str(root))) == 6
    assert len(calls) == 1


def test_read_dataset_uses_the_arrow_path_when_partitions_do_unify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast path is actually taken on a declared-type root (#190).

    Pinned by making the per-file fallback raise: a root written through
    ``write_table``, which applies the declared physical types, must never reach
    it. Without this the fallback could quietly become the only path and the
    measured 8.1x would be gone with every test still green.
    """
    root = tmp_path / "items"
    _write(root / "a.parquet", [{"k": "1", "principal_amount": "100"}])
    _write(root / "b.parquet", [{"k": "2", "principal_amount": "200"}])

    def _explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("fell back to the per-file path")

    monkeypatch.setattr(storage, "read_table", _explode)

    table = storage.read_dataset(str(root)).sort_values("k").reset_index(drop=True)

    assert table["k"].to_list() == ["1", "2"]


def test_read_dataset_projection_reindexes_a_column_absent_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A column no partition has comes back null, *on the Arrow path* (#190).

    Projecting a name Arrow does not have raises ``ArrowInvalid``, which the
    fallback would swallow — so handing ``to_table`` the unprojectable name
    still produces the right frame, just via the slow path, on every read that
    names a not-yet-existing column. The fallback is made fatal here so this
    pins the fast path rather than the answer.
    """
    root = tmp_path / "items"
    _write(root / "a.parquet", [{"k": "1"}])

    def _explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("fell back to the per-file path")

    monkeypatch.setattr(storage, "read_table", _explode)

    table = storage.read_dataset(str(root), columns=["k", "nowhere"])

    assert list(table.columns) == ["k", "nowhere"]
    assert table["nowhere"].isna().all()


def test_read_dataset_still_skips_orphaned_tempfiles(tmp_path: Path) -> None:
    """The Arrow path reads the filtered path list, never the directory (#68).

    A directory-backed dataset would pick up a crash-orphaned ``tmp*.parquet``
    and die on it, which is the whole reason ``iter_partition_paths`` filters.
    """
    root = tmp_path / "items"
    _write(root / "a.parquet", [{"k": "1"}])
    (root / "tmpabc123.parquet").write_bytes(b"")

    assert storage.read_dataset(str(root))["k"].to_list() == ["1"]


def test_read_dataset_honors_the_partition_filter(tmp_path: Path) -> None:
    """Partition selection happens before Arrow sees a path list."""
    root = tmp_path / "items"
    _write(root / "date=2026-01-01" / "part-0000.parquet", [{"k": "old"}])
    _write(root / "date=2026-01-02" / "part-0000.parquet", [{"k": "new"}])

    table = storage.read_dataset(str(root), partition_filter={"date": "2026-01-02"})

    assert table["k"].to_list() == ["new"]


def test_read_table_projects_without_reading_the_other_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection is pushed into the parquet read, not applied after it (#190).

    Measured on data/genwindow-eval-apr: ``documents`` is 12,206.6 MB and its
    ``accession_number`` column is 0.2988 MB of compressed chunks, so the read
    a dedup scan wants is 0.0024% of what the old path transferred.

    ``pd.read_parquet`` is made fatal because the *result* here is not evidence
    of anything: the pre-#190 implementation handed pandas the whole object and
    a ``columns=`` list, and it returned exactly these columns too. Without
    this the entire rewrite could be reverted with the suite still green.
    """

    def _explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("read_table must not read through pandas (#190)")

    path = tmp_path / "t.parquet"
    _write(path, [{"k": "1", "text": "x" * 100}])
    monkeypatch.setattr(storage.pd, "read_parquet", _explode)

    assert list(storage.read_table(path, ["k"]).columns) == ["k"]
    assert list(storage.read_table(path, ["text", "k"]).columns) == ["text", "k"]
    # And a requested-but-absent column still comes back, null, rather than
    # being silently dropped: ParquetFile.read does not raise on one.
    tolerant = storage.read_table(path, ["k", "missing"])
    assert list(tolerant.columns) == ["k", "missing"]
    assert tolerant["missing"].isna().all()


def test_count_table_rows_reads_only_the_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish guard's row count must not download the object (#190).

    Materialising the table would give the same 3, so reading any column group
    is made fatal: the count has to come from footer metadata. This runs once
    per final table on every publish, against objects measured at 12.2 GB.
    """

    def _explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("count_table_rows must read only the footer (#190)")

    path = tmp_path / "t.parquet"
    _write(path, [{"k": "1"}, {"k": "2"}, {"k": "3"}])
    monkeypatch.setattr(pq.ParquetFile, "read", _explode)

    assert storage.count_table_rows(path) == 3
    assert storage.count_table_rows(tmp_path / "absent.parquet") is None


def test_arrow_filesystem_leaves_local_paths_alone() -> None:
    """Local reads need no filesystem object; only S3 gets one."""
    filesystem, resolved = storage.arrow_filesystem("/artifacts/x/y.parquet")

    assert filesystem is None
    assert resolved == "/artifacts/x/y.parquet"


def test_arrow_filesystem_builds_s3_on_the_configured_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The S3FileSystem is the second credentialed object #71 had to feed (#190).

    It must resolve the same profile as the boto3 client, which is why the
    Session is memoized separately and this reads through it rather than
    building its own.
    """
    built: list[dict[str, object]] = []
    storage.configure_s3_profile("analysis")
    monkeypatch.setattr(
        storage.pyarrow.fs,
        "S3FileSystem",
        lambda **kwargs: built.append(kwargs) or "FS",
    )

    filesystem, resolved = storage.arrow_filesystem("s3://bucket/a/b.parquet")

    assert filesystem == "FS"
    assert resolved == "bucket/a/b.parquet"
    # The profile is the point: the autouse fixture's FakeSession records what
    # it was constructed with, so this fails if the filesystem resolves any
    # other profile than the configured one. Stubbing boto3_session here
    # instead — as this test used to — mocked away the only thing it claims.
    assert FakeSession.created == ["analysis"]
    assert built == [
        {
            "access_key": "AK-analysis",
            "secret_key": "SK-analysis",
            "session_token": "TOK-analysis",
            "connect_timeout": storage.S3_CLIENT_CONFIG.connect_timeout,
            "request_timeout": storage.S3_CLIENT_CONFIG.read_timeout,
            "retry_strategy": built[0]["retry_strategy"],
            "region": "us-east-2",
        }
    ]
    # Bounded rather than pyarrow's unbounded defaults, matching what boto3
    # gets from S3_CLIENT_CONFIG: moving reads here would otherwise have
    # dropped #112's timeout and retry mitigation on the main read path.
    assert isinstance(
        built[0]["retry_strategy"], storage.pyarrow.fs.AwsStandardS3RetryStrategy
    )


def test_the_s3_branch_of_every_reader_actually_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every read test is local, so the branch #190 is *about* never ran (#190).

    On a local path ``arrow_filesystem`` returns ``None`` and the whole
    credentialed branch short-circuits, which left
    ``ParquetFile(resolved, filesystem=...)`` — the bucket/key handoff this
    change turns on — unexecuted by the suite. A typo in the stripped path or a
    dropped ``filesystem=`` would have shipped green.

    Swapping in a LocalFileSystem keeps that branch honest with no network: the
    readers are handed an ``s3://`` URI, take the S3 path through
    ``strip_s3_scheme``, and read through a real pyarrow filesystem object.
    """
    root = tmp_path / "bucket-root"
    _write(root / "items" / "a.parquet", [{"k": "1", "text": "x"}])
    _write(root / "items" / "b.parquet", [{"k": "2", "text": "y"}])

    def fake_filesystem(path: object) -> tuple[object, str]:
        # Same shape arrow_filesystem returns for S3: a filesystem, and a path
        # with the scheme stripped and rooted at the "bucket".
        stripped = storage.strip_s3_scheme(path)
        assert not str(stripped).startswith("s3://")
        return pyarrow.fs.LocalFileSystem(), str(root / stripped.split("/", 1)[1])

    monkeypatch.setattr(storage, "arrow_filesystem", fake_filesystem)
    monkeypatch.setattr(
        storage, "iter_partition_paths", lambda *a, **k: iter(_S3_ITEM_PATHS)
    )

    assert storage.read_table("s3://bucket/items/a.parquet", ["k"])["k"].to_list() == [
        "1"
    ]
    assert storage.count_table_rows("s3://bucket/items/a.parquet") == 1
    assert storage.count_table_rows("s3://bucket/items/absent.parquet") is None
    table = storage.read_dataset("s3://bucket/items").sort_values("k")
    assert table["k"].to_list() == ["1", "2"]


#: The two partitions the S3-branch test's stubbed listing returns.
_S3_ITEM_PATHS = ["s3://bucket/items/a.parquet", "s3://bucket/items/b.parquet"]


def test_arrow_filesystem_refuses_a_session_with_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling through to pyarrow's own chain would re-open #71 (#190).

    ``S3FileSystem()`` with no keys is not anonymous — it resolves its own
    chain, and that chain disagrees with botocore's: botocore drops the
    environment provider when a profile is set explicitly, while pyarrow checks
    the environment first. So a profile that resolves nothing plus ambient
    environment keys for another account would fail every write loudly and read
    every partition silently from the wrong account.
    """

    class CredentiallessSession:
        region_name = None

        def get_credentials(self: Self) -> None:
            return None

    monkeypatch.setattr(
        storage, "boto3_session", lambda *a, **k: CredentiallessSession()
    )

    with pytest.raises(RuntimeError, match="No AWS credentials"):
        storage.arrow_filesystem("s3://bucket/a/b.parquet")


def test_arrow_filesystem_warns_when_credentials_expire_before_a_scan_could(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A static key triple means a long scan can outlive its credentials (#190).

    ``get_frozen_credentials`` refreshes, so each filesystem starts fresh — but
    ``S3FileSystem`` never re-signs, and botocore's advisory refresh window is
    900s. A full-corpus read takes longer than that, and what the operator sees
    is pyarrow's ``AWS Error UNKNOWN (HTTP status 400)``, which reads like
    anything but an expiry. This does not widen the window; it names the cause.
    """
    from collections import namedtuple
    from datetime import UTC, datetime, timedelta

    frozen = namedtuple("Frozen", "access_key secret_key token")

    class ExpiringCredentials:
        _expiry_time = datetime.now(UTC) + timedelta(seconds=60)

        def get_frozen_credentials(self: Self) -> object:
            return frozen("AK", "SK", "TOK")

    class ExpiringSession:
        region_name = None

        def get_credentials(self: Self) -> object:
            return ExpiringCredentials()

    monkeypatch.setattr(storage, "boto3_session", lambda *a, **k: ExpiringSession())
    monkeypatch.setattr(storage.pyarrow.fs, "S3FileSystem", lambda **kwargs: "FS")

    with caplog.at_level("WARNING"):
        storage.arrow_filesystem("s3://bucket/a/b.parquet")

    assert "expire in" in caplog.text
