"""Tests for the storage read path and its S3 client factory."""

from __future__ import annotations

from typing import Self

import pytest

from cdt import ingest, storage


class FakeSession:
    """Records the profile it was constructed with and hands back a marker."""

    created: list[str | None] = []

    def __init__(self: Self, profile_name: str | None = None) -> None:
        """Record the requested profile."""
        self.profile_name = profile_name
        FakeSession.created.append(profile_name)

    def client(self: Self, name: str, config: object = None) -> str:
        """Return a marker standing in for a boto3 S3 client."""
        del config
        return f"{name}:{self.profile_name!r}"


@pytest.fixture(autouse=True)
def _reset_s3_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a clean client cache and a fake Session."""
    monkeypatch.setattr(storage, "_S3_CLIENTS", {})
    monkeypatch.setattr(storage, "_CONFIGURED_S3_PROFILE", "")
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
