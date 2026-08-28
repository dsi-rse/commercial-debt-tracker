"""Tests for CDT settings resolution."""

from __future__ import annotations

import importlib
from pathlib import Path

import dotenv

import cdt.settings


def test_settings_uses_default_data_dir_when_env_missing(monkeypatch) -> None:  # noqa: ANN001
    """Importing settings should not require DATA_DIR to be predefined."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    reloaded = importlib.reload(cdt.settings)

    assert reloaded.DATA_DIR == (reloaded.PROJECT_ROOT / "data").resolve()

    monkeypatch.undo()
    importlib.reload(cdt.settings)


def test_settings_resolves_configured_data_dir(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    """Configured DATA_DIR should still be resolved relative to the project root."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "processor"))
    reloaded = importlib.reload(cdt.settings)

    assert reloaded.DATA_DIR == (tmp_path / "processor").resolve()

    monkeypatch.undo()
    importlib.reload(cdt.settings)


def test_sixk_triage_model_defaults_and_overrides(monkeypatch) -> None:  # noqa: ANN001
    """The 6-K triage model id follows the extractor's env-override pattern."""
    monkeypatch.delenv("SIXK_TRIAGE_MODEL", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    reloaded = importlib.reload(cdt.settings)
    assert reloaded.SIXK_TRIAGE_MODEL == reloaded.DEFAULT_SIXK_TRIAGE_MODEL

    monkeypatch.setenv("SIXK_TRIAGE_MODEL", "openai/gpt-5.6-terra")
    reloaded = importlib.reload(cdt.settings)
    assert reloaded.SIXK_TRIAGE_MODEL == "openai/gpt-5.6-terra"

    monkeypatch.undo()
    importlib.reload(cdt.settings)
