"""Argument-parsing and logging helpers for the ``cdt`` CLI."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path


def configure_logging(*, quiet: bool, log_file: Path | None = None) -> None:
    """Configure CLI logging."""
    level = logging.WARNING if quiet else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_date(value: str) -> date:
    """Parse an ISO date for argparse."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"expected YYYY-MM-DD, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed
