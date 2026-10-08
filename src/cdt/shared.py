"""Adapters over ``idi_ftm2j_shared`` used by CDT."""

from __future__ import annotations

import logging
from typing import Self

from idi_ftm2j_shared.failures import FailureClassifier
from idi_ftm2j_shared.failures import FailureRegistry as _SharedFailureRegistry
from idi_ftm2j_shared.logs import TqdmLoggingHandler
from idi_ftm2j_shared.logs import get_logger as _shared_get_logger

__all__ = [
    "FailureClassifier",
    "FailureRegistry",
    "get_logger",
    "log_stage_complete",
    "log_stage_start",
]


def log_stage_start(logger: logging.Logger, stage: str, **details: object) -> None:
    """Log ``Starting stage: <stage> | key=value ...``, the run's stage-start line."""
    logger.info("Starting stage: %s%s", stage, _stage_details(details))


def log_stage_complete(logger: logging.Logger, stage: str, **details: object) -> None:
    """Log ``Completed stage: <stage> | key=value ...``, the run's stage-end line."""
    logger.info("Completed stage: %s%s", stage, _stage_details(details))


def _stage_details(details: dict[str, object]) -> str:
    text = " ".join(f"{key}={value}" for key, value in details.items())
    return f" | {text}" if text else ""


def get_logger(name: str) -> logging.Logger:
    """Return the shared package's logger for ``name``, routed through the root logger.

    The shared package gives each logger its own console handler, its own
    level and no propagation. CDT configures the root logger instead
    (``cdt.cli_support.configure_logging``: level, format, ``--log-file``), so
    the console handler is removed and level and propagation are deferred to
    the root. Any other handler the shared package attached is kept.
    """
    logger = _shared_get_logger(name)
    for handler in list(logger.handlers):
        if isinstance(handler, TqdmLoggingHandler):
            logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


class FailureRegistry(_SharedFailureRegistry):
    """Shared FailureRegistry plus ``discard``, until the shared package has one.

    Without ``discard`` a filing that succeeds on a --force retry stays
    registered, so later runs keep skipping it. Delete this subclass once
    idi-ftm2j-shared ships a discard method.
    """

    def discard(self: Self, key: tuple[str, str]) -> None:
        """Remove ``key`` if registered, counting it toward the next flush."""
        with self._lock:
            if key not in self._entries:
                return
            self._entries.remove(key)
            self._reasons.pop(key, None)
            self._pending += 1
            if self._pending >= self._flush_every:
                self.flush()
