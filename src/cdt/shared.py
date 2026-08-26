"""Shared-package adapters used by CDT.

Prefers ``idi_ftm2j_shared`` when available and falls back to local compatible
implementations so CDT remains runnable while the shared package integration is
rolled out.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Self

MIN_FAILURE_KEY_PARTS = 2

try:
    from idi_ftm2j_shared.failures import FailureClassifier
    from idi_ftm2j_shared.failures import FailureRegistry as _SharedFailureRegistry
    from idi_ftm2j_shared.logs import get_logger
    from idi_ftm2j_shared.storage import load_json, save_json

    if hasattr(_SharedFailureRegistry, "discard"):
        FailureRegistry = _SharedFailureRegistry
    else:

        class FailureRegistry(_SharedFailureRegistry):  # type: ignore[no-redef]
            """Registry with removal, until the shared package grows one.

            Without ``discard``, a filing that succeeds on a --force retry stays
            registered forever: every later normal run keeps skipping it and
            failures.json permanently over-reports (#67). Delete this subclass
            once idi-ftm2j-shared ships a discard method.
            """

            def discard(self: Self, key: tuple[str, str]) -> None:
                """Remove a key so a successfully retried entity is retried again."""
                with self._lock:
                    if key not in self._entries:
                        return
                    self._entries.remove(key)
                    self._reasons.pop(key, None)
                    self._pending += 1
                    if self._pending >= self._flush_every:
                        self.flush()

except ModuleNotFoundError:

    class FailureClassifier(ABC):
        """Fallback base class for permanent failure classification."""

        @property
        @abstractmethod
        def do_not_retry(self: Self) -> frozenset[StrEnum]:
            """Return failure types that should not be retried."""

        def is_retryable(self: Self, failure_type: StrEnum) -> bool:
            """Return whether the given failure type should be retried."""
            return failure_type not in self.do_not_retry

        @abstractmethod
        def classify_from_response(
            self: Self, response: dict, **kwargs: object
        ) -> StrEnum:
            """Classify a failure from an API response."""

    class FailureRegistry:
        """Fallback persistent registry for permanent failures."""

        def __init__(
            self: Self,
            file_path: str,
            classifier: FailureClassifier,
            flush_every: int = 10,
        ) -> None:
            """Initialize the registry."""
            self.file_path = file_path
            self._classifier = classifier
            self._flush_every = flush_every
            self._pending = 0
            self._entries: set[tuple[str, str]] = set()
            self._reasons: dict[tuple[str, str], str] = {}
            self._lock = RLock()
            self.load()

        def load(self: Self) -> None:
            """Load entries from disk if present."""
            if not self.file_path or not Path(self.file_path).exists():
                self._entries = set()
                self._reasons = {}
                return
            try:
                data = load_json(self.file_path, return_type="dict")
            except json.JSONDecodeError:
                self._entries = set()
                self._reasons = {}
                return
            if not isinstance(data, dict):
                self._entries = set()
                self._reasons = {}
                return
            entries = data.get("entries", [])
            reasons = data.get("reasons", {})
            self._entries = {
                tuple(entry[:MIN_FAILURE_KEY_PARTS])
                for entry in entries
                if isinstance(entry, list) and len(entry) >= MIN_FAILURE_KEY_PARTS
            }
            self._reasons = {
                entry: str(reasons.get(" ".join(entry), "")) for entry in self._entries
            }

        def save(self: Self) -> None:
            """Persist entries to disk."""
            if not self.file_path:
                return
            save_json(
                self.file_path,
                {
                    "entries": [list(entry) for entry in sorted(self._entries)],
                    "reasons": {
                        " ".join(entry): self._reasons.get(entry, "")
                        for entry in sorted(self._entries)
                    },
                },
            )

        def add(self: Self, key: tuple[str, str], failure_type: StrEnum) -> None:
            """Add a non-retryable failure to the registry."""
            if self._classifier.is_retryable(failure_type):
                return
            with self._lock:
                if key in self._entries:
                    return
                self._entries.add(key)
                self._reasons[key] = str(failure_type)
                self._pending += 1
                if self._pending >= self._flush_every:
                    self.flush()

        def discard(self: Self, key: tuple[str, str]) -> None:
            """Remove a key so a successfully retried entity is retried again."""
            with self._lock:
                if key not in self._entries:
                    return
                self._entries.remove(key)
                self._reasons.pop(key, None)
                self._pending += 1
                if self._pending >= self._flush_every:
                    self.flush()

        def flush(self: Self) -> None:
            """Persist buffered failures."""
            with self._lock:
                self.save()
                self._pending = 0

        def __contains__(self: Self, key: tuple[str, str]) -> bool:
            """Return whether the key is already marked non-retryable."""
            return key in self._entries

    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Return a basic fallback logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    def load_json(file_path: str, return_type: str = "dict") -> dict | list:
        """Load JSON from a local path, returning an empty container if missing."""
        path = Path(file_path)
        if not path.exists():
            return {} if return_type == "dict" else []
        return json.loads(path.read_text(encoding="utf-8"))

    def save_json(file_path: str, data: dict | list) -> None:
        """Persist JSON to a local path."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
