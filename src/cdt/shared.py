"""Adapters over ``idi_ftm2j_shared`` used by CDT."""

from __future__ import annotations

from typing import Self

from idi_ftm2j_shared.failures import FailureClassifier
from idi_ftm2j_shared.failures import FailureRegistry as _SharedFailureRegistry
from idi_ftm2j_shared.logs import get_logger

__all__ = ["FailureClassifier", "FailureRegistry", "get_logger"]


class FailureRegistry(_SharedFailureRegistry):
    """Registry with removal, until the shared package grows one.

    Without ``discard``, a filing that succeeds on a --force retry stays
    registered forever: every later normal run keeps skipping it and
    failures.json permanently over-reports. Delete this subclass once
    idi-ftm2j-shared ships a discard method.
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
