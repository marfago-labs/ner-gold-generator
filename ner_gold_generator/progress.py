"""CLI progress reporting on stderr."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Progress:
    """Lightweight stderr logger for batch gold builds.

    Attributes:
        quiet: When ``True``, :meth:`log` is a no-op.
    """

    quiet: bool = False

    def log(self, message: str) -> None:
        """Print *message* to stderr unless :attr:`quiet` is set."""
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    @staticmethod
    def bar(done: int, total: int, *, width: int = 28) -> str:
        """Return a fixed-width ASCII progress bar prefix.

        Args:
            done: Completed items (clamped visually against *total*).
            total: Total items; ``0`` yields an empty bar.
            width: Inner bar width in characters.

        Returns:
            String like ``"[########----------------] 3/10"``.
        """
        if total <= 0:
            return "[--------------------] 0/0"
        filled = int(width * done / total)
        return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total}"
