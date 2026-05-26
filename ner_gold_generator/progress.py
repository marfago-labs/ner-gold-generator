"""CLI progress reporting on stderr."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


def short_model_label(model_id: str, *, max_len: int = 32) -> str:
    """Compact model slug for progress lines (drops vendor prefix when needed)."""
    model_id = model_id.strip()
    if len(model_id) <= max_len:
        return model_id
    leaf = model_id.rsplit("/", 1)[-1]
    if len(leaf) <= max_len:
        return leaf
    return f"{leaf[: max_len - 1]}…"


def _stderr_columns() -> int:
    try:
        return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)
    except OSError:
        return 100


def _fit_status_line(prefix: str, detail: str, width: int) -> str:
    """Fit *prefix* and *detail* on one line, ellipsis in the middle if needed."""
    if not detail:
        return prefix
    sep = " "
    budget = width - len(prefix) - len(sep)
    if budget >= len(detail):
        return f"{prefix}{sep}{detail}"
    if budget <= 1:
        return prefix[:width]
    keep = budget - 1
    head = keep // 2
    tail = keep - head - 1
    return f"{prefix}{sep}{detail[:head]}…{detail[-tail:]}"


@dataclass
class Progress:
    """Lightweight stderr logger and document counters for batch gold builds.

    Attributes:
        quiet: When ``True``, :meth:`log` and status updates are no-ops.
        verbose: When ``True``, :meth:`verbose_log` prints step details.
    """

    quiet: bool = False
    verbose: bool = False

    def log(self, message: str) -> None:
        """Print *message* to stderr unless :attr:`quiet` is set."""
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    def verbose_log(self, message: str) -> None:
        """Print a verbose step line when :attr:`verbose` is enabled."""
        if self.verbose and not self.quiet:
            self.log(f"verbose: {message}")

    @staticmethod
    def status_prefix(done: int, total: int) -> str:
        """Return a compact counter like ``3/10 (30%)``."""
        if total <= 0:
            return "0/0 (0%)"
        clamped = min(max(done, 0), total)
        pct = int(100 * clamped / total)
        return f"{clamped}/{total} ({pct}%)"

    # Backward-compatible alias for tests/callers.
    bar = status_prefix

    def update_bar(self, done: int, total: int, *, detail: str = "") -> None:
        """Print document progress as ``count (pct%) detail`` on stderr."""
        if self.quiet or total <= 0:
            return
        prefix = self.status_prefix(done, total)
        line = _fit_status_line(prefix, detail.rstrip(), _stderr_columns())
        print(line, file=sys.stderr, flush=True)

    def finish_bar(self) -> None:
        """No-op (kept for callers that clear legacy in-place bars)."""

    @contextmanager
    def track(
        self,
        total: int,
        *,
        label: str = "documents",
    ) -> Iterator[Callable[[int, str], None]]:
        """Context manager that logs progress after each completed item.

        Yields a ``tick(done, detail)`` callback where *done* is the number of
        finished items (0..*total*) and *detail* is optional status text.
        """
        if not self.quiet:
            self.log(f"{label}: {total} total")

        def tick(done: int, detail: str = "") -> None:
            if detail:
                self.update_bar(done, total, detail=detail)

        try:
            yield tick
        except BaseException:
            self.finish_bar()
            raise
        else:
            self.finish_bar()
