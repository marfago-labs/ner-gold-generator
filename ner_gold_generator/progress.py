"""CLI progress reporting on stderr (Rich when interactive, plain otherwise)."""

from __future__ import annotations

import re
import shutil
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ner_gold_generator.logger import get_logger

_LOGGER = get_logger(__name__)


def short_model_label(model_id: str, *, max_len: int = 32) -> str:
    """Compact model slug for progress lines (drops vendor prefix when needed)."""
    model_id = model_id.strip()
    if len(model_id) <= max_len:
        return model_id
    leaf = model_id.rsplit("/", 1)[-1]
    if len(leaf) <= max_len:
        return leaf
    return f"{leaf[: max_len - 1]}…"


def format_skip_error(error: str, *, max_len: int = 200) -> str:
    """Return a short, human-readable skip reason for terminal output."""
    attempt_parts: list[str] = []
    calls = re.search(r"llm_calls=(\d+)/(\d+)", error)
    if calls:
        attempt_parts.append(f"attempt {calls.group(1)}/{calls.group(2)}")
    reason = re.search(r"reason=(\w+)", error)
    if reason and reason.group(1) == "validation":
        val = re.search(r"validation_failures=(\d+)/(\d+)", error)
        if val:
            attempt_parts.append(f"validation {val.group(1)}/{val.group(2)}")
    elif reason and reason.group(1) == "provider":
        prov = re.search(r"provider_failures=(\d+)/(\d+)", error)
        if prov:
            attempt_parts.append(f"provider {prov.group(1)}/{prov.group(2)}")

    if "  Surface " in error:
        detail = error[error.index("  Surface ") + 2 :].strip()
    elif ").\n  " in error:
        detail = error.split(").\n  ", 1)[-1].strip()
    else:
        detail = error.replace("\n", " ").strip()
        if "reason=validation" in error and "Surface " not in detail:
            detail = "validation failed after retries"
        elif "reason=provider" in error and "invalid JSON" not in detail:
            detail = "provider/API failed after retries"

    prefix = f"({' · '.join(attempt_parts)}) " if attempt_parts else ""
    text = f"{prefix}{detail}".strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


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


def render_bar(done: int, total: int, *, width: int = 28) -> str:
    """Return an ASCII progress bar like ``[████░░░░]`` for *done* of *total*."""
    width = max(8, width)
    if total <= 0:
        return "[" + "?" * width + "]"
    clamped = min(max(done, 0), total)
    filled = int(width * clamped / total)
    filled = min(filled, width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class Progress:
    """stderr progress helper for batch gold builds.

      Uses Rich on interactive stderr (TTY) for a stable progress bar plus log lines.
    Falls back to plain ASCII lines when not a TTY (tests, pipes) or when ``quiet``.
    """

    quiet: bool = False
    verbose: bool = False
    use_bar: bool = True
    use_rich: bool | None = None
    _tty_bar: bool | None = None
    _bar_width: int = 28
    _bar_line_active: bool = field(default=False, repr=False)
    _last_tty_width: int = field(default=0, repr=False)
    _rich_console: Any = field(default=None, repr=False)
    _rich_progress: Any = field(default=None, repr=False)
    _rich_task_id: Any = field(default=None, repr=False)
    _track_done: int = field(default=0, repr=False)
    _track_total: int = field(default=0, repr=False)
    _update_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _use_rich(self) -> bool:
        if self.quiet or not self.use_bar:
            return False
        if self._tty_bar is False:
            return False
        if self.use_rich is not None:
            return self.use_rich
        if self._tty_bar is True:
            return True
        return sys.stderr.isatty()

    def log(self, message: str) -> None:
        """Print *message* to stderr unless :attr:`quiet` is set."""
        if self.quiet:
            return
        if self._rich_console is not None:
            self._rich_console.print(message)
            return
        print(message, file=sys.stderr, flush=True)

    def log_skip(self, doc_id: str, error: str) -> None:
        """Print a document skip without breaking an active Rich progress bar."""
        reason = format_skip_error(error)
        if self.quiet:
            return
        if self._rich_console is not None:
            self._rich_console.print(
                f"[yellow]skipped[/] [bold]{doc_id}[/]: {reason}",
            )
        else:
            self.log(f"skipped {doc_id}: {reason}")

    def verbose_log(self, message: str) -> None:
        """Print a verbose step line when :attr:`verbose` is enabled."""
        if self.verbose and not self.quiet:
            _LOGGER.debug(f"verbose: {message}")

    def set_activity(self, detail: str) -> None:
        """Update in-flight status without advancing the completed count."""
        if self.quiet:
            return
        detail = detail.rstrip()
        if self._rich_progress is not None and self._rich_task_id is not None:
            with self._update_lock:
                self._rich_progress.update(self._rich_task_id, activity=detail)
            return
        if self._track_total > 0:
            self.update_bar(self._track_done, self._track_total, detail=detail)

    @staticmethod
    def status_prefix(done: int, total: int) -> str:
        """Return a compact counter like ``3/10 (30%)``."""
        if total <= 0:
            return "0/0 (0%)"
        clamped = min(max(done, 0), total)
        pct = int(100 * clamped / total)
        return f"{clamped}/{total} ({pct}%)"

    bar = status_prefix

    def _use_tty_refresh(self) -> bool:
        if self._tty_bar is not None:
            return self._tty_bar
        return sys.stderr.isatty()

    def _bar_for(self, done: int, total: int) -> str:
        cols = _stderr_columns()
        width = min(self._bar_width, max(8, cols - 36))
        return render_bar(done, total, width=width)

    def update_bar(self, done: int, total: int, *, detail: str = "") -> None:
        """Print document progress (plain mode only; Rich uses :meth:`track`)."""
        if self.quiet or total <= 0 or self._rich_progress is not None:
            return
        prefix = self.status_prefix(done, total)
        cols = _stderr_columns()
        detail = detail.rstrip()

        if self.use_bar and self._use_tty_refresh():
            bar = self._bar_for(done, total)
            line = f"\r{bar} {prefix}"
            if detail:
                remaining = cols - len(line) - 1
                if remaining > 0:
                    line += " " + detail[:remaining]
            pad = max(self._last_tty_width - len(line), 0)
            print(line + (" " * pad), end="", file=sys.stderr, flush=True)
            self._last_tty_width = len(line) + pad
            self._bar_line_active = True
            return

        if self.use_bar:
            core = f"{self._bar_for(done, total)} {prefix}"
            line = _fit_status_line(core, detail, cols)
        else:
            line = _fit_status_line(prefix, detail, cols)
        print(line, file=sys.stderr, flush=True)

    def finish_bar(self) -> None:
        """End an in-place TTY progress line with a newline (plain mode)."""
        if self._bar_line_active:
            print(file=sys.stderr, flush=True)
            self._bar_line_active = False
            self._last_tty_width = 0

    @contextmanager
    def track(
        self,
        total: int,
        *,
        label: str = "documents",
    ) -> Iterator[Callable[[int, str], None]]:
        """Context manager yielding ``tick(done, detail)`` after each completed item."""
        if self.quiet:

            def tick_noop(done: int, detail: str = "") -> None:
                del done, detail

            yield tick_noop
            return

        if self._use_rich():
            yield from self._track_rich(total, label=label)
            return

        yield from self._track_plain(total, label=label)

    def _track_rich(
        self,
        total: int,
        *,
        label: str,
    ) -> Iterator[Callable[[int, str], None]]:
        from rich.console import Console
        from rich.progress import BarColumn, TextColumn, TimeElapsedColumn
        from rich.progress import Progress as RichProgress

        console = Console(stderr=True, soft_wrap=True)
        self._rich_console = console
        with RichProgress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("|"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("|"),
            TextColumn("{task.fields[activity]}", justify="left"),
            console=console,
            transient=False,
        ) as rich_prog:
            self._rich_progress = rich_prog
            self._track_total = total
            self._track_done = 0
            description = f"{label}: {total} total"
            task_id = rich_prog.add_task(description, total=total, activity="")
            self._rich_task_id = task_id

            def tick(done: int, detail: str = "") -> None:
                self._track_done = done
                with self._update_lock:
                    rich_prog.update(
                        task_id,
                        completed=done,
                        description=description,
                        activity=detail,
                    )

            try:
                yield tick
            finally:
                self._rich_progress = None
                self._rich_console = None
                self._rich_task_id = None
                self._track_total = 0
                self._track_done = 0

    def _track_plain(
        self,
        total: int,
        *,
        label: str,
    ) -> Iterator[Callable[[int, str], None]]:
        self._track_total = total
        self._track_done = 0
        self.log(f"{label}: {total} total")
        if self.use_bar and total > 0:
            self.update_bar(0, total, detail="starting…")

        def tick(done: int, detail: str = "") -> None:
            self._track_done = done
            self.update_bar(done, total, detail=detail)

        try:
            yield tick
        except BaseException:
            self.finish_bar()
            raise
        else:
            self.finish_bar()
        finally:
            self._track_total = 0
            self._track_done = 0
