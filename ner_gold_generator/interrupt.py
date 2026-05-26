"""User-initiated cancellation (Ctrl+C) during gold builds."""

from __future__ import annotations

from typing import Any

INTERRUPT_EXIT_CODE = 130


class BuildInterrupted(Exception):
    """Build was cancelled; may include partially completed gold rows.

    Attributes:
        examples: Completed rows in document order (indices may be sparse gaps
            filled only for completed docs in synthetic parallel mode).
        completed: Number of documents finished.
        total: Total documents requested.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        *,
        completed: int,
        total: int,
    ) -> None:
        self.examples = examples
        self.completed = completed
        self.total = total
        super().__init__(f"interrupted after {completed}/{total} documents")
