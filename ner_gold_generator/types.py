"""Shared datatypes for proposed and validated entity spans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProposedEntity:
    """Entity span proposed before anchoring to source text.

    Attributes:
        text: Verbatim surface string as proposed by an LLM or manual file.
        label: Normalized label name (see :func:`validate.normalize_entity_label`).
    """

    text: str
    label: str


@dataclass(frozen=True, slots=True)
class ValidatedEntity:
    """Entity span anchored to source text with character offsets.

    Invariant: ``source_text[start:end] == text`` for the document being labeled.

    Attributes:
        text: Verbatim substring of the source document.
        label: Allowed unified label (lowercase, underscores).
        start: Inclusive start offset (0-based).
        end: Exclusive end offset.
    """

    text: str
    label: str
    start: int
    end: int
