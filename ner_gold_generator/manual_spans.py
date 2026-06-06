"""Manual span types and anchoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ner_gold_generator.validate import index_unique_span


@dataclass(frozen=True, slots=True)
class ManualSpan:
    """One manual entity: gold ``text``/``label``, optional unique ``match`` substring."""

    text: str
    label: str
    match: str | None = None


def index_manual_span(text: str, span: ManualSpan) -> tuple[int, int]:
    """Anchor *span* in *text* using a unique ``match`` key when provided.

    When ``match`` is set, it must occur exactly once; ``text`` must appear inside
    that matched region (typically as a prefix). Otherwise ``text`` must be unique.
    """
    key = span.match if span.match else span.text
    located = index_unique_span(text, key)
    if located is None:
        raise ValueError(f"Surface {key!r} not found or not unique in text")
    start, end = located
    if span.match:
        inner_start = text.find(span.text, start, end)
        if inner_start < 0:
            raise ValueError(
                f"Entity text {span.text!r} not found within match {span.match!r} in text"
            )
        return inner_start, inner_start + len(span.text)
    return start, end


def parse_span_list(value: Any, *, doc_key: str) -> list[ManualSpan]:
    """Parse one document's span list from manual annotation JSON/YAML."""
    if not isinstance(value, list):
        raise ValueError(f"Manual spans for {doc_key!r} must be a list")
    spans: list[ManualSpan] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid span entry for {doc_key!r}")
        text = item.get("text")
        label = item.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            raise ValueError(f"Span needs text and label for {doc_key!r}")
        match_raw = item.get("match")
        match = match_raw.strip() if isinstance(match_raw, str) and match_raw.strip() else None
        spans.append(ManualSpan(text=text.strip(), label=label.strip(), match=match))
    return spans
