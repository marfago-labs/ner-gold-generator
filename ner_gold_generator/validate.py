"""Validate labels and anchor entity spans to source text."""

from __future__ import annotations

from ner_gold_generator.types import ProposedEntity, ValidatedEntity

_ALLOWED_LABELS = frozenset(
    {
        "person",
        "organization",
        "location",
        "model",
        "dataset",
        "benchmark",
        "metric",
        "method",
        "number",
        "year",
        "arxiv_id",
        "acronym",
        "miscellaneous",
    }
)


def normalize_entity_label(raw: str) -> str | None:
    """Map a raw label string to an allowed unified label.

    Args:
        raw: Label from manual files or LLM output (any casing/spacing).

    Returns:
        Canonical label name, or ``None`` if not in the allow-list.
        Aliases ``org``/``orgs`` → ``organization``, ``misc`` → ``miscellaneous``.
    """
    key = raw.strip().lower().replace(" ", "_")
    aliases = {
        "org": "organization",
        "misc": "miscellaneous",
        "orgs": "organization",
    }
    key = aliases.get(key, key)
    if key in _ALLOWED_LABELS:
        return key
    return None


def index_unique_span(text: str, surface: str) -> tuple[int, int] | None:
    """Locate *surface* in *text* when it occurs exactly once.

    Args:
        text: Full document body.
        surface: Verbatim substring to find.

    Returns:
        ``(start, end)`` with exclusive *end*, or ``None`` if *surface* is empty,
        missing, or appears more than once.
    """
    if not surface:
        return None
    start = text.find(surface)
    if start < 0:
        return None
    if text.find(surface, start + 1) >= 0:
        return None
    return start, start + len(surface)


def validate_proposed_entities(
    text: str,
    proposed: list[ProposedEntity],
) -> list[ValidatedEntity]:
    """Filter and anchor proposed entities against *text*.

    Drops entities with unknown labels, empty surfaces, or non-unique spans.
    Deduplicates identical ``(start, end, label)`` keys.

    Args:
        text: Source document used for substring checks.
        proposed: Candidate spans (typically post-consensus).

    Returns:
        Validated entities sorted by ``(start, end, label)``.
    """
    validated: list[ValidatedEntity] = []
    seen: set[tuple[int, int, str]] = set()
    for item in proposed:
        label = normalize_entity_label(item.label)
        if label is None:
            continue
        surface = item.text.strip()
        if not surface:
            continue
        span = index_unique_span(text, surface)
        if span is None:
            continue
        start, end = span
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        validated.append(ValidatedEntity(text=surface, label=label, start=start, end=end))
    validated.sort(key=lambda e: (e.start, e.end, e.label))
    return validated
