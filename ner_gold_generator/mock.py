"""Deterministic entity extraction for tests and ``--mock`` LLM mode."""

from __future__ import annotations

import re

from ner_gold_generator.types import ProposedEntity

_KNOWN: tuple[tuple[str, str], ...] = (
    ("GPT-3", "model"),
    ("BERT", "model"),
    ("CLIP", "model"),
    ("ImageNet", "dataset"),
    ("GLUE", "dataset"),
    ("BLEU", "metric"),
    ("OpenAI", "organization"),
)


def mock_extract_entities(text: str) -> list[ProposedEntity]:
    """Scan *text* for known ML tokens and unique four-digit years.

    Used by :class:`llm_client.MockChatClient` and direct unit tests. Does not
    call external APIs.

    Args:
        text: Document body to scan.

    Returns:
        Proposed entities for each known token present and each year that
        appears exactly once in *text*.
    """
    found: list[ProposedEntity] = []
    seen: set[tuple[str, str]] = set()
    for surface, label in _KNOWN:
        if surface not in text:
            continue
        key = (surface, label)
        if key in seen:
            continue
        seen.add(key)
        found.append(ProposedEntity(text=surface, label=label))
    for match in re.finditer(r"\b(19|20)\d{2}\b", text):
        year = match.group(0)
        key = (year, "year")
        if key in seen:
            continue
        if text.count(year) == 1:
            seen.add(key)
            found.append(ProposedEntity(text=year, label="year"))
    return found
