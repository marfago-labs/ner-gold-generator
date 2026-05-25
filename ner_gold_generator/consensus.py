"""Majority voting across multiple LLM annotation passes."""

from __future__ import annotations

from collections import Counter

from ner_gold_generator.types import ProposedEntity


def consensus_entities(
    runs: list[list[ProposedEntity]],
    *,
    min_votes: int,
) -> list[ProposedEntity]:
    """Keep ``(text, label)`` pairs that appear in at least *min_votes* runs.

    Each run may list the same pair at most once. Vote keys use stripped surface
    text and normalized labels. When multiple surface variants tie for a key,
    the most frequent surface string in agreeing runs is kept.

    Args:
        runs: One list of proposed entities per LLM pass.
        min_votes: Minimum number of runs that must contain the pair.

    Returns:
        De-duplicated proposed entities sorted by ``(text.lower(), label)``.

    Raises:
        ValueError: If *min_votes* is less than 1.
    """
    if min_votes < 1:
        raise ValueError("min_votes must be >= 1")
    if not runs:
        return []

    vote_counts: Counter[tuple[str, str]] = Counter()
    surface_counts: dict[tuple[str, str], Counter[str]] = {}

    for run in runs:
        seen_in_run: set[tuple[str, str]] = set()
        for ent in run:
            label = ent.label.strip().lower().replace(" ", "_")
            surface = ent.text.strip()
            if not surface or not label:
                continue
            key = (surface, label)
            if key in seen_in_run:
                continue
            seen_in_run.add(key)
            vote_counts[key] += 1
            surface_counts.setdefault(key, Counter())
            surface_counts[key][surface] += 1

    accepted: list[ProposedEntity] = []
    for (surface, label), count in vote_counts.items():
        if count < min_votes:
            continue
        best_surface = surface_counts[(surface, label)].most_common(1)[0][0]
        accepted.append(ProposedEntity(text=best_surface, label=label))

    accepted.sort(key=lambda e: (e.text.lower(), e.label))
    return accepted
