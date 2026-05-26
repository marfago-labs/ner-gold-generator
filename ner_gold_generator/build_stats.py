"""Summarize gold JSONL rows after a build-gold run."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _int_stats(values: list[int]) -> tuple[int, float, int]:
    if not values:
        return 0, 0.0, 0
    return min(values), sum(values) / len(values), max(values)


def _format_label_counts(counts: Counter[str]) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{label}={counts[label]}" for label in sorted(counts))


def summarize_gold_build(
    examples: list[dict[str, Any]],
    *,
    source: str,
    mode: str,
) -> list[str]:
    """Return human-readable statistic lines for a completed build.

    Args:
        examples: Gold row dicts about to be written as JSONL.
        source: CLI ``--source`` value.
        mode: ``manual``, ``llm``, or ``synthetic``.

    Returns:
        Lines without trailing newlines (caller prints them).
    """
    n_docs = len(examples)
    if n_docs == 0:
        return ["statistics: (no examples produced)"]

    entity_counts: list[int] = []
    text_lengths: list[int] = []
    label_counts: Counter[str] = Counter()

    for row in examples:
        entities = row.get("entities")
        if isinstance(entities, list):
            entity_counts.append(len(entities))
            for ent in entities:
                if isinstance(ent, dict) and isinstance(ent.get("label"), str):
                    label_counts[ent["label"]] += 1
        text = row.get("text")
        if isinstance(text, str):
            text_lengths.append(len(text))

    total_entities = sum(entity_counts)
    ent_min, ent_avg, ent_max = _int_stats(entity_counts)
    txt_min, txt_avg, txt_max = _int_stats(text_lengths)

    lines = [
        "statistics:",
        f"  documents: {n_docs}",
        f"  entities: {total_entities} (avg {ent_avg:.1f}/doc, min {ent_min}, max {ent_max})",
        f"  text length (chars): avg {txt_avg:.0f}, min {txt_min}, max {txt_max}",
        f"  labels: {_format_label_counts(label_counts)}",
    ]

    if source == "synthetic":
        lines.extend(_synthetic_stats(examples))
    elif mode == "llm":
        lines.extend(_llm_labeling_stats(examples))

    return lines


def _synthetic_stats(examples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    meta0 = examples[0].get("synthetic_meta")
    if not isinstance(meta0, dict):
        return lines

    seed = meta0.get("seed")
    engine = meta0.get("engine")
    llm_model = meta0.get("llm_model")
    if seed is not None:
        lines.append(f"  synthetic seed: {seed}")
    if isinstance(engine, str) and engine:
        lines.append(f"  synthetic engine (config): {engine}")

    engines: Counter[str] = Counter()
    planned_total = 0
    used_total = 0
    missing_total = 0
    duplicate_surfaces = 0

    for row in examples:
        meta = row.get("synthetic_meta")
        if not isinstance(meta, dict):
            continue
        text_engine = meta.get("text_engine")
        if isinstance(text_engine, str):
            engines[text_engine] += 1
        usage = meta.get("entity_usage")
        if not isinstance(usage, dict):
            continue
        planned_total += int(usage.get("planned", 0))
        used_total += int(usage.get("used_exactly_once", 0))
        missing = usage.get("missing")
        if isinstance(missing, list):
            missing_total += len(missing)
        duplicates = usage.get("duplicate_in_text")
        if isinstance(duplicates, list):
            duplicate_surfaces += len(duplicates)

    if engines:
        engine_parts = ", ".join(f"{k}={engines[k]}" for k in sorted(engines))
        lines.append(f"  text engine (actual): {engine_parts}")
    if isinstance(llm_model, str) and llm_model:
        lines.append(f"  llm model: {llm_model}")
    if planned_total:
        lines.append(
            f"  entity plan: {used_total}/{planned_total} surfaces used exactly once "
            f"({missing_total} missing, {duplicate_surfaces} duplicate)"
        )

    return lines


def _llm_labeling_stats(examples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    sample_runs: list[int] = []
    min_votes: list[int] = []
    for row in examples:
        meta = row.get("llm_meta")
        if not isinstance(meta, dict):
            continue
        if isinstance(meta.get("sample_runs"), int):
            sample_runs.append(meta["sample_runs"])
        if isinstance(meta.get("min_votes"), int):
            min_votes.append(meta["min_votes"])
    if sample_runs:
        uniq = sorted(set(sample_runs))
        if len(uniq) == 1:
            lines.append(f"  llm sample runs per doc: {uniq[0]}")
        else:
            lines.append(f"  llm sample runs per doc: {uniq}")
    if min_votes:
        uniq = sorted(set(min_votes))
        if len(uniq) == 1:
            lines.append(f"  llm min votes: {uniq[0]}")
        else:
            lines.append(f"  llm min votes: {uniq}")
    return lines
