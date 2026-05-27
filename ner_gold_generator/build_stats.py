"""Summarize gold JSONL rows after a build-gold run."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ner_gold_generator.lexical_diversity import (
    compute_lexical_diversity,
    format_lexical_diversity_lines,
)
from ner_gold_generator.progress import format_skip_error


def _int_stats(values: list[int]) -> tuple[int, float, int]:
    if not values:
        return 0, 0.0, 0
    return min(values), sum(values) / len(values), max(values)


def _format_label_counts(counts: Counter[str]) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{label}={counts[label]}" for label in sorted(counts))


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def summarize_gold_build(
    examples: list[dict[str, Any]],
    *,
    source: str,
    mode: str,
    elapsed_seconds: float | None = None,
    requested_documents: int | None = None,
    skipped_documents: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return human-readable statistic lines for a completed build.

    Args:
        examples: Gold row dicts about to be written as JSONL.
        source: CLI ``--source`` value.
        mode: ``manual``, ``llm``, or ``synthetic``.
        elapsed_seconds: Wall-clock time for the build (if known).
        requested_documents: Planned document count (for partial runs).
        skipped_documents: Synthetic documents skipped after LLM retries.

    Returns:
        Lines without trailing newlines (caller prints them). When *examples*
        contain ``text`` fields, appends lexical diversity (distinct-1/2) lines
        from :mod:`ner_gold_generator.lexical_diversity` (report-only).
    """
    n_docs = len(examples)
    if n_docs == 0:
        lines = ["statistics: (no examples produced)"]
        if requested_documents is not None:
            lines.append(f"  requested documents: {requested_documents}")
        if elapsed_seconds is not None:
            lines.append(f"  elapsed: {_format_elapsed(elapsed_seconds)}")
        return lines

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
    ]
    if requested_documents is not None and requested_documents != n_docs:
        lines.append(f"  completed: {n_docs}/{requested_documents} documents")
    if elapsed_seconds is not None:
        lines.append(f"  elapsed: {_format_elapsed(elapsed_seconds)}")
        if n_docs > 0:
            lines.append(f"  throughput: {elapsed_seconds / n_docs:.1f}s per document")
    lines.extend(
        [
            f"  entities: {total_entities} (avg {ent_avg:.1f}/doc, min {ent_min}, max {ent_max})",
            f"  text length (chars): avg {txt_avg:.0f}, min {txt_min}, max {txt_max}",
            f"  labels: {_format_label_counts(label_counts)}",
        ]
    )

    if skipped_documents:
        lines.append(f"  skipped documents: {len(skipped_documents)}")
        for skip in skipped_documents[:5]:
            doc_id = skip.get("id", "?")
            err = format_skip_error(str(skip.get("error", "")))
            lines.append(f"    - {doc_id}: {err}")
        if len(skipped_documents) > 5:
            lines.append(f"    ... and {len(skipped_documents) - 5} more")

    diversity = compute_lexical_diversity(examples)
    if diversity is not None:
        lines.extend(format_lexical_diversity_lines(diversity))

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
    parallelism = meta0.get("llm_parallelism")
    if seed is not None:
        lines.append(f"  synthetic seed: {seed}")
    if isinstance(engine, str) and engine:
        lines.append(f"  synthetic engine (config): {engine}")
    if isinstance(parallelism, int):
        lines.append(f"  llm parallelism: {parallelism}")

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
