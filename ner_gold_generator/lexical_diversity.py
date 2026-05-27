"""Lexical diversity metrics for generated gold text (reporting only, not gating)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens for diversity statistics."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def distinct_n(tokens: list[str], n: int) -> float:
    """Return unique n-grams / total n-grams in *tokens* (0.0 when undefined)."""
    if n < 1 or len(tokens) < n:
        return 0.0
    if n == 1:
        return len(set(tokens)) / len(tokens)
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


@dataclass(frozen=True, slots=True)
class LexicalDiversityStats:
    """Distinct-1/2 per document and over the full corpus."""

    per_doc_distinct_1: tuple[float, ...]
    per_doc_distinct_2: tuple[float, ...]
    corpus_distinct_1: float
    corpus_distinct_2: float
    corpus_token_count: int

    @property
    def distinct_1_mean(self) -> float:
        if not self.per_doc_distinct_1:
            return 0.0
        return sum(self.per_doc_distinct_1) / len(self.per_doc_distinct_1)

    @property
    def distinct_2_mean(self) -> float:
        if not self.per_doc_distinct_2:
            return 0.0
        return sum(self.per_doc_distinct_2) / len(self.per_doc_distinct_2)


def compute_lexical_diversity(examples: list[dict[str, Any]]) -> LexicalDiversityStats | None:
    """Compute distinct-1/2 for all string ``text`` fields in *examples*.

    Returns ``None`` when no non-empty text rows exist. Used by
    :func:`ner_gold_generator.build_stats.summarize_gold_build` for stderr
    statistics only; failures never depend on these scores.
    """
    per_doc_d1: list[float] = []
    per_doc_d2: list[float] = []
    corpus_tokens: list[str] = []
    for row in examples:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = tokenize_words(text)
        if not tokens:
            continue
        per_doc_d1.append(distinct_n(tokens, 1))
        per_doc_d2.append(distinct_n(tokens, 2))
        corpus_tokens.extend(tokens)
    if not per_doc_d1:
        return None
    return LexicalDiversityStats(
        per_doc_distinct_1=tuple(per_doc_d1),
        per_doc_distinct_2=tuple(per_doc_d2),
        corpus_distinct_1=distinct_n(corpus_tokens, 1),
        corpus_distinct_2=distinct_n(corpus_tokens, 2),
        corpus_token_count=len(corpus_tokens),
    )


def format_lexical_diversity_lines(stats: LexicalDiversityStats) -> list[str]:
    """Human-readable summary lines for :func:`summarize_gold_build`."""
    d1 = stats.per_doc_distinct_1
    d2 = stats.per_doc_distinct_2
    d1_min, d1_max = min(d1), max(d1)
    d2_min, d2_max = min(d2), max(d2)
    return [
        "  lexical diversity (quality, not a gate):",
        (
            f"    distinct-1 per doc: mean {stats.distinct_1_mean:.3f} "
            f"(min {d1_min:.3f}, max {d1_max:.3f})"
        ),
        (
            f"    distinct-2 per doc: mean {stats.distinct_2_mean:.3f} "
            f"(min {d2_min:.3f}, max {d2_max:.3f})"
        ),
        (
            f"    corpus: distinct-1={stats.corpus_distinct_1:.3f} "
            f"distinct-2={stats.corpus_distinct_2:.3f} "
            f"({stats.corpus_token_count} tokens)"
        ),
    ]
