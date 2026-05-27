"""Tests for lexical diversity metrics."""

from __future__ import annotations

from ner_gold_generator.build_stats import summarize_gold_build
from ner_gold_generator.lexical_diversity import (
    compute_lexical_diversity,
    distinct_n,
    format_lexical_diversity_lines,
    tokenize_words,
)


def test_distinct_n_unigrams_and_bigrams() -> None:
    tokens = tokenize_words("the cat sat on the mat")
    assert distinct_n(tokens, 1) == distinct_n(["the", "cat", "sat", "on", "the", "mat"], 1)
    assert distinct_n(tokens, 1) < 1.0  # repeated "the"
    assert distinct_n(tokens, 2) <= 1.0
    assert distinct_n(tokens, 2) > 0.0


def test_distinct_n_empty() -> None:
    assert distinct_n([], 1) == 0.0
    assert distinct_n(["only"], 2) == 0.0


def test_compute_lexical_diversity_two_docs() -> None:
    examples = [
        {"text": "alpha beta gamma delta"},
        {"text": "alpha beta gamma epsilon"},
    ]
    stats = compute_lexical_diversity(examples)
    assert stats is not None
    assert len(stats.per_doc_distinct_1) == 2
    assert stats.corpus_token_count == 8
    assert stats.corpus_distinct_1 < stats.distinct_1_mean


def test_compute_lexical_diversity_skips_empty() -> None:
    assert compute_lexical_diversity([{"text": ""}, {"text": "   "}]) is None


def test_compute_lexical_diversity_skips_non_word_text() -> None:
    assert compute_lexical_diversity([{"text": "--- !!! ..."}]) is None


def test_summarize_includes_lexical_diversity() -> None:
    examples = [{"text": "one two three four five", "entities": []}]
    text = "\n".join(summarize_gold_build(examples, source="synthetic", mode="synthetic"))
    assert "lexical diversity" in text
    assert "distinct-1 per doc" in text
    assert "corpus:" in text


def test_format_lexical_diversity_lines() -> None:
    stats = compute_lexical_diversity([{"text": "hello world hello"}])
    assert stats is not None
    lines = format_lexical_diversity_lines(stats)
    assert any("not a gate" in line for line in lines)
