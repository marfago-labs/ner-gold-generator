"""Tests for post-build gold statistics."""

from __future__ import annotations

from ner_gold_generator.build_stats import summarize_gold_build


def test_summarize_empty() -> None:
    lines = summarize_gold_build([], source="synthetic", mode="synthetic")
    assert lines == ["statistics: (no examples produced)"]


def test_summarize_includes_skipped_documents() -> None:
    examples = [{"text": "ok", "entities": []}]
    skipped = [
        {
            "id": "synthetic-00000042-00003",
            "document_index": 3,
            "document_seed": 1,
            "error": "Surface 'pre-training' appears 2 times (expected 1)",
        }
    ]
    text = "\n".join(
        summarize_gold_build(
            examples,
            source="synthetic",
            mode="synthetic",
            skipped_documents=skipped,
        )
    )
    assert "skipped documents: 1" in text
    assert "synthetic-00000042-00003" in text
    assert "appears 2 times" in text


def test_summarize_includes_elapsed_and_throughput() -> None:
    examples = [{"text": "hi", "entities": []}]
    lines = summarize_gold_build(
        examples,
        source="synthetic",
        mode="synthetic",
        elapsed_seconds=12.5,
        requested_documents=5,
    )
    text = "\n".join(lines)
    assert "elapsed: 12.5s" in text
    assert "throughput: 12.5s per document" in text
    assert "completed: 1/5 documents" in text


def test_summarize_synthetic_rows() -> None:
    examples = [
        {
            "text": "a" * 200,
            "entities": [
                {"text": "BERT", "label": "model"},
                {"text": "GLUE", "label": "dataset"},
            ],
            "synthetic_meta": {
                "seed": 42,
                "engine": "llm",
                "llm_model": "meta-llama/llama-3.3-70b-instruct:free",
                "text_engine": "llm",
                "entity_usage": {
                    "planned": 2,
                    "used_exactly_once": 2,
                    "missing": [],
                    "duplicate_in_text": [],
                },
            },
        },
        {
            "text": "b" * 300,
            "entities": [{"text": "ViT", "label": "model"}],
            "synthetic_meta": {
                "seed": 42,
                "engine": "llm",
                "llm_model": "meta-llama/llama-3.3-70b-instruct:free",
                "text_engine": "llm",
                "entity_usage": {
                    "planned": 1,
                    "used_exactly_once": 1,
                    "missing": [],
                    "duplicate_in_text": [],
                },
            },
        },
    ]
    lines = summarize_gold_build(examples, source="synthetic", mode="synthetic")
    text = "\n".join(lines)
    assert "documents: 2" in text
    assert "entities: 3" in text
    assert "model=2" in text
    assert "dataset=1" in text
    assert "entity plan: 3/3" in text
    assert "llm model: meta-llama" in text


def test_summarize_synthetic_with_plan_gaps() -> None:
    examples = [
        {
            "text": "x",
            "entities": [],
            "synthetic_meta": {
                "seed": 1,
                "engine": "llm",
                "text_engine": "procedural",
                "entity_usage": {
                    "planned": 3,
                    "used_exactly_once": 2,
                    "missing": ["BERT"],
                    "duplicate_in_text": ["GLUE"],
                },
            },
        }
    ]
    text = "\n".join(summarize_gold_build(examples, source="synthetic", mode="synthetic"))
    assert "2/3 surfaces" in text
    assert "1 missing" in text
    assert "1 duplicate" in text
    assert "procedural=1" in text


def test_summarize_llm_labeling_meta() -> None:
    examples = [
        {
            "text": "t",
            "entities": [{"text": "A", "label": "model"}],
            "llm_meta": {"sample_runs": 3, "min_votes": 2},
        },
        {
            "text": "u",
            "entities": [],
            "llm_meta": {"sample_runs": 5, "min_votes": 2},
        },
    ]
    text = "\n".join(summarize_gold_build(examples, source="arxiv", mode="llm"))
    assert "llm sample runs per doc: [3, 5]" in text
    assert "llm min votes: 2" in text


def test_summarize_manual_rows() -> None:
    examples = [
        {
            "text": "OpenAI released GPT-3.",
            "entities": [{"text": "OpenAI", "label": "organization"}],
        }
    ]
    lines = summarize_gold_build(examples, source="arxiv", mode="manual")
    assert any("documents: 1" in line for line in lines)
    assert any("organization=1" in line for line in lines)
    assert not any("synthetic seed" in line for line in lines)
