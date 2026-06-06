"""Unit tests for manual span parsing and anchoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner_gold_generator.manual_spans import ManualSpan, index_manual_span, parse_span_list

# --- parse_span_list ---


def test_parse_span_list_single_span() -> None:
    raw = [{"text": "Transformer", "label": "model"}]
    spans = parse_span_list(raw, doc_key="1706.03762")
    assert spans == [ManualSpan("Transformer", "model")]


def test_parse_span_list_with_match_and_strips_whitespace() -> None:
    raw = [
        {
            "text": "  GPT-3  ",
            "label": " model ",
            "match": "  GPT-3 achieves  ",
        }
    ]
    spans = parse_span_list(raw, doc_key="2005.14165")
    assert spans == [ManualSpan("GPT-3", "model", match="GPT-3 achieves")]


def test_parse_span_list_blank_match_treated_as_none() -> None:
    raw = [{"text": "BERT", "label": "model", "match": "   "}]
    spans = parse_span_list(raw, doc_key="x")
    assert spans[0].match is None


def test_parse_span_list_multiple_spans_preserves_order() -> None:
    raw = [
        {"text": "A", "label": "model"},
        {"text": "B", "label": "dataset"},
    ]
    spans = parse_span_list(raw, doc_key="doc")
    assert [s.text for s in spans] == ["A", "B"]


def test_parse_span_list_not_a_list_raises() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        parse_span_list({"text": "A", "label": "model"}, doc_key="bad")


def test_parse_span_list_invalid_entry_type_raises() -> None:
    with pytest.raises(ValueError, match="Invalid span entry"):
        parse_span_list(["not-a-dict"], doc_key="bad")


def test_parse_span_list_missing_text_or_label_raises() -> None:
    with pytest.raises(ValueError, match="text and label"):
        parse_span_list([{"text": "only-text"}], doc_key="bad")
    with pytest.raises(ValueError, match="text and label"):
        parse_span_list([{"label": "model"}], doc_key="bad")
    with pytest.raises(ValueError, match="text and label"):
        parse_span_list([{"text": 42, "label": "model"}], doc_key="bad")


# --- index_manual_span (no match) ---


def test_index_manual_span_unique_text_without_match() -> None:
    text = "The Transformer architecture is novel."
    span = ManualSpan("Transformer", "model")
    start, end = index_manual_span(text, span)
    assert text[start:end] == "Transformer"
    assert start == text.index("Transformer")


def test_index_manual_span_duplicate_text_without_match_raises() -> None:
    text = "GPT-3 here and GPT-3 there."
    span = ManualSpan("GPT-3", "model")
    with pytest.raises(ValueError, match="not found or not unique"):
        index_manual_span(text, span)


def test_index_manual_span_missing_text_raises() -> None:
    with pytest.raises(ValueError, match="not found or not unique"):
        index_manual_span("no entities here", ManualSpan("BERT", "model"))


# --- index_manual_span (with match) ---


def test_index_manual_span_match_selects_correct_occurrence() -> None:
    text = "We train GPT-3 early. Later GPT-3 achieves strong results."
    span = ManualSpan("GPT-3", "model", match="GPT-3 achieves")
    start, end = index_manual_span(text, span)
    assert text[start:end] == "GPT-3"
    assert start == text.index("GPT-3 achieves")
    assert start != text.index("GPT-3")


def test_index_manual_span_match_full_phrase_as_text() -> None:
    text = "Results: GPT-3 achieves state of the art."
    span = ManualSpan("GPT-3 achieves", "metric", match="GPT-3 achieves")
    start, end = index_manual_span(text, span)
    assert text[start:end] == "GPT-3 achieves"


def test_index_manual_span_duplicate_match_raises() -> None:
    text = "GPT-3 achieves X. Again GPT-3 achieves Y."
    span = ManualSpan("GPT-3", "model", match="GPT-3 achieves")
    with pytest.raises(ValueError, match="not found or not unique"):
        index_manual_span(text, span)


def test_index_manual_span_match_not_in_text_raises() -> None:
    text = "Only GPT-3 is mentioned."
    span = ManualSpan("GPT-3", "model", match="GPT-3 achieves")
    with pytest.raises(ValueError, match="not found or not unique"):
        index_manual_span(text, span)


def test_index_manual_span_text_not_inside_match_region_raises() -> None:
    text = "The phrase GPT-3 achieves results is here."
    # match is unique but entity text does not appear inside matched slice
    span = ManualSpan("BERT", "model", match="GPT-3 achieves")
    with pytest.raises(ValueError, match="not found within match"):
        index_manual_span(text, span)


def test_index_manual_span_text_offset_inside_match() -> None:
    text = "Score on GPT-3 achieves benchmark: 90%."
    span = ManualSpan("achieves", "metric", match="GPT-3 achieves")
    start, end = index_manual_span(text, span)
    assert text[start:end] == "achieves"


# --- integration with real GPT-3 abstract snippet ---


_GPT3_ABSTRACT_SNIPPET = (
    "Specifically, we train GPT-3, an autoregressive language model. "
    "For all tasks, GPT-3 is applied without fine-tuning. "
    "GPT-3 achieves strong performance on many NLP datasets."
)


def test_index_manual_span_gpt3_benchmark_offsets_match_ner_dataset() -> None:
    """Builtin config uses match='GPT-3 achieves' → span at second results mention."""
    span = ManualSpan("GPT-3", "model", match="GPT-3 achieves")
    start, end = index_manual_span(_GPT3_ABSTRACT_SNIPPET, span)
    assert _GPT3_ABSTRACT_SNIPPET[start:end] == "GPT-3"
    assert _GPT3_ABSTRACT_SNIPPET[start : start + len("GPT-3 achieves")] == "GPT-3 achieves"


def test_load_json_manual_file_with_match(tmp_path: Path) -> None:
    from ner_gold_generator.manual import load_manual_annotations

    path = tmp_path / "manual.json"
    path.write_text(
        json.dumps(
            {
                "arxiv-2005.14165": [
                    {"text": "GPT-3", "label": "model", "match": "GPT-3 achieves"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ann = load_manual_annotations(path)
    assert ann["arxiv-2005.14165"][0] == ManualSpan("GPT-3", "model", match="GPT-3 achieves")
    start, end = index_manual_span(_GPT3_ABSTRACT_SNIPPET, ann["arxiv-2005.14165"][0])
    assert _GPT3_ABSTRACT_SNIPPET[start:end] == "GPT-3"
