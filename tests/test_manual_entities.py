"""Tests for arXiv benchmark YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ner_gold_generator.manual_entities import (
    DEFAULT_ARXIV_BENCHMARK_PATH,
    PAPER_ENTITIES,
    PAPER_ORDER,
    default_arxiv_benchmark_path,
    load_arxiv_benchmark,
)
from ner_gold_generator.manual_spans import index_manual_span, parse_span_list


def test_default_arxiv_benchmark_path() -> None:
    path = default_arxiv_benchmark_path()
    assert path.name == "arxiv_benchmark.yaml"
    assert path.is_file()


def test_load_arxiv_benchmark_builtin() -> None:
    cfg = load_arxiv_benchmark()
    assert len(cfg.paper_order) == 10
    assert "1706.03762" in cfg.paper_entities
    gpt3 = cfg.paper_entities["2005.14165"][0]
    assert gpt3.text == "GPT-3"
    assert gpt3.label == "model"
    assert gpt3.match == "GPT-3 achieves"


def test_module_attrs_match_config() -> None:
    cfg = load_arxiv_benchmark()
    assert list(cfg.paper_order) == PAPER_ORDER
    assert PAPER_ENTITIES["1706.03762"][0].label == "model"
    assert len(PAPER_ENTITIES) == len(cfg.paper_entities)


def test_all_builtin_papers_have_entities() -> None:
    cfg = load_arxiv_benchmark()
    for arxiv_id in cfg.paper_order:
        assert arxiv_id in cfg.paper_entities
        assert len(cfg.paper_entities[arxiv_id]) >= 1


def test_builtin_transformer_span_uses_canonical_surface() -> None:
    """Canonical model name with match key for the introducing phrase."""
    span = load_arxiv_benchmark().paper_entities["1706.03762"][0]
    assert span.text == "Transformer"
    assert span.match == "network architecture, the Transformer"


def test_load_arxiv_benchmark_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_arxiv_benchmark(tmp_path / "missing.yaml")


def test_load_arxiv_benchmark_top_level_not_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not a dict\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_arxiv_benchmark(path)


def test_load_arxiv_benchmark_invalid_order(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("papers:\n  '1':\n    - text: A\n      label: model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paper_order"):
        load_arxiv_benchmark(path)


def test_load_arxiv_benchmark_empty_papers_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump({"paper_order": ["1"], "papers": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="papers must be a non-empty"):
        load_arxiv_benchmark(path)


def test_load_arxiv_benchmark_order_without_papers(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump({"paper_order": ["1"], "papers": {"2": [{"text": "A", "label": "model"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing from papers"):
        load_arxiv_benchmark(path)


def test_load_arxiv_benchmark_invalid_span_in_paper(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump(
            {
                "paper_order": ["1"],
                "papers": {"1": [{"text": "A"}]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text and label"):
        load_arxiv_benchmark(path)


def test_load_arxiv_benchmark_custom_yaml_with_match(tmp_path: Path) -> None:
    text = "Alpha GPT-3 beta. Gamma GPT-3 achieves delta."
    path = tmp_path / "bench.yaml"
    path.write_text(
        yaml.dump(
            {
                "paper_order": ["99.00001"],
                "papers": {
                    "99.00001": [
                        {"text": "GPT-3", "label": "model", "match": "GPT-3 achieves"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_arxiv_benchmark(path)
    span = cfg.paper_entities["99.00001"][0]
    start, end = index_manual_span(text, span)
    assert text[start:end] == "GPT-3"
    assert text[start : start + len("GPT-3 achieves")] == "GPT-3 achieves"


def test_builtin_config_is_yaml() -> None:
    assert DEFAULT_ARXIV_BENCHMARK_PATH.suffix.lower() in {".yaml", ".yml"}


def test_parse_span_list_reexported_via_benchmark_loader(tmp_path: Path) -> None:
    """Benchmark loader uses the same span parser as manual files."""
    spans = parse_span_list(
        [{"text": "ViT", "label": "model", "match": "Vision Transformer (ViT)"}],
        doc_key="x",
    )
    assert spans[0].match == "Vision Transformer (ViT)"
