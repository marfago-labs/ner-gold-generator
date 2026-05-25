"""Tests for manual annotation loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner_gold_generator.manual import (
    load_manual_annotations,
    resolve_manual_spans,
)


def test_resolve_builtin_arxiv() -> None:
    spans = resolve_manual_spans(
        "arxiv-1706.03762",
        source="arxiv",
        file_annotations={},
    )
    assert spans is not None
    assert any(s[1] == "model" for s in spans)


def test_load_manual_json(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "doc-1": [{"text": "A", "label": "model"}],
            }
        ),
        encoding="utf-8",
    )
    ann = load_manual_annotations(path)
    assert ann["doc-1"] == [("A", "model")]


def test_load_manual_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manual_annotations(tmp_path / "nope.json")


def test_load_manual_invalid_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_manual_annotations(path)


def test_resolve_manual_file_key(tmp_path: Path) -> None:
    ann = {"doc-1": [("A", "model")]}
    assert resolve_manual_spans("doc-1", source="folder", file_annotations=ann) == [("A", "model")]


def test_load_manual_none_returns_empty() -> None:
    assert load_manual_annotations(None) == {}


def test_load_manual_invalid_span_entry(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": [{"text": "A"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="text and label"):
        load_manual_annotations(path)


def test_load_manual_yaml(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "doc-1:\n  - text: YAMLSpan\n    label: model\n",
        encoding="utf-8",
    )
    ann = load_manual_annotations(path)
    assert ann["doc-1"] == [("YAMLSpan", "model")]
