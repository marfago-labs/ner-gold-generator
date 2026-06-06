"""Tests for manual annotation loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner_gold_generator.manual import (
    ManualSpan,
    load_manual_annotations,
    resolve_manual_spans,
)
from ner_gold_generator.manual_entities import PAPER_ENTITIES


def test_resolve_builtin_arxiv() -> None:
    spans = resolve_manual_spans(
        "arxiv-1706.03762",
        source="arxiv",
        file_annotations={},
    )
    assert spans is not None
    assert any(s.label == "model" for s in spans)


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
    assert ann["doc-1"] == [ManualSpan("A", "model")]


def test_load_manual_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manual_annotations(tmp_path / "nope.json")


def test_load_manual_invalid_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_manual_annotations(path)


def test_resolve_manual_file_key(tmp_path: Path) -> None:
    ann = {"doc-1": [ManualSpan("A", "model")]}
    assert resolve_manual_spans("doc-1", source="folder", file_annotations=ann) == [
        ManualSpan("A", "model")
    ]


def test_load_manual_none_returns_empty() -> None:
    assert load_manual_annotations(None) == {}


def test_load_manual_invalid_span_entry(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": [{"text": "A"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="text and label"):
        load_manual_annotations(path)


def test_load_manual_yaml_import_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("doc-1:\n  - text: A\n    label: model\n", encoding="utf-8")

    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match="PyYAML"):
        load_manual_annotations(path)


def test_load_manual_top_level_not_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_manual_annotations(path)


def test_resolve_manual_bare_arxiv_key() -> None:
    ann = {"1706.03762": [ManualSpan("Transformer", "model")]}
    spans = resolve_manual_spans("arxiv-1706.03762", source="arxiv", file_annotations=ann)
    assert spans == [ManualSpan("Transformer", "model")]


def test_resolve_manual_invalid_span_type(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": ["not-a-dict"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid span"):
        load_manual_annotations(path)


def test_load_manual_yaml_with_match_field(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "arxiv-2005.14165:\n  - text: GPT-3\n    label: model\n    match: GPT-3 achieves\n",
        encoding="utf-8",
    )
    ann = load_manual_annotations(path)
    assert ann["arxiv-2005.14165"] == [ManualSpan("GPT-3", "model", match="GPT-3 achieves")]


def test_resolve_manual_file_overrides_builtin() -> None:
    file_ann = {
        "1706.03762": [ManualSpan("OVERRIDE", "model")],
    }
    spans = resolve_manual_spans(
        "arxiv-1706.03762",
        source="arxiv",
        file_annotations=file_ann,
    )
    assert spans == [ManualSpan("OVERRIDE", "model")]


def test_resolve_manual_unknown_doc_returns_none() -> None:
    assert (
        resolve_manual_spans(
            "arxiv-9999.99999",
            source="arxiv",
            file_annotations={},
        )
        is None
    )


def test_builtin_arxiv_annotations_returns_mutable_copy() -> None:
    from ner_gold_generator.manual import builtin_arxiv_annotations

    first = builtin_arxiv_annotations()
    second = builtin_arxiv_annotations()
    assert first is not second
    first["1706.03762"].append(ManualSpan("X", "model"))
    assert len(second["1706.03762"]) == len(PAPER_ENTITIES["1706.03762"])


def test_load_manual_yaml(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "doc-1:\n  - text: YAMLSpan\n    label: model\n",
        encoding="utf-8",
    )
    ann = load_manual_annotations(path)
    assert ann["doc-1"] == [ManualSpan("YAMLSpan", "model")]
