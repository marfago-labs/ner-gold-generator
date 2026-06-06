"""Tests for procedural synthetic default YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_gold_generator.synthetic_config import (
    _DEFAULT_LEXICON,
    _DEFAULT_TEMPLATES,
    default_synthetic_defaults_path,
    load_synthetic_defaults,
    parse_synthetic_config_dict,
)


def test_default_synthetic_defaults_path() -> None:
    path = default_synthetic_defaults_path()
    assert path.name == "synthetic_defaults.yaml"
    assert path.is_file()


def test_load_synthetic_defaults() -> None:
    templates, lexicon = load_synthetic_defaults()
    assert len(templates) >= 4
    assert "Transformer" in lexicon["model"]
    assert templates == _DEFAULT_TEMPLATES
    assert lexicon == _DEFAULT_LEXICON


def test_parse_synthetic_config_uses_yaml_defaults() -> None:
    cfg = parse_synthetic_config_dict({})
    assert cfg.templates == _DEFAULT_TEMPLATES
    assert "GPT-3" in cfg.lexicon["model"]


def test_load_synthetic_defaults_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_synthetic_defaults(tmp_path / "nope.yaml")


def test_load_synthetic_defaults_empty_lexicon(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("templates:\n  - '{e0}'\nlexicon: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lexicon"):
        load_synthetic_defaults(path)
