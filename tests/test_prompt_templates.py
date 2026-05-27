"""Tests for prompt template rendering."""

from __future__ import annotations

import pytest

from ner_gold_generator.prompt_templates import PromptTemplateError, render_template


def test_render_template_substitutes_placeholders() -> None:
    out = render_template("Hello {name}, len={n}", name="world", n=3)
    assert out == "Hello world, len=3"


def test_render_template_missing_key_raises() -> None:
    with pytest.raises(PromptTemplateError):
        render_template("Missing {key}")
