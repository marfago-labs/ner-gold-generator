"""Render prompt templates from config files with ``str.format`` placeholders."""

from __future__ import annotations


class PromptTemplateError(ValueError):
    """Raised when a template references a missing placeholder."""


def render_template(template: str, **kwargs: str | int) -> str:
    """Render *template* with ``str.format``; missing keys raise :class:`PromptTemplateError`."""
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise PromptTemplateError(f"Missing placeholder in prompt template: {exc}") from exc
