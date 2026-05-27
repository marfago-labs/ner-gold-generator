"""Helpers for LLM-related tests."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PROMPTS = _REPO / "tests" / "fixtures" / "prompts"


def make_llm_config(**overrides: object) -> dict[str, object]:
    """Minimal ``llm`` block for synthetic tests (stateless by default)."""
    base: dict[str, object] = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "agent_mode": "stateless",
        "system_prompt": (_PROMPTS / "synthetic_system.txt").read_text(encoding="utf-8"),
        "user_prompt": (_PROMPTS / "synthetic_user.txt").read_text(encoding="utf-8"),
        "feedback_prompt": (_PROMPTS / "synthetic_feedback.txt").read_text(encoding="utf-8"),
    }
    base.update(overrides)
    return base
