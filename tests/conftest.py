"""Shared pytest fixtures for ner-gold-generator."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.llm_test_helpers import make_llm_config

_REPO = Path(__file__).resolve().parent.parent
_PROMPTS = _REPO / "tests" / "fixtures" / "prompts"


@pytest.fixture
def synthetic_llm_prompts() -> dict[str, str]:
    """Inline LLM prompt strings for tests (mirrors config files)."""
    cfg = make_llm_config()
    return {k: str(v) for k, v in cfg.items() if k.endswith("_prompt") or k == "agent_mode"}


@pytest.fixture
def labeling_prompt_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point labeling prompts at test fixtures."""
    monkeypatch.setenv(
        "LLM_GOLD_SYSTEM_PROMPT_FILE",
        str(_PROMPTS / "labeling_system.txt"),
    )
    monkeypatch.setenv(
        "LLM_GOLD_USER_PROMPT_FILE",
        str(_PROMPTS / "labeling_user.txt"),
    )
