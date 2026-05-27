"""Tests for structured LLM / Agno logging."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_gold_generator import llm_logging
from ner_gold_generator.synthetic_compose import collect_validation_issues
from ner_gold_generator.synthetic_generators import PlannedEntity


def test_log_llm_prompt_includes_doc_and_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "t.log"
    monkeypatch.setenv("NER_GOLD_LOG_FILE", str(log_path))
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "0")
    monkeypatch.setenv("NER_GOLD_LOG_LEVEL", "DEBUG")
    llm_logging._LOGGER.handlers.clear()
    llm_logging._LOGGER = llm_logging.get_logger("test_llm_logging_prompt")

    llm_logging.log_llm_prompt(
        model_id="m",
        system="sys",
        user="usr",
        temperature=0.1,
        engine="agno",
        document_id="doc-1",
        turn=2,
        turn_budget=5,
        prompt_kind="feedback",
    )
    log_text = log_path.read_text(encoding="utf-8")
    assert "doc=doc-1" in log_text
    assert "feedback" in log_text


def test_collect_validation_issues_includes_length() -> None:
    plan = [PlannedEntity("A", "model")]
    issues = collect_validation_issues("x", plan, min_chars=10, max_chars=20)
    assert any("length is" in i for i in issues)


def test_log_synthetic_failure_includes_length(
    tmp_path: Path,  # type: ignore[name-defined]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "fail.log"
    monkeypatch.setenv("NER_GOLD_LOG_FILE", str(log_path))
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "0")
    llm_logging._LOGGER.handlers.clear()
    llm_logging._LOGGER = llm_logging.get_logger("test_llm_logging_fail")

    llm_logging.log_synthetic_failure(
        document_id="d1",
        model_id="m",
        attempt=1,
        call_budget=3,
        failure_kind="validation",
        message="too short",
        text="hi",
        agent_mode="per_doc",
        turn=1,
        document_type="news_article",
    )
    text = log_path.read_text(encoding="utf-8")
    assert "document_type=news_article" in text
    assert "text length: 2 chars" in text
