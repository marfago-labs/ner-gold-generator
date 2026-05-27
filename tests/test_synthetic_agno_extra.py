"""Additional coverage for synthetic_agno and compose helpers."""

from __future__ import annotations

import asyncio

import pytest
from tests.llm_test_helpers import make_llm_config

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.synthetic_agno import (
    generate_text_with_agno_async,
    parse_text_from_run,
    verify_agno_installation,
)
from ner_gold_generator.synthetic_compose import (
    build_validation_feedback,
    collect_validation_issues,
    generate_text_with_llm_async,
)
from ner_gold_generator.synthetic_config import load_synthetic_config, parse_synthetic_config_dict
from ner_gold_generator.synthetic_generators import PlannedEntity

_REPO = __import__("pathlib").Path(__file__).resolve().parent.parent


class _FakeRun:
    def __init__(self, content: str) -> None:
        self.content = content


class _FlakyAgent:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)

    async def arun(
        self, message: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> object:
        del message, session_id, user_id
        item = self._outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _FakeRun(str(item))


def test_build_validation_feedback_renders_issues() -> None:
    config = parse_synthetic_config_dict({"llm": make_llm_config()})
    msg = build_validation_feedback(
        ["length is 5 characters, target is 10-20 characters"],
        config=config,
        min_chars=10,
        max_chars=20,
    )
    assert "length is 5" in msg
    assert "10-20" in msg


def test_agno_provider_retry_then_success() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(
                agent_mode="per_doc",
                provider_max_attempts=2,
                validation_max_attempts=2,
            ),
            "text_length": {"min_chars": 5, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = '{"text": "We report results for BERT on GLUE with strong scores."}'
    rate_err = openai.RateLimitError(
        "rate",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
        body=None,
    )

    def factory() -> _FlakyAgent:
        return _FlakyAgent([rate_err, good])

    text = asyncio.run(
        generate_text_with_agno_async(
            plan,
            config,
            settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
            document_id="doc-retry",
            document_type="news",
            style_description="News",
            agent_factory=factory,
        )
    )
    assert "BERT" in text


def test_parse_text_from_run_non_json_string() -> None:
    text, raw = parse_text_from_run(_FakeRun("plain prose BERT only"))
    assert text == "plain prose BERT only"
    assert raw == "plain prose BERT only"


def test_load_repo_synthetic_llm_config() -> None:
    cfg = load_synthetic_config(_REPO / "configs" / "synthetic_llm.json")
    assert cfg.llm.agent_mode == "per_doc"
    assert cfg.llm.user_prompt
    assert cfg.llm.feedback_prompt


def test_stateless_async_generate() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(agent_mode="stateless"),
            "text_length": {"min_chars": 5, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    class _Client:
        async def complete_json(
            self, *, system: str, user: str, temperature: float, call_label: str | None = None
        ) -> dict:
            del system, user, temperature, call_label
            return {"text": "We study BERT on GLUE with additional context for length."}

    text = asyncio.run(
        generate_text_with_llm_async(
            plan,
            config,
            settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
            mock=False,
            client=_Client(),
            document_id="doc-stateless",
        )
    )
    assert "BERT" in text


def test_verify_agno_installation_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import ner_gold_generator.synthetic_agno as agno_mod

    monkeypatch.setattr(agno_mod, "Agent", None)
    with pytest.raises(ImportError, match="Agno is required"):
        verify_agno_installation()


def test_create_synthetic_agent_uses_factory() -> None:
    from ner_gold_generator.synthetic_agno import create_synthetic_agent

    class _Stub:
        pass

    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(agent_mode="per_doc")}
    )
    agent = create_synthetic_agent(
        config,
        LlmGoldSettings.model_construct(openrouter_api_key="k"),
        document_id="doc-1",
        agent_factory=lambda: _Stub(),
    )
    assert isinstance(agent, _Stub)


def test_collect_validation_issues_empty_text() -> None:
    issues = collect_validation_issues("", [PlannedEntity("A", "model")], min_chars=1, max_chars=99)
    assert issues[0] == "empty text from LLM"


def test_collect_validation_issues_too_long() -> None:
    issues = collect_validation_issues(
        "x" * 50, [PlannedEntity("x", "model")], min_chars=1, max_chars=10
    )
    assert any("length is" in i for i in issues)
