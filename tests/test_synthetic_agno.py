"""Tests for Agno per-document synthetic generation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.llm_test_helpers import make_llm_config

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.synthetic_agno import (
    extract_run_text,
    generate_text_with_agno_async,
    parse_text_from_run,
)
from ner_gold_generator.synthetic_compose import assemble_mock_llm_text
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict
from ner_gold_generator.synthetic_generators import PlannedEntity


class _FakeRun:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeAgent:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.messages: list[str] = []

    async def arun(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> _FakeRun:
        del session_id, user_id
        self.messages.append(message)
        raw = self._responses.pop(0)
        return _FakeRun(raw)


def test_extract_run_text_from_json_string() -> None:
    run = _FakeRun('{"text": "Hello BERT world"}')
    text, raw = parse_text_from_run(run)
    assert "Hello" in text
    assert raw


def test_agno_feedback_loop_fixes_entities() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(
                agent_mode="per_doc",
                provider_max_attempts=1,
                validation_max_attempts=3,
            ),
            "text_length": {"min_chars": 10, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model"), PlannedEntity("GLUE", "dataset")]
    good = assemble_mock_llm_text(plan, config)
    bad = '{"text": "no entities"}'

    def factory() -> _FakeAgent:
        return _FakeAgent([bad, good])

    text = asyncio.run(
        generate_text_with_agno_async(
            plan,
            config,
            settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
            document_id="synthetic-test-agno-001",
            document_type="scientific_abstract",
            style_description="Scientific abstract",
            agent_factory=factory,
        )
    )
    assert "BERT" in text


def test_agno_exhaustion_raises() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(
                agent_mode="per_doc",
                provider_max_attempts=1,
                validation_max_attempts=2,
            ),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    def factory() -> _FakeAgent:
        return _FakeAgent(['{"text": "missing"}', '{"text": "still missing"}'])

    with pytest.raises(ValueError, match="reason=validation"):
        asyncio.run(
            generate_text_with_agno_async(
                plan,
                config,
                settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
                document_id="synthetic-test-agno-002",
                document_type="news_article",
                style_description="News",
                agent_factory=factory,
            )
        )


def test_extract_run_text_plain_string() -> None:
    assert extract_run_text("plain") == "plain"


def test_build_agno_sqlite_db_uses_db_url_for_memory() -> None:
    from unittest.mock import MagicMock, patch

    from ner_gold_generator.synthetic_agno import build_agno_sqlite_db
    from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

    pytest.importorskip("agno")
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(agno_db_path=":memory:")}
    )
    with patch("agno.db.sqlite.SqliteDb", MagicMock()) as mock_db:
        build_agno_sqlite_db(config, document_id="doc-1")
    mock_db.assert_called_once_with(db_url="sqlite:///:memory:")


def test_build_agno_sqlite_db_per_document_file(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from ner_gold_generator.synthetic_agno import build_agno_sqlite_db
    from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

    pytest.importorskip("agno")
    db_file = tmp_path / "sessions.db"
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(agno_db_path=str(db_file))}
    )
    with patch("agno.db.sqlite.SqliteDb", MagicMock()) as mock_db:
        build_agno_sqlite_db(config, document_id="synthetic-00001-00002")
    called_path = mock_db.call_args.kwargs["db_file"]
    assert "synthetic-00001-00002" in str(called_path)


def test_extract_run_text_none_and_pydantic() -> None:
    from pydantic import BaseModel

    class _Out(BaseModel):
        text: str = "hi"

    class _Run:
        content = _Out()

    assert extract_run_text(None) == ""
    assert "hi" in extract_run_text(_Run())


def test_extract_run_text_from_messages() -> None:
    class _Msg:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    class _Run:
        messages = [_Msg("user", "hi"), _Msg("assistant", '{"text": "from messages"}')]

    text, _ = parse_text_from_run(_Run())
    assert "from messages" in text


def test_extract_run_text_from_dict_messages() -> None:
    class _Run:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": '{"text": "dict msg"}'},
        ]

    text, _ = parse_text_from_run(_Run())
    assert "dict msg" in text


def test_agno_sync_run_path() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(agent_mode="per_doc", provider_max_attempts=1),
            "text_length": {"min_chars": 10, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = assemble_mock_llm_text(plan, config)

    class _SyncAgent:
        def run(
            self, message: str, *, session_id: str | None = None, user_id: str | None = None
        ) -> _FakeRun:
            del message, session_id, user_id
            return _FakeRun(good)

    text = asyncio.run(
        generate_text_with_agno_async(
            plan,
            config,
            settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
            document_id="sync-agent",
            document_type="blog_post",
            style_description="Blog",
            agent_factory=lambda: _SyncAgent(),
        )
    )
    assert "BERT" in text


def test_agno_non_retryable_provider_fails() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(agent_mode="per_doc", provider_max_attempts=2),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    class _FailAgent:
        async def arun(
            self, message: str, *, session_id: str | None = None, user_id: str | None = None
        ) -> object:
            del message, session_id, user_id
            raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="reason=provider"):
        asyncio.run(
            generate_text_with_agno_async(
                plan,
                config,
                settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
                document_id="fail-fast",
                document_type="news_article",
                style_description="News",
                agent_factory=lambda: _FailAgent(),
            )
        )


def test_extract_run_text_dict_content() -> None:
    class _Run:
        content = {"text": "dict body"}

    text, _ = parse_text_from_run(_Run())
    assert "dict body" in text


def test_agno_rate_limit_retries_then_succeeds() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(agent_mode="per_doc", provider_max_attempts=2),
            "text_length": {"min_chars": 10, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = assemble_mock_llm_text(plan, config)
    rate_err = openai.RateLimitError(
        "rate",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
        body=None,
    )

    class _RateAgent:
        def __init__(self) -> None:
            self._n = 0

        async def arun(
            self, message: str, *, session_id: str | None = None, user_id: str | None = None
        ) -> _FakeRun:
            del message, session_id, user_id
            self._n += 1
            if self._n == 1:
                raise rate_err
            return _FakeRun(good)

    text = asyncio.run(
        generate_text_with_agno_async(
            plan,
            config,
            settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
            document_id="rate-retry",
            document_type="transcript",
            style_description="Transcript",
            agent_factory=lambda: _RateAgent(),
            on_status=lambda _: None,
        )
    )
    assert "BERT" in text
