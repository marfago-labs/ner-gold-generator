"""Async coverage for generate_text_with_llm_async stateless paths."""

from __future__ import annotations

import asyncio

import pytest
from tests.llm_test_helpers import make_llm_config

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.llm_client import InvalidLlmResponseError, LlmRequestError
from ner_gold_generator.synthetic_compose import (
    assemble_mock_llm_text,
    generate_text_with_llm_async,
    wait_for_retry_async,
)
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict
from ner_gold_generator.synthetic_generators import PlannedEntity


class _AsyncClient:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls = 0

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict:
        del system, user, temperature, call_label
        self.calls += 1
        return await self._handler(self)


def _run(coro):
    return asyncio.run(coro)


def test_async_mock_shortcut() -> None:
    config = parse_synthetic_config_dict({"engine": "llm", "llm": make_llm_config()})
    plan = [PlannedEntity("BERT", "model")]
    text = _run(generate_text_with_llm_async(plan, config, settings=LlmGoldSettings(), mock=True))
    assert text


def test_async_requires_document_id_for_per_doc() -> None:
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(agent_mode="per_doc")}
    )
    with pytest.raises(ValueError, match="document_id"):
        _run(
            generate_text_with_llm_async(
                [PlannedEntity("A", "model")],
                config,
                settings=LlmGoldSettings.model_construct(openrouter_api_key="k"),
                mock=False,
            )
        )


def test_async_requires_client_for_stateless() -> None:
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(agent_mode="stateless")}
    )
    with pytest.raises(ValueError, match="AsyncChatClient"):
        _run(
            generate_text_with_llm_async(
                [PlannedEntity("A", "model")],
                config,
                settings=LlmGoldSettings(),
                mock=False,
            )
        )


def test_async_validation_budget() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=5, validation_max_attempts=2),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    async def handler(_self):
        return {"text": "no entities here at all"}

    client = _AsyncClient(handler)
    with pytest.raises(ValueError, match="reason=validation"):
        _run(
            generate_text_with_llm_async(
                plan,
                config,
                settings=LlmGoldSettings(),
                mock=False,
                client=client,
                document_id="d1",
            )
        )
    assert client.calls == 2


def test_async_provider_retry_exhausted() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=2, validation_max_attempts=5),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    async def handler(_self):
        raise LlmRequestError("rate limit", retryable=True, retry_after_seconds=0.0)

    client = _AsyncClient(handler)
    waits: list[float] = []

    with pytest.raises(ValueError, match="reason=provider"):
        _run(
            generate_text_with_llm_async(
                plan,
                config,
                settings=LlmGoldSettings(),
                mock=False,
                client=client,
                document_id="d2",
                on_wait=lambda remaining, msg: waits.append(remaining),
            )
        )
    assert client.calls == 2


def test_async_non_retryable_provider() -> None:
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(provider_max_attempts=3)}
    )
    plan = [PlannedEntity("BERT", "model")]

    async def handler(_self):
        raise LlmRequestError("auth failed", retryable=False)

    client = _AsyncClient(handler)
    with pytest.raises(ValueError, match="reason=provider"):
        _run(
            generate_text_with_llm_async(
                plan,
                config,
                settings=LlmGoldSettings(),
                mock=False,
                client=client,
                document_id="d3",
            )
        )
    assert client.calls == 1


def test_async_invalid_json_provider() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=2, validation_max_attempts=5),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    async def handler(_self):
        raise InvalidLlmResponseError("invalid JSON", raw="{")

    client = _AsyncClient(handler)
    with pytest.raises(ValueError, match="reason=provider"):
        _run(
            generate_text_with_llm_async(
                plan,
                config,
                settings=LlmGoldSettings(),
                mock=False,
                client=client,
                document_id="d4",
                on_status=lambda s: None,
            )
        )
    assert client.calls == 2


def test_async_generic_retryable_then_success() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=2, validation_max_attempts=2),
            "text_length": {"min_chars": 5, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = assemble_mock_llm_text(plan, config)
    state = {"n": 0}

    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    async def handler(_self):
        state["n"] += 1
        if state["n"] == 1:
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            )
        return {"text": good}

    client = _AsyncClient(handler)
    statuses: list[str] = []
    elapsed: list[tuple[int, float, str]] = []
    text = _run(
        generate_text_with_llm_async(
            plan,
            config,
            settings=LlmGoldSettings(),
            mock=False,
            client=client,
            document_id="d5",
            request_delay_seconds=0.0,
            on_status=statuses.append,
            on_attempt=lambda a, b: None,
            on_call_elapsed=lambda a, s, o: elapsed.append((a, s, o)),
        )
    )
    assert "BERT" in text
    assert client.calls == 2
    assert statuses


def test_wait_for_retry_async_invokes_on_wait() -> None:
    exc = LlmRequestError("rate", retryable=True, retry_after_seconds=1.0)
    seen: list[float] = []

    async def _go():
        await wait_for_retry_async(exc, on_wait=lambda r, _m: seen.append(r))

    _run(_go())
    assert seen


def test_async_generic_non_retryable_fails_fast() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(provider_max_attempts=3)}
    )
    plan = [PlannedEntity("BERT", "model")]

    async def handler(_self):
        raise openai.AuthenticationError(
            "auth",
            response=httpx.Response(401, request=httpx.Request("POST", "https://example.com")),
            body=None,
        )

    client = _AsyncClient(handler)
    with pytest.raises(ValueError, match="reason=provider"):
        _run(
            generate_text_with_llm_async(
                plan,
                config,
                settings=LlmGoldSettings(),
                mock=False,
                client=client,
                document_id="d7",
            )
        )
    assert client.calls == 1


def test_async_success_with_callbacks() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(),
            "text_length": {"min_chars": 5, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = assemble_mock_llm_text(plan, config)

    async def handler(_self):
        return {"text": good}

    client = _AsyncClient(handler)
    text = _run(
        generate_text_with_llm_async(
            plan,
            config,
            settings=LlmGoldSettings(),
            mock=False,
            client=client,
            document_id="d6",
            on_call_elapsed=lambda *_: None,
        )
    )
    assert text
