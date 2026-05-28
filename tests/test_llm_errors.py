"""Tests for OpenRouter / LLM error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.llm_test_helpers import make_llm_config

from ner_gold_generator.llm_client import (
    LlmRequestError,
    OpenRouterChatClient,
    compute_rate_limit_wait_seconds,
    describe_llm_api_error,
    raise_llm_request_error,
    retry_after_seconds_from_error,
    verify_llm_installation,
    wait_for_retry,
)
from ner_gold_generator.synthetic_compose import generate_text_with_llm
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict
from ner_gold_generator.synthetic_generators import PlannedEntity, plan_entities_for_document


def _http_response(status: int, body: dict | None = None):
    httpx = pytest.importorskip("httpx")
    return httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


class _NotFoundClient:
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict:
        del system, user, temperature
        openai = pytest.importorskip("openai")
        raise openai.NotFoundError(
            "missing",
            response=_http_response(
                404,
                {"error": {"message": "No endpoints found for bad/model:free.", "code": 404}},
            ),
            body={"error": {"message": "No endpoints found for bad/model:free.", "code": 404}},
        )


def test_retry_after_seconds_invalid_metadata() -> None:
    class _Exc:
        body = {"error": {"metadata": {"retry_after_seconds": "not-a-number"}}}

    assert retry_after_seconds_from_error(_Exc()) is None


def test_retry_after_seconds_invalid_header() -> None:
    class _Headers:
        def get(self, _key: str) -> str:
            return "wait"

    class _Response:
        headers = _Headers()

    class _Exc:
        response = _Response()

    assert retry_after_seconds_from_error(_Exc()) is None


def test_describe_api_timeout_retryable() -> None:
    openai = pytest.importorskip("openai")
    from unittest.mock import MagicMock

    exc = openai.APITimeoutError(request=MagicMock())
    msg, retryable, code = describe_llm_api_error(exc, model_id="slow/model:free")
    assert retryable
    assert code is None
    assert "timed out" in msg.lower()


def test_describe_api_connection_error() -> None:
    openai = pytest.importorskip("openai")
    from unittest.mock import MagicMock

    exc = openai.APIConnectionError(request=MagicMock())
    msg, retryable, code = describe_llm_api_error(exc, model_id="m")
    assert retryable
    assert code is None
    assert "connection failed" in msg.lower()


def test_describe_not_found_not_retryable() -> None:
    openai = pytest.importorskip("openai")
    exc = openai.NotFoundError(
        "x",
        response=_http_response(
            404, {"error": {"message": "No endpoints found for x.", "code": 404}}
        ),
        body={"error": {"message": "No endpoints found for x.", "code": 404}},
    )
    msg, retryable, code = describe_llm_api_error(exc, model_id="bad/model:free")
    assert not retryable
    assert code == 404
    assert "bad/model:free" in msg
    assert "openrouter.ai/models" in msg


def test_raise_llm_request_error_wraps_not_found() -> None:
    openai = pytest.importorskip("openai")
    exc = openai.NotFoundError(
        "x",
        response=_http_response(404, {"error": {"message": "missing", "code": 404}}),
        body={"error": {"message": "missing", "code": 404}},
    )
    with pytest.raises(LlmRequestError) as err:
        raise_llm_request_error(exc, model_id="bad/model:free")
    assert not err.value.retryable
    assert err.value.status_code == 404


def test_openrouter_client_invalid_json_raises_value_error() -> None:
    pytest.importorskip("openai")
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="not-json"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    with patch("openai.OpenAI", return_value=mock_client):
        client = OpenRouterChatClient(api_key="k", model_id="test/model:free")
        with pytest.raises(ValueError, match="invalid JSON"):
            client.complete_json(system="s", user="u", temperature=0.0)


def test_generate_text_with_llm_model_not_found_fails_fast() -> None:
    import random

    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(model="bad/model:free", max_attempts=3),
        }
    )
    plan = plan_entities_for_document(random.Random(0), config)
    with pytest.raises(ValueError, match="No OpenRouter endpoint"):
        generate_text_with_llm(
            plan,
            config,
            settings=None,
            mock=False,
            client=_NotFoundClient(),
        )


def test_describe_rate_limit_message_is_short() -> None:
    openai = pytest.importorskip("openai")
    body = {
        "error": {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "model is temporarily rate-limited upstream.",
                "retry_after_seconds": 4,
            },
        }
    }
    exc = openai.RateLimitError(
        "rate",
        response=_http_response(429, body),
        body=body,
    )
    msg, retryable, code = describe_llm_api_error(
        exc, model_id="meta-llama/llama-3.3-70b-instruct:free"
    )
    assert retryable and code == 429
    assert "rate-limited upstream" in msg
    assert "retry after" in msg
    assert "user_id" not in msg
    assert len(msg) < 300


def test_describe_connection_and_auth_errors() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    conn = openai.APIConnectionError(request=request)
    msg, retryable, _ = describe_llm_api_error(conn, model_id="m")
    assert retryable
    assert "connection" in msg.lower()

    auth = openai.AuthenticationError(
        "auth",
        response=_http_response(401),
        body=None,
    )
    msg2, retryable2, code2 = describe_llm_api_error(auth, model_id="m")
    assert not retryable2
    assert code2 == 401


def test_describe_api_status_retryable() -> None:
    openai = pytest.importorskip("openai")
    exc = openai.APIStatusError(
        "server",
        response=_http_response(503, {"error": {"message": "overloaded"}}),
        body={"error": {"message": "overloaded"}},
    )
    msg, retryable, code = describe_llm_api_error(exc, model_id="m")
    assert retryable
    assert code == 503
    assert "503" in msg


def test_verify_llm_installation_ok() -> None:
    pytest.importorskip("openai")
    verify_llm_installation()


def test_verify_llm_installation_missing_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="httpx"):
        verify_llm_installation()


def test_verify_llm_installation_missing_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="openai"):
        verify_llm_installation()


def test_verify_llm_installation_missing_jiter(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "jiter" or name.startswith("jiter."):
            raise ImportError("cannot import name 'from_json' from 'jiter'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="jiter"):
        verify_llm_installation()


def test_main_returns_1_on_bare_errno2(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator import build_gold as bg

    def boom(*_args: object, **_kwargs: object) -> int:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(bg, "_run_build", boom)
    code = bg.main(["--source", "synthetic", "--synthetic-config", "x.json", "--quiet"])
    assert code == 1


def test_main_returns_1_on_config_error(tmp_path: Path) -> None:
    from ner_gold_generator.build_gold import main

    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"engine": "llm", "num_documents": 1}), encoding="utf-8")
    code = main(
        [
            "--source",
            "synthetic",
            "--synthetic-config",
            str(cfg),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--quiet",
        ]
    )
    assert code == 1


def test_synthetic_requires_api_key_when_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator.labeling import LlmGoldSettings
    from ner_gold_generator.synthetic import build_examples_synthetic
    from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = parse_synthetic_config_dict(
        {
            "num_documents": 1,
            "engine": "llm",
            "llm": make_llm_config(),
        }
    )
    settings = LlmGoldSettings.model_construct(openrouter_api_key="", mock_llm=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_examples_synthetic(config, settings=settings, mock=False)


def test_describe_agno_wrapped_rate_limit_string() -> None:
    now = 1_700_000_000.0
    reset_ms = int((now + 120.0) * 1000)
    message = (
        "Rate limit error from OpenAI API: Error code: 429 - "
        "{'error': {'message': 'Rate limit exceeded: @ratelimit/too-many-requests.', "
        "'code': 429, 'metadata': {'headers': {'X-RateLimit-Remaining': '0', "
        "'X-RateLimit-Reset': '" + str(reset_ms) + "'}}}}}"
    )

    class _AgnoWrapped(Exception):
        pass

    exc = _AgnoWrapped(message)
    msg, retryable, code = describe_llm_api_error(exc, model_id="openai/gpt-oss-120b:free")
    assert retryable
    assert code == 429
    assert "retry after" in msg
    assert retry_after_seconds_from_error(exc, now=now) == pytest.approx(120.0, abs=0.01)


def test_resolve_rate_limit_wait_seconds_default_when_unparseable() -> None:
    from ner_gold_generator.llm_client import (
        DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
        resolve_rate_limit_wait_seconds,
    )

    class _Bare429(Exception):
        pass

    wait = resolve_rate_limit_wait_seconds(_Bare429("HTTP 429 too many requests"))
    assert wait == pytest.approx(DEFAULT_RATE_LIMIT_BACKOFF_SECONDS + 0.5, abs=0.1)


def test_retry_after_from_openrouter_x_ratelimit_reset_ms() -> None:
    openai = pytest.importorskip("openai")
    now = 1_700_000_000.0
    reset_ms = int((now + 45.0) * 1000)
    body = {
        "error": {
            "message": "Rate limit exceeded",
            "code": 429,
            "metadata": {
                "headers": {
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_ms),
                }
            },
        }
    }
    exc = openai.RateLimitError("rate", response=_http_response(429, body), body=body)
    delay = retry_after_seconds_from_error(exc, now=now)
    assert delay == pytest.approx(45.0, abs=0.01)


def test_retry_after_prefers_explicit_seconds_over_reset() -> None:
    class _Exc:
        body = {
            "error": {
                "metadata": {
                    "retry_after_seconds": 7,
                    "headers": {"X-RateLimit-Reset": "9999999999999"},
                }
            }
        }

    assert retry_after_seconds_from_error(_Exc(), now=0.0) == 7.0


def test_compute_rate_limit_wait_seconds_caps_long_reset() -> None:
    assert compute_rate_limit_wait_seconds(5000.0, cap_seconds=120.0) == 120.0


def test_retry_after_from_openrouter_body() -> None:
    openai = pytest.importorskip("openai")
    exc = openai.RateLimitError(
        "rate",
        response=_http_response(
            429,
            {
                "error": {
                    "metadata": {"retry_after_seconds": 29},
                }
            },
        ),
        body={
            "error": {
                "metadata": {"retry_after_seconds": 29},
            }
        },
    )
    assert retry_after_seconds_from_error(exc) == 29.0


def test_wait_for_retry_ticks_countdown(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("ner_gold_generator.llm_client.time.sleep", lambda s: sleeps.append(s))
    waits: list[tuple[float, str]] = []
    exc = LlmRequestError("rate", retryable=True, retry_after_seconds=12.0)
    wait_for_retry(exc, on_wait=lambda w, d: waits.append((w, d)), tick_seconds=5.0)
    assert sum(sleeps) == pytest.approx(12.5, abs=0.1)
    assert len(waits) >= 2
    assert waits[0][1].startswith("rate limited")


def test_wait_for_retry_calls_on_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("ner_gold_generator.llm_client.time.sleep", fake_sleep)
    waits: list[tuple[float, str]] = []

    exc = LlmRequestError("rate", retryable=True, retry_after_seconds=10.0)
    wait_for_retry(exc, on_wait=lambda w, d: waits.append((w, d)))
    assert waits
    assert slept


def test_generate_text_with_llm_retries_rate_limit() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(max_attempts=2),
            "text_length": {"min_chars": 10, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    good = {"text": "We study BERT on GLUE and report strong results across benchmarks."}

    class _ThenOkClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            self.calls += 1
            if self.calls == 1:
                openai = pytest.importorskip("openai")
                raise openai.RateLimitError(
                    "rate",
                    response=_http_response(429),
                    body=None,
                )
            return good

    text = generate_text_with_llm(
        plan,
        config,
        settings=None,
        mock=False,
        client=_ThenOkClient(),
    )
    assert "BERT" in text
