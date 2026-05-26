"""OpenAI-compatible chat clients for OpenRouter and deterministic mocks."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn, Protocol

from ner_gold_generator.logger import get_logger

_STDERR_LOG_LOCK = threading.Lock()
_LOGGER = get_logger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/models"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 529})
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_LLM_INSTALL_HINT = "uv sync --extra llm"
_JITER_REINSTALL_HINT = "uv pip install --force-reinstall jiter"


def verify_llm_installation() -> None:
    """Ensure optional LLM dependencies are importable before live API calls.

    Raises:
        ImportError: With setup commands when ``openai``, ``httpx``, or ``jiter``
            is missing or broken (common on Windows when ``jiter`` lacks
            ``__init__.py`` after a partial install).
    """
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"LLM support requires httpx. Run: {_LLM_INSTALL_HINT}") from exc
    try:
        from jiter import from_json  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Broken or incomplete jiter install (OpenAI SDK needs jiter). "
            f"Run: {_JITER_REINSTALL_HINT}  (or {_LLM_INSTALL_HINT})"
        ) from exc
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"LLM support requires openai. Run: {_LLM_INSTALL_HINT}") from exc


class InvalidLlmResponseError(ValueError):
    """Model returned a response that could not be parsed as expected JSON.

    Attributes:
        raw: Raw model text when available (for debug logs).
    """

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class LlmRequestError(Exception):
    """OpenRouter / chat API failure with a user-facing message.

    Attributes:
        retryable: When ``True``, callers may retry the same request.
        model_id: Model slug from config or env.
        status_code: HTTP status when available.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        model_id: str = "",
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.model_id = model_id
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _api_error_message(exc: object) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    return str(exc)


def _upstream_error_hint(exc: object) -> str:
    """Provider-specific detail from OpenRouter error metadata (if any)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            meta = err.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("raw"), str):
                return meta["raw"].strip()
    return ""


def retry_after_seconds_from_error(exc: BaseException) -> float | None:
    """Parse Retry-After / OpenRouter metadata from an SDK error."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            meta = err.get("metadata")
            if isinstance(meta, dict):
                for key in ("retry_after_seconds", "retry_after_seconds_raw"):
                    raw = meta.get(key)
                    if raw is not None:
                        try:
                            return max(0.0, float(raw))
                        except (TypeError, ValueError):
                            pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
    return None


def describe_llm_api_error(exc: BaseException, *, model_id: str) -> tuple[str, bool, int | None]:
    """Map an SDK exception to ``(message, retryable, status_code)``."""
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            NotFoundError,
            RateLimitError,
        )
    except ImportError:
        return (f"LLM request failed for {model_id!r}: {exc}", False, None)

    if isinstance(exc, APITimeoutError):
        return (
            f"OpenRouter request timed out for {model_id!r} "
            "(reasoning/free models can be slow — try a faster model or raise llm.timeout_seconds)",
            True,
            None,
        )
    if isinstance(exc, APIConnectionError):
        return (f"OpenRouter connection failed: {exc}", True, None)
    if isinstance(exc, RateLimitError):
        upstream = _upstream_error_hint(exc)
        retry = retry_after_seconds_from_error(exc)
        parts = [f"Rate limit (429) on model {model_id!r}"]
        if upstream:
            parts.append(upstream)
        else:
            parts.append(_api_error_message(exc) or "too many requests")
        if retry is not None:
            parts.append(f"retry after ~{retry:.0f}s")
        return (" — ".join(parts), True, 429)
    if isinstance(exc, AuthenticationError):
        return (
            "OpenRouter authentication failed — check OPENROUTER_API_KEY in .env",
            False,
            401,
        )
    if isinstance(exc, NotFoundError):
        return (
            f"No OpenRouter endpoint for model {model_id!r}. "
            f"Update llm.model in your synthetic config. "
            f"Browse free models: {_OPENROUTER_MODELS_URL}",
            False,
            404,
        )
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        detail = _api_error_message(exc)
        retryable = code in _RETRYABLE_STATUS_CODES
        return (f"OpenRouter API error {code} for {model_id!r}: {detail}", retryable, code)
    return (f"LLM request failed for {model_id!r}: {exc}", False, None)


def raise_llm_request_error(exc: BaseException, *, model_id: str) -> NoReturn:
    """Re-raise *exc* as :class:`LlmRequestError` with classified metadata."""
    message, retryable, status_code = describe_llm_api_error(exc, model_id=model_id)
    raise LlmRequestError(
        message,
        retryable=retryable,
        model_id=model_id,
        status_code=status_code,
        retry_after_seconds=retry_after_seconds_from_error(exc),
    ) from exc


def wait_for_retry(
    exc: LlmRequestError,
    *,
    on_wait: Callable[[float, str], None] | None = None,
    cap_seconds: float = 120.0,
    tick_seconds: float = 5.0,
) -> None:
    """Sleep for rate-limit backoff when *exc* carries ``retry_after_seconds``."""
    if not exc.retryable or exc.retry_after_seconds is None:
        return
    wait = min(max(1.0, exc.retry_after_seconds + 0.5), cap_seconds)
    remaining = wait
    while remaining > 0:
        step = min(max(tick_seconds, 1.0), remaining)
        if on_wait is not None:
            on_wait(remaining, f"rate limited — {remaining:.0f}s left")
        time.sleep(step)
        remaining -= step


class ChatClient(Protocol):
    """Protocol for a single-turn JSON object response from a chat model."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model.

        Args:
            system: System prompt (labeling rules).
            user: User prompt including the source text block.
            temperature: Sampling temperature for this pass.
            call_label: Optional tag printed with live prompts (e.g. retry attempt).

        Returns:
            Parsed top-level JSON object (typically ``{"entities": [...]}``).
        """
        ...


class AsyncChatClient(Protocol):
    """Async protocol for a single-turn JSON object response from a chat model."""

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model (async)."""
        ...


def strip_json_fence(raw: str) -> str:
    """Remove optional Markdown ```json fences from model output.

    Args:
        raw: Raw model text, possibly fenced.

    Returns:
        Inner JSON string suitable for :func:`json.loads`.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and _FENCE_RE.match(lines[0]):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_entities_json(raw: str) -> list[dict[str, str]]:
    """Parse an ``entities`` array from model JSON text.

    Args:
        raw: JSON string, optionally Markdown-fenced.

    Returns:
        List of dicts with string ``text`` and ``label`` keys. Invalid entries
        are skipped.

    Raises:
        ValueError: If JSON is not an object or lacks an ``entities`` array.
        json.JSONDecodeError: If JSON is malformed.
    """
    payload = json.loads(strip_json_fence(raw))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Missing entities array")
    out: list[dict[str, str]] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
        if isinstance(text, str) and isinstance(label, str):
            out.append({"text": text, "label": label})
    return out


def log_llm_prompt(
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    call_label: str | None = None,
) -> None:
    """Print the full system and user prompts sent to a live LLM (stderr)."""
    header = f"--- LLM prompt ({model_id}, temperature={temperature}"
    if call_label:
        header += f", {call_label}"
    payload = "\n".join(
        (
            f"{header}) ---",
            "SYSTEM:",
            system,
            "USER:",
            user,
            "--- end LLM prompt ---",
        )
    )
    with _STDERR_LOG_LOCK:
        _LOGGER.debug(payload)


def log_llm_response(
    *,
    model_id: str,
    raw: str,
    call_label: str | None = None,
) -> None:
    """Print the raw model text returned from a live LLM call (stderr)."""
    header = f"--- LLM response ({model_id}"
    if call_label:
        header += f", {call_label}"
    payload = "\n".join(
        (
            f"{header}) ---",
            raw if raw.strip() else "(empty)",
            "--- end LLM response ---",
        )
    )
    with _STDERR_LOG_LOCK:
        _LOGGER.debug(payload)


class OpenRouterChatClient:
    """Live LLM calls via OpenRouter (OpenAI-compatible API).

    Requires the optional ``openai`` package (``uv sync --extra llm``).

    Args:
        api_key: OpenRouter API key.
        model_id: OpenRouter model slug (e.g. ``vendor/model:free``).
        base_url: API base URL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        base_url: str = _OPENROUTER_BASE,
        timeout_seconds: float = 120.0,
    ) -> None:
        verify_llm_installation()
        from openai import OpenAI

        self._model_id = model_id
        # max_retries=0: we handle 429/backoff in generate_text_with_llm (visible progress).
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout_seconds,
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict[str, Any]:
        """Request a JSON object response from OpenRouter."""
        log_llm_prompt(
            model_id=self._model_id,
            system=system,
            user=user,
            temperature=temperature,
            call_label=call_label,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            choices = response.choices
            if not choices:
                raise InvalidLlmResponseError(
                    f"Model {self._model_id!r} returned no choices "
                    "(empty or malformed OpenRouter response)",
                    raw="",
                )
            raw = choices[0].message.content or ""
            log_llm_response(model_id=self._model_id, raw=raw, call_label=call_label)
            return json.loads(strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            raise InvalidLlmResponseError(
                f"Model {self._model_id!r} returned invalid JSON: {exc}",
                raw=raw,
            ) from exc
        except InvalidLlmResponseError:
            raise
        except Exception as exc:
            raise_llm_request_error(exc, model_id=self._model_id)


class AsyncOpenRouterChatClient:
    """Async live LLM calls via OpenRouter (OpenAI-compatible API).

    Uses the OpenAI SDK async client (``AsyncOpenAI``) against the OpenRouter
    base URL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        base_url: str = _OPENROUTER_BASE,
        timeout_seconds: float = 120.0,
    ) -> None:
        verify_llm_installation()
        from openai import AsyncOpenAI

        self._model_id = model_id
        # max_retries=0: callers handle retry/backoff.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout_seconds,
        )

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict[str, Any]:
        """Request a JSON object response from OpenRouter (async)."""
        log_llm_prompt(
            model_id=self._model_id,
            system=system,
            user=user,
            temperature=temperature,
            call_label=call_label,
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model_id,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            choices = response.choices
            if not choices:
                raise InvalidLlmResponseError(
                    f"Model {self._model_id!r} returned no choices "
                    "(empty or malformed OpenRouter response)",
                    raw="",
                )
            raw = choices[0].message.content or ""
            log_llm_response(model_id=self._model_id, raw=raw, call_label=call_label)
            return json.loads(strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            raise InvalidLlmResponseError(
                f"Model {self._model_id!r} returned invalid JSON: {exc}",
                raw=raw,
            ) from exc
        except InvalidLlmResponseError:
            raise
        except Exception as exc:
            raise_llm_request_error(exc, model_id=self._model_id)


def build_async_chat_client(
    settings: object,
    *,
    model_id: str,
    timeout_seconds: float = 120.0,
) -> AsyncChatClient:
    """Build an async OpenRouter client from settings.

    Kept as a small wrapper to make asyncio-based callers easy to test by
    monkeypatching this factory.
    """
    api_key = getattr(settings, "openrouter_api_key", "")
    return AsyncOpenRouterChatClient(
        api_key=str(api_key),
        model_id=model_id,
        timeout_seconds=timeout_seconds,
    )


class MockChatClient:
    """Deterministic JSON responses for tests and ``--mock`` CLI runs.

    When no canned *responses* are supplied, extracts entities from the
    ``---TEXT---`` block in the user prompt via :func:`mock.mock_extract_entities`.

    Args:
        responses: Optional list of JSON payloads returned in order per call.
    """

    def __init__(self, *, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_index = 0

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict[str, Any]:
        """Return the next canned response or mock-extracted entities."""
        del system, temperature, call_label
        if self._responses:
            idx = min(self._call_index, len(self._responses) - 1)
            self._call_index += 1
            return self._responses[idx]
        body = user.split("---TEXT---", 1)[1].strip() if "---TEXT---" in user else user
        from ner_gold_generator.mock import mock_extract_entities

        entities = mock_extract_entities(body)
        return {"entities": [{"text": e.text, "label": e.label} for e in entities]}

    def reset(self) -> None:
        """Reset the canned-response cursor to the first item."""
        self._call_index = 0
