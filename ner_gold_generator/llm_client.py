"""OpenAI-compatible chat clients for OpenRouter and deterministic mocks."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)


class ChatClient(Protocol):
    """Protocol for a single-turn JSON object response from a chat model."""

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        """Return a parsed JSON object from the model.

        Args:
            system: System prompt (labeling rules).
            user: User prompt including the source text block.
            temperature: Sampling temperature for this pass.

        Returns:
            Parsed top-level JSON object (typically ``{"entities": [...]}``).
        """
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


class OpenRouterChatClient:
    """Live LLM calls via OpenRouter (OpenAI-compatible API).

    Requires the optional ``openai`` package (``uv sync --extra llm``).

    Args:
        api_key: OpenRouter API key.
        model_id: Model slug (e.g. ``openai/gpt-4o-mini``).
        base_url: API base URL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        base_url: str = _OPENROUTER_BASE,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._base_url = base_url

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        """Request a JSON object response from OpenRouter."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        response = client.chat.completions.create(
            model=self._model_id,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return json.loads(strip_json_fence(raw))


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

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        """Return the next canned response or mock-extracted entities."""
        del system, temperature
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
