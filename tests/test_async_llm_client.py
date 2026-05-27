"""Tests for async OpenRouter client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ner_gold_generator.llm_client import AsyncOpenRouterChatClient, InvalidLlmResponseError


async def _async_success() -> None:
    pytest.importorskip("openai")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"text": "ok"}'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = AsyncOpenRouterChatClient(api_key="k", model_id="m")
        out = await client.complete_json(system="s", user="u", temperature=0.0)
    assert out == {"text": "ok"}


async def _async_invalid() -> None:
    pytest.importorskip("openai")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="not-json"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = AsyncOpenRouterChatClient(api_key="k", model_id="m")
        with pytest.raises(InvalidLlmResponseError):
            await client.complete_json(system="s", user="u", temperature=0.0)


def test_async_complete_json_success() -> None:
    asyncio.run(_async_success())


def test_async_complete_json_invalid() -> None:
    asyncio.run(_async_invalid())


async def _async_no_choices() -> None:
    pytest.importorskip("openai")
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = AsyncOpenRouterChatClient(api_key="k", model_id="m")
        with pytest.raises(InvalidLlmResponseError, match="no choices"):
            await client.complete_json(system="s", user="u", temperature=0.0)


def test_async_complete_json_no_choices() -> None:
    asyncio.run(_async_no_choices())
