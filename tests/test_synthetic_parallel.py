"""Tests for synthetic LLM async parallelism behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.synthetic import build_examples_synthetic
from ner_gold_generator.synthetic_compose import assemble_mock_llm_text
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict


def test_parallelism_one_runs_sequential_no_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator import synthetic as syn_mod

    config = parse_synthetic_config_dict(
        {
            "num_documents": 2,
            "engine": "llm",
            "llm": {"model": "test/model", "parallelism": 1},
            "entities_per_document": {"distribution": "fixed", "fixed": 1},
        }
    )
    monkeypatch.setattr(
        "ner_gold_generator.llm_client.verify_llm_installation",
        lambda: None,
    )
    monkeypatch.setattr(syn_mod, "build_async_chat_client", lambda *_a, **_k: MagicMock())

    async def _fast_async(plan: list, cfg: object, **_kw: object) -> str:
        await asyncio.sleep(0)
        return assemble_mock_llm_text(plan, cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(syn_mod, "generate_text_with_llm_async", _fast_async)

    with patch.object(syn_mod.asyncio, "create_task") as create_task:
        examples = build_examples_synthetic(
            config,
            settings=LlmGoldSettings.model_construct(
                openrouter_api_key="key",
                mock_llm=False,
            ),
            mock=False,
        )
    create_task.assert_not_called()
    assert len(examples) == 2


def test_parallel_worker_failure_cancels_pending_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator import synthetic as syn_mod

    config = parse_synthetic_config_dict(
        {
            "num_documents": 2,
            "engine": "llm",
            "llm": {"model": "test/model", "parallelism": 2},
            "entities_per_document": {"distribution": "fixed", "fixed": 1},
        }
    )
    monkeypatch.setattr("ner_gold_generator.llm_client.verify_llm_installation", lambda: None)
    monkeypatch.setattr(syn_mod, "build_async_chat_client", lambda *_a, **_k: MagicMock())

    calls = {"n": 0}

    async def _flaky_async(plan: list, cfg: object, **_kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated worker failure")
        return assemble_mock_llm_text(plan, cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(syn_mod, "generate_text_with_llm_async", _flaky_async)

    result = build_examples_synthetic(
        config,
        settings=LlmGoldSettings.model_construct(
            openrouter_api_key="key",
            mock_llm=False,
        ),
        mock=False,
    )
    assert len(result.examples) == 1
    assert len(result.skipped) == 1
    assert "simulated worker failure" in result.skipped[0].error


def test_openrouter_empty_choices_raises_clear_error() -> None:
    pytest.importorskip("openai")
    from unittest.mock import MagicMock, patch

    from ner_gold_generator.llm_client import OpenRouterChatClient

    mock_response = MagicMock()
    mock_response.choices = []
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    with patch("openai.OpenAI", return_value=mock_client):
        client = OpenRouterChatClient(api_key="k", model_id="m")
        with pytest.raises(ValueError, match="no choices"):
            client.complete_json(system="s", user="u", temperature=0.0)
