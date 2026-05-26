"""Tests for synthetic LLM parallelism (no leaks, no spurious thread pool)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.synthetic import build_examples_synthetic
from ner_gold_generator.synthetic_compose import assemble_mock_llm_text
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict


def test_parallelism_one_runs_sequential_no_thread_pool(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(syn_mod, "build_chat_client", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(
        syn_mod,
        "generate_text_with_llm",
        lambda plan, cfg, **_kw: assemble_mock_llm_text(plan, cfg),
    )

    with patch.object(syn_mod, "ThreadPoolExecutor") as pool_cls:
        examples = build_examples_synthetic(
            config,
            settings=LlmGoldSettings.model_construct(
                openrouter_api_key="key",
                mock_llm=False,
            ),
            mock=False,
        )
    pool_cls.assert_not_called()
    assert len(examples) == 2


def test_parallel_worker_failure_shuts_down_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator import synthetic as syn_mod

    config = parse_synthetic_config_dict(
        {
            "num_documents": 2,
            "engine": "llm",
            "llm": {"model": "test/model", "parallelism": 2},
            "entities_per_document": {"distribution": "fixed", "fixed": 1},
        }
    )
    monkeypatch.setattr(
        "ner_gold_generator.llm_client.verify_llm_installation",
        lambda: None,
    )
    monkeypatch.setattr(syn_mod, "build_chat_client", lambda *_a, **_k: MagicMock())

    calls = {"n": 0}

    def _flaky(plan: list, cfg: object, **_kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated worker failure")
        return assemble_mock_llm_text(plan, cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(syn_mod, "generate_text_with_llm", _flaky)

    shutdown_calls: list[tuple[bool, bool]] = []
    real_executor = ThreadPoolExecutor

    class _TrackingExecutor:
        def __init__(self, max_workers: int) -> None:
            self._inner = real_executor(max_workers=max_workers)

        def submit(self, *args: object, **kwargs: object):
            return self._inner.submit(*args, **kwargs)

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))
            self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(syn_mod, "ThreadPoolExecutor", _TrackingExecutor)

    with pytest.raises(ValueError, match="simulated worker failure"):
        build_examples_synthetic(
            config,
            settings=LlmGoldSettings.model_construct(
                openrouter_api_key="key",
                mock_llm=False,
            ),
            mock=False,
        )

    assert shutdown_calls
    assert shutdown_calls[-1] == (False, True)


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
