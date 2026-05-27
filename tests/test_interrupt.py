"""Tests for graceful Ctrl+C (KeyboardInterrupt) handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ner_gold_generator.build_gold import (
    INTERRUPT_EXIT_CODE,
    _handle_build_interrupted,
    build_examples_manual,
    main,
)
from ner_gold_generator.interrupt import BuildInterrupted
from ner_gold_generator.progress import Progress
from ner_gold_generator.sources import SourceDocument


def test_build_interrupted_carries_partial_examples() -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    exc = BuildInterrupted(rows, completed=2, total=5)
    assert exc.examples == rows
    assert exc.completed == 2
    assert exc.total == 5
    assert "2/5" in str(exc)


def test_handle_build_interrupted_prints_statistics_when_not_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "partial.jsonl"
    args = SimpleNamespace(
        output=out,
        quiet=False,
        source="synthetic",
        mode="synthetic",
    )
    exc = BuildInterrupted(
        [{"text": "x", "entities": [], "synthetic_meta": {"seed": 1, "engine": "llm"}}],
        completed=1,
        total=3,
    )
    code = _handle_build_interrupted(
        exc,
        args=args,
        progress=Progress(quiet=False),
        elapsed_seconds=4.2,
    )
    assert code == 130
    err = capsys.readouterr().err
    assert "statistics:" in err
    assert "elapsed: 4.2s" in err
    assert "completed: 1/3 documents" in err


def test_handle_build_interrupted_writes_partial_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "partial.jsonl"
    exc = BuildInterrupted([{"id": "doc-1", "text": "x", "entities": []}], completed=1, total=3)
    args = SimpleNamespace(output=out)
    code = _handle_build_interrupted(exc, args=args, progress=Progress(quiet=True))
    assert code == INTERRUPT_EXIT_CODE
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "doc-1"


def test_handle_build_interrupted_skips_write_when_empty(tmp_path: Path) -> None:
    out = tmp_path / "empty.jsonl"
    exc = BuildInterrupted([], completed=0, total=2)
    args = SimpleNamespace(output=out)
    code = _handle_build_interrupted(exc, args=args, progress=Progress(quiet=True))
    assert code == INTERRUPT_EXIT_CODE
    assert not out.exists()


def test_build_examples_manual_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [
        SourceDocument(doc_id="d1", text="Alice works at OpenAI.", source="folder"),
        SourceDocument(doc_id="d2", text="Bob uses BERT.", source="folder"),
    ]
    manual = {"d1": [("OpenAI", "ORG")], "d2": [("BERT", "MODEL")]}

    def _resolve(doc_id: str, **kwargs: object) -> list | None:
        del kwargs
        if doc_id == "d2":
            raise KeyboardInterrupt
        return manual.get(doc_id)

    monkeypatch.setattr(
        "ner_gold_generator.build_gold.resolve_manual_spans",
        _resolve,
    )
    with pytest.raises(BuildInterrupted) as raised:
        build_examples_manual(
            docs,
            source="folder",
            manual_path=None,
            progress=Progress(quiet=True),
        )
    assert len(raised.value.examples) == 1
    assert raised.value.examples[0]["id"] == "d1"
    assert raised.value.total == 2


def test_build_examples_synthetic_procedural_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_gold_generator import synthetic as syn_mod
    from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

    config = parse_synthetic_config_dict({"num_documents": 3, "engine": "procedural"})
    calls = {"n": 0}
    original = syn_mod.assemble_procedural_text

    def _assemble(*args: object, **kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(syn_mod, "assemble_procedural_text", _assemble)
    with pytest.raises(BuildInterrupted) as raised:
        syn_mod.build_examples_synthetic(config, progress=Progress(quiet=True))
    assert len(raised.value.examples) == 1
    assert raised.value.total == 3


def test_build_examples_synthetic_llm_parallel_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.llm_test_helpers import make_llm_config

    from ner_gold_generator import synthetic as syn_mod
    from ner_gold_generator.labeling import LlmGoldSettings
    from ner_gold_generator.synthetic_compose import assemble_mock_llm_text
    from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

    config = parse_synthetic_config_dict(
        {
            "num_documents": 3,
            "engine": "llm",
            "llm": make_llm_config(model="test/model", parallelism=2),
            "entities_per_document": {"distribution": "fixed", "fixed": 1},
        }
    )
    monkeypatch.setattr(
        "ner_gold_generator.llm_client.verify_llm_installation",
        lambda: None,
    )
    monkeypatch.setattr(syn_mod, "build_async_chat_client", lambda *_a, **_k: MagicMock())

    async def _fast_llm_async(plan: list, cfg: object, **_kwargs: object) -> str:
        return assemble_mock_llm_text(plan, cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(syn_mod, "generate_text_with_llm_async", _fast_llm_async)

    orig_as_completed = syn_mod.asyncio.as_completed

    def _interrupt_after_first(tasks: list) -> object:
        iterator = orig_as_completed(tasks)
        yield next(iterator)
        raise KeyboardInterrupt

    monkeypatch.setattr(syn_mod.asyncio, "as_completed", _interrupt_after_first)

    with pytest.raises(BuildInterrupted) as raised:
        syn_mod.build_examples_synthetic(
            config,
            settings=LlmGoldSettings.model_construct(
                openrouter_api_key="key",
                mock_llm=False,
            ),
            mock=False,
            progress=Progress(quiet=True),
        )
    assert 1 <= len(raised.value.examples) < 3
    assert raised.value.total == 3


def test_main_returns_130_on_build_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "out.jsonl"
    partial = [{"id": "p1", "text": "hi", "entities": []}]

    def _run_build(*_args: object, **_kwargs: object) -> int:
        raise BuildInterrupted(partial, completed=1, total=2)

    monkeypatch.setattr("ner_gold_generator.build_gold._run_build", _run_build)
    code = main(["--source", "arxiv", "--output", str(out), "--quiet"])
    assert code == 130
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8").strip())["id"] == "p1"
