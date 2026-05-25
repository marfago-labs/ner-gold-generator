"""Tests for build_gold / build_arxiv CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ner_gold_generator.build_arxiv as build_arxiv_mod
import ner_gold_generator.build_gold as build_gold_mod
import ner_gold_generator.manual_entities as manual_mod
from ner_gold_generator import env as env_module
from ner_gold_generator.progress import Progress


def test_progress_bar_format() -> None:
    assert Progress.bar(0, 10) == "[----------------------------] 0/10"
    assert Progress.bar(10, 10).endswith("10/10")


def test_build_arxiv_gold_llm_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "1706.03762.json").write_text(
        json.dumps(
            {
                "arxiv_id": "1706.03762",
                "abstract": "OpenAI GPT-3 on ImageNet in 2020.",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(manual_mod, "PAPER_ORDER", ["1706.03762"])
    code = build_arxiv_mod.main(
        [
            "--mode",
            "llm",
            "--mock",
            "--quiet",
            "--input",
            str(arxiv_dir),
            "--output",
            str(out),
            "--samples",
            "2",
            "--min-votes",
            "2",
        ]
    )
    assert code == 0
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["gold_source"] == "llm"
    assert row["source"] == "arxiv"


def test_build_gold_youtube_llm(tmp_path: Path) -> None:
    yt = tmp_path / "yt"
    yt.mkdir()
    (yt / "v1.json").write_text(
        json.dumps({"video_id": "v1", "transcript": "OpenAI and GPT-3."}),
        encoding="utf-8",
    )
    out = tmp_path / "yt_gold.jsonl"
    code = build_gold_mod.main(
        [
            "--source",
            "youtube",
            "--input",
            str(yt),
            "--mode",
            "llm",
            "--mock",
            "--quiet",
            "--output",
            str(out),
            "--limit",
            "1",
        ]
    )
    assert code == 0
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["id"] == "youtube-v1"


def test_manual_non_arxiv_requires_file(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("hello", encoding="utf-8")
    code = build_gold_mod.main(
        [
            "--source",
            "folder",
            "--input",
            str(folder),
            "--mode",
            "manual",
            "--quiet",
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )
    assert code == 2


def test_manual_with_json_file(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("OpenAI rocks", encoding="utf-8")
    manual = tmp_path / "manual.json"
    manual.write_text(
        json.dumps(
            {
                "file-a.txt": [{"text": "OpenAI", "label": "organization"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    code = build_gold_mod.main(
        [
            "--source",
            "folder",
            "--input",
            str(folder),
            "--manual",
            str(manual),
            "--mode",
            "manual",
            "--quiet",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["entities"][0]["text"] == "OpenAI"


def test_build_arxiv_gold_manual_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_gold_mod.load_documents("arxiv", tmp_path / "missing")


def test_load_project_env_reads_file(tmp_path: Path) -> None:
    env_module._ENV_LOADED = False
    dotenv = tmp_path / ".env"
    dotenv.write_text("MOCK_LLM=true\n", encoding="utf-8")
    assert env_module.load_project_env(env_path=dotenv) is True
    env_module._ENV_LOADED = False


def test_main_manual_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    abstract = (
        "We propose a new simple network architecture, the Transformer, based solely on "
        "attention. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German "
        "translation task. On the WMT 2014 English-to-French translation task, our model "
        "establishes a new single-model state-of-the-art BLEU score of 41.8."
    )
    (arxiv_dir / "1706.03762.json").write_text(
        json.dumps({"arxiv_id": "1706.03762", "abstract": abstract}),
        encoding="utf-8",
    )
    out = tmp_path / "gold.jsonl"
    monkeypatch.setattr(manual_mod, "PAPER_ORDER", ["1706.03762"])
    code = build_arxiv_mod.main(
        [
            "--mode",
            "manual",
            "--quiet",
            "--input",
            str(arxiv_dir),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()


def test_ensure_arxiv_argv_maps_arxiv_dir() -> None:
    argv = build_arxiv_mod._ensure_arxiv_argv(["--arxiv-dir", "/tmp/raw"])
    assert "--input" in argv
    assert "/tmp/raw" in argv
    assert "--source" in argv


def test_resolve_input_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YOUTUBE_INPUT_DIR", str(tmp_path))
    assert build_gold_mod.resolve_input_path("youtube", None) == tmp_path


def test_build_examples_manual_skip_no_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "9999.00001.json").write_text(
        json.dumps({"arxiv_id": "9999.00001", "abstract": "No curated spans."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(manual_mod, "PAPER_ORDER", ["9999.00001"])
    with pytest.raises(ValueError, match="No examples produced"):
        build_gold_mod.build_examples_manual(
            build_gold_mod.load_documents("arxiv", arxiv_dir),
            source="arxiv",
            manual_path=None,
            progress=Progress(quiet=True),
        )


def test_progress_log(capsys) -> None:
    Progress(quiet=False).log("status")
    assert "status" in capsys.readouterr().err


def test_load_project_env_without_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    env_module._ENV_LOADED = False
    dotenv = tmp_path / ".env"
    dotenv.write_text("X=1\n", encoding="utf-8")
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object):
        if name == "dotenv":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert env_module.load_project_env(env_path=dotenv) is False
    env_module._ENV_LOADED = False


def test_manual_bad_surface_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "1706.03762.json").write_text(
        json.dumps({"arxiv_id": "1706.03762", "abstract": "short"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(manual_mod, "PAPER_ORDER", ["1706.03762"])
    with pytest.raises(ValueError, match="Surface"):
        build_gold_mod.build_examples_manual(
            build_gold_mod.load_documents("arxiv", arxiv_dir),
            source="arxiv",
            manual_path=None,
            progress=Progress(quiet=True),
        )


def test_build_manual_empty_docs() -> None:
    with pytest.raises(ValueError, match="No documents"):
        build_gold_mod.build_examples_manual(
            [],
            source="arxiv",
            manual_path=None,
        )


def test_build_llm_empty_docs() -> None:
    with pytest.raises(ValueError, match="No documents"):
        build_gold_mod.build_examples_llm(
            [],
            settings=build_gold_mod.LlmGoldSettings(mock_llm=True),
            num_samples=1,
            min_votes=1,
            mock=True,
        )
