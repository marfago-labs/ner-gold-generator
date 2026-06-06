"""Tests for build_gold / build_arxiv CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ner_gold_generator.build_arxiv as build_arxiv_mod
import ner_gold_generator.build_gold as build_gold_mod
import ner_gold_generator.manual_entities as manual_mod
from ner_gold_generator import env as env_module
from ner_gold_generator.build_gold import _cli_error_message
from ner_gold_generator.progress import (
    Progress,
    _fit_status_line,
    format_skip_error,
    render_bar,
    short_model_label,
)


def test_cli_error_message_errno2_hint() -> None:
    msg = _cli_error_message(FileNotFoundError(2, "No such file or directory"))
    assert "jiter" in msg
    assert "uv sync" in msg


def test_short_model_label() -> None:
    assert (
        short_model_label("nvidia/nemotron-3-nano-30b-a3b:free") == "nemotron-3-nano-30b-a3b:free"
    )
    long_id = "meta-llama/llama-3.3-70b-instruct:free-extra-suffix"
    assert len(short_model_label(long_id, max_len=20)) <= 20


def test_fit_status_line_shows_ends() -> None:
    prefix = Progress.status_prefix(0, 5)
    detail = "synthetic-00000042-00000 LLM 1/3 nvidia/nemotron-3-nano-30b-a3b:free"
    line = _fit_status_line(prefix, detail, 80)
    assert "synthetic" in line
    assert ":free" in line


def test_progress_status_prefix_format() -> None:
    assert Progress.status_prefix(0, 10) == "0/10 (0%)"
    assert Progress.status_prefix(10, 10) == "10/10 (100%)"


def test_format_skip_error_extracts_validation_surfaces() -> None:
    err = (
        "Synthetic LLM failed (model='m', reason=validation, "
        "llm_calls=7/7, provider_failures=0/2, validation_failures=5/5).\n"
        "  Surface 'pre-training' appears 2 times (expected 1)"
    )
    formatted = format_skip_error(err)
    assert "(attempt 7/7 · validation 5/5)" in formatted
    assert "Surface 'pre-training' appears 2 times (expected 1)" in formatted


def test_render_bar_half_and_full() -> None:
    assert render_bar(0, 10, width=10) == "[----------]"
    assert render_bar(5, 10, width=10) == "[#####-----]"
    assert render_bar(10, 10, width=10) == "[##########]"


def test_progress_track_updates_bar(capsys) -> None:
    prog = Progress(quiet=False)
    prog._tty_bar = False  # line-based output in tests (no \\r refresh)
    with prog.track(2, label="test") as tick:
        tick(1, detail="first")
        tick(2, detail="second")
    err = capsys.readouterr().err
    assert "test: 2 total" in err
    assert "[#" in err
    assert "(50%)" in err
    assert "first" in err
    assert "second" in err


def test_progress_update_bar_skips_when_quiet(capsys) -> None:
    Progress(quiet=True).update_bar(1, 2, detail="hidden")
    assert capsys.readouterr().err == ""


def test_progress_status_prefix_zero_total() -> None:
    assert Progress.status_prefix(0, 0) == "0/0 (0%)"


def test_fit_status_line_truncates_long_detail() -> None:
    prefix = "1/5 (20%)"
    detail = "x" * 200
    line = _fit_status_line(prefix, detail, 40)
    assert line.startswith(prefix)
    assert "…" in line
    assert len(line) <= 40


def test_progress_update_bar_prints_line(capsys) -> None:
    prog = Progress(quiet=False)
    prog._tty_bar = False
    prog.update_bar(1, 2, detail="working")
    err = capsys.readouterr().err
    assert "[#" in err
    assert "1/2 (50%)" in err
    assert "working" in err


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


def test_main_prints_statistics_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = Path(__file__).resolve().parent.parent / "configs" / "synthetic_smoke.yaml"
    out = tmp_path / "gold.jsonl"
    code = build_gold_mod.main(
        [
            "--source",
            "synthetic",
            "--synthetic-config",
            str(smoke),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "statistics:" in captured.err
    assert "documents:" in captured.err
    assert "entity plan:" in captured.err
    assert "elapsed:" in captured.err


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


def test_resolve_input_path_arxiv_legacy_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARXIV_RAW_DIR", str(tmp_path))
    monkeypatch.delenv("ARXIV_INPUT_DIR", raising=False)
    assert build_gold_mod.resolve_input_path("arxiv", None) == tmp_path


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


def test_progress_verbose_log(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "1")
    monkeypatch.setenv("NER_GOLD_LOG_FILE", "")
    import ner_gold_generator.progress as progress_mod

    progress_mod._LOGGER.handlers.clear()
    progress_mod._LOGGER = progress_mod.get_logger(progress_mod.__name__)
    Progress(quiet=False, verbose=True).verbose_log("loading docs")
    assert "verbose: loading docs" in capsys.readouterr().err


def test_progress_verbose_log_disabled(capsys) -> None:
    Progress(quiet=False, verbose=False).verbose_log("hidden")
    assert "hidden" not in capsys.readouterr().err


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


def test_build_manual_gpt3_match_disambiguation(tmp_path: Path) -> None:
    """Duplicate 'GPT-3' in text requires match= in manual YAML."""
    abstract = (
        "We train GPT-3 early. For all tasks, GPT-3 is applied. GPT-3 achieves strong performance."
    )
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "2005.14165.json").write_text(
        json.dumps({"arxiv_id": "2005.14165", "abstract": abstract}),
        encoding="utf-8",
    )
    manual = tmp_path / "spans.yaml"
    manual.write_text(
        "2005.14165:\n  - text: GPT-3\n    label: model\n    match: GPT-3 achieves\n",
        encoding="utf-8",
    )
    docs = build_gold_mod.load_documents("arxiv", arxiv_dir)
    examples = build_gold_mod.build_examples_manual(
        docs,
        source="arxiv",
        manual_path=manual,
        progress=Progress(quiet=True),
    )
    assert len(examples) == 1
    model_ent = next(e for e in examples[0]["entities"] if e["label"] == "model")
    assert model_ent["text"] == "GPT-3"
    assert abstract[model_ent["start"] : model_ent["end"]] == "GPT-3"
    assert abstract[model_ent["start"] : model_ent["start"] + len("GPT-3 achieves")] == (
        "GPT-3 achieves"
    )


def test_build_manual_duplicate_gpt3_without_match_fails(tmp_path: Path) -> None:
    abstract = "GPT-3 here. GPT-3 there."
    arxiv_dir = tmp_path / "arxiv"
    arxiv_dir.mkdir()
    (arxiv_dir / "2005.14165.json").write_text(
        json.dumps({"arxiv_id": "2005.14165", "abstract": abstract}),
        encoding="utf-8",
    )
    manual = tmp_path / "spans.yaml"
    manual.write_text(
        "2005.14165:\n  - text: GPT-3\n    label: model\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found or not unique"):
        build_gold_mod.build_examples_manual(
            build_gold_mod.load_documents("arxiv", arxiv_dir),
            source="arxiv",
            manual_path=manual,
            progress=Progress(quiet=True),
        )


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


def test_main_verbose_prints_step_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "1")
    monkeypatch.setenv("NER_GOLD_LOG_FILE", "")
    import ner_gold_generator.progress as progress_mod

    progress_mod._LOGGER.handlers.clear()
    progress_mod._LOGGER = progress_mod.get_logger(progress_mod.__name__)
    import ner_gold_generator.llm_logging as llm_logging_mod

    llm_logging_mod._LOGGER.handlers.clear()
    llm_logging_mod._LOGGER = llm_logging_mod.get_logger(llm_logging_mod.__name__)
    smoke = Path(__file__).resolve().parent.parent / "configs" / "synthetic_smoke.yaml"
    out = tmp_path / "gold.jsonl"
    code = build_gold_mod.main(
        [
            "--source",
            "synthetic",
            "--synthetic-config",
            str(smoke),
            "--output",
            str(out),
            "--verbose",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "verbose: starting build" in err
    assert "verbose: loading synthetic config" in err
    assert "verbose:" in err
