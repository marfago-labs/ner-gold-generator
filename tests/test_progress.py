"""Tests for progress helpers."""

from __future__ import annotations

import pytest

from ner_gold_generator.progress import (
    Progress,
    _fit_status_line,
    _stderr_columns,
    format_skip_error,
    render_bar,
    short_model_label,
)


def test_short_model_label_truncates() -> None:
    long_id = "vendor/" + "x" * 40
    assert len(short_model_label(long_id, max_len=20)) <= 20


def test_format_skip_error_validation_and_provider() -> None:
    err = (
        "Synthetic LLM failed (model='m', reason=validation, llm_calls=3/5, "
        "validation_failures=2/3).\n  Surface 'BERT' appears 2 times"
    )
    out = format_skip_error(err)
    assert "validation" in out
    assert "Surface" in out or "BERT" in out

    prov = "Synthetic LLM failed (model='m', reason=provider, provider_failures=2/2)"
    assert "provider" in format_skip_error(prov)


def test_render_bar_edge_cases() -> None:
    assert "?" in render_bar(0, 0)
    assert render_bar(5, 10).startswith("[")


def test_fit_status_line_and_stderr_columns() -> None:
    line = _fit_status_line("prefix:", "detail text here", 40)
    assert "prefix:" in line
    assert _stderr_columns() >= 60


def test_progress_track_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ner_gold_generator.progress.sys.stderr.isatty", lambda: False)
    prog = Progress(quiet=False)
    with prog.track(2, label="test") as tick:
        tick(1, "step one")
        tick(2, "done")
    prog.log_skip("doc-1", "failed")


def test_progress_verbose_log_quiet() -> None:
    prog = Progress(quiet=True)
    prog.verbose_log("hidden")


def test_progress_rich_track() -> None:
    pytest.importorskip("rich")
    prog = Progress(quiet=False, use_rich=True)
    with prog.track(2, label="docs") as tick:
        tick(1, "one")
        prog.log("status line")
        prog.log_skip("d1", "reason=provider provider_failures=1/2")
        tick(2)
    prog.update_bar(1, 2, detail="x" * 200)


def test_progress_set_activity_does_not_advance_completed() -> None:
    pytest.importorskip("rich")
    prog = Progress(quiet=False, use_rich=True)
    with prog.track(3, label="synthetic") as tick:
        prog.set_activity("doc-a: calling LLM…")
        tick(0, "")
        prog.set_activity("doc-b: rate limited — 12s left")
        tick(1, "doc-a done")
        prog.set_activity("")
        tick(2, "doc-b done")


def test_format_skip_error_fallbacks() -> None:
    assert "validation failed" in format_skip_error("reason=validation foo")
    assert "provider" in format_skip_error("reason=provider provider_failures=1/1")
    long_err = "x" * 300
    assert len(format_skip_error(long_err)) <= 200


def test_progress_plain_track_exception_finishes_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ner_gold_generator.progress.sys.stderr.isatty", lambda: False)
    prog = Progress(quiet=False, use_rich=False)
    with pytest.raises(RuntimeError), prog.track(1, label="docs") as tick:
        tick(1)
        raise RuntimeError("boom")


def test_progress_tty_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ner_gold_generator.progress.sys.stderr.isatty", lambda: True)
    prog = Progress(quiet=False, use_rich=False, _tty_bar=True)
    prog.update_bar(1, 3, detail="working")
    prog.finish_bar()
