"""Tests for corpus source loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner_gold_generator.sources import (
    default_input_dir,
    load_documents,
)


def test_load_arxiv_json(tmp_path: Path) -> None:
    (tmp_path / "1706.03762.json").write_text(
        json.dumps({"arxiv_id": "1706.03762", "abstract": "Transformer on WMT."}),
        encoding="utf-8",
    )
    docs = load_documents("arxiv", tmp_path)
    assert len(docs) == 1
    assert docs[0].doc_id == "arxiv-1706.03762"
    assert "Transformer" in docs[0].text


def test_load_youtube_prefers_transcript(tmp_path: Path) -> None:
    (tmp_path / "abc.json").write_text(
        json.dumps(
            {
                "video_id": "abc",
                "title": "Talk",
                "description": "Desc",
                "transcript": "OpenAI discusses GPT-3.",
            }
        ),
        encoding="utf-8",
    )
    docs = load_documents("youtube", tmp_path)
    assert docs[0].text == "OpenAI discusses GPT-3."
    assert docs[0].doc_id == "youtube-abc"


def test_load_youtube_title_description_fallback(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(
        json.dumps({"video_id": "x", "title": "T", "description": "D"}),
        encoding="utf-8",
    )
    docs = load_documents("youtube", tmp_path)
    assert docs[0].text == "T\n\nD"


def test_load_folder_txt_and_md(tmp_path: Path) -> None:
    (tmp_path / "post.md").write_text("# Blog\n\nCLIP model.", encoding="utf-8")
    (tmp_path / "note.txt").write_text("BERT fine-tuning.", encoding="utf-8")
    docs = load_documents("folder", tmp_path)
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert any("CLIP model." in t for t in texts)


def test_load_folder_json(tmp_path: Path) -> None:
    (tmp_path / "article.json").write_text(
        json.dumps({"slug": "my-post", "content": "ImageNet results."}),
        encoding="utf-8",
    )
    docs = load_documents("folder", tmp_path)
    assert docs[0].doc_id == "file-my-post"
    assert docs[0].text == "ImageNet results."


def test_load_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"id": "b1", "text": "GLUE benchmark."}) + "\n",
        encoding="utf-8",
    )
    docs = load_documents("jsonl", path)
    assert docs[0].doc_id == "b1"


def test_load_documents_unsupported_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        load_documents("blogs", tmp_path)


def test_default_input_dir_arxiv() -> None:
    root = Path(__file__).resolve().parents[1]
    path = default_input_dir("arxiv", root)
    assert path.name == "arxiv"
    assert "text-compressor" in str(path)


def test_default_input_dir_youtube() -> None:
    root = Path(__file__).resolve().parents[1]
    assert default_input_dir("youtube", root).name == "youtube"


def test_default_input_dir_unknown() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="No default"):
        default_input_dir("folder", root)


def test_load_arxiv_skips_non_object_json(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("[1, 2]\n", encoding="utf-8")
    (tmp_path / "ok.json").write_text(
        json.dumps({"arxiv_id": "1", "abstract": "text"}),
        encoding="utf-8",
    )
    assert len(load_documents("arxiv", tmp_path)) == 1


def test_load_documents_not_a_directory(tmp_path: Path) -> None:
    file = tmp_path / "notadir.txt"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="directory"):
        load_documents("arxiv", file)


def test_load_arxiv_custom_text_field(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(
        json.dumps({"arxiv_id": "x", "abstract": "A", "note": "custom body"}),
        encoding="utf-8",
    )
    docs = load_documents("arxiv", tmp_path, text_field="note")
    assert docs[0].text == "custom body"


def test_load_folder_skips_empty_txt(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("BERT", encoding="utf-8")
    docs = load_documents("folder", tmp_path)
    assert len(docs) == 1


def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents("jsonl", tmp_path / "nope.jsonl")


def test_load_jsonl_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_documents("jsonl", path)


def test_load_folder_json_missing_text(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="text field"):
        load_documents("folder", tmp_path)


def test_youtube_no_text_raises(tmp_path: Path) -> None:
    (tmp_path / "z.json").write_text(json.dumps({"video_id": "z"}), encoding="utf-8")
    with pytest.raises(ValueError, match="transcript"):
        load_documents("youtube", tmp_path)


def test_load_documents_limit(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text(f"text {i}", encoding="utf-8")
    docs = load_documents("folder", tmp_path, limit=2)
    assert len(docs) == 2
