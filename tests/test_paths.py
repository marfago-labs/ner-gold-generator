"""Tests for monorepo dataset path defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_gold_generator.paths import dataset_dir, default_output_path, labs_root


def test_labs_root_is_parent_of_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "ner-gold-generator"
    repo.mkdir()
    assert labs_root(repo_root=repo) == tmp_path


def test_dataset_dir_default_sibling_ner_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "ner-gold-generator"
    repo.mkdir()
    monkeypatch.delenv("NER_DATASET_DIR", raising=False)
    assert dataset_dir(repo_root=repo) == tmp_path / "ner-dataset" / "datasets"


def test_dataset_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom-datasets"
    monkeypatch.setenv("NER_DATASET_DIR", str(custom))
    assert dataset_dir() == custom


def test_default_output_path_joins_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "ner-gold-generator"
    repo.mkdir()
    monkeypatch.delenv("NER_DATASET_DIR", raising=False)
    assert default_output_path("arxiv_gold.jsonl", repo_root=repo) == (
        tmp_path / "ner-dataset" / "datasets" / "arxiv_gold.jsonl"
    )
