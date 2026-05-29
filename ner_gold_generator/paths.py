"""Monorepo path defaults for gold JSONL output."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def labs_root(*, repo_root: Path | None = None) -> Path:
    """Return the parent directory of the ner-gold-generator repo (marfago-labs layout).

    Args:
        repo_root: Override repository root (defaults to this package's parent).

    Returns:
        Path to the directory that typically contains sibling repos such as
        ``text-compressor``, ``ner-detector``, and ``ner-dataset``.
    """
    root = repo_root or _REPO_ROOT
    return root.parent


def dataset_dir(*, repo_root: Path | None = None) -> Path:
    """Directory for generated gold JSONL files.

    Resolution order:

    1. ``NER_DATASET_DIR`` environment variable when set.
    2. ``{labs_root}/ner-dataset/datasets`` (sibling dataset repo).

    The directory is created by callers when writing output; this function does
    not mkdir.
    """
    override = os.environ.get("NER_DATASET_DIR", "").strip()
    if override:
        return Path(override)
    return labs_root(repo_root=repo_root) / "ner-dataset" / "datasets"


def default_output_path(filename: str, *, repo_root: Path | None = None) -> Path:
    """Return a gold JSONL path under :func:`dataset_dir`.

    Args:
        filename: Basename such as ``arxiv_gold.jsonl`` or ``synthetic_mixed.jsonl``.
        repo_root: Optional ner-gold-generator root for tests.
    """
    return dataset_dir(repo_root=repo_root) / filename
