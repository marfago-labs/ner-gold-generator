"""Manual gold annotations (built-in arXiv + optional JSON/YAML file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ner_gold_generator.manual_entities as arxiv_manual


def builtin_arxiv_annotations() -> dict[str, list[tuple[str, str]]]:
    """Return built-in arXiv benchmark spans keyed by bare arXiv id.

    Returns:
        Copy of :data:`manual_entities.PAPER_ENTITIES`.
    """
    return dict(arxiv_manual.PAPER_ENTITIES)


def load_manual_annotations(path: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Load manual spans from a JSON or YAML file.

    File format: mapping from document id to a list of objects with ``text`` and
    ``label`` keys. Keys may be full ids (``arxiv-1706.03762``) or bare arXiv ids.

    Args:
        path: Annotation file path, or ``None`` for an empty mapping.

    Returns:
        Document id → list of ``(surface, label)`` tuples.

    Raises:
        FileNotFoundError: If *path* is set but not a file.
        ImportError: If the file is YAML and PyYAML is not installed.
        ValueError: If the file structure is invalid.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    if path is None:
        return out
    if not path.is_file():
        raise FileNotFoundError(f"Manual annotations not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML manual files: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manual annotations must be a mapping in {path}")
    for key, value in raw.items():
        spans = _parse_span_list(value, doc_key=str(key))
        out[str(key)] = spans
    return out


def _parse_span_list(value: Any, *, doc_key: str) -> list[tuple[str, str]]:
    """Parse one document's span list from manual annotation JSON."""
    if not isinstance(value, list):
        raise ValueError(f"Manual spans for {doc_key!r} must be a list")
    spans: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid span entry for {doc_key!r}")
        text = item.get("text")
        label = item.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            raise ValueError(f"Span needs text and label for {doc_key!r}")
        spans.append((text.strip(), label.strip()))
    return spans


def resolve_manual_spans(
    doc_id: str,
    *,
    source: str,
    file_annotations: dict[str, list[tuple[str, str]]],
) -> list[tuple[str, str]] | None:
    """Resolve manual spans for a document id.

    Lookup order: exact *doc_id* in *file_annotations*, bare arXiv id in file,
    then built-in arXiv table when *source* is ``arxiv``.

    Args:
        doc_id: Canonical id (e.g. ``arxiv-1706.03762``, ``file-posts/a.md``).
        source: Corpus type passed to the CLI ``--source``.
        file_annotations: Spans loaded via :func:`load_manual_annotations`.

    Returns:
        Span list for the document, or ``None`` if no manual gold is defined.
    """
    if doc_id in file_annotations:
        return file_annotations[doc_id]
    bare = doc_id.removeprefix("arxiv-")
    if source == "arxiv" and bare in file_annotations:
        return file_annotations[bare]
    if source == "arxiv":
        builtin = builtin_arxiv_annotations()
        if bare in builtin:
            return builtin[bare]
        if doc_id in builtin:
            return builtin[doc_id]
    return None
