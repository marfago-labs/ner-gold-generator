"""Manual gold annotations (built-in arXiv + optional JSON/YAML file)."""

from __future__ import annotations

import json
from pathlib import Path

import ner_gold_generator.manual_entities as arxiv_manual
from ner_gold_generator.manual_spans import ManualSpan, index_manual_span, parse_span_list

__all__ = [
    "ManualSpan",
    "builtin_arxiv_annotations",
    "index_manual_span",
    "load_manual_annotations",
    "parse_span_list",
    "resolve_manual_spans",
]


def builtin_arxiv_annotations() -> dict[str, list[ManualSpan]]:
    """Return built-in arXiv benchmark spans keyed by bare arXiv id.

    Returns:
        Copy of spans from ``configs/arxiv_benchmark.yaml``.
    """
    return {key: list(spans) for key, spans in arxiv_manual.PAPER_ENTITIES.items()}


def load_manual_annotations(path: Path | None) -> dict[str, list[ManualSpan]]:
    """Load manual spans from a JSON or YAML file.

    File format: mapping from document id to a list of objects with ``text`` and
    ``label`` keys. Optional ``match`` disambiguates when ``text`` is not unique.
    Keys may be full ids (``arxiv-1706.03762``) or bare arXiv ids.

    Args:
        path: Annotation file path, or ``None`` for an empty mapping.

    Returns:
        Document id → list of :class:`ManualSpan` entries.

    Raises:
        FileNotFoundError: If *path* is set but not a file.
        ImportError: If the file is YAML and PyYAML is not installed.
        ValueError: If the file structure is invalid.
    """
    out: dict[str, list[ManualSpan]] = {}
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
        spans = parse_span_list(value, doc_key=str(key))
        out[str(key)] = spans
    return out


def resolve_manual_spans(
    doc_id: str,
    *,
    source: str,
    file_annotations: dict[str, list[ManualSpan]],
) -> list[ManualSpan] | None:
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
