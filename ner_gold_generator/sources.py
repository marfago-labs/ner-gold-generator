"""Load source documents from arXiv, YouTube, folders, or JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_TEXT_JSON_KEYS = (
    "text",
    "abstract",
    "transcript",
    "content",
    "body",
    "markdown",
    "post",
    "article",
)
_ID_JSON_KEYS = (
    "id",
    "arxiv_id",
    "video_id",
    "slug",
    "name",
    "doc_id",
)

SUPPORTED_SOURCES = frozenset({"arxiv", "youtube", "folder", "jsonl"})
"""Corpus types accepted by :func:`load_documents` and the CLI ``--source`` flag."""


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One text item ready for gold labeling.

    Attributes:
        doc_id: Stable example id (e.g. ``arxiv-1706.03762``, ``file-posts/a.md``).
        text: Document body passed to manual or LLM labeling.
        source: Corpus tag: ``arxiv``, ``youtube``, ``folder``, or ``jsonl``.
        path: Optional filesystem path of the source file.
    """

    doc_id: str
    text: str
    source: str
    path: Path | None = None


def default_input_dir(source: str, repo_root: Path) -> Path:
    """Return the default text-compressor raw directory for a corpus.

    Args:
        source: ``arxiv`` or ``youtube`` only.
        repo_root: Path to the ``ner-gold-generator`` repository root.

    Returns:
        ``{repo_parent}/text-compressor/data/raw/{arxiv|youtube}``.

    Raises:
        ValueError: If *source* has no bundled default (use ``--input``).
    """
    base = repo_root.parent / "text-compressor" / "data" / "raw"
    if source == "arxiv":
        return base / "arxiv"
    if source == "youtube":
        return base / "youtube"
    msg = f"No default input path for source {source!r}; use --input"
    raise ValueError(msg)


def load_documents(
    source: str,
    input_path: Path,
    *,
    limit: int | None = None,
    text_field: str | None = None,
    glob_pattern: str = "*",
) -> list[SourceDocument]:
    """Load documents for a corpus type from disk.

    Args:
        source: One of :data:`SUPPORTED_SOURCES`.
        input_path: Directory (arxiv/youtube/folder) or JSONL file path.
        limit: If set, return only the first *limit* documents after sorting.
        text_field: Override JSON/JSONL field used as document body.
        glob_pattern: Glob under *input_path* for ``folder`` source (default ``*``).

    Returns:
        Loaded documents in stable sorted order.

    Raises:
        ValueError: Unsupported *source* or missing text fields.
        FileNotFoundError: Missing input directory or JSONL file.
    """
    key = source.strip().lower()
    if key not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source {source!r}; choose from {sorted(SUPPORTED_SOURCES)}")
    if key == "jsonl":
        docs = _load_jsonl(input_path, text_field=text_field)
    elif not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    elif key == "arxiv":
        docs = _load_arxiv_dir(input_path, text_field=text_field)
    elif key == "youtube":
        docs = _load_youtube_dir(input_path, text_field=text_field)
    else:
        docs = _load_folder(input_path, glob_pattern=glob_pattern, text_field=text_field)
    if limit is not None:
        return docs[:limit]
    return docs


def _load_json_file(path: Path, *, text_field: str | None) -> tuple[str, str]:
    """Load one JSON file and return ``(doc_id, text)``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    doc_id = _extract_id(raw, fallback=path.stem)
    text = _extract_text(raw, text_field=text_field, path=path)
    return doc_id, text


def _extract_id(raw: dict[str, object], *, fallback: str) -> str:
    """Pick the first non-empty id field from a JSON object."""
    for key in _ID_JSON_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _extract_text(
    raw: dict[str, object],
    *,
    text_field: str | None,
    path: Path,
) -> str:
    """Extract document body text from a JSON object."""
    if text_field:
        value = raw.get(text_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty field {text_field!r} in {path}")
        return value.strip()
    for key in _TEXT_JSON_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"No text field in {path}; tried {list(_TEXT_JSON_KEYS)} (override with --text-field)"
    )


def _load_arxiv_dir(
    directory: Path,
    *,
    text_field: str | None,
) -> list[SourceDocument]:
    """Load ``*.json`` arXiv paper files from *directory*."""
    docs: list[SourceDocument] = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        arxiv_id = str(raw.get("arxiv_id", path.stem)).strip()
        text = _extract_text(raw, text_field=text_field or "abstract", path=path)
        docs.append(
            SourceDocument(
                doc_id=f"arxiv-{arxiv_id}",
                text=text,
                source="arxiv",
                path=path,
            )
        )
    return docs


def _load_youtube_dir(
    directory: Path,
    *,
    text_field: str | None,
) -> list[SourceDocument]:
    """Load ``*.json`` YouTube metadata/transcript files from *directory*."""
    docs: list[SourceDocument] = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        video_id = str(raw.get("video_id", path.stem)).strip()
        if text_field:
            text = _extract_text(raw, text_field=text_field, path=path)
        else:
            text = _youtube_text(raw, path=path)
        docs.append(
            SourceDocument(
                doc_id=f"youtube-{video_id}",
                text=text,
                source="youtube",
                path=path,
            )
        )
    return docs


def _youtube_text(raw: dict[str, object], *, path: Path) -> str:
    """Prefer transcript; else join title and description."""
    transcript = raw.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return transcript.strip()
    title = raw.get("title")
    description = raw.get("description")
    parts: list[str] = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    if parts:
        return "\n\n".join(parts)
    raise ValueError(f"No transcript, title, or description in {path}")


def _load_folder(
    directory: Path,
    *,
    glob_pattern: str,
    text_field: str | None,
) -> list[SourceDocument]:
    """Load ``.txt``, ``.md``, and ``.json`` files under *directory*."""
    docs: list[SourceDocument] = []
    paths = sorted(directory.glob(glob_pattern))
    for path in paths:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            rel = path.relative_to(directory).as_posix()
            docs.append(
                SourceDocument(
                    doc_id=f"file-{rel}",
                    text=text,
                    source="folder",
                    path=path,
                )
            )
        elif suffix == ".json":
            doc_id, text = _load_json_file(path, text_field=text_field)
            docs.append(
                SourceDocument(
                    doc_id=f"file-{doc_id}",
                    text=text,
                    source="folder",
                    path=path,
                )
            )
    return docs


def _load_jsonl(path: Path, *, text_field: str | None) -> list[SourceDocument]:
    """Load one JSON object per non-empty line from *path*."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    field = text_field or "text"
    docs: list[SourceDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid JSONL line {line_no} in {path}")
            doc_id = _extract_id(raw, fallback=f"line-{line_no}")
            text_value = raw.get(field)
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(f"Missing or empty {field!r} on line {line_no} in {path}")
            docs.append(
                SourceDocument(
                    doc_id=str(doc_id),
                    text=text_value.strip(),
                    source="jsonl",
                    path=path,
                )
            )
    return docs
