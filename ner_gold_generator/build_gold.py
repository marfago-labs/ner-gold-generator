"""Build NER gold JSONL from multiple corpus sources.

Provides the ``build-gold`` CLI (:func:`main`) and helpers to assemble manual or
LLM-labeled examples written as JSONL for downstream benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import ner_gold_generator.manual_entities as arxiv_manual
from ner_gold_generator.env import load_project_env
from ner_gold_generator.labeling import (
    LlmGoldSettings,
    default_min_votes,
    entities_to_gold_dict,
    label_text_with_llm_consensus,
)
from ner_gold_generator.manual import load_manual_annotations, resolve_manual_spans
from ner_gold_generator.progress import Progress
from ner_gold_generator.sources import SourceDocument, default_input_dir, load_documents
from ner_gold_generator.validate import index_unique_span

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT.parent / "ner-detector" / "benchmark" / "datasets" / "arxiv_gold.jsonl"


def _index_entity(text: str, surface: str) -> tuple[int, int]:
    """Anchor *surface* in *text* or raise if not uniquely found."""
    span = index_unique_span(text, surface)
    if span is None:
        raise ValueError(f"Surface {surface!r} not found or not unique in text")
    return span


def _filter_arxiv_builtin_order(docs: list[SourceDocument]) -> list[SourceDocument]:
    """Keep only built-in benchmark papers in :data:`manual_entities.PAPER_ORDER`."""
    by_id = {doc.doc_id: doc for doc in docs}
    filtered: list[SourceDocument] = []
    for arxiv_id in arxiv_manual.PAPER_ORDER:
        doc_id = f"arxiv-{arxiv_id}"
        if doc_id in by_id:
            filtered.append(by_id[doc_id])
    return filtered


def build_examples_manual(
    docs: list[SourceDocument],
    *,
    source: str,
    manual_path: Path | None,
    progress: Progress | None = None,
) -> list[dict]:
    """Build gold example dicts from manual span tables.

    For ``source=arxiv`` without ``manual_path``, uses built-in spans and
    :data:`manual_entities.PAPER_ORDER`. Documents without manual spans are
    skipped.

    Args:
        docs: Loaded source documents.
        source: CLI corpus type.
        manual_path: Optional JSON/YAML annotation file.
        progress: Optional stderr progress helper.

    Returns:
        List of gold row dicts (``id``, ``text``, ``entities``, ``source``).

    Raises:
        ValueError: If no documents or no examples are produced.
    """
    file_ann = load_manual_annotations(manual_path)
    use_builtin_order = source == "arxiv" and manual_path is None
    if use_builtin_order:
        docs = _filter_arxiv_builtin_order(docs)
    if not docs:
        raise ValueError("No documents to annotate in manual mode")

    prog = progress or Progress()
    total = len(docs)
    prog.log(f"build_gold: mode=manual source={source} documents={total}")

    examples: list[dict] = []
    for idx, doc in enumerate(docs, start=1):
        spans = resolve_manual_spans(doc.doc_id, source=source, file_annotations=file_ann)
        if spans is None:
            prog.log(f"{Progress.bar(idx, total)} {doc.doc_id}  skipped (no manual spans)")
            continue
        entities = []
        for surface, label in spans:
            start, end = _index_entity(doc.text, surface)
            entities.append({"text": surface, "label": label, "start": start, "end": end})
        row: dict = {
            "id": doc.doc_id,
            "text": doc.text,
            "entities": entities,
            "source": doc.source,
        }
        examples.append(row)
        prog.log(f"{Progress.bar(idx, total)} {doc.doc_id}  {len(entities)} entities (manual)")
    if not examples:
        raise ValueError(
            "No examples produced in manual mode; add --manual or use source=arxiv "
            "with built-in annotations"
        )
    return examples


def build_examples_llm(
    docs: list[SourceDocument],
    *,
    settings: LlmGoldSettings,
    num_samples: int | None,
    min_votes: int | None,
    mock: bool,
    progress: Progress | None = None,
) -> list[dict]:
    """Build gold example dicts via multi-sample LLM consensus.

    Args:
        docs: Loaded source documents.
        settings: OpenRouter / LLM configuration.
        num_samples: Override pass count per document.
        min_votes: Override consensus threshold.
        mock: Use deterministic mock LLM.
        progress: Optional stderr progress helper.

    Returns:
        Gold rows including ``gold_source`` and ``llm_meta`` when applicable.

    Raises:
        ValueError: If *docs* is empty.
    """
    if not docs:
        raise ValueError("No documents to label")

    prog = progress or Progress()
    n_samples = num_samples if num_samples is not None else settings.llm_gold_samples
    votes = (
        min_votes
        if min_votes is not None
        else default_min_votes(n_samples, settings.llm_gold_min_votes)
    )
    model = "mock" if mock or settings.mock_llm else settings.llm_gold_model
    total = len(docs)
    source = docs[0].source if docs else "unknown"
    prog.log(
        f"build_gold: mode=llm source={source} documents={total} "
        f"samples={n_samples} min_votes={votes} model={model}"
    )

    examples: list[dict] = []
    for idx, doc in enumerate(docs, start=1):
        prefix = f"{Progress.bar(idx, total)} {doc.doc_id}"

        def on_sample(sample_idx: int, sample_total: int, *, pfx: str = prefix) -> None:
            prog.log(f"{pfx}  LLM pass {Progress.bar(sample_idx, sample_total)}")

        result = label_text_with_llm_consensus(
            doc.text,
            settings=settings,
            num_samples=num_samples,
            min_votes=min_votes,
            mock=mock,
            on_sample=on_sample,
        )
        examples.append(
            {
                "id": doc.doc_id,
                "text": doc.text,
                "entities": entities_to_gold_dict(result.entities),
                "source": doc.source,
                "gold_source": "llm",
                "llm_meta": {
                    "samples": result.sample_runs,
                    "min_votes": result.min_votes,
                    "per_run_counts": result.raw_run_counts,
                    "model": settings.llm_gold_model,
                },
            }
        )
        prog.log(f"{prefix}  done: {len(result.entities)} entities (runs={result.raw_run_counts})")
    return examples


def resolve_input_path(source: str, input_path: Path | None) -> Path:
    """Resolve CLI ``--input`` from explicit path, env, or defaults.

    Checks ``{SOURCE}_INPUT_DIR``, then ``ARXIV_RAW_DIR`` for arXiv, then
    :func:`sources.default_input_dir`.

    Args:
        source: Corpus type (``arxiv``, ``youtube``, …).
        input_path: Explicit ``--input`` value, if any.

    Returns:
        Resolved filesystem path.
    """
    if input_path is not None:
        return input_path
    env_key = f"{source.upper()}_INPUT_DIR"
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return Path(env_val)
    if source == "arxiv":
        legacy = os.environ.get("ARXIV_RAW_DIR", "").strip()
        if legacy:
            return Path(legacy)
    return default_input_dir(source, _REPO_ROOT)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse ``build-gold`` command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("arxiv", "youtube", "folder", "jsonl"),
        default="arxiv",
        help="Input corpus type (default: arxiv)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory or JSONL file (default: text-compressor raw path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--mode",
        choices=("manual", "llm"),
        default="manual",
        help="manual=curated spans; llm=multi-sample OpenRouter consensus",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=None,
        help="JSON/YAML manual spans by document id (required for manual non-arxiv)",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="JSON/JSONL field name for body text (e.g. transcript, content)",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob under --input for folder source (default: *)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="LLM passes per document",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=None,
        help="Minimum agreeing LLM passes",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Deterministic mock LLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress on stderr",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``build-gold``.

    Loads ``.env``, reads documents, labels them, and writes JSONL to
    ``--output``.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``2`` for usage errors, other non-zero on failure.
    """
    load_project_env()
    args = _parse_args(argv)
    settings = LlmGoldSettings()
    progress = Progress(quiet=args.quiet)

    if args.mode == "manual" and args.source != "arxiv" and args.manual is None:
        print(
            "error: manual mode for non-arxiv sources requires --manual annotations file",
            file=sys.stderr,
        )
        return 2

    input_path = resolve_input_path(args.source, args.input)
    docs = load_documents(
        args.source,
        input_path,
        limit=args.limit,
        text_field=args.text_field,
        glob_pattern=args.glob,
    )

    if args.mode == "manual":
        examples = build_examples_manual(
            docs,
            source=args.source,
            manual_path=args.manual,
            progress=progress,
        )
    else:
        examples = build_examples_llm(
            docs,
            settings=settings,
            num_samples=args.samples,
            min_votes=args.min_votes,
            mock=args.mock or settings.mock_llm,
            progress=progress,
        )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(ex, ensure_ascii=False) for ex in examples]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.quiet:
        sys.stderr.flush()
    print(f"Wrote {out} ({len(examples)} examples, source={args.source}, mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
