"""Backward-compatible CLI entry point for arXiv gold only.

Exposes ``build-arxiv-gold``, which delegates to :func:`build_gold.main` with
``--source arxiv`` and a default output path under ``ner-detector``.
"""

from __future__ import annotations

from pathlib import Path

from ner_gold_generator.build_gold import main as build_gold_main

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT.parent / "ner-detector" / "benchmark" / "datasets" / "arxiv_gold.jsonl"


def _ensure_arxiv_argv(argv: list[str] | None) -> list[str]:
    """Inject arXiv defaults and map legacy ``--arxiv-dir`` to ``--input``.

    Args:
        argv: Raw CLI arguments (without program name).

    Returns:
        Argument list with ``--source arxiv``, default ``--output``, and
        ``--arxiv-dir`` rewritten when present.
    """
    args = list(argv or [])
    if "--source" not in args:
        args = ["--source", "arxiv", *args]
    if "--output" not in args:
        args = [*args, "--output", str(_DEFAULT_OUTPUT)]
    if "--arxiv-dir" in args:
        idx = args.index("--arxiv-dir")
        if idx + 1 < len(args):
            path = args[idx + 1]
            args = [
                *args[:idx],
                "--input",
                path,
                *args[idx + 2 :],
            ]
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``build-arxiv-gold``.

    See :func:`build_gold.main` for exit codes and behavior.
    """
    return build_gold_main(_ensure_arxiv_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
