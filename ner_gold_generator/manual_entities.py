"""Load built-in arXiv benchmark manual spans from YAML config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ner_gold_generator.manual_spans import ManualSpan, parse_span_list

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARXIV_BENCHMARK_PATH = _REPO_ROOT / "configs" / "arxiv_benchmark.yaml"


@dataclass(frozen=True, slots=True)
class ArxivBenchmarkConfig:
    """Curated arXiv benchmark papers and manual entity spans."""

    paper_order: tuple[str, ...]
    paper_entities: dict[str, tuple[ManualSpan, ...]]


def default_arxiv_benchmark_path(*, repo_root: Path | None = None) -> Path:
    """Return the default ``configs/arxiv_benchmark.yaml`` path."""
    root = repo_root or _REPO_ROOT
    return root / "configs" / "arxiv_benchmark.yaml"


def load_arxiv_benchmark(path: Path | None = None) -> ArxivBenchmarkConfig:
    """Load paper order and manual spans from YAML.

    Args:
        path: Benchmark config file (defaults to :data:`DEFAULT_ARXIV_BENCHMARK_PATH`).

    Returns:
        Parsed benchmark configuration.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the file structure is invalid.
    """
    config_path = path or DEFAULT_ARXIV_BENCHMARK_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"ArXiv benchmark config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"ArXiv benchmark config must be a mapping: {config_path}")
    return _parse_arxiv_benchmark(raw, config_path=config_path)


def _parse_arxiv_benchmark(raw: dict[str, Any], *, config_path: Path) -> ArxivBenchmarkConfig:
    order_raw = raw.get("paper_order")
    papers_raw = raw.get("papers")
    if not isinstance(order_raw, list) or not order_raw:
        raise ValueError(f"paper_order must be a non-empty list in {config_path}")
    if not isinstance(papers_raw, dict) or not papers_raw:
        raise ValueError(f"papers must be a non-empty mapping in {config_path}")

    paper_order = tuple(str(item) for item in order_raw)
    paper_entities: dict[str, tuple[ManualSpan, ...]] = {}
    for arxiv_id, spans_raw in papers_raw.items():
        parsed = parse_span_list(spans_raw, doc_key=str(arxiv_id))
        paper_entities[str(arxiv_id)] = tuple(parsed)

    missing = [pid for pid in paper_order if pid not in paper_entities]
    if missing:
        raise ValueError(f"paper_order ids missing from papers in {config_path}: {missing}")

    return ArxivBenchmarkConfig(paper_order=paper_order, paper_entities=paper_entities)


def _init_module_benchmark() -> ArxivBenchmarkConfig:
    return load_arxiv_benchmark()


_benchmark = _init_module_benchmark()

# Backward-compatible module attributes (tests may monkeypatch these).
PAPER_ORDER: list[str] = list(_benchmark.paper_order)
PAPER_ENTITIES: dict[str, list[ManualSpan]] = {
    key: list(spans) for key, spans in _benchmark.paper_entities.items()
}
