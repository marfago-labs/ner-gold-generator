"""Synthetic gold JSONL generation (entity-first, deterministic structured values)."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from ner_gold_generator.interrupt import BuildInterrupted
from ner_gold_generator.labeling import LlmGoldSettings, build_chat_client
from ner_gold_generator.progress import Progress, short_model_label
from ner_gold_generator.synthetic_compose import (
    assemble_procedural_text,
    audit_plan_entity_usage,
    config_summary_for_meta,
    entities_from_plan,
    generate_text_with_llm,
)
from ner_gold_generator.synthetic_config import (
    SyntheticConfig,
    load_synthetic_config,
    validate_synthetic_config,
)
from ner_gold_generator.synthetic_generators import plan_entities_for_document


def _build_synthetic_row(
    *,
    config: SyntheticConfig,
    doc_id: str,
    doc_idx: int,
    doc_seed: int,
    plan: list,
    text: str,
    text_engine: str,
) -> dict:
    entities = entities_from_plan(text, plan)
    usage = audit_plan_entity_usage(text, plan)
    return {
        "id": doc_id,
        "text": text,
        "entities": entities,
        "source": "synthetic",
        "gold_source": "synthetic",
        "synthetic_meta": {
            **config_summary_for_meta(config),
            "document_index": doc_idx,
            "document_seed": doc_seed,
            "planned_entity_count": len(plan),
            "text_engine": text_engine,
            "entity_usage": usage.to_dict(),
        },
    }


def build_examples_synthetic(
    config: SyntheticConfig,
    *,
    settings: LlmGoldSettings | None = None,
    mock: bool = False,
    progress: Progress | None = None,
) -> list[dict]:
    """Generate gold rows from a synthetic config (procedural or LLM glue).

    Entity spans come from the pre-planned surfaces, not from free-form LLM NER.

    Args:
        config: Parsed synthetic specification.
        settings: LLM settings when ``engine=llm``.
        mock: Force procedural mock text even for LLM engine.
        progress: Optional stderr progress helper.

    Returns:
        Gold example dicts with ``gold_source=synthetic`` and ``synthetic_meta``.

    Raises:
        ValueError: If ``engine`` is unknown or LLM validation fails.
    """
    validate_synthetic_config(config)
    if config.num_documents < 1:
        raise ValueError("num_documents must be at least 1")
    engine = config.engine.lower()
    if engine not in {"procedural", "llm"}:
        raise ValueError(f"Unknown synthetic engine: {config.engine!r}")

    llm_settings = settings or LlmGoldSettings()
    rng = random.Random(config.seed)
    prog = progress or Progress()
    total = config.num_documents
    prog.log(f"build_gold: mode=synthetic engine={engine} documents={total} seed={config.seed}")
    use_procedural = engine == "procedural" or mock or llm_settings.mock_llm
    if engine == "llm" and not use_procedural and not llm_settings.openrouter_api_key.strip():
        raise ValueError("engine=llm requires OPENROUTER_API_KEY in .env (or use --mock for tests)")

    examples_by_idx: dict[int, dict] = {}
    docs: list[tuple[int, int, str, list]] = []
    for doc_idx in range(total):
        doc_seed = rng.randint(0, 2**31 - 1)
        doc_rng = random.Random(doc_seed)
        plan = plan_entities_for_document(doc_rng, config)
        doc_id = f"synthetic-{config.seed:08d}-{doc_idx:05d}"
        docs.append((doc_idx, doc_seed, doc_id, plan))

    if engine == "llm" and not use_procedural:
        build_chat_client(
            llm_settings,
            mock=False,
            model_id=config.llm.model.strip(),
            timeout_seconds=config.llm.timeout_seconds,
        )
        prog.verbose_log(
            f"initialized OpenRouter client (model={config.llm.model.strip()}, timeout={config.llm.timeout_seconds:.0f}s)"
        )
    try:
        with prog.track(total, label=f"synthetic ({engine})") as tick:
            if use_procedural:
                for doc_idx, doc_seed, doc_id, plan in docs:
                    doc_rng = random.Random(doc_seed)
                    text_engine = "procedural"
                    prog.verbose_log(
                        f"{doc_id}: planned {len(plan)} entities (seed={doc_seed}, engine={text_engine})"
                    )
                    tick(doc_idx, f"{doc_id} generating ({text_engine})…")
                    text = assemble_procedural_text(plan, config, doc_rng)
                    row = _build_synthetic_row(
                        config=config,
                        doc_id=doc_id,
                        doc_idx=doc_idx,
                        doc_seed=doc_seed,
                        plan=plan,
                        text=text,
                        text_engine=text_engine,
                    )
                    examples_by_idx[doc_idx] = row
                    prog.verbose_log(f"{doc_id}: finalized with {len(row['entities'])} entities")
                    tick(doc_idx + 1, f"{doc_id} {len(row['entities'])} entities ({text_engine})")
            else:
                text_engine = "llm"
                model_short = short_model_label(config.llm.model)
                max_workers = min(max(1, config.llm.parallelism), total)
                prog.verbose_log(f"LLM parallelism={max_workers}")
                for _, doc_seed, doc_id, plan in docs:
                    prog.verbose_log(
                        f"{doc_id}: planned {len(plan)} entities (seed={doc_seed}, engine={text_engine})"
                    )

                def run_one(
                    doc_idx: int, doc_seed: int, doc_id: str, plan: list
                ) -> tuple[int, dict, list[str]]:
                    elapsed_logs: list[str] = []
                    worker_chat = build_chat_client(
                        llm_settings,
                        mock=False,
                        model_id=config.llm.model.strip(),
                        timeout_seconds=config.llm.timeout_seconds,
                    )

                    def on_call_elapsed(attempt: int, seconds: float, outcome: str) -> None:
                        elapsed_logs.append(
                            f"{doc_id}: LLM call {attempt}/{config.llm.max_attempts} "
                            f"{outcome} in {seconds:.1f}s"
                        )

                    text = generate_text_with_llm(
                        plan,
                        config,
                        settings=llm_settings,
                        mock=False,
                        client=worker_chat,
                        on_status=None,
                        on_call_elapsed=on_call_elapsed,
                        request_delay_seconds=config.llm.request_delay_seconds,
                    )
                    row = _build_synthetic_row(
                        config=config,
                        doc_id=doc_id,
                        doc_idx=doc_idx,
                        doc_seed=doc_seed,
                        plan=plan,
                        text=text,
                        text_engine="llm",
                    )
                    return doc_idx, row, elapsed_logs

                _run_llm_documents_parallel(
                    docs,
                    run_one=run_one,
                    max_workers=max_workers,
                    examples_by_idx=examples_by_idx,
                    prog=prog,
                    model_short=model_short,
                    tick=tick,
                    total=total,
                )
    except KeyboardInterrupt as exc:
        raise BuildInterrupted(
            _ordered_partial_examples(examples_by_idx, total),
            completed=len(examples_by_idx),
            total=total,
        ) from exc
    return [examples_by_idx[i] for i in range(total)]


def _ordered_partial_examples(examples_by_idx: dict[int, dict], total: int) -> list[dict]:
    """Return completed rows in document order (omit not-yet-finished indices)."""
    return [examples_by_idx[i] for i in range(total) if i in examples_by_idx]


def _record_completed_llm_doc(
    *,
    examples_by_idx: dict[int, dict],
    doc_idx: int,
    row: dict,
    elapsed_logs: list[str],
    prog: Progress,
    model_short: str,
    tick: Callable[[int, str], None],
    completed: int,
) -> int:
    """Store one finished LLM document and update progress (main thread only)."""
    examples_by_idx[doc_idx] = row
    for line in elapsed_logs:
        prog.verbose_log(line)
    completed += 1
    doc_id = row["id"]
    prog.verbose_log(
        f"{doc_id}: finalized with {len(row['entities'])} entities (model={model_short})"
    )
    tick(completed, f"{doc_id} {len(row['entities'])} entities (llm)")
    return completed


def _cancel_futures(futures: list[Future]) -> None:
    """Best-effort cancel for tasks not yet running."""
    for future in futures:
        future.cancel()


def _shutdown_executor(pool: ThreadPoolExecutor, *, cancel_pending: bool) -> None:
    """Stop the pool without leaving worker threads dangling."""
    pool.shutdown(wait=not cancel_pending, cancel_futures=cancel_pending)


def _run_llm_documents_parallel(
    docs: list[tuple[int, int, str, list]],
    *,
    run_one: Callable[[int, int, str, list], tuple[int, dict, list[str]]],
    max_workers: int,
    examples_by_idx: dict[int, dict],
    prog: Progress,
    model_short: str,
    tick: Callable[[int, str], None],
    total: int,
) -> None:
    """Run LLM glue for each document with a bounded thread pool.

    Progress updates and shared dict writes happen on the main thread only.
    The pool is always shut down (and pending work cancelled) on error or interrupt.
    When *max_workers* is 1, documents run sequentially (no thread pool).
    """
    if max_workers <= 1:
        completed = 0
        for doc_idx, doc_seed, doc_id, plan in docs:
            doc_idx, row, elapsed_logs = run_one(doc_idx, doc_seed, doc_id, plan)
            completed = _record_completed_llm_doc(
                examples_by_idx=examples_by_idx,
                doc_idx=doc_idx,
                row=row,
                elapsed_logs=elapsed_logs,
                prog=prog,
                model_short=model_short,
                tick=tick,
                completed=completed,
            )
        return

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures: list[Future[tuple[int, dict, list[str]]]] = [
        pool.submit(run_one, doc_idx, doc_seed, doc_id, plan)
        for doc_idx, doc_seed, doc_id, plan in docs
    ]
    completed = 0
    try:
        for future in as_completed(futures):
            doc_idx, row, elapsed_logs = future.result()
            completed = _record_completed_llm_doc(
                examples_by_idx=examples_by_idx,
                doc_idx=doc_idx,
                row=row,
                elapsed_logs=elapsed_logs,
                prog=prog,
                model_short=model_short,
                tick=tick,
                completed=completed,
            )
    except KeyboardInterrupt as exc:
        prog.log("interrupted: cancelling in-flight LLM calls…")
        _cancel_futures(futures)
        _shutdown_executor(pool, cancel_pending=True)
        raise BuildInterrupted(
            _ordered_partial_examples(examples_by_idx, total),
            completed=len(examples_by_idx),
            total=total,
        ) from exc
    except Exception:
        _cancel_futures(futures)
        _shutdown_executor(pool, cancel_pending=True)
        raise
    else:
        _shutdown_executor(pool, cancel_pending=False)


def build_examples_from_config_path(
    config_path: Path,
    *,
    settings: LlmGoldSettings | None = None,
    mock: bool = False,
    progress: Progress | None = None,
) -> list[dict]:
    """Load config from disk and run :func:`build_examples_synthetic`."""
    config = load_synthetic_config(config_path)
    return build_examples_synthetic(
        config,
        settings=settings,
        mock=mock,
        progress=progress,
    )
