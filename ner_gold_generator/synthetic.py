"""Synthetic gold JSONL generation (entity-first, deterministic structured values).

Plans entity surfaces from a YAML config and seed, generates document text
(procedural templates or LLM prose), validates each planned surface appears
exactly once, then exports gold offsets from the plan—not from LLM NER.

Primary entry points:

- :func:`build_examples_from_config_path` — load ``configs/synthetic_*.yaml``
- :func:`build_examples_synthetic` — run from an in-memory :class:`~ner_gold_generator.synthetic_config.SyntheticConfig`

See ``docs/synthetic.md`` for config fields, document profiles, and logging.
"""

from __future__ import annotations

import asyncio
import random
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ner_gold_generator.document_profiles import (
    get_document_profile,
    load_document_profiles,
    merge_config_for_profile,
    resolve_profiles_dir,
    sample_document_type,
)
from ner_gold_generator.interrupt import BuildInterrupted
from ner_gold_generator.labeling import LlmGoldSettings, build_chat_client
from ner_gold_generator.llm_client import build_async_chat_client
from ner_gold_generator.logger import get_logger
from ner_gold_generator.progress import Progress, short_model_label
from ner_gold_generator.synthetic_compose import (
    assemble_procedural_text,
    audit_plan_entity_usage,
    config_summary_for_meta,
    entities_from_plan,
    generate_text_with_llm_async,
)
from ner_gold_generator.synthetic_config import (
    SyntheticConfig,
    load_synthetic_config,
    validate_synthetic_config,
)
from ner_gold_generator.synthetic_generators import plan_entities_for_document

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SkippedDocument:
    """A synthetic document that failed after all LLM retries."""

    id: str
    document_index: int
    document_seed: int
    error: str

    def to_dict(self) -> dict:
        """Serialize for stderr skip summaries and structured logs."""
        return {
            "id": self.id,
            "document_index": self.document_index,
            "document_seed": self.document_seed,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class SyntheticBuildResult:
    """Outcome of a synthetic build (successful rows plus skipped documents)."""

    examples: list[dict]
    skipped: list[SkippedDocument]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.examples)


@dataclass(frozen=True, slots=True)
class _DocWorkItem:
    doc_idx: int
    doc_seed: int
    doc_id: str
    plan: list
    doc_config: SyntheticConfig
    document_type: str
    style_description: str


def _build_synthetic_row(
    *,
    config: SyntheticConfig,
    doc_id: str,
    doc_idx: int,
    doc_seed: int,
    plan: list,
    text: str,
    text_engine: str,
    document_type: str,
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
            "document_type": document_type,
            "domain": document_type,
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
    config_path: Path | None = None,
    document_log_dir: str | Path | None = None,
) -> SyntheticBuildResult:
    """Generate gold rows from a synthetic config (procedural or LLM glue).

    Entity spans come from the pre-planned surfaces, not from free-form LLM NER.

    Args:
        config: Parsed synthetic specification.
        settings: LLM settings when ``engine=llm``.
        mock: Force procedural mock text even for LLM engine.
        progress: Optional stderr progress helper.
        config_path: Path the config was loaded from (for profile resolution).
        document_log_dir: Override ``llm.document_log_dir`` (CLI ``--document-log-dir``).

    Returns:
        Successful gold rows and any documents skipped after LLM retries.

    Raises:
        ValueError: If ``engine`` is unknown or every document fails.
    """
    validate_synthetic_config(config)
    if config.num_documents < 1:
        raise ValueError("num_documents must be at least 1")

    from ner_gold_generator.llm_logging import configure_document_log_dir

    log_dir_raw = document_log_dir if document_log_dir is not None else config.llm.document_log_dir
    log_dir_path = configure_document_log_dir(
        str(log_dir_raw).strip() if log_dir_raw else None,
        relative_to=config_path.parent if config_path is not None else None,
    )

    engine = config.engine.lower()
    if engine not in {"procedural", "llm"}:
        raise ValueError(f"Unknown synthetic engine: {config.engine!r}")

    llm_settings = settings or LlmGoldSettings()
    rng = random.Random(config.seed)
    prog = progress or Progress()
    total = config.num_documents
    profiles = load_document_profiles(resolve_profiles_dir(config, config_path))
    prog.log(f"build_gold: mode=synthetic engine={engine} documents={total} seed={config.seed}")
    if log_dir_path is not None:
        prog.log(f"document LLM logs: {log_dir_path}")
    use_procedural = engine == "procedural" or mock or llm_settings.mock_llm
    if engine == "llm" and not use_procedural and not llm_settings.openrouter_api_key.strip():
        raise ValueError("engine=llm requires OPENROUTER_API_KEY in .env (or use --mock for tests)")

    examples_by_idx: dict[int, dict] = {}
    skipped: list[SkippedDocument] = []
    docs: list[_DocWorkItem] = []
    for doc_idx in range(total):
        doc_seed = rng.randint(0, 2**31 - 1)
        doc_rng = random.Random(doc_seed)
        doc_type = sample_document_type(config, doc_rng)
        profile = get_document_profile(config, doc_type, profiles=profiles, config_path=config_path)
        doc_config = merge_config_for_profile(config, profile)
        plan = plan_entities_for_document(doc_rng, doc_config)
        doc_id = f"synthetic-{config.seed:08d}-{doc_idx:05d}"
        docs.append(
            _DocWorkItem(
                doc_idx=doc_idx,
                doc_seed=doc_seed,
                doc_id=doc_id,
                plan=plan,
                doc_config=doc_config,
                document_type=profile.profile_id,
                style_description=profile.style_description,
            )
        )

    if engine == "llm" and not use_procedural:
        agent_mode = config.llm.agent_mode
        _LOGGER.info(
            "synthetic batch start engine=llm agent_mode=%s parallelism=%s model=%s",
            agent_mode,
            config.llm.parallelism,
            config.llm.model.strip(),
        )
        if agent_mode == "stateless":
            build_chat_client(
                llm_settings,
                mock=False,
                model_id=config.llm.model.strip(),
                timeout_seconds=config.llm.timeout_seconds,
            )
            prog.verbose_log(
                f"initialized OpenRouter client (model={config.llm.model.strip()}, "
                f"timeout={config.llm.timeout_seconds:.0f}s)"
            )
        else:
            prog.verbose_log(f"LLM agent_mode={agent_mode} (per-document Agno sessions)")
    try:
        with prog.track(total, label=f"synthetic ({engine})") as tick:
            if use_procedural:
                for item in docs:
                    doc_rng = random.Random(item.doc_seed)
                    text_engine = "procedural"
                    prog.verbose_log(
                        f"{item.doc_id}: planned {len(item.plan)} entities "
                        f"(seed={item.doc_seed}, type={item.document_type}, engine={text_engine})"
                    )
                    tick(item.doc_idx, f"{item.doc_id} generating ({text_engine})…")
                    text = assemble_procedural_text(item.plan, item.doc_config, doc_rng)
                    row = _build_synthetic_row(
                        config=config,
                        doc_id=item.doc_id,
                        doc_idx=item.doc_idx,
                        doc_seed=item.doc_seed,
                        plan=item.plan,
                        text=text,
                        text_engine=text_engine,
                        document_type=item.document_type,
                    )
                    examples_by_idx[item.doc_idx] = row
                    prog.verbose_log(
                        f"{item.doc_id}: finalized with {len(row['entities'])} entities"
                    )
                    tick(
                        item.doc_idx + 1,
                        f"{item.doc_id} {len(row['entities'])} entities ({text_engine})",
                    )
            else:
                text_engine = "llm"
                model_short = short_model_label(config.llm.model)
                max_workers = min(max(1, config.llm.parallelism), total)
                prog.verbose_log(f"LLM parallelism={max_workers}")
                for item in docs:
                    prog.verbose_log(
                        f"{item.doc_id}: planned {len(item.plan)} entities "
                        f"(seed={item.doc_seed}, type={item.document_type}, engine={text_engine})"
                    )
                llm_out, llm_skipped = asyncio.run(
                    _run_llm_documents_async(
                        docs,
                        config=config,
                        llm_settings=llm_settings,
                        max_workers=max_workers,
                        prog=prog,
                        model_short=model_short,
                        tick=tick,
                    )
                )
                examples_by_idx.update(llm_out)
                skipped.extend(llm_skipped)
    except KeyboardInterrupt as exc:
        raise BuildInterrupted(
            _ordered_partial_examples(examples_by_idx, total),
            completed=len(examples_by_idx),
            total=total,
        ) from exc
    examples = [examples_by_idx[i] for i in sorted(examples_by_idx)]
    if not examples and skipped:
        first = skipped[0].error.replace("\n", " ")
        raise ValueError(f"All {len(skipped)} synthetic documents failed (first error: {first})")
    if skipped:
        prog.log(f"skipped {len(skipped)}/{total} documents after LLM retries (see gold.log)")
    if engine == "llm" and not use_procedural:
        _LOGGER.info(
            "synthetic batch done succeeded=%s skipped=%s (see gold.log for failure details)",
            len(examples),
            len(skipped),
        )
    return SyntheticBuildResult(examples=examples, skipped=skipped)


def _ordered_partial_examples(examples_by_idx: dict[int, dict], total: int) -> list[dict]:
    """Return completed rows in document order (omit not-yet-finished indices)."""
    return [examples_by_idx[i] for i in range(total) if i in examples_by_idx]


class _InflightActivity:
    """Aggregate per-document in-flight status into one progress activity line."""

    def __init__(self, prog: Progress) -> None:
        self._prog = prog
        self._lock = threading.Lock()
        self._docs: dict[str, str] = {}

    def set(self, doc_id: str, message: str | None) -> None:
        with self._lock:
            if message is None:
                self._docs.pop(doc_id, None)
            else:
                self._docs[doc_id] = message
            summary = self._summarize_locked()
        self._prog.set_activity(summary)

    def _summarize_locked(self) -> str:
        if not self._docs:
            return ""
        if len(self._docs) == 1:
            doc_id, msg = next(iter(self._docs.items()))
            return f"{doc_id}: {msg}"
        return f"{len(self._docs)} documents in flight"


async def _run_llm_documents_async(
    docs: list[_DocWorkItem],
    *,
    config: SyntheticConfig,
    llm_settings: LlmGoldSettings,
    max_workers: int,
    prog: Progress,
    model_short: str,
    tick: Callable[[int, str], None],
) -> tuple[dict[int, dict], list[SkippedDocument]]:
    """Run LLM glue for each document using asyncio concurrency."""

    semaphore = asyncio.Semaphore(max(1, max_workers))
    completed = 0
    out: dict[int, dict] = {}
    skipped: list[SkippedDocument] = []
    use_agno = config.llm.agent_mode == "per_doc"
    inflight = _InflightActivity(prog)

    async def run_one_async(
        item: _DocWorkItem,
    ) -> tuple[int, str, int, dict | None, list[str], str | None]:
        elapsed_logs: list[str] = []

        try:
            async with semaphore:
                client = None
                if not use_agno:
                    client = build_async_chat_client(
                        llm_settings,
                        model_id=config.llm.model.strip(),
                        timeout_seconds=config.llm.timeout_seconds,
                    )

                def on_call_elapsed(attempt: int, seconds: float, outcome: str) -> None:
                    elapsed_logs.append(
                        f"{item.doc_id}: LLM call {attempt}/{config.llm.max_attempts} "
                        f"{outcome} in {seconds:.1f}s"
                    )

                def on_status(status: str) -> None:
                    inflight.set(item.doc_id, status)
                    prog.verbose_log(f"{item.doc_id}: {status}")

                def on_wait(remaining: float, _detail: str) -> None:
                    inflight.set(item.doc_id, f"rate limited — {remaining:.0f}s left")

                try:
                    text, final_plan = await generate_text_with_llm_async(
                        item.plan,
                        item.doc_config,
                        settings=llm_settings,
                        mock=False,
                        client=client,
                        on_status=on_status,
                        on_wait=on_wait,
                        on_call_elapsed=on_call_elapsed,
                        request_delay_seconds=config.llm.request_delay_seconds,
                        document_id=item.doc_id,
                        document_type=item.document_type,
                        style_description=item.style_description,
                        doc_seed=item.doc_seed,
                    )
                finally:
                    inflight.set(item.doc_id, None)

            row = _build_synthetic_row(
                config=config,
                doc_id=item.doc_id,
                doc_idx=item.doc_idx,
                doc_seed=item.doc_seed,
                plan=final_plan,
                text=text,
                text_engine="llm",
                document_type=item.document_type,
            )
            return item.doc_idx, item.doc_id, item.doc_seed, row, elapsed_logs, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err = str(exc).strip() or repr(exc)
            one_line = err.replace("\n", " ").strip()
            _LOGGER.warning("%s: skipped after LLM retries: %s", item.doc_id, one_line)
            return item.doc_idx, item.doc_id, item.doc_seed, None, elapsed_logs, err

    def _record_result(
        doc_idx: int,
        doc_id: str,
        doc_seed: int,
        row: dict | None,
        elapsed_logs: list[str],
        error: str | None,
    ) -> None:
        nonlocal completed
        for line in elapsed_logs:
            prog.verbose_log(line)
        completed += 1
        if row is not None:
            out[doc_idx] = row
            prog.verbose_log(
                f"{doc_id}: finalized with {len(row['entities'])} entities (model={model_short})"
            )
            tick(completed, f"{doc_id} {len(row['entities'])} entities (llm)")
            return
        skipped.append(
            SkippedDocument(
                id=doc_id,
                document_index=doc_idx,
                document_seed=doc_seed,
                error=error or "unknown error",
            )
        )
        prog.log_skip(doc_id, error or "unknown error")
        tick(completed, f"{doc_id} skipped")

    if max_workers <= 1:
        for item in docs:
            doc_idx, doc_id, doc_seed, row, elapsed_logs, error = await run_one_async(item)
            _record_result(doc_idx, doc_id, doc_seed, row, elapsed_logs, error)
        return out, skipped

    tasks = [asyncio.create_task(run_one_async(item)) for item in docs]
    try:
        for task in asyncio.as_completed(tasks):
            doc_idx, doc_id, doc_seed, row, elapsed_logs, error = await task
            _record_result(doc_idx, doc_id, doc_seed, row, elapsed_logs, error)
    except asyncio.CancelledError as exc:
        for task in tasks:
            task.cancel()
        raise exc
    except KeyboardInterrupt as exc:
        prog.log("interrupted: cancelling in-flight LLM calls…")
        for task in tasks:
            task.cancel()
        raise BuildInterrupted(
            _ordered_partial_examples(out, len(docs)),
            completed=len(out),
            total=len(docs),
        ) from exc
    return out, skipped


def build_examples_from_config_path(
    config_path: Path,
    *,
    settings: LlmGoldSettings | None = None,
    mock: bool = False,
    progress: Progress | None = None,
    document_log_dir: str | Path | None = None,
) -> SyntheticBuildResult:
    """Load config from disk and run :func:`build_examples_synthetic`."""
    config = load_synthetic_config(config_path)
    return build_examples_synthetic(
        config,
        settings=settings,
        mock=mock,
        progress=progress,
        config_path=config_path,
        document_log_dir=document_log_dir,
    )
