"""Structured logging for LLM and Agno synthetic generation.

Writes prompt/response/failure traces to ``gold.log`` (via :mod:`ner_gold_generator.logger`)
and optionally to per-document files under ``llm.document_log_dir`` or
``--document-log-dir``.

Configure per-document logs with :func:`configure_document_log_dir` before
starting a synthetic batch. Paths are resolved relative to the synthetic YAML
config directory when not absolute.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ner_gold_generator.logger import get_logger

if TYPE_CHECKING:
    from ner_gold_generator.synthetic_compose import PlanEntityUsageAudit
from ner_gold_generator.synthetic_generators import PlannedEntity

_LOG_LOCK = threading.Lock()
_LOGGER = get_logger(__name__)
_DOCUMENT_LOG_DIR: Path | None = None

EngineKind = Literal["stateless", "agno"]
PromptKind = Literal["initial", "feedback"]


def configure_document_log_dir(
    path: Path | str | None,
    *,
    relative_to: Path | None = None,
) -> Path | None:
    """Write per-document synthetic LLM traces to ``{dir}/{document_id}.log``."""
    global _DOCUMENT_LOG_DIR
    if path is None or not str(path).strip():
        _DOCUMENT_LOG_DIR = None
        return None
    log_dir = Path(str(path).strip())
    if not log_dir.is_absolute() and relative_to is not None:
        log_dir = (relative_to / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _DOCUMENT_LOG_DIR = log_dir
    return log_dir


def _document_log_path(document_id: str) -> Path | None:
    if _DOCUMENT_LOG_DIR is None:
        return None
    safe = document_id.replace("/", "_").replace("\\", "_")
    return _DOCUMENT_LOG_DIR / f"{safe}.log"


def _append_document_log(document_id: str | None, level: str, body: str) -> None:
    if not document_id:
        return
    log_path = _document_log_path(document_id)
    if log_path is None:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = f"{stamp} {level.upper()}\n{body.rstrip()}\n\n"
    with _LOG_LOCK, log_path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def log_llm_prompt(
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    engine: EngineKind = "stateless",
    document_id: str | None = None,
    turn: int | None = None,
    turn_budget: int | None = None,
    prompt_kind: PromptKind = "initial",
    call_label: str | None = None,
) -> None:
    """Log full system and user prompts to the debug log file."""
    engine_label = "Agno" if engine == "agno" else "LLM"
    header = f"--- {engine_label} prompt ({model_id}, temperature={temperature}"
    if document_id:
        header += f", doc={document_id}"
    if turn is not None and turn_budget is not None:
        header += f", turn={turn}/{turn_budget}"
    header += f", {prompt_kind}"
    if call_label:
        header += f", {call_label}"
    payload = "\n".join(
        (
            f"{header}) ---",
            "SYSTEM:",
            system,
            "USER:",
            user,
            f"--- end {engine_label} prompt ---",
        )
    )
    with _LOG_LOCK:
        _LOGGER.debug(payload)
    _append_document_log(document_id, "DEBUG", payload)


def log_llm_response(
    *,
    model_id: str,
    raw: str,
    engine: EngineKind = "stateless",
    document_id: str | None = None,
    turn: int | None = None,
    elapsed_seconds: float | None = None,
    outcome: str = "ok",
    call_label: str | None = None,
) -> None:
    """Log raw model response text to the debug log file."""
    engine_label = "Agno" if engine == "agno" else "LLM"
    header = f"--- {engine_label} response ({model_id}"
    if document_id:
        header += f", doc={document_id}"
    if turn is not None:
        header += f", turn={turn}"
    if call_label:
        header += f", {call_label}"
    trailer = f"--- end {engine_label} response ---"
    if elapsed_seconds is not None:
        trailer += f" elapsed={elapsed_seconds:.1f}s outcome={outcome}"
    payload = "\n".join(
        (
            f"{header}) ---",
            raw if raw.strip() else "(empty)",
            trailer,
        )
    )
    with _LOG_LOCK:
        _LOGGER.debug(payload)
    _append_document_log(document_id, "DEBUG", payload)


def log_agno_session_start(
    *,
    document_id: str,
    document_type: str,
    model_id: str,
    session_id: str,
    provider_max_attempts: int,
    validation_max_attempts: int,
) -> None:
    """Log start of an Agno per-document synthetic session."""
    body = (
        "synthetic session start "
        f"document_type={document_type} model={model_id} agent_mode=per_doc "
        f"session_id={session_id} provider_max={provider_max_attempts} "
        f"validation_max={validation_max_attempts}"
    )
    _LOGGER.info(
        "synthetic session start doc=%s document_type=%s model=%s agent_mode=per_doc "
        "session_id=%s provider_max=%s validation_max=%s",
        document_id,
        document_type,
        model_id,
        session_id,
        provider_max_attempts,
        validation_max_attempts,
    )
    _append_document_log(document_id, "INFO", body)


def log_agno_turn_start(
    *,
    document_id: str,
    turn: int,
    turn_budget: int,
    prompt_kind: PromptKind,
    message_chars: int,
) -> None:
    """Log an Agno turn before the model responds (initial or feedback)."""
    body = (
        f"agno turn start turn={turn}/{turn_budget} "
        f"prompt_kind={prompt_kind} message_chars={message_chars}"
    )
    _LOGGER.debug(
        "agno turn start doc=%s turn=%s/%s prompt_kind=%s message_chars=%s",
        document_id,
        turn,
        turn_budget,
        prompt_kind,
        message_chars,
    )
    _append_document_log(document_id, "DEBUG", body)


def log_agno_turn_success(
    *,
    document_id: str,
    turn: int,
    text_len: int,
    used_exactly_once: int,
    planned: int,
) -> None:
    """Log successful validation after an Agno turn."""
    body = (
        f"synthetic session success turns={turn} text_len={text_len} "
        f"entities={used_exactly_once}/{planned}"
    )
    _LOGGER.info(
        "synthetic session success doc=%s turns=%s text_len=%s entities=%s/%s",
        document_id,
        turn,
        text_len,
        used_exactly_once,
        planned,
    )
    _append_document_log(document_id, "INFO", body)


def log_synthetic_failure(
    *,
    document_id: str | None,
    model_id: str,
    attempt: int,
    call_budget: int,
    failure_kind: str,
    message: str,
    text: str | None = None,
    plan: list[PlannedEntity] | None = None,
    audit: PlanEntityUsageAudit | None = None,
    agent_mode: str | None = None,
    turn: int | None = None,
    document_type: str | None = None,
) -> None:
    """Write a failing generation attempt to the log file for debugging."""
    header = f"synthetic failure ({failure_kind})"
    if document_id:
        header += f" doc={document_id}"
    if document_type:
        header += f" document_type={document_type}"
    if agent_mode:
        header += f" agent_mode={agent_mode}"
    if turn is not None:
        header += f" turn={turn}"
    header += f" model={model_id} attempt={attempt}/{call_budget}"
    lines = [header, f"errors: {message}"]
    if plan:
        lines.append("planned entities:")
        for ent in plan:
            lines.append(f"  - [{ent.label}] {ent.surface}")
    if audit is not None:
        lines.append(f"occurrence_counts: {audit.occurrence_counts}")
        if audit.missing:
            lines.append(f"missing: {list(audit.missing)}")
        if audit.duplicate_in_text:
            lines.append(f"duplicate_in_text: {list(audit.duplicate_in_text)}")
    if text is not None:
        lines.append(f"text length: {len(text)} chars")
    lines.append("generated text:")
    lines.append(text if text else "(empty)")
    payload = "\n".join(lines)
    with _LOG_LOCK:
        _LOGGER.warning(payload)
    _append_document_log(document_id, "WARNING", payload)
