"""Per-document Agno agent sessions for synthetic LLM text generation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

from pydantic import BaseModel, Field

from ner_gold_generator.labeling import LlmGoldSettings
from ner_gold_generator.llm_client import (
    describe_llm_api_error,
    strip_json_fence,
)
from ner_gold_generator.llm_logging import (
    log_agno_session_start,
    log_agno_turn_start,
    log_agno_turn_success,
    log_llm_prompt,
    log_llm_response,
    log_synthetic_failure,
)
from ner_gold_generator.synthetic_compose import (
    _format_synthetic_llm_failure,
    _llm_call_budget,
    audit_plan_entity_usage,
    build_synthetic_user_prompt,
    build_validation_feedback,
    collect_validation_issues,
    wait_for_retry_async,
)
from ner_gold_generator.synthetic_config import SyntheticConfig
from ner_gold_generator.synthetic_generators import PlannedEntity

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover - optional llm extra
    Agent = None  # type: ignore[misc, assignment]


class SyntheticTextOutput(BaseModel):
    """Structured output schema for synthetic document generation."""

    text: str = Field(..., description="Full document text")


class AgnoAgentProtocol(Protocol):
    """Minimal agent surface used by tests (fake agents)."""

    def run(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> object: ...

    async def arun(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> object: ...


def verify_agno_installation() -> None:
    if Agent is None:
        raise ImportError(
            "Agno is required for llm.agent_mode=per_doc. Install with: uv sync --extra llm"
        )
    try:
        import sqlalchemy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for Agno SqliteDb sessions (llm.agent_mode=per_doc). "
            "Install with: uv sync --extra llm"
        ) from exc


def extract_run_text(run: object) -> str:
    """Return assistant text from an Agno run."""
    if run is None:
        return ""
    if isinstance(run, str):
        return run.strip()
    content = getattr(run, "content", None)
    if content is not None:
        if isinstance(content, BaseModel):
            return content.model_dump_json()
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, str) and content.strip():
            return content.strip()
    messages = getattr(run, "messages", None) or []
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role != "assistant":
            continue
        text = getattr(msg, "content", None)
        if text is None and isinstance(msg, dict):
            text = msg.get("content")
        if isinstance(text, BaseModel):
            return text.model_dump_json()
        if text is not None and str(text).strip():
            return str(text).strip()
    return ""


def parse_text_from_run(run: object) -> tuple[str, str]:
    """Parse document text from an Agno run; return ``(text, raw)``."""
    raw = extract_run_text(run)
    if not raw.strip():
        return "", raw
    try:
        payload = json.loads(strip_json_fence(raw))
        if isinstance(payload, dict):
            return str(payload.get("text", "")).strip(), raw
    except json.JSONDecodeError:
        pass
    return raw.strip(), raw


def build_agno_sqlite_db(
    config: SyntheticConfig,
    *,
    document_id: str | None = None,
) -> object:
    """Return an Agno :class:`SqliteDb` for one synthetic document session.

    Agno's ``SqliteDb(db_file=...)`` treats paths literally and breaks on ``:memory:``
    (it resolves to a disk path). Use ``db_url=sqlite:///:memory:`` for in-memory sessions.
    When *document_id* is set and a file path is configured, use a per-document file so
    parallel builds do not contend on one SQLite file.
    """
    from agno.db.sqlite import SqliteDb

    raw = config.llm.agno_db_path.strip()
    if not raw or raw == ":memory:":
        return SqliteDb(db_url="sqlite:///:memory:")
    if raw.startswith("sqlite:"):
        return SqliteDb(db_url=raw)

    path = Path(raw)
    if document_id:
        safe_id = document_id.replace("/", "_").replace("\\", "_")
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            path = path.parent / f"{path.stem}-{safe_id}{path.suffix}"
        else:
            path = path / f"{safe_id}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteDb(db_file=str(path.resolve()))


def create_synthetic_agent(
    config: SyntheticConfig,
    settings: LlmGoldSettings,
    *,
    document_id: str | None = None,
    agent_factory: Callable[[], AgnoAgentProtocol] | None = None,
) -> AgnoAgentProtocol:
    """Build an Agno agent for one synthetic document session."""
    if agent_factory is not None:
        return agent_factory()
    verify_agno_installation()
    from agno.models.openrouter import OpenRouterResponses

    model_id = config.llm.model.strip()
    agent = Agent(
        model=OpenRouterResponses(id=model_id, api_key=settings.openrouter_api_key),
        db=build_agno_sqlite_db(config, document_id=document_id),
        instructions=config.llm.system_prompt,
        output_schema=SyntheticTextOutput,
        add_history_to_context=True,
        num_history_runs=config.llm.validation_max_attempts,
        markdown=False,
        reasoning=False,
    )
    return agent


async def _run_agent_async(
    agent: AgnoAgentProtocol,
    message: str,
    *,
    session_id: str,
) -> object:
    arun = getattr(agent, "arun", None)
    if callable(arun):
        return await arun(message, session_id=session_id, user_id="synthetic")
    return await asyncio.to_thread(
        agent.run,
        message,
        session_id=session_id,
        user_id="synthetic",
    )


def _raise_agno_failed(
    *,
    model_id: str,
    reason: str,
    last_error: str,
    llm_calls: int,
    provider_max_attempts: int,
    validation_max_attempts: int,
    provider_failures: int,
    validation_failures: int,
    document_id: str | None,
    last_text: str | None,
    plan: list[PlannedEntity],
    document_type: str,
    hint: str = "",
) -> NoReturn:
    log_synthetic_failure(
        document_id=document_id,
        model_id=model_id,
        attempt=llm_calls,
        call_budget=_llm_call_budget(provider_max_attempts, validation_max_attempts),
        failure_kind=f"final_{reason}",
        message=last_error,
        text=last_text,
        plan=plan,
        agent_mode="per_doc",
        document_type=document_type,
    )
    raise ValueError(
        _format_synthetic_llm_failure(
            model_id=model_id,
            reason=reason,
            last_error=last_error,
            llm_calls=llm_calls,
            provider_max_attempts=provider_max_attempts,
            validation_max_attempts=validation_max_attempts,
            provider_failures=provider_failures,
            validation_failures=validation_failures,
            hint=hint,
        )
    )


async def generate_text_with_agno_async(
    plan: list[PlannedEntity],
    config: SyntheticConfig,
    *,
    settings: LlmGoldSettings,
    document_id: str,
    document_type: str,
    style_description: str,
    agent_factory: Callable[[], AgnoAgentProtocol] | None = None,
    on_attempt: Callable[[int, int], None] | None = None,
    on_wait: Callable[[float, str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Generate synthetic document text via a per-document Agno session."""
    model_id = config.llm.model.strip()
    if not model_id:
        raise ValueError("engine=llm requires llm.model in the synthetic config file")

    provider_max = config.llm.provider_max_attempts
    validation_max = config.llm.validation_max_attempts
    call_budget = _llm_call_budget(provider_max, validation_max)
    min_chars = config.text_length.min_chars
    max_chars = config.text_length.max_chars

    agent = create_synthetic_agent(
        config,
        settings,
        document_id=document_id,
        agent_factory=agent_factory,
    )
    session_id = document_id
    log_agno_session_start(
        document_id=document_id,
        document_type=document_type,
        model_id=model_id,
        session_id=session_id,
        provider_max_attempts=provider_max,
        validation_max_attempts=validation_max,
    )

    provider_failures = 0
    validation_failures = 0
    llm_calls = 0
    last_error = "no attempts"
    last_error_kind = "provider"
    last_text: str | None = None
    feedback_message: str | None = None

    for turn in range(1, validation_max + 1):
        if on_attempt is not None:
            on_attempt(turn, validation_max)

        if turn == 1:
            message = build_synthetic_user_prompt(
                plan,
                config=config,
                document_type=document_type,
                style_description=style_description,
            )
            prompt_kind: str = "initial"
        else:
            message = feedback_message or ""
            prompt_kind = "feedback"

        log_agno_turn_start(
            document_id=document_id,
            turn=turn,
            turn_budget=validation_max,
            prompt_kind=prompt_kind,  # type: ignore[arg-type]
            message_chars=len(message),
        )

        turn_succeeded = False
        for _provider_try in range(provider_max):
            llm_calls += 1
            if on_status is not None:
                on_status(f"calling Agno (turn {turn}/{validation_max})…")
            log_llm_prompt(
                model_id=model_id,
                system=config.llm.system_prompt,
                user=message,
                temperature=config.llm.temperature,
                engine="agno",
                document_id=document_id,
                turn=turn,
                turn_budget=validation_max,
                prompt_kind=prompt_kind,  # type: ignore[arg-type]
            )
            started = time.perf_counter()
            try:
                run = await _run_agent_async(agent, message, session_id=session_id)
                elapsed = time.perf_counter() - started
                text, raw = parse_text_from_run(run)
                log_llm_response(
                    model_id=model_id,
                    raw=raw,
                    engine="agno",
                    document_id=document_id,
                    turn=turn,
                    elapsed_seconds=elapsed,
                    outcome="ok",
                )
                turn_succeeded = True
                break
            except Exception as exc:
                elapsed = time.perf_counter() - started
                msg, retryable, _ = describe_llm_api_error(exc, model_id=model_id)
                log_llm_response(
                    model_id=model_id,
                    raw=msg,
                    engine="agno",
                    document_id=document_id,
                    turn=turn,
                    elapsed_seconds=elapsed,
                    outcome="error",
                )
                last_error = msg
                last_error_kind = "provider"
                provider_failures += 1
                log_synthetic_failure(
                    document_id=document_id,
                    model_id=model_id,
                    attempt=llm_calls,
                    call_budget=call_budget,
                    failure_kind="provider",
                    message=msg,
                    plan=plan,
                    agent_mode="per_doc",
                    turn=turn,
                    document_type=document_type,
                )
                if not retryable or provider_failures >= provider_max:
                    break
                from ner_gold_generator.llm_client import (
                    LlmRequestError,
                    retry_after_seconds_from_error,
                )

                wrapped = LlmRequestError(
                    msg,
                    retryable=True,
                    model_id=model_id,
                    retry_after_seconds=retry_after_seconds_from_error(exc),
                )
                await wait_for_retry_async(wrapped, on_wait=on_wait)
                continue

        if not turn_succeeded:
            break

        last_text = text
        issues = collect_validation_issues(
            text,
            plan,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if not issues:
            audit = audit_plan_entity_usage(text, plan)
            log_agno_turn_success(
                document_id=document_id,
                turn=turn,
                text_len=len(text),
                used_exactly_once=audit.used_exactly_once,
                planned=audit.planned,
            )
            return text

        last_error = "; ".join(issues)
        last_error_kind = "validation"
        validation_failures += 1
        audit = audit_plan_entity_usage(text, plan)
        log_synthetic_failure(
            document_id=document_id,
            model_id=model_id,
            attempt=llm_calls,
            call_budget=call_budget,
            failure_kind="validation",
            message=last_error,
            text=text,
            plan=plan,
            audit=audit,
            agent_mode="per_doc",
            turn=turn,
            document_type=document_type,
        )
        if validation_failures >= validation_max:
            break
        feedback_message = build_validation_feedback(
            issues,
            config=config,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if on_status is not None:
            on_status(f"retry: {last_error[:72]}")

    _raise_agno_failed(
        model_id=model_id,
        reason=last_error_kind,
        last_error=last_error,
        llm_calls=llm_calls,
        provider_max_attempts=provider_max,
        validation_max_attempts=validation_max,
        provider_failures=provider_failures,
        validation_failures=validation_failures,
        document_id=document_id,
        last_text=last_text,
        plan=plan,
        document_type=document_type,
    )
