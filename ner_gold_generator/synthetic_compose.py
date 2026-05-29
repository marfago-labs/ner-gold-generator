"""Assemble synthetic document text and validate planned entity surfaces.

Handles procedural template filling, mock LLM text, stateless OpenRouter retries,
and shared validation helpers used by :mod:`ner_gold_generator.synthetic_agno`.

Key functions:

- :func:`plan_entities_for_document` lives in :mod:`ner_gold_generator.synthetic_generators`
- :func:`collect_validation_issues` — length + exactly-once surface checks
- :func:`generate_text_with_llm_async` — LLM glue (Agno delegates to ``synthetic_agno``)
- :func:`entities_from_plan` — anchor gold spans after validation passes
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from ner_gold_generator.labeling import LlmGoldSettings, build_chat_client
from ner_gold_generator.llm_client import (
    RATE_LIMIT_WAIT_CAP_SECONDS,
    AsyncChatClient,
    ChatClient,
    InvalidLlmResponseError,
    LlmRequestError,
    MockChatClient,
    compute_rate_limit_wait_seconds,
    describe_llm_api_error,
    retry_after_seconds_from_error,
)
from ner_gold_generator.llm_logging import log_synthetic_failure
from ner_gold_generator.prompt_templates import render_template
from ner_gold_generator.synthetic_config import SyntheticConfig
from ner_gold_generator.synthetic_generators import PlannedEntity
from ner_gold_generator.validate import index_unique_span

if TYPE_CHECKING:
    from ner_gold_generator.synthetic_agno import AgnoAgentProtocol


def _llm_call_budget(provider_max_attempts: int, validation_max_attempts: int) -> int:
    return provider_max_attempts + validation_max_attempts


def _format_synthetic_llm_failure(
    *,
    model_id: str,
    reason: str,
    last_error: str,
    llm_calls: int,
    provider_max_attempts: int,
    validation_max_attempts: int,
    provider_failures: int,
    validation_failures: int,
    hint: str = "",
) -> str:
    """Build a structured failure message (parsed by :func:`format_skip_error`)."""
    call_budget = _llm_call_budget(provider_max_attempts, validation_max_attempts)
    return (
        "Synthetic LLM failed "
        f"(model={model_id!r}, reason={reason}, "
        f"llm_calls={llm_calls}/{call_budget}, "
        f"provider_failures={provider_failures}/{provider_max_attempts}, "
        f"validation_failures={validation_failures}/{validation_max_attempts}).\n"
        f"  {last_error}{hint}"
    )


def _raise_synthetic_llm_failed(
    *,
    model_id: str,
    reason: str,
    last_error: str,
    llm_calls: int,
    provider_max_attempts: int,
    validation_max_attempts: int,
    provider_failures: int,
    validation_failures: int,
    hint: str = "",
    document_id: str | None = None,
    last_text: str | None = None,
    plan: list[PlannedEntity] | None = None,
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


def _handle_provider_response_error(
    exc: ValueError,
    *,
    document_id: str | None,
    model_id: str,
    attempt: int,
    call_budget: int,
    plan: list[PlannedEntity],
) -> tuple[str, str]:
    """Log a parse/API response error; return ``(message, raw_text)``."""
    message = str(exc).strip() or repr(exc)
    raw_text = exc.raw if isinstance(exc, InvalidLlmResponseError) else ""
    failure_kind = (
        "provider_json"
        if "invalid JSON" in message or "no choices" in message
        else "provider_parse"
    )
    log_synthetic_failure(
        document_id=document_id,
        model_id=model_id,
        attempt=attempt,
        call_budget=call_budget,
        failure_kind=failure_kind,
        message=message,
        text=raw_text,
        plan=plan,
    )
    return message, raw_text


def collect_validation_issues(
    text: str,
    plan: list[PlannedEntity],
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Return human-readable validation errors (empty list when valid)."""
    issues: list[str] = []
    if not text:
        issues.append("empty text from LLM")
        return issues
    n = len(text)
    if n < min_chars or n > max_chars:
        issues.append(f"length is {n} characters, target is {min_chars}-{max_chars} characters")
    issues.extend(validate_text_contains_plan(text, plan))
    return issues


def build_validation_feedback(
    issues: list[str],
    *,
    config: SyntheticConfig,
    min_chars: int,
    max_chars: int,
    document_type: str | None = None,
) -> str:
    """Render the feedback user message from config template."""
    bullet = "\n".join(f"- {e}" for e in issues)
    if (
        document_type
        and document_type.strip().lower() == "transcript"
        and any("expected 1" in issue for issue in issues)
    ):
        bullet += (
            "\n- Transcript: use Host:/Guest: (or similar) for turn labels only; "
            "each required surface must appear once—do not repeat planned names "
            "as speaker prefixes or in intro and outro."
        )
    return render_template(
        config.llm.feedback_prompt,
        issues=bullet,
        min_chars=min_chars,
        max_chars=max_chars,
    )


def _validate_llm_text_or_log(
    text: str,
    plan: list[PlannedEntity],
    *,
    document_id: str | None,
    model_id: str,
    attempt: int,
    call_budget: int,
    min_chars: int,
    max_chars: int,
    agent_mode: str | None = None,
    document_type: str | None = None,
) -> str | None:
    """Return an error message when validation fails (and log details); else ``None``."""
    issues = collect_validation_issues(text, plan, min_chars=min_chars, max_chars=max_chars)
    if not issues:
        return None
    msg = "; ".join(issues)
    audit = audit_plan_entity_usage(text, plan) if text else None
    log_synthetic_failure(
        document_id=document_id,
        model_id=model_id,
        attempt=attempt,
        call_budget=call_budget,
        failure_kind="validation",
        message=msg,
        text=text,
        plan=plan,
        audit=audit,
        agent_mode=agent_mode,
        document_type=document_type,
    )
    return msg


_FILLER_SENTENCES = (
    "Further analysis confirms the trend across multiple runs.",
    "Ablations indicate that hyperparameters were tuned on validation data.",
    "We release code and checkpoints for reproducibility.",
    "Limitations include domain shift and label noise in weak supervision.",
)

_PLACEHOLDER_RE = re.compile(r"\{e(\d+)\}")


def _max_placeholder_index(template: str) -> int:
    indices = [int(m.group(1)) for m in _PLACEHOLDER_RE.finditer(template)]
    return max(indices) if indices else -1


def fill_template(template: str, entities: list[PlannedEntity]) -> str | None:
    """Substitute ``{e0}``, ``{e1}``, … with planned entity surfaces.

    Returns ``None`` if *entities* has fewer items than the template requires.
    """
    max_idx = _max_placeholder_index(template)
    if max_idx < 0:
        return template
    need = max_idx + 1
    if len(entities) < need:
        return None
    out = template
    for i in range(need):
        out = out.replace(f"{{e{i}}}", entities[i].surface)
    return out


def _single_entity_sentence(entity: PlannedEntity) -> str:
    return f"We discuss {entity.surface} in the context of this work."


def assemble_procedural_text(
    plan: list[PlannedEntity],
    config: SyntheticConfig,
    rng: random.Random,
) -> str:
    """Build document text from templates and filler without an LLM."""
    if not plan:
        return "No entities were planned for this document."

    parts: list[str] = []
    idx = 0
    templates = list(config.templates)
    rng.shuffle(templates)
    while idx < len(plan):
        remaining = len(plan) - idx
        suitable = [
            t
            for t in templates
            if (_max_placeholder_index(t) + 1) <= remaining and _max_placeholder_index(t) >= 0
        ]
        if suitable:
            template = rng.choice(suitable)
            need = _max_placeholder_index(template) + 1
            chunk = plan[idx : idx + need]
            filled = fill_template(template, chunk)
            if filled is not None:
                parts.append(filled)
                idx += need
                continue
        parts.append(_single_entity_sentence(plan[idx]))
        idx += 1

    text = " ".join(parts)
    min_chars = config.text_length.min_chars
    max_chars = config.text_length.max_chars
    filler_i = 0
    while len(text) < min_chars and filler_i < len(_FILLER_SENTENCES) * 4:
        text = f"{text} {_FILLER_SENTENCES[filler_i % len(_FILLER_SENTENCES)]}"
        filler_i += 1
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "."
        for ent in plan:
            if ent.surface not in text:
                text = f"{text} Reference: {ent.surface}."
    return text.strip()


def assemble_mock_llm_text(plan: list[PlannedEntity], config: SyntheticConfig) -> str:
    """Deterministic pseudo-LLM body for tests (procedural assembly)."""
    rng = random.Random(0)
    return assemble_procedural_text(plan, config, rng)


def _format_entity_list(plan: list[PlannedEntity]) -> str:
    return "\n".join(f"- [{ent.label}] {ent.surface}" for ent in plan)


_TRANSCRIPT_USER_PROMPT_ADDENDUM = """\
Transcript-specific rules:
- Prefix each turn with a role label such as "Host:" or "Guest:" only. Never use a required entity surface as a turn prefix.
- Each required entity surface must appear exactly once in the full document. After the first mention of a person or organization from the list, use pronouns or role words (the host, the guest, they, the company) instead of repeating that surface.
- Do not repeat a required organization or person name in both an introduction and a sign-off."""


def build_synthetic_user_prompt(
    plan: list[PlannedEntity],
    *,
    config: SyntheticConfig,
    document_type: str,
    style_description: str,
) -> str:
    """Render the initial user prompt from config template."""
    min_chars = config.text_length.min_chars
    max_chars = config.text_length.max_chars
    body = render_template(
        config.llm.user_prompt,
        document_type=document_type,
        domain=document_type,
        style_description=style_description,
        min_chars=min_chars,
        max_chars=max_chars,
        entity_list=_format_entity_list(plan),
    )
    if document_type.strip().lower() == "transcript":
        body = f"{body}\n\n{_TRANSCRIPT_USER_PROMPT_ADDENDUM}"
    return body


def count_surface_occurrences(text: str, surface: str) -> int:
    """Count non-overlapping literal occurrences of *surface* in *text* (regex-escaped)."""
    if not surface:
        return 0
    return len(re.findall(re.escape(surface), text))


@dataclass(frozen=True, slots=True)
class PlanEntityUsageAudit:
    """How many planned entity surfaces appear in generated text."""

    planned: int
    used_exactly_once: int
    missing: tuple[str, ...]
    duplicate_in_text: tuple[str, ...]
    occurrence_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable audit for ``synthetic_meta.entity_usage``."""
        return {
            "planned": self.planned,
            "used_exactly_once": self.used_exactly_once,
            "missing": list(self.missing),
            "duplicate_in_text": list(self.duplicate_in_text),
            "occurrence_counts": dict(self.occurrence_counts),
        }


def audit_plan_entity_usage(text: str, plan: list[PlannedEntity]) -> PlanEntityUsageAudit:
    """Audit planned surfaces via literal regex counts (exactly-once = accepted)."""
    missing: list[str] = []
    duplicate: list[str] = []
    counts: dict[str, int] = {}
    used_once = 0
    for ent in plan:
        n = count_surface_occurrences(text, ent.surface)
        counts[ent.surface] = n
        if n == 1:
            used_once += 1
        elif n == 0:
            missing.append(ent.surface)
        else:
            duplicate.append(ent.surface)
    return PlanEntityUsageAudit(
        planned=len(plan),
        used_exactly_once=used_once,
        missing=tuple(missing),
        duplicate_in_text=tuple(duplicate),
        occurrence_counts=counts,
    )


def validate_text_contains_plan(text: str, plan: list[PlannedEntity]) -> list[str]:
    """Return error messages if any planned entity is missing or not unique."""
    audit = audit_plan_entity_usage(text, plan)
    errors: list[str] = []
    for surface in audit.missing:
        errors.append(f"Surface {surface!r} missing (0 occurrences)")
    for surface in audit.duplicate_in_text:
        n = audit.occurrence_counts[surface]
        errors.append(f"Surface {surface!r} appears {n} times (expected 1)")
    return errors


class _SyncChatClientAdapter:
    """Adapt a sync :class:`ChatClient` for :func:`generate_text_with_llm_async`."""

    def __init__(self, client: ChatClient) -> None:
        self._client = client

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        call_label: str | None = None,
    ) -> dict:
        return self._client.complete_json(
            system=system,
            user=user,
            temperature=temperature,
            call_label=call_label,
        )


def generate_text_with_llm(
    plan: list[PlannedEntity],
    config: SyntheticConfig,
    *,
    settings: LlmGoldSettings,
    mock: bool,
    client: ChatClient | None = None,
    on_attempt: Callable[[int, int], None] | None = None,
    on_wait: Callable[[float, str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_call_elapsed: Callable[[int, float, str], None] | None = None,
    request_delay_seconds: float = 0.0,
    document_id: str | None = None,
    document_type: str | None = None,
    style_description: str | None = None,
) -> str:
    """Request prose from an LLM (or mock) embedding all planned entities."""
    if config.llm.agent_mode == "per_doc" and not mock:
        raise ValueError(
            "generate_text_with_llm does not support llm.agent_mode=per_doc; "
            "use generate_text_with_llm_async"
        )

    async def _run() -> str:
        async_client: AsyncChatClient | None = None
        if not mock:
            chat = client or build_chat_client(
                settings,
                mock=False,
                model_id=config.llm.model.strip(),
            )
            async_client = _SyncChatClientAdapter(chat)
        text, _plan = await generate_text_with_llm_async(
            plan,
            config,
            settings=settings,
            mock=mock,
            client=async_client,
            on_attempt=on_attempt,
            on_wait=on_wait,
            on_status=on_status,
            on_call_elapsed=on_call_elapsed,
            request_delay_seconds=request_delay_seconds,
            document_id=document_id,
            document_type=document_type,
            style_description=style_description,
        )
        return text

    return asyncio.run(_run())


async def wait_for_retry_async(
    exc: LlmRequestError,
    *,
    on_wait: Callable[[float, str], None] | None = None,
    cap_seconds: float = RATE_LIMIT_WAIT_CAP_SECONDS,
    tick_seconds: float = 5.0,
) -> None:
    """Async sleep for rate-limit backoff when *exc* is retryable (429)."""
    if not exc.retryable:
        return
    retry_after = exc.retry_after_seconds
    if retry_after is None and exc.status_code == 429:
        from ner_gold_generator.llm_client import DEFAULT_RATE_LIMIT_BACKOFF_SECONDS

        retry_after = DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
    if retry_after is None:
        return
    wait = compute_rate_limit_wait_seconds(retry_after, cap_seconds=cap_seconds)
    remaining = wait
    while remaining > 0:
        step = min(max(tick_seconds, 1.0), remaining)
        if on_wait is not None:
            on_wait(remaining, f"rate limited — {remaining:.0f}s left")
        await asyncio.sleep(step)
        remaining -= step


async def generate_text_with_llm_async(
    plan: list[PlannedEntity],
    config: SyntheticConfig,
    *,
    settings: LlmGoldSettings,
    mock: bool,
    client: AsyncChatClient | None = None,
    on_attempt: Callable[[int, int], None] | None = None,
    on_wait: Callable[[float, str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_call_elapsed: Callable[[int, float, str], None] | None = None,
    request_delay_seconds: float = 0.0,
    document_id: str | None = None,
    document_type: str | None = None,
    style_description: str | None = None,
    agent_factory: Callable[[], AgnoAgentProtocol] | None = None,
    doc_seed: int | None = None,
) -> tuple[str, list[PlannedEntity]]:
    """Async variant of :func:`generate_text_with_llm` (LLM engine only).

    Returns:
        Generated text and the entity plan used (may differ from *plan* after Agno restarts).
    """
    if mock:
        text = assemble_mock_llm_text(plan, config)
        return text, plan

    model_id = config.llm.model.strip()
    if not model_id:
        raise ValueError("engine=llm requires llm.model in the synthetic config file")
    doc_type = document_type or config.effective_document_type()
    style = style_description or config.domain.replace("_", " ")
    min_chars = config.text_length.min_chars
    max_chars = config.text_length.max_chars

    if config.llm.agent_mode == "per_doc":
        from ner_gold_generator.synthetic_agno import generate_text_with_agno_async

        if not document_id:
            raise ValueError("document_id is required for llm.agent_mode=per_doc")
        return await generate_text_with_agno_async(
            plan,
            config,
            settings=settings,
            document_id=document_id,
            document_type=doc_type,
            style_description=style,
            agent_factory=agent_factory,
            on_attempt=on_attempt,
            on_wait=on_wait,
            on_status=on_status,
            doc_seed=doc_seed,
            doc_config=config,
        )

    if client is None:
        raise ValueError("AsyncChatClient is required for llm.agent_mode=stateless")
    system_prompt = config.llm.system_prompt.strip()
    if not system_prompt:
        raise ValueError("engine=llm requires llm.system_prompt or llm.prompt_file")
    user = build_synthetic_user_prompt(
        plan,
        config=config,
        document_type=doc_type,
        style_description=style,
    )
    last_error = "no attempts"
    last_error_kind = "provider"
    provider_failures = 0
    validation_failures = 0
    provider_max_attempts = config.llm.provider_max_attempts
    validation_max_attempts = config.llm.validation_max_attempts
    call_budget = _llm_call_budget(provider_max_attempts, validation_max_attempts)
    attempt = 0
    last_text: str | None = None
    while True:
        attempt += 1
        if attempt > 1 and request_delay_seconds > 0:
            if on_status is not None:
                on_status(f"waiting {request_delay_seconds:.0f}s before retry")
            await asyncio.sleep(request_delay_seconds)
        if on_attempt is not None:
            on_attempt(attempt, call_budget)
        if on_status is not None:
            on_status("calling OpenRouter…")
        started = time.perf_counter()
        try:
            payload = await client.complete_json(
                system=system_prompt,
                user=user,
                temperature=config.llm.temperature,
                call_label=f"attempt {attempt}/{call_budget}",
            )
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "ok")
        except LlmRequestError as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            if not exc.retryable:
                log_synthetic_failure(
                    document_id=document_id,
                    model_id=model_id,
                    attempt=attempt,
                    call_budget=call_budget,
                    failure_kind="provider",
                    message=str(exc),
                    plan=plan,
                )
                _raise_synthetic_llm_failed(
                    model_id=model_id,
                    reason="provider",
                    last_error=str(exc),
                    llm_calls=attempt,
                    provider_max_attempts=provider_max_attempts,
                    validation_max_attempts=validation_max_attempts,
                    provider_failures=provider_failures + 1,
                    validation_failures=validation_failures,
                    document_id=document_id,
                    last_text=last_text,
                    plan=plan,
                )
            last_error = str(exc)
            last_error_kind = "provider"
            provider_failures += 1
            log_synthetic_failure(
                document_id=document_id,
                model_id=model_id,
                attempt=attempt,
                call_budget=call_budget,
                failure_kind="provider",
                message=last_error,
                plan=plan,
            )
            if provider_failures >= provider_max_attempts:
                break
            await wait_for_retry_async(exc, on_wait=on_wait)
            continue
        except ValueError as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            last_error, raw_text = _handle_provider_response_error(
                exc,
                document_id=document_id,
                model_id=model_id,
                attempt=attempt,
                call_budget=call_budget,
                plan=plan,
            )
            last_text = raw_text or last_text
            last_error_kind = "provider"
            provider_failures += 1
            if provider_failures >= provider_max_attempts:
                break
            if on_status is not None:
                on_status(f"retry: {last_error[:72]}")
            continue
        except Exception as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            msg, retryable, _ = describe_llm_api_error(exc, model_id=model_id)
            if not retryable:
                log_synthetic_failure(
                    document_id=document_id,
                    model_id=model_id,
                    attempt=attempt,
                    call_budget=call_budget,
                    failure_kind="provider",
                    message=msg,
                    plan=plan,
                )
                _raise_synthetic_llm_failed(
                    model_id=model_id,
                    reason="provider",
                    last_error=msg,
                    llm_calls=attempt,
                    provider_max_attempts=provider_max_attempts,
                    validation_max_attempts=validation_max_attempts,
                    provider_failures=provider_failures + 1,
                    validation_failures=validation_failures,
                    document_id=document_id,
                    last_text=last_text,
                    plan=plan,
                )
            last_error = msg
            last_error_kind = "provider"
            provider_failures += 1
            log_synthetic_failure(
                document_id=document_id,
                model_id=model_id,
                attempt=attempt,
                call_budget=call_budget,
                failure_kind="provider",
                message=last_error,
                plan=plan,
            )
            if provider_failures >= provider_max_attempts:
                break
            wrapped = LlmRequestError(
                msg,
                retryable=True,
                model_id=model_id,
                retry_after_seconds=retry_after_seconds_from_error(exc),
            )
            await wait_for_retry_async(wrapped, on_wait=on_wait)
            continue

        text = str(payload.get("text", "")).strip()
        last_text = text
        validation_error = _validate_llm_text_or_log(
            text,
            plan,
            document_id=document_id,
            model_id=model_id,
            attempt=attempt,
            call_budget=call_budget,
            min_chars=min_chars,
            max_chars=max_chars,
            agent_mode="stateless",
            document_type=doc_type,
        )
        if validation_error is None:
            return text, plan
        last_error = validation_error
        last_error_kind = "validation"
        validation_failures += 1
        if validation_failures >= validation_max_attempts:
            break
        if on_status is not None:
            on_status(f"retry: {last_error[:72]}")
    hint = ""
    if "rate limit" in last_error.lower() and provider_max_attempts < 3:
        hint = (
            " Tip: increase llm.provider_max_attempts "
            "(or llm.max_attempts for legacy configs) for 429 retries."
        )
    _raise_synthetic_llm_failed(
        model_id=model_id,
        reason=last_error_kind,
        last_error=last_error,
        llm_calls=attempt,
        provider_max_attempts=provider_max_attempts,
        validation_max_attempts=validation_max_attempts,
        provider_failures=provider_failures,
        validation_failures=validation_failures,
        hint=hint,
        document_id=document_id,
        last_text=last_text,
        plan=plan,
    )
    raise AssertionError("unreachable")


def mock_llm_client_for_plan(plan: list[PlannedEntity], config: SyntheticConfig) -> MockChatClient:
    """Canned JSON response that embeds entities procedurally."""
    text = assemble_mock_llm_text(plan, config)
    return MockChatClient(responses=[{"text": text}])


def entities_from_plan(text: str, plan: list[PlannedEntity]) -> list[dict]:
    """Convert a validated plan into gold entity dicts anchored in *text*."""
    entities: list[dict] = []
    for ent in plan:
        span = index_unique_span(text, ent.surface)
        if span is None:
            raise ValueError(f"Planned surface not unique in text: {ent.surface!r}")
        start, end = span
        entities.append({"text": ent.surface, "label": ent.label, "start": start, "end": end})
    entities.sort(key=lambda e: (e["start"], e["end"], e["label"]))
    return entities


def config_summary_for_meta(config: SyntheticConfig) -> dict:
    """Compact config snapshot for JSONL ``synthetic_meta``."""
    return {
        "seed": config.seed,
        "num_documents": config.num_documents,
        "engine": config.engine,
        "domain": config.domain,
        "document_type": config.effective_document_type(),
        "text_length": {
            "min_chars": config.text_length.min_chars,
            "max_chars": config.text_length.max_chars,
        },
        "llm_model": config.llm.model,
        "llm_agent_mode": config.llm.agent_mode,
        "llm_parallelism": config.llm.parallelism,
        "llm_provider_max_attempts": config.llm.provider_max_attempts,
        "llm_validation_max_attempts": config.llm.validation_max_attempts,
    }


def config_summary_json(config: SyntheticConfig) -> str:
    """Compact JSON snapshot of :func:`config_summary_for_meta` for logging."""
    return json.dumps(config_summary_for_meta(config), sort_keys=True)
