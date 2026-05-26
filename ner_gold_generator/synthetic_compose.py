"""Assemble synthetic document text around a planned entity list."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from ner_gold_generator.labeling import LlmGoldSettings, build_chat_client
from ner_gold_generator.llm_client import (
    AsyncChatClient,
    ChatClient,
    InvalidLlmResponseError,
    LlmRequestError,
    MockChatClient,
    describe_llm_api_error,
    retry_after_seconds_from_error,
    wait_for_retry,
)
from ner_gold_generator.logger import get_logger
from ner_gold_generator.synthetic_config import SyntheticConfig
from ner_gold_generator.synthetic_generators import PlannedEntity
from ner_gold_generator.validate import index_unique_span

_LOGGER = get_logger(__name__)


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
    _log_synthetic_failure(
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


def _log_synthetic_failure(
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
) -> None:
    """Write a failing LLM attempt to ``gold.log`` for debugging."""
    header = f"synthetic failure ({failure_kind})"
    if document_id:
        header += f" doc={document_id}"
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
    lines.append("generated text:")
    lines.append(text if text else "(empty)")
    _LOGGER.warning("\n".join(lines))


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
    _log_synthetic_failure(
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


def _validate_llm_text_or_log(
    text: str,
    plan: list[PlannedEntity],
    *,
    document_id: str | None,
    model_id: str,
    attempt: int,
    call_budget: int,
) -> str | None:
    """Return an error message when validation fails (and log details); else ``None``."""
    if not text:
        _log_synthetic_failure(
            document_id=document_id,
            model_id=model_id,
            attempt=attempt,
            call_budget=call_budget,
            failure_kind="validation",
            message="empty text from LLM",
            text="",
            plan=plan,
        )
        return "empty text from LLM"
    errors = validate_text_contains_plan(text, plan)
    if errors:
        msg = "; ".join(errors)
        _log_synthetic_failure(
            document_id=document_id,
            model_id=model_id,
            attempt=attempt,
            call_budget=call_budget,
            failure_kind="validation",
            message=msg,
            text=text,
            plan=plan,
        )
        return msg
    audit = audit_plan_entity_usage(text, plan)
    if audit.used_exactly_once != audit.planned:
        msg = f"audit mismatch: used={audit.used_exactly_once} planned={audit.planned}"
        _log_synthetic_failure(
            document_id=document_id,
            model_id=model_id,
            attempt=attempt,
            call_budget=call_budget,
            failure_kind="validation",
            message=msg,
            text=text,
            plan=plan,
            audit=audit,
        )
        return msg
    return None


_FILLER_SENTENCES = (
    "Further analysis confirms the trend across multiple runs.",
    "Ablations indicate that hyperparameters were tuned on validation data.",
    "We release code and checkpoints for reproducibility.",
    "Limitations include domain shift and label noise in weak supervision.",
)

_DEFAULT_SYNTHETIC_SYSTEM = """You generate short documents for NER benchmark gold data.

Rules:
- Follow the domain style and length limits given in the user message.
- Put the full document in the JSON "text" field.
- Include EVERY required entity surface EXACTLY once in "text", character-for-character.
- Do not alter spelling, punctuation, or spacing of required surfaces.
- Do not add extra named entities beyond the required list.
- Respond with a single JSON object only: {"text": "<full document text>"}
- No markdown fences and no text outside the JSON object.
"""

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


def build_synthetic_user_prompt(
    plan: list[PlannedEntity],
    *,
    domain: str,
    min_chars: int,
    max_chars: int,
) -> str:
    """User prompt with domain, length, and required verbatim entity surfaces."""
    lines = [
        "Write one document for the benchmark.",
        f"Domain style: {domain}",
        f"Target length: {min_chars}-{max_chars} characters.",
        (
            "Required entities (each surface must appear exactly once in the "
            'JSON "text" field, verbatim):'
        ),
    ]
    for ent in plan:
        lines.append(f"- [{ent.label}] {ent.surface}")
    return "\n".join(lines)


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
) -> str:
    """Request prose from an LLM (or mock) embedding all planned entities."""
    if mock:
        return assemble_mock_llm_text(plan, config)

    model_id = config.llm.model.strip()
    if not model_id:
        raise ValueError("engine=llm requires llm.model in the synthetic config file")
    system_prompt = (
        config.llm.system_prompt.strip()
        if config.llm.system_prompt.strip()
        else _DEFAULT_SYNTHETIC_SYSTEM
    )
    chat = client or build_chat_client(settings, mock=False, model_id=model_id)
    user = build_synthetic_user_prompt(
        plan,
        domain=config.domain,
        min_chars=config.text_length.min_chars,
        max_chars=config.text_length.max_chars,
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
            time.sleep(request_delay_seconds)
        if on_attempt is not None:
            on_attempt(attempt, call_budget)
        if on_status is not None:
            on_status("calling OpenRouter…")
        started = time.perf_counter()
        try:
            payload = chat.complete_json(
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
                _log_synthetic_failure(
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
            _log_synthetic_failure(
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
            wait_for_retry(exc, on_wait=on_wait)
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
                _log_synthetic_failure(
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
            _log_synthetic_failure(
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
            wait_for_retry(wrapped, on_wait=on_wait)
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
        )
        if validation_error is None:
            return text
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


async def wait_for_retry_async(
    exc: LlmRequestError,
    *,
    on_wait: Callable[[float, str], None] | None = None,
    cap_seconds: float = 120.0,
    tick_seconds: float = 5.0,
) -> None:
    """Async sleep for rate-limit backoff when *exc* carries ``retry_after_seconds``."""
    if not exc.retryable or exc.retry_after_seconds is None:
        return
    wait = min(max(1.0, exc.retry_after_seconds + 0.5), cap_seconds)
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
    client: AsyncChatClient,
    on_attempt: Callable[[int, int], None] | None = None,
    on_wait: Callable[[float, str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_call_elapsed: Callable[[int, float, str], None] | None = None,
    request_delay_seconds: float = 0.0,
    document_id: str | None = None,
) -> str:
    """Async variant of :func:`generate_text_with_llm` (LLM engine only)."""
    if mock:
        return assemble_mock_llm_text(plan, config)

    model_id = config.llm.model.strip()
    if not model_id:
        raise ValueError("engine=llm requires llm.model in the synthetic config file")
    system_prompt = (
        config.llm.system_prompt.strip()
        if config.llm.system_prompt.strip()
        else _DEFAULT_SYNTHETIC_SYSTEM
    )
    user = build_synthetic_user_prompt(
        plan,
        domain=config.domain,
        min_chars=config.text_length.min_chars,
        max_chars=config.text_length.max_chars,
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
                _log_synthetic_failure(
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
            _log_synthetic_failure(
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
                _log_synthetic_failure(
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
            _log_synthetic_failure(
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
        )
        if validation_error is None:
            return text
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
        "text_length": {
            "min_chars": config.text_length.min_chars,
            "max_chars": config.text_length.max_chars,
        },
        "llm_model": config.llm.model,
        "llm_parallelism": config.llm.parallelism,
        "llm_provider_max_attempts": config.llm.provider_max_attempts,
        "llm_validation_max_attempts": config.llm.validation_max_attempts,
    }


def config_summary_json(config: SyntheticConfig) -> str:
    return json.dumps(config_summary_for_meta(config), sort_keys=True)
