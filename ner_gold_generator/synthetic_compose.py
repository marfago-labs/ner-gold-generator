"""Assemble synthetic document text around a planned entity list."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from ner_gold_generator.labeling import LlmGoldSettings, build_chat_client
from ner_gold_generator.llm_client import (
    ChatClient,
    LlmRequestError,
    MockChatClient,
    describe_llm_api_error,
    retry_after_seconds_from_error,
    wait_for_retry,
)
from ner_gold_generator.synthetic_config import SyntheticConfig
from ner_gold_generator.synthetic_generators import PlannedEntity
from ner_gold_generator.validate import index_unique_span

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
    max_attempts = config.llm.max_attempts
    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and request_delay_seconds > 0:
            if on_status is not None:
                on_status(f"waiting {request_delay_seconds:.0f}s before retry")
            time.sleep(request_delay_seconds)
        if on_attempt is not None:
            on_attempt(attempt, max_attempts)
        if on_status is not None:
            on_status("calling OpenRouter…")
        started = time.perf_counter()
        try:
            payload = chat.complete_json(
                system=system_prompt,
                user=user,
                temperature=config.llm.temperature,
                call_label=f"attempt {attempt}/{max_attempts}",
            )
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "ok")
        except LlmRequestError as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            if not exc.retryable:
                raise ValueError(str(exc)) from exc
            last_error = str(exc)
            wait_for_retry(exc, on_wait=on_wait)
            continue
        except ValueError as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            last_error = str(exc)
            if on_status is not None:
                on_status(f"retry: {last_error[:72]}")
            continue
        except Exception as exc:
            if on_call_elapsed is not None:
                on_call_elapsed(attempt, time.perf_counter() - started, "error")
            msg, retryable, _ = describe_llm_api_error(exc, model_id=model_id)
            if not retryable:
                raise ValueError(msg) from exc
            last_error = msg
            wrapped = LlmRequestError(
                msg,
                retryable=True,
                model_id=model_id,
                retry_after_seconds=retry_after_seconds_from_error(exc),
            )
            wait_for_retry(wrapped, on_wait=on_wait)
            continue

        text = str(payload.get("text", "")).strip()
        if not text:
            last_error = "empty text from LLM"
            if on_status is not None:
                on_status(f"retry: {last_error}")
            continue
        errors = validate_text_contains_plan(text, plan)
        if not errors:
            audit = audit_plan_entity_usage(text, plan)
            if audit.used_exactly_once != audit.planned:
                last_error = (
                    f"audit mismatch: used={audit.used_exactly_once} planned={audit.planned}"
                )
                if on_status is not None:
                    on_status(f"retry: {last_error[:72]}")
                continue
            return text
        last_error = "; ".join(errors)
        if on_status is not None:
            on_status(f"retry: {last_error[:72]}")
    hint = ""
    if "rate limit" in last_error.lower() and max_attempts < 3:
        hint = " Tip: increase llm.max_attempts in your synthetic config for 429 retries."
    raise ValueError(
        f"Synthetic LLM failed (model={model_id!r}, attempts={max_attempts}).\n  {last_error}{hint}"
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
    }


def config_summary_json(config: SyntheticConfig) -> str:
    return json.dumps(config_summary_for_meta(config), sort_keys=True)
