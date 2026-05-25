"""Multi-sample LLM entity labeling with majority consensus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ner_gold_generator.consensus import consensus_entities
from ner_gold_generator.llm_client import (
    ChatClient,
    MockChatClient,
    OpenRouterChatClient,
)
from ner_gold_generator.mock import mock_extract_entities
from ner_gold_generator.types import ProposedEntity, ValidatedEntity
from ner_gold_generator.validate import normalize_entity_label, validate_proposed_entities

_DEFAULT_LABELS = (
    "person",
    "organization",
    "location",
    "model",
    "dataset",
    "benchmark",
    "metric",
    "method",
    "number",
    "year",
    "arxiv_id",
    "acronym",
    "miscellaneous",
)

_SYSTEM_PROMPT = """You annotate named entities in scientific text for NER evaluation gold data.

Rules:
- Copy entity text EXACTLY as it appears in the source (character-for-character).
- Do not paraphrase, normalize, or invent spans that are not in the text.
- Do not include paper titles unless they appear verbatim in the text block.
- Prefer high-precision: omit uncertain entities.
- Use only these labels: {labels}

Respond with JSON only:
{{"entities": [{{"text": "<verbatim span>", "label": "<label>"}}]}}
"""


class LlmGoldSettings(BaseSettings):
    """OpenRouter and LLM labeling settings (environment + ``.env``).

    Environment variables use the aliases in ``Field(alias=...)`` (e.g.
    ``OPENROUTER_API_KEY``, ``LLM_GOLD_SAMPLES``). See project
    ``docs/configuration.md``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    llm_gold_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="LLM_GOLD_MODEL",
    )
    llm_gold_samples: int = Field(default=3, alias="LLM_GOLD_SAMPLES")
    llm_gold_min_votes: int = Field(default=0, alias="LLM_GOLD_MIN_VOTES")
    llm_gold_temperature: float = Field(default=0.4, alias="LLM_GOLD_TEMPERATURE")
    mock_llm: bool = Field(default=False, alias="MOCK_LLM")


@dataclass(frozen=True, slots=True)
class LlmLabelResult:
    """Outcome of multi-sample labeling for one document.

    Attributes:
        entities: Spans that passed consensus and substring validation.
        sample_runs: Number of LLM passes executed.
        min_votes: Vote threshold used for consensus.
        raw_run_counts: Entity count per pass (before consensus).
    """

    entities: list[ValidatedEntity]
    sample_runs: int
    min_votes: int
    raw_run_counts: list[int]


def default_min_votes(num_samples: int, configured: int) -> int:
    """Compute the minimum agreeing LLM passes.

    Args:
        num_samples: Total LLM passes for one document.
        configured: ``LLM_GOLD_MIN_VOTES`` (``0`` means auto majority).

    Returns:
        ``configured`` capped at *num_samples*, or ``(num_samples // 2) + 1``
        when *configured* is ``0``.
    """
    if configured > 0:
        return min(configured, num_samples)
    return (num_samples // 2) + 1


def build_chat_client(
    settings: LlmGoldSettings,
    *,
    mock: bool = False,
    mock_client: ChatClient | None = None,
) -> ChatClient:
    """Construct a chat client for labeling.

    Args:
        settings: API key, model, and mock flags.
        mock: Force mock client even if an API key is set.
        mock_client: Inject a custom client (tests).

    Returns:
        *mock_client*, :class:`~llm_client.MockChatClient`, or
        :class:`~llm_client.OpenRouterChatClient`.
    """
    if mock_client is not None:
        return mock_client
    if mock or settings.mock_llm or not settings.openrouter_api_key.strip():
        return MockChatClient()
    return OpenRouterChatClient(
        api_key=settings.openrouter_api_key.strip(),
        model_id=settings.llm_gold_model,
        base_url=settings.openrouter_base_url,
    )


def _user_prompt(text: str, *, labels: tuple[str, ...]) -> str:
    """Build the user message containing the source text block."""
    label_list = ", ".join(labels)
    return (
        f"Extract named entities from the text below.\n\n"
        f"Allowed labels: {label_list}\n\n"
        f"---TEXT---\n{text}\n---END---"
    )


def _parse_run(payload: dict[str, Any]) -> list[ProposedEntity]:
    """Parse one LLM JSON payload into proposed entities."""
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return []
    proposed: list[ProposedEntity] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            continue
        norm = normalize_entity_label(label)
        if norm is None:
            continue
        proposed.append(ProposedEntity(text=text.strip(), label=norm))
    return proposed


def extract_entities_once(
    client: ChatClient,
    text: str,
    *,
    labels: tuple[str, ...] = _DEFAULT_LABELS,
    temperature: float = 0.4,
) -> list[ProposedEntity]:
    """Run a single LLM annotation pass on *text*.

    Args:
        client: Chat backend (live or mock).
        text: Document body to annotate.
        labels: Allowed label names for the prompt.
        temperature: Sampling temperature for this pass.

    Returns:
        Proposed entities with normalized labels (not yet anchored).
    """
    system = _SYSTEM_PROMPT.format(labels=", ".join(labels))
    user = _user_prompt(text, labels=labels)
    payload = client.complete_json(system=system, user=user, temperature=temperature)
    return _parse_run(payload)


def label_text_with_llm_consensus(
    text: str,
    *,
    settings: LlmGoldSettings | None = None,
    num_samples: int | None = None,
    min_votes: int | None = None,
    labels: tuple[str, ...] = _DEFAULT_LABELS,
    temperature: float | None = None,
    mock: bool = False,
    client: ChatClient | None = None,
    on_sample: Callable[[int, int], None] | None = None,
) -> LlmLabelResult:
    """Label *text* with multiple LLM passes and majority consensus.

    After consensus, drops spans that are not exact unique substrings of *text*
    via :func:`validate.validate_proposed_entities`.

    Args:
        text: Document body.
        settings: OpenRouter settings; defaults to :class:`LlmGoldSettings()`.
        num_samples: LLM passes (overrides ``llm_gold_samples``).
        min_votes: Vote threshold (overrides env / auto majority).
        labels: Allowed labels in the prompt.
        temperature: Per-pass temperature (overrides settings).
        mock: Use :class:`~llm_client.MockChatClient`.
        client: Inject a chat client (tests).
        on_sample: Optional ``(current_pass, total_passes)`` callback.

    Returns:
        Validated entities and run metadata.

    Raises:
        ValueError: If *num_samples* is less than 1.
    """
    cfg = settings or LlmGoldSettings()
    n = num_samples if num_samples is not None else cfg.llm_gold_samples
    if n < 1:
        raise ValueError("num_samples must be >= 1")
    votes = min_votes if min_votes is not None else default_min_votes(n, cfg.llm_gold_min_votes)
    temp = temperature if temperature is not None else cfg.llm_gold_temperature
    chat = client or build_chat_client(cfg, mock=mock)

    runs: list[list[ProposedEntity]] = []
    for sample_idx in range(n):
        if on_sample is not None:
            on_sample(sample_idx + 1, n)
        runs.append(
            extract_entities_once(
                chat,
                text,
                labels=labels,
                temperature=temp,
            )
        )

    agreed = consensus_entities(runs, min_votes=votes)
    validated = validate_proposed_entities(text, agreed)
    return LlmLabelResult(
        entities=validated,
        sample_runs=n,
        min_votes=votes,
        raw_run_counts=[len(r) for r in runs],
    )


def entities_to_gold_dict(entities: list[ValidatedEntity]) -> list[dict[str, str | int]]:
    """Serialize validated entities for JSONL gold output.

    Args:
        entities: Anchored spans from labeling or manual indexing.

    Returns:
        List of dicts with ``text``, ``label``, ``start``, ``end`` keys.
    """
    return [
        {
            "text": ent.text,
            "label": ent.label,
            "start": ent.start,
            "end": ent.end,
        }
        for ent in entities
    ]


__all__ = [
    "LlmGoldSettings",
    "LlmLabelResult",
    "label_text_with_llm_consensus",
    "mock_extract_entities",
    "extract_entities_once",
    "entities_to_gold_dict",
    "default_min_votes",
    "build_chat_client",
]
