"""Load and validate synthetic gold generation configuration (YAML/JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentTypeWeight:
    """Weighted document type for mixed synthetic runs."""

    type: str
    weight: float


@dataclass(frozen=True, slots=True)
class TextLengthConfig:
    min_chars: int = 200
    max_chars: int = 800


@dataclass(frozen=True, slots=True)
class EntitiesPerDocumentConfig:
    distribution: str = "uniform"
    min_count: int = 3
    max_count: int = 8
    fixed_count: int = 5
    poisson_lambda: float = 6.0


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    formats: tuple[str, ...] = ()
    year_min: int = 2015
    year_max: int = 2026
    value_min: float = 0.1
    value_max: float = 99.9
    decimal_separators: tuple[str, ...] = (".", ",")
    thousands_separators: tuple[str, ...] = (",", ".", "")
    currency_codes: tuple[str, ...] = ("USD", "EUR", "GBP")
    currency_symbols: tuple[str, ...] = ("$", "€", "£")


@dataclass(frozen=True, slots=True)
class LlmSyntheticConfig:
    model: str = ""
    max_attempts: int = 3
    provider_max_attempts: int = 3
    validation_max_attempts: int = 3
    temperature: float = 0.4
    timeout_seconds: float = 120.0
    request_delay_seconds: float = 0.0
    parallelism: int = 3
    agent_mode: str = "per_doc"
    agno_db_path: str = ""
    system_prompt: str = ""
    prompt_file: str = ""
    user_prompt: str = ""
    user_prompt_file: str = ""
    feedback_prompt: str = ""
    feedback_prompt_file: str = ""


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Full synthetic dataset specification."""

    seed: int = 42
    num_documents: int = 10
    engine: str = "procedural"
    domain: str = "scientific_abstract"
    document_type: str = ""
    document_types: tuple[DocumentTypeWeight, ...] = ()
    document_profiles_dir: str = "document_profiles"
    text_length: TextLengthConfig = field(default_factory=TextLengthConfig)
    entities_per_document: EntitiesPerDocumentConfig = field(
        default_factory=EntitiesPerDocumentConfig
    )
    label_weights: dict[str, float] = field(default_factory=dict)
    lexicon: dict[str, tuple[str, ...]] = field(default_factory=dict)
    generators: dict[str, GeneratorConfig] = field(default_factory=dict)
    templates: tuple[str, ...] = field(default_factory=tuple)
    llm: LlmSyntheticConfig = field(default_factory=LlmSyntheticConfig)

    def normalized_label_weights(self) -> dict[str, float]:
        if not self.label_weights:
            return {
                "model": 0.15,
                "dataset": 0.12,
                "benchmark": 0.1,
                "metric": 0.12,
                "organization": 0.1,
                "method": 0.08,
                "number": 0.08,
                "date": 0.06,
                "year": 0.05,
                "percentage": 0.05,
                "currency": 0.04,
                "person": 0.03,
                "location": 0.02,
            }
        total = sum(self.label_weights.values())
        if total <= 0:
            raise ValueError("label_weights must sum to a positive value")
        return {k: v / total for k, v in self.label_weights.items()}

    def effective_document_type(self) -> str:
        """Primary document type id (explicit field, mixed default, or domain alias)."""
        if self.document_type.strip():
            return self.document_type.strip()
        if self.document_types:
            return self.document_types[0].type
        return self.domain.strip() or "scientific_abstract"


_DEFAULT_TEMPLATES = (
    "We propose {e0} and evaluate it on {e1}, achieving {e2} on the {e3} benchmark.",
    "Compared with {e0}, our approach on {e1} improves {e2} as measured by {e3}.",
    "Experiments on {e0} using {e1} report {e2} for the {e3} task in {e4}.",
    "The method {e0} was tested on {e1}; results include {e2} and {e3} from {e4}.",
)

_DEFAULT_LEXICON: dict[str, tuple[str, ...]] = {
    "model": ("Transformer", "BERT", "GPT-3", "ResNet", "ViT"),
    "dataset": ("ImageNet", "CIFAR-10", "GLUE", "SQuAD", "WMT 2014"),
    "benchmark": ("WMT 2014 English-to-German", "SQuAD", "GLUE", "ImageNet"),
    "metric": ("28.4 BLEU", "F1 score", "accuracy", "perplexity"),
    "organization": ("OpenAI", "Google", "Meta", "Microsoft"),
    "method": ("fine-tuning", "pre-training", "attention mechanism"),
    "person": ("Geoffrey Hinton", "Yann LeCun"),
    "location": ("United States", "Europe"),
}


def _parse_text_length(raw: dict[str, Any]) -> TextLengthConfig:
    return TextLengthConfig(
        min_chars=int(raw.get("min_chars", 200)),
        max_chars=int(raw.get("max_chars", 800)),
    )


def _parse_entities_per_document(raw: dict[str, Any]) -> EntitiesPerDocumentConfig:
    return EntitiesPerDocumentConfig(
        distribution=str(raw.get("distribution", "uniform")),
        min_count=int(raw.get("min", raw.get("min_count", 3))),
        max_count=int(raw.get("max", raw.get("max_count", 8))),
        fixed_count=int(raw.get("fixed", raw.get("fixed_count", 5))),
        poisson_lambda=float(raw.get("lambda", raw.get("poisson_lambda", 6.0))),
    )


def _parse_generator(label: str, raw: dict[str, Any]) -> GeneratorConfig:
    formats_raw = raw.get("formats", ())
    formats = tuple(str(x) for x in formats_raw) if isinstance(formats_raw, list) else ()
    return GeneratorConfig(
        formats=formats,
        year_min=int(raw.get("year_min", 2015)),
        year_max=int(raw.get("year_max", 2026)),
        value_min=float(raw.get("min", raw.get("value_min", 0.1))),
        value_max=float(raw.get("max", raw.get("value_max", 99.9))),
        decimal_separators=tuple(raw.get("decimal_separators", [".", ","])),
        thousands_separators=tuple(raw.get("thousands_separators", [",", ".", ""])),
        currency_codes=tuple(raw.get("codes", raw.get("currency_codes", ["USD", "EUR", "GBP"]))),
        currency_symbols=tuple(raw.get("symbols", raw.get("currency_symbols", ["$", "€", "£"]))),
    )


def _parse_llm(raw: dict[str, Any]) -> LlmSyntheticConfig:
    max_attempts = max(1, int(raw.get("max_attempts", 3)))
    provider_attempts = int(raw.get("provider_max_attempts", max_attempts))
    validation_attempts = int(raw.get("validation_max_attempts", max_attempts))
    agent_mode = str(raw.get("agent_mode", "per_doc")).strip().lower()
    if agent_mode not in {"per_doc", "stateless"}:
        raise ValueError(f"llm.agent_mode must be 'per_doc' or 'stateless', got {agent_mode!r}")
    return LlmSyntheticConfig(
        model=str(raw.get("model", "")).strip(),
        max_attempts=max_attempts,
        provider_max_attempts=max(1, provider_attempts),
        validation_max_attempts=max(1, validation_attempts),
        temperature=float(raw.get("temperature", 0.4)),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        request_delay_seconds=float(raw.get("request_delay_seconds", 0.0)),
        parallelism=max(1, int(raw.get("parallelism", 3))),
        agent_mode=agent_mode,
        agno_db_path=str(raw.get("agno_db_path", "")).strip(),
        system_prompt=str(raw.get("system_prompt", "")),
        prompt_file=str(raw.get("prompt_file", "")).strip(),
        user_prompt=str(raw.get("user_prompt", "")),
        user_prompt_file=str(raw.get("user_prompt_file", "")).strip(),
        feedback_prompt=str(raw.get("feedback_prompt", "")),
        feedback_prompt_file=str(raw.get("feedback_prompt_file", "")).strip(),
    )


def _resolve_prompt_path(path_str: str, config_path: Path) -> Path:
    prompt_path = Path(path_str)
    if not prompt_path.is_absolute():
        prompt_path = (config_path.parent / prompt_path).resolve()
    return prompt_path


def _load_prompt_file(path_str: str, config_path: Path, label: str) -> str:
    if not path_str.strip():
        return ""
    prompt_path = _resolve_prompt_path(path_str, config_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Synthetic {label} prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def parse_document_types(raw: object) -> tuple[DocumentTypeWeight, ...]:
    """Parse ``document_types`` list from config JSON."""
    if not isinstance(raw, list):
        return ()
    out: list[DocumentTypeWeight] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        type_id = str(item.get("type", "")).strip()
        if not type_id:
            continue
        out.append(DocumentTypeWeight(type=type_id, weight=float(item.get("weight", 1.0))))
    return tuple(out)


def validate_synthetic_config(config: SyntheticConfig) -> None:
    """Validate cross-field constraints on a parsed config."""
    if config.engine.lower() == "llm" and not config.llm.model.strip():
        raise ValueError("engine=llm requires llm.model in the synthetic config file")
    if config.engine.lower() == "llm":
        if not config.llm.system_prompt.strip():
            raise ValueError("engine=llm requires llm.system_prompt or llm.prompt_file")
        if not config.llm.user_prompt.strip():
            raise ValueError("engine=llm requires llm.user_prompt or llm.user_prompt_file")
        if not config.llm.feedback_prompt.strip():
            raise ValueError("engine=llm requires llm.feedback_prompt or llm.feedback_prompt_file")


def load_synthetic_config(path: Path) -> SyntheticConfig:
    """Load synthetic config from YAML or JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML synthetic configs: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Synthetic config must be a mapping in {path}")
    config = parse_synthetic_config_dict(raw)
    llm = config.llm
    system_prompt = llm.system_prompt
    prompt_file = llm.prompt_file
    if llm.prompt_file:
        system_prompt = _load_prompt_file(llm.prompt_file, path, "system")
        prompt_file = str(_resolve_prompt_path(llm.prompt_file, path))
    user_prompt = llm.user_prompt
    user_prompt_file = llm.user_prompt_file
    if llm.user_prompt_file:
        user_prompt = _load_prompt_file(llm.user_prompt_file, path, "user")
        user_prompt_file = str(_resolve_prompt_path(llm.user_prompt_file, path))
    feedback_prompt = llm.feedback_prompt
    feedback_prompt_file = llm.feedback_prompt_file
    if llm.feedback_prompt_file:
        feedback_prompt = _load_prompt_file(llm.feedback_prompt_file, path, "feedback")
        feedback_prompt_file = str(_resolve_prompt_path(llm.feedback_prompt_file, path))
    config = replace(
        config,
        llm=replace(
            llm,
            system_prompt=system_prompt,
            prompt_file=prompt_file,
            user_prompt=user_prompt,
            user_prompt_file=user_prompt_file,
            feedback_prompt=feedback_prompt,
            feedback_prompt_file=feedback_prompt_file,
        ),
    )
    validate_synthetic_config(config)
    return config


def parse_synthetic_config_dict(raw: dict[str, Any]) -> SyntheticConfig:
    """Parse a config mapping into :class:`SyntheticConfig`."""
    text_length = _parse_text_length(raw.get("text_length", {}) or {})
    epd = _parse_entities_per_document(raw.get("entities_per_document", {}) or {})
    gens_raw = raw.get("generators", {}) or {}
    generators: dict[str, GeneratorConfig] = {}
    if isinstance(gens_raw, dict):
        for label, spec in gens_raw.items():
            if isinstance(spec, dict):
                generators[str(label)] = _parse_generator(str(label), spec)

    lexicon_raw = raw.get("lexicon", {}) or {}
    lexicon: dict[str, tuple[str, ...]] = dict(_DEFAULT_LEXICON)
    if isinstance(lexicon_raw, dict):
        for label, values in lexicon_raw.items():
            if isinstance(values, list):
                lexicon[str(label)] = tuple(str(v) for v in values)

    templates_raw = raw.get("templates", ())
    if isinstance(templates_raw, list) and templates_raw:
        templates = tuple(str(t) for t in templates_raw)
    else:
        templates = _DEFAULT_TEMPLATES

    label_weights_raw = raw.get("label_weights", {}) or {}
    label_weights: dict[str, float] = {}
    if isinstance(label_weights_raw, dict):
        for label, weight in label_weights_raw.items():
            label_weights[str(label)] = float(weight)

    domain = str(raw.get("domain", "scientific_abstract"))
    document_type = str(raw.get("document_type", "")).strip()
    if not document_type and domain:
        document_type = domain

    return SyntheticConfig(
        seed=int(raw.get("seed", 42)),
        num_documents=int(raw.get("num_documents", 10)),
        engine=str(raw.get("engine", "procedural")),
        domain=domain,
        document_type=document_type,
        document_types=parse_document_types(raw.get("document_types")),
        document_profiles_dir=str(raw.get("document_profiles_dir", "document_profiles")),
        text_length=text_length,
        entities_per_document=epd,
        label_weights=label_weights,
        lexicon=lexicon,
        generators=generators,
        templates=templates,
        llm=_parse_llm(raw.get("llm", {}) or {}),
    )
