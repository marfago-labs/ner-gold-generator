"""Load and validate synthetic gold generation configuration (YAML/JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


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
    system_prompt: str = ""
    prompt_file: str = ""


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Full synthetic dataset specification."""

    seed: int = 42
    num_documents: int = 10
    engine: str = "procedural"
    domain: str = "scientific_abstract"
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
    return LlmSyntheticConfig(
        model=str(raw.get("model", "")).strip(),
        max_attempts=max_attempts,
        provider_max_attempts=max(1, provider_attempts),
        validation_max_attempts=max(1, validation_attempts),
        temperature=float(raw.get("temperature", 0.4)),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        request_delay_seconds=float(raw.get("request_delay_seconds", 0.0)),
        parallelism=max(1, int(raw.get("parallelism", 3))),
        system_prompt=str(raw.get("system_prompt", "")),
        prompt_file=str(raw.get("prompt_file", "")).strip(),
    )


def validate_synthetic_config(config: SyntheticConfig) -> None:
    """Validate cross-field constraints on a parsed config."""
    if config.engine.lower() == "llm" and not config.llm.model.strip():
        raise ValueError("engine=llm requires llm.model in the synthetic config file")


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
    if config.llm.prompt_file:
        prompt_path = Path(config.llm.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (path.parent / prompt_path).resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Synthetic prompt file not found: {prompt_path}")
        config = replace(
            config,
            llm=replace(
                config.llm,
                system_prompt=prompt_path.read_text(encoding="utf-8").strip(),
                prompt_file=str(prompt_path),
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

    return SyntheticConfig(
        seed=int(raw.get("seed", 42)),
        num_documents=int(raw.get("num_documents", 10)),
        engine=str(raw.get("engine", "procedural")),
        domain=str(raw.get("domain", "scientific_abstract")),
        text_length=text_length,
        entities_per_document=epd,
        label_weights=label_weights,
        lexicon=lexicon,
        generators=generators,
        templates=templates,
        llm=_parse_llm(raw.get("llm", {}) or {}),
    )
