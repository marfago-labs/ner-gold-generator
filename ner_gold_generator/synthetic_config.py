"""Load and validate synthetic gold generation configuration (YAML/JSON).

Repository layout (relative to the package root)::

    configs/
      arxiv_benchmark.yaml        # built-in arXiv manual spans
      synthetic_defaults.yaml     # default procedural templates + lexicon
      synthetic_llm.yaml          # example batch spec
      prompts/                    # LLM prompt templates (not inlined in Python)
        synthetic_llm_system_prompt.txt
        synthetic_llm_user_prompt.txt
        synthetic_llm_feedback_prompt.txt
        labeling_system_prompt.txt
        labeling_user_prompt.txt
      document_profiles/          # per-genre YAML overrides

Synthetic YAML files reference prompts with paths relative to the config file,
for example ``prompt_file: prompts/synthetic_llm_system_prompt.txt``.
:func:`load_synthetic_config` reads those files and inlines the text into
:class:`LlmSyntheticConfig` before validation.

Omitted YAML fields receive code defaults (see :class:`SyntheticConfig` and
:class:`TextLengthConfig`). When ``document_types`` or ``document_type`` is set,
:func:`ner_gold_generator.document_profiles.merge_config_for_profile` may
override ``text_length``, lexicon, ``label_weights``, and templates per document.
"""

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
    """Inclusive character bounds enforced on LLM output and in prompts."""

    min_chars: int = 200
    max_chars: int = 800


@dataclass(frozen=True, slots=True)
class EntitiesPerDocumentConfig:
    """How many planned entities to sample per document."""

    distribution: str = "uniform"
    min_count: int = 3
    max_count: int = 8
    fixed_count: int = 5
    poisson_lambda: float = 6.0


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Per-label options for structured surface generators (date, number, currency, …)."""

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
    """OpenRouter / Agno settings for ``engine: llm``.

    Prompt prose lives under ``configs/prompts/`` and is loaded via
    ``prompt_file``, ``user_prompt_file``, and ``feedback_prompt_file`` (paths
    relative to the synthetic config file). After loading, the file contents
    are available as ``system_prompt``, ``user_prompt``, and ``feedback_prompt``.
    """

    model: str = ""
    max_attempts: int = 3
    provider_max_attempts: int = 3
    validation_max_attempts: int = 3
    temperature: float = 0.4
    timeout_seconds: float = 120.0
    request_delay_seconds: float = 0.0
    parallelism: int = 3
    agent_mode: str = "per_doc"
    max_restarts: int = 1
    agno_db_path: str = ""
    document_log_dir: str = ""  # per-document LLM trace files, e.g. logs/
    system_prompt: str = ""
    prompt_file: str = ""  # e.g. prompts/synthetic_llm_system_prompt.txt
    user_prompt: str = ""  # filled from user_prompt_file at load time
    user_prompt_file: str = ""  # e.g. prompts/synthetic_llm_user_prompt.txt
    feedback_prompt: str = ""  # filled from feedback_prompt_file at load time
    feedback_prompt_file: str = ""  # e.g. prompts/synthetic_llm_feedback_prompt.txt


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Full synthetic dataset specification.

    Top-level YAML or JSON may omit many fields; defaults apply (for example
    ``text_length`` 200–800 chars, built-in ``label_weights`` when empty).
    Per-document genre profiles can override length and sampling; see
    :mod:`ner_gold_generator.document_profiles`.
    """

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
        """Return label weights summing to 1.0 (built-in mix when ``label_weights`` is empty)."""
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


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYNTHETIC_DEFAULTS_PATH = _REPO_ROOT / "configs" / "synthetic_defaults.yaml"


def default_synthetic_defaults_path(*, repo_root: Path | None = None) -> Path:
    """Return ``configs/synthetic_defaults.yaml`` under the repository root."""
    root = repo_root or _REPO_ROOT
    return root / "configs" / "synthetic_defaults.yaml"


def load_synthetic_defaults(
    path: Path | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Load default procedural templates and lexicon from YAML.

    Raises:
        FileNotFoundError: If the defaults file is missing.
        ValueError: If the file structure is invalid.
    """
    config_path = path or _DEFAULT_SYNTHETIC_DEFAULTS_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Synthetic defaults config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for synthetic defaults: pip install pyyaml") from exc
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Synthetic defaults must be a mapping: {config_path}")

    templates_raw = raw.get("templates", ())
    if not isinstance(templates_raw, list) or not templates_raw:
        raise ValueError(f"templates must be a non-empty list in {config_path}")
    templates = tuple(str(t) for t in templates_raw)

    lexicon_raw = raw.get("lexicon", {}) or {}
    if not isinstance(lexicon_raw, dict) or not lexicon_raw:
        raise ValueError(f"lexicon must be a non-empty mapping in {config_path}")
    lexicon: dict[str, tuple[str, ...]] = {}
    for label, values in lexicon_raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"lexicon[{label!r}] must be a non-empty list in {config_path}")
        lexicon[str(label)] = tuple(str(v) for v in values)
    return templates, lexicon


_DEFAULT_TEMPLATES, _DEFAULT_LEXICON = load_synthetic_defaults()


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
        max_restarts=max(0, int(raw.get("max_restarts", 1))),
        agno_db_path=str(raw.get("agno_db_path", "")).strip(),
        document_log_dir=str(raw.get("document_log_dir", "")).strip(),
        system_prompt=str(raw.get("system_prompt", "")),
        prompt_file=str(raw.get("prompt_file", "")).strip(),
        user_prompt=str(raw.get("user_prompt", "")),
        user_prompt_file=str(raw.get("user_prompt_file", "")).strip(),
        feedback_prompt=str(raw.get("feedback_prompt", "")),
        feedback_prompt_file=str(raw.get("feedback_prompt_file", "")).strip(),
    )


def _resolve_prompt_path(path_str: str, config_path: Path) -> Path:
    """Resolve a prompt path relative to the synthetic config file directory."""
    prompt_path = Path(path_str)
    if not prompt_path.is_absolute():
        prompt_path = (config_path.parent / prompt_path).resolve()
    return prompt_path


def _load_prompt_file(path_str: str, config_path: Path, label: str) -> str:
    """Read a prompt template from disk (``label`` is ``system`` / ``user`` / ``feedback``)."""
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
    """Load synthetic config from YAML or JSON and inline prompt files.

    Resolves ``llm.prompt_file``, ``llm.user_prompt_file``, and
    ``llm.feedback_prompt_file`` relative to *path*'s parent directory (typically
    ``configs/prompts/...``). Raises :class:`FileNotFoundError` when a referenced
    prompt file is missing.
    """
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
    """Parse a config mapping into :class:`SyntheticConfig` without reading prompt files.

    Use :func:`load_synthetic_config` for on-disk configs so prompt paths are loaded.
    """
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
