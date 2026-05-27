"""Load and merge per-genre document profiles for synthetic generation.

Profiles live under ``configs/document_profiles/`` (or ``document_profiles_dir``
relative to the synthetic JSON file). Each YAML/JSON file defines
``style_description``, optional ``templates``, ``lexicon``, ``label_weights``,
and ``text_length`` overrides.

:func:`merge_config_for_profile` applies those overrides per document before
planning entities and calling the LLM. See ``docs/synthetic.md`` for merge
precedence and sparse-config defaults.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from ner_gold_generator.synthetic_config import (
    _DEFAULT_LEXICON,
    _DEFAULT_TEMPLATES,
    SyntheticConfig,
    TextLengthConfig,
    _parse_text_length,
)


@dataclass(frozen=True, slots=True)
class DocumentProfile:
    """Genre-specific overrides for synthetic document generation."""

    profile_id: str
    style_description: str
    templates: tuple[str, ...] = ()
    lexicon: dict[str, tuple[str, ...]] = field(default_factory=dict)
    label_weights: dict[str, float] | None = None
    text_length: TextLengthConfig | None = None


def _load_profile_file(path: Path) -> DocumentProfile:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML document profiles: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Document profile must be a mapping: {path}")
    profile_id = str(raw.get("id", path.stem))
    style = str(raw.get("style_description", "")).strip()
    if not style:
        raise ValueError(f"Document profile {path} requires style_description")
    templates_raw = raw.get("templates", ())
    templates = tuple(str(t) for t in templates_raw) if isinstance(templates_raw, list) else ()
    lexicon: dict[str, tuple[str, ...]] = {}
    lexicon_raw = raw.get("lexicon", {}) or {}
    if isinstance(lexicon_raw, dict):
        for label, values in lexicon_raw.items():
            if isinstance(values, list):
                lexicon[str(label)] = tuple(str(v) for v in values)
    label_weights: dict[str, float] | None = None
    lw_raw = raw.get("label_weights")
    if isinstance(lw_raw, dict) and lw_raw:
        label_weights = {str(k): float(v) for k, v in lw_raw.items()}
    text_length: TextLengthConfig | None = None
    tl_raw = raw.get("text_length")
    if isinstance(tl_raw, dict):
        text_length = _parse_text_length(tl_raw)
    return DocumentProfile(
        profile_id=profile_id,
        style_description=style,
        templates=templates,
        lexicon=lexicon,
        label_weights=label_weights,
        text_length=text_length,
    )


def load_document_profiles(profiles_dir: Path) -> dict[str, DocumentProfile]:
    """Load all profile files from *profiles_dir* (``.yaml`` / ``.json``)."""
    if not profiles_dir.is_dir():
        return {}
    profiles: dict[str, DocumentProfile] = {}
    for path in sorted(profiles_dir.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        profile = _load_profile_file(path)
        profiles[profile.profile_id] = profile
    return profiles


def resolve_profiles_dir(config: SyntheticConfig, config_path: Path | None) -> Path:
    """Resolve ``document_profiles_dir`` relative to the synthetic config file."""
    raw_dir = config.document_profiles_dir.strip() or "document_profiles"
    profiles_path = Path(raw_dir)
    if not profiles_path.is_absolute() and config_path is not None:
        profiles_path = (config_path.parent / profiles_path).resolve()
    elif not profiles_path.is_absolute():
        profiles_path = Path("configs") / profiles_path
    return profiles_path


def sample_document_type(config: SyntheticConfig, rng: random.Random) -> str:
    """Pick a document type id for one document (deterministic from *rng*)."""
    if config.document_types:
        types = config.document_types
        total = sum(max(0.0, t.weight) for t in types)
        if total <= 0:
            raise ValueError("document_types weights must sum to a positive value")
        pick = rng.random() * total
        acc = 0.0
        for entry in types:
            acc += max(0.0, entry.weight)
            if pick <= acc:
                return entry.type
        return types[-1].type
    if config.document_type.strip():
        return config.document_type.strip()
    return config.domain.strip() or "scientific_abstract"


def get_document_profile(
    config: SyntheticConfig,
    profile_id: str,
    *,
    profiles: dict[str, DocumentProfile] | None = None,
    config_path: Path | None = None,
) -> DocumentProfile:
    """Return the profile for *profile_id*, or a synthetic one from ``domain``."""
    if profiles is None:
        profiles = load_document_profiles(resolve_profiles_dir(config, config_path))
    if profile_id in profiles:
        return profiles[profile_id]
    # Fallback: use domain string as style when no profile file exists.
    return DocumentProfile(
        profile_id=profile_id,
        style_description=config.domain or profile_id.replace("_", " "),
    )


def merge_config_for_profile(
    base: SyntheticConfig,
    profile: DocumentProfile,
) -> SyntheticConfig:
    """Return a copy of *base* with profile overrides applied.

    Lexicon keys are merged (base, then profile). ``text_length``,
    ``label_weights``, and ``templates`` use the profile when set; otherwise
    the base config (or built-in defaults) remains. ``domain`` and
    ``document_type`` are set to ``profile.profile_id`` for logging and prompts.
    """
    lexicon = dict(_DEFAULT_LEXICON)
    lexicon.update(base.lexicon)
    lexicon.update(profile.lexicon)
    templates = profile.templates if profile.templates else base.templates
    if not templates:
        templates = _DEFAULT_TEMPLATES
    label_weights = profile.label_weights if profile.label_weights else base.label_weights
    text_length = profile.text_length if profile.text_length else base.text_length
    return replace(
        base,
        domain=profile.profile_id,
        document_type=profile.profile_id,
        lexicon=lexicon,
        templates=templates,
        label_weights=label_weights or base.label_weights,
        text_length=text_length,
    )
