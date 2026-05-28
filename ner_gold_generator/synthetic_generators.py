"""Deterministic generators for structured synthetic entity surfaces.

Plans labels and literal strings from ``SyntheticConfig`` (lexicon, generators,
``label_weights``) and a per-document seed. Used by :mod:`ner_gold_generator.synthetic`
before text assembly or LLM prompting.

See :func:`plan_entities_for_document` and :class:`PlannedEntity`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from ner_gold_generator.synthetic_config import GeneratorConfig, SyntheticConfig
from ner_gold_generator.validate import normalize_entity_label


@dataclass(frozen=True, slots=True)
class PlannedEntity:
    """Entity to embed verbatim in generated text."""

    surface: str
    label: str


def _gen_cfg(config: SyntheticConfig, label: str) -> GeneratorConfig:
    return config.generators.get(label, GeneratorConfig())


def _random_date(rng: random.Random, cfg: GeneratorConfig) -> date:
    start = date(cfg.year_min, 1, 1)
    end = date(cfg.year_max, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, max(0, delta_days)))


def _format_date(d: date, fmt: str) -> str:
    mapping = {
        "%d": f"{d.day:02d}",
        "%m": f"{d.month:02d}",
        "%Y": str(d.year),
        "%b": d.strftime("%b"),
        "%B": d.strftime("%B"),
    }
    out = fmt
    for token, value in sorted(mapping.items(), key=lambda x: -len(x[0])):
        out = out.replace(token, value)
    return out


def generate_date_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "date")
    d = _random_date(rng, cfg)
    formats = cfg.formats or (
        "%d/%m/%Y",
        "%d %b %Y",
        "%d-%m-%Y",
        "%B %d, %Y",
    )
    return _format_date(d, rng.choice(formats))


def generate_year_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "year")
    return str(rng.randint(cfg.year_min, cfg.year_max))


def _format_number_value(
    value: float,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> str:
    dec_sep = rng.choice(cfg.decimal_separators) if cfg.decimal_separators else "."
    thou_sep = rng.choice(cfg.thousands_separators) if cfg.thousands_separators else ""
    if rng.random() < 0.5:
        whole = int(value)
        frac = int(round((value - whole) * 10))
        body = f"{whole}{dec_sep}{frac}" if frac else str(whole)
    else:
        body = f"{value:.1f}".replace(".", dec_sep)
    if thou_sep and rng.random() < 0.4 and value >= 1000:
        parts = body.split(dec_sep, 1)
        whole_part = parts[0]
        if len(whole_part) > 3:
            whole_part = thou_sep.join(
                [whole_part[max(0, i - 3) : i] for i in range(len(whole_part), 0, -3)][::-1]
            )
        body = dec_sep.join([whole_part, parts[1]]) if len(parts) > 1 else whole_part
    return body


def generate_number_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "number")
    value = rng.uniform(cfg.value_min, cfg.value_max)
    return _format_number_value(value, rng, cfg)


def generate_percentage_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "percentage")
    value = rng.uniform(cfg.value_min, cfg.value_max)
    body = _format_number_value(value, rng, cfg)
    return f"{body}%"


def generate_currency_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "currency")
    value = rng.uniform(cfg.value_min, cfg.value_max * 100)
    amount = _format_number_value(value, rng, cfg)
    style = rng.randint(0, 2)
    code = rng.choice(cfg.currency_codes) if cfg.currency_codes else "USD"
    symbol = rng.choice(cfg.currency_symbols) if cfg.currency_symbols else "$"
    if style == 0:
        return f"{symbol}{amount}"
    if style == 1:
        return f"{amount} {code}"
    return f"{code} {amount}"


def generate_arxiv_id_surface(rng: random.Random, config: SyntheticConfig) -> str:
    cfg = _gen_cfg(config, "arxiv_id")
    year = rng.randint(cfg.year_min, cfg.year_max) % 100
    num = rng.randint(10000, 99999)
    variant = rng.randint(0, 2)
    if variant == 0:
        return f"arXiv:{year:02d}{num:05d}"
    if variant == 1:
        return f"arXiv:{year:02d}{num:05d}v{rng.randint(1, 3)}"
    return f"{year:02d}{num:05d}"


def generate_metric_surface(rng: random.Random, config: SyntheticConfig) -> str:
    lex = config.lexicon.get("metric", ())
    if lex and rng.random() < 0.6:
        return rng.choice(lex)
    value = rng.uniform(1.0, 99.9)
    unit = rng.choice(("BLEU", "F1", "accuracy", "ROUGE-L"))
    return f"{value:.1f} {unit}"


def generate_lexical_surface(
    label: str,
    rng: random.Random,
    config: SyntheticConfig,
    *,
    used: set[str],
) -> str:
    """Sample a lexical entity surface from lexicon, avoiding *used*."""
    entries = config.lexicon.get(label, ())
    if entries:
        candidates = [e for e in entries if e not in used]
        if candidates:
            return rng.choice(candidates)
    fallback = {
        "model": "NeuralNet-X",
        "dataset": "CustomSet",
        "benchmark": "EvalBench",
        "organization": "ResearchLab",
        "method": "backpropagation",
        "person": "A. Researcher",
        "location": "Cambridge",
        "acronym": "NLP",
        "miscellaneous": "baseline",
    }
    base = fallback.get(label, "Entity")
    suffix = 1
    candidate = base
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def generate_surface_for_label(
    label: str,
    rng: random.Random,
    config: SyntheticConfig,
    *,
    used: set[str],
) -> str:
    """Generate a unique surface string for *label*."""
    normalized = normalize_entity_label(label)
    if normalized is None:
        raise ValueError(f"Unsupported synthetic label: {label!r}")
    label = normalized
    for _ in range(32):
        if label == "date":
            surface = generate_date_surface(rng, config)
        elif label == "year":
            surface = generate_year_surface(rng, config)
        elif label == "number":
            surface = generate_number_surface(rng, config)
        elif label == "percentage":
            surface = generate_percentage_surface(rng, config)
        elif label == "currency":
            surface = generate_currency_surface(rng, config)
        elif label == "arxiv_id":
            surface = generate_arxiv_id_surface(rng, config)
        elif label == "metric":
            surface = generate_metric_surface(rng, config)
        else:
            surface = generate_lexical_surface(label, rng, config, used=used)
        if surface and surface not in used:
            return surface
    raise RuntimeError(f"Could not generate unique surface for label {label!r}")


def sample_entity_count(rng: random.Random, config: SyntheticConfig) -> int:
    """Sample how many entities to plan for one document."""
    epd = config.entities_per_document
    if epd.distribution == "fixed":
        return max(1, epd.fixed_count)
    if epd.distribution == "poisson":
        import math

        lam = epd.poisson_lambda
        l_exp = math.exp(-lam)
        k = 0
        p = 1.0
        while p > l_exp:
            k += 1
            p *= rng.random()
        count = max(0, k - 1)
        return max(epd.min_count, min(epd.max_count, count))
    lo = max(1, epd.min_count)
    hi = max(lo, epd.max_count)
    return rng.randint(lo, hi)


def sample_label(rng: random.Random, weights: dict[str, float]) -> str:
    """Weighted random choice of entity label."""
    labels = list(weights.keys())
    probs = [weights[label] for label in labels]
    return rng.choices(labels, weights=probs, k=1)[0]


def plan_entities_for_document(
    rng: random.Random,
    config: SyntheticConfig,
) -> list[PlannedEntity]:
    """Build an entity plan (surfaces + labels) for one synthetic document."""
    count = sample_entity_count(rng, config)
    weights = config.normalized_label_weights()
    used: set[str] = set()
    plan: list[PlannedEntity] = []
    for _ in range(count):
        label = sample_label(rng, weights)
        surface = generate_surface_for_label(label, rng, config, used=used)
        used.add(surface)
        plan.append(PlannedEntity(surface=surface, label=label))
    return plan
