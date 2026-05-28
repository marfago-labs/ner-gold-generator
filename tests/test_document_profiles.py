"""Tests for document profile loading and merging."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ner_gold_generator.document_profiles import (
    get_document_profile,
    load_document_profiles,
    merge_config_for_profile,
    resolve_profiles_dir,
    sample_document_type,
)
from ner_gold_generator.synthetic_config import parse_synthetic_config_dict

_REPO = Path(__file__).resolve().parent.parent


def test_load_document_profiles_from_repo() -> None:
    profiles = load_document_profiles(_REPO / "configs" / "document_profiles")
    assert "news_article" in profiles
    assert "scientific_abstract" in profiles
    assert profiles["news_article"].style_description


def test_transcript_profile_avoids_speaker_role_lexicon() -> None:
    profiles = load_document_profiles(_REPO / "configs" / "document_profiles")
    transcript = profiles["transcript"]
    people = transcript.lexicon.get("person", ())
    blocked = {"Host", "Guest", "Chris"}
    assert blocked.isdisjoint(set(people))
    assert any(" " in name for name in people)


def test_merge_transcript_profile_uses_distinct_lexicon() -> None:
    base = parse_synthetic_config_dict({"document_type": "transcript"})
    profile = get_document_profile(
        base,
        "transcript",
        profiles=load_document_profiles(_REPO / "configs" / "document_profiles"),
    )
    merged = merge_config_for_profile(base, profile)
    assert "Northwind Research" in merged.lexicon.get("organization", ())
    assert "Host" not in merged.lexicon.get("person", ())


def test_sample_document_type_mixed_deterministic() -> None:
    config = parse_synthetic_config_dict(
        {
            "document_types": [
                {"type": "news_article", "weight": 1.0},
                {"type": "blog_post", "weight": 0.0},
            ]
        }
    )
    rng = random.Random(42)
    assert sample_document_type(config, rng) == "news_article"


def test_merge_config_for_profile_overrides_lexicon() -> None:
    base = parse_synthetic_config_dict({"document_type": "news_article"})
    profile = get_document_profile(
        base,
        "news_article",
        profiles=load_document_profiles(_REPO / "configs" / "document_profiles"),
    )
    merged = merge_config_for_profile(base, profile)
    assert merged.document_type == "news_article"
    assert "Reuters" in merged.lexicon.get("organization", ())


def test_resolve_profiles_dir_relative_to_config(tmp_path: Path) -> None:
    config = parse_synthetic_config_dict({"document_profiles_dir": "profiles"})
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text("{}", encoding="utf-8")
    resolved = resolve_profiles_dir(config, cfg_file)
    assert resolved == (tmp_path / "profiles").resolve()


def test_load_profile_json_and_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not a dict", encoding="utf-8")
    with pytest.raises(ValueError):
        from ner_gold_generator.document_profiles import _load_profile_file

        _load_profile_file(bad)

    yaml_path = tmp_path / "needs_yaml.yaml"
    yaml_path.write_text("id: yaml_profile\nstyle_description: YAML style\n", encoding="utf-8")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="PyYAML"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", fake_import)
        from ner_gold_generator.document_profiles import _load_profile_file

        _load_profile_file(yaml_path)

    no_style = tmp_path / "empty.json"
    no_style.write_text(json.dumps({"id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="style_description"):
        from ner_gold_generator.document_profiles import _load_profile_file

        _load_profile_file(no_style)

    good = tmp_path / "ok.json"
    good.write_text(
        json.dumps({"id": "custom", "style_description": "Custom style", "templates": ["{e0}"]}),
        encoding="utf-8",
    )
    from ner_gold_generator.document_profiles import _load_profile_file

    p = _load_profile_file(good)
    assert p.profile_id == "custom"


def test_load_document_profiles_missing_dir(tmp_path: Path) -> None:
    assert load_document_profiles(tmp_path / "nope") == {}


def test_sample_document_type_weighted_pick_and_fallbacks() -> None:
    config = parse_synthetic_config_dict({"document_type": "blog_post"})
    assert sample_document_type(config, random.Random(0)) == "blog_post"

    config_domain = parse_synthetic_config_dict({"domain": "custom_domain"})
    assert sample_document_type(config_domain, random.Random(0)) == "custom_domain"

    mixed = parse_synthetic_config_dict(
        {
            "document_types": [
                {"type": "a", "weight": 1.0},
                {"type": "b", "weight": 1.0},
            ]
        }
    )
    assert sample_document_type(mixed, random.Random(0.99)) in {"a", "b"}


def test_get_document_profile_fallback_and_merge_empty_templates(tmp_path: Path) -> None:
    config = parse_synthetic_config_dict({"document_type": "unknown_type"})
    profile = get_document_profile(config, "unknown_type", profiles={})
    assert profile.profile_id == "unknown_type"
    merged = merge_config_for_profile(config, profile)
    assert merged.templates


def test_resolve_profiles_dir_absolute() -> None:
    config = parse_synthetic_config_dict({"document_profiles_dir": "/abs/profiles"})
    assert resolve_profiles_dir(config, None) == Path("/abs/profiles")


def test_load_document_profiles_skips_non_profile_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
    prof = tmp_path / "p.json"
    prof.write_text(
        json.dumps({"id": "p1", "style_description": "Style", "templates": ["{e0}"]}),
        encoding="utf-8",
    )
    loaded = load_document_profiles(tmp_path)
    assert "p1" in loaded


def test_sample_document_type_requires_positive_weights() -> None:
    config = parse_synthetic_config_dict({"document_types": [{"type": "a", "weight": 0.0}]})
    with pytest.raises(ValueError, match="positive"):
        sample_document_type(config, random.Random(0))
