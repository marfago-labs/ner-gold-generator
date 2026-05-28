"""Tests for synthetic gold generation."""

from __future__ import annotations

import builtins
import json
import random
from pathlib import Path

import pytest
from tests.llm_test_helpers import make_llm_config

from ner_gold_generator.build_gold import main
from ner_gold_generator.llm_client import LlmRequestError
from ner_gold_generator.synthetic import build_examples_synthetic
from ner_gold_generator.synthetic_compose import (
    assemble_mock_llm_text,
    assemble_procedural_text,
    audit_plan_entity_usage,
    count_surface_occurrences,
    entities_from_plan,
    fill_template,
    generate_text_with_llm,
    mock_llm_client_for_plan,
    validate_text_contains_plan,
)
from ner_gold_generator.synthetic_config import (
    SyntheticConfig,
    load_synthetic_config,
    parse_synthetic_config_dict,
    validate_synthetic_config,
)
from ner_gold_generator.synthetic_generators import (
    PlannedEntity,
    generate_arxiv_id_surface,
    generate_currency_surface,
    generate_date_surface,
    generate_number_surface,
    generate_percentage_surface,
    generate_surface_for_label,
    generate_year_surface,
    plan_entities_for_document,
    sample_entity_count,
    sample_label,
)
from ner_gold_generator.validate import index_unique_span, normalize_entity_label

_REPO = Path(__file__).resolve().parent.parent
_SMOKE_CONFIG = _REPO / "configs" / "synthetic_smoke.yaml"


def test_normalize_new_synthetic_labels() -> None:
    assert normalize_entity_label("date") == "date"
    assert normalize_entity_label("currency") == "currency"
    assert normalize_entity_label("percentage") == "percentage"


def test_structured_generators_deterministic() -> None:
    rng = random.Random(99)
    config = parse_synthetic_config_dict({"seed": 1})
    d1 = generate_date_surface(rng, config)
    rng2 = random.Random(99)
    assert generate_date_surface(rng2, config) == d1
    assert generate_year_surface(random.Random(1), config)
    assert "%" in generate_percentage_surface(random.Random(2), config)
    assert generate_currency_surface(random.Random(3), config)
    assert generate_number_surface(random.Random(4), config)
    assert "arXiv" in generate_arxiv_id_surface(
        random.Random(5), config
    ) or generate_arxiv_id_surface(random.Random(6), config)


def test_plan_entities_unique_surfaces() -> None:
    config = parse_synthetic_config_dict(
        {
            "entities_per_document": {"distribution": "fixed", "fixed": 6},
            "label_weights": {"model": 1.0, "dataset": 1.0, "date": 1.0},
        }
    )
    plan = plan_entities_for_document(random.Random(7), config)
    surfaces = [e.surface for e in plan]
    assert len(surfaces) == len(set(surfaces))


def test_fill_template_and_procedural_text() -> None:
    plan = [
        PlannedEntity("BERT", "model"),
        PlannedEntity("GLUE", "dataset"),
        PlannedEntity("2024", "year"),
        PlannedEntity("12.5%", "percentage"),
    ]
    config = parse_synthetic_config_dict({"text_length": {"min_chars": 50, "max_chars": 2000}})
    text = assemble_procedural_text(plan, config, random.Random(1))
    assert "BERT" in text
    assert "GLUE" in text
    errors = validate_text_contains_plan(text, plan)
    assert not errors
    entities = entities_from_plan(text, plan)
    assert len(entities) == len(plan)
    for ent in entities:
        assert index_unique_span(text, ent["text"]) is not None


def test_build_examples_synthetic_smoke_config() -> None:
    config = load_synthetic_config(_SMOKE_CONFIG)
    examples = build_examples_synthetic(config)
    assert len(examples) == 3
    for row in examples:
        assert row["source"] == "synthetic"
        assert row["gold_source"] == "synthetic"
        assert "synthetic_meta" in row
        usage = row["synthetic_meta"]["entity_usage"]
        assert usage["used_exactly_once"] == usage["planned"]
        for ent in row["entities"]:
            span = index_unique_span(row["text"], ent["text"])
            assert span == (ent["start"], ent["end"])


def test_audit_plan_entity_usage_regex_counts() -> None:
    plan = [
        PlannedEntity("BERT", "model"),
        PlannedEntity("GLUE", "dataset"),
        PlannedEntity("BERT", "model"),
    ]
    text = "We compare BERT on GLUE and cite BERT again."
    audit = audit_plan_entity_usage(text, plan)
    assert audit.planned == 3
    assert audit.used_exactly_once == 1
    assert "GLUE" not in audit.missing
    assert audit.occurrence_counts["BERT"] == 2
    assert count_surface_occurrences(text, "GLUE") == 1
    assert count_surface_occurrences(text, "") == 0
    assert validate_text_contains_plan(text, [PlannedEntity("GLUE", "dataset")]) == []
    assert "BERT" in str(validate_text_contains_plan(text, plan))


def test_build_examples_synthetic_llm_builds_worker_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ner_gold_generator import synthetic as syn_mod
    from ner_gold_generator.labeling import LlmGoldSettings

    config = parse_synthetic_config_dict(
        {
            "num_documents": 2,
            "engine": "llm",
            "llm": make_llm_config(),
            "entities_per_document": {"distribution": "fixed", "fixed": 2},
        }
    )
    build_calls: list[int] = []

    class _PromptClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, temperature, call_label
            self.calls += 1
            if self.calls == 1:
                openai = pytest.importorskip("openai")
                raise openai.RateLimitError(
                    "rate",
                    response=pytest.importorskip("httpx").Response(
                        429,
                        headers={"retry-after": "0"},
                        request=pytest.importorskip("httpx").Request(
                            "POST", "https://openrouter.ai/api/v1/chat/completions"
                        ),
                    ),
                    body={
                        "error": {
                            "metadata": {"retry_after_seconds": 0.1},
                        }
                    },
                )
            plan: list[PlannedEntity] = []
            for line in user.splitlines():
                if line.startswith("- ["):
                    label, surface = line[3:].split("] ", 1)
                    plan.append(PlannedEntity(surface, label))
            return {"text": assemble_mock_llm_text(plan, config)}

    def _fake_build_async(*_args: object, **_kwargs: object) -> _PromptClient:
        build_calls.append(1)
        return _PromptClient()

    monkeypatch.setattr(syn_mod, "build_async_chat_client", _fake_build_async)
    monkeypatch.setattr(
        "ner_gold_generator.llm_client.verify_llm_installation",
        lambda: None,
    )
    examples = build_examples_synthetic(
        config,
        settings=LlmGoldSettings.model_construct(
            openrouter_api_key="test-key",
            mock_llm=False,
        ),
        mock=False,
    )
    assert len(build_calls) == 2
    assert len(examples) == 2


def test_build_examples_synthetic_llm_mock() -> None:
    config = parse_synthetic_config_dict(
        {
            "num_documents": 2,
            "engine": "llm",
            "llm": make_llm_config(),
            "entities_per_document": {"distribution": "fixed", "fixed": 3},
        }
    )
    examples = build_examples_synthetic(config, mock=True)
    assert len(examples) == 2
    assert all(e["entities"] for e in examples)


def test_parse_llm_timeout_and_request_delay() -> None:
    cfg = parse_synthetic_config_dict(
        {"llm": {"timeout_seconds": 45.0, "request_delay_seconds": 3.0, "parallelism": 4}}
    )
    assert cfg.llm.timeout_seconds == 45.0
    assert cfg.llm.request_delay_seconds == 3.0
    assert cfg.llm.parallelism == 4


def test_load_synthetic_config_with_prompt_file(tmp_path: Path) -> None:
    (tmp_path / "system.txt").write_text("You are a strict generator.", encoding="utf-8")
    (tmp_path / "user.txt").write_text("User {entity_list}", encoding="utf-8")
    (tmp_path / "feedback.txt").write_text("Fix {issues}", encoding="utf-8")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "engine": "llm",
                "llm": {
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "prompt_file": "system.txt",
                    "user_prompt_file": "user.txt",
                    "feedback_prompt_file": "feedback.txt",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_synthetic_config(cfg_path)
    assert cfg.llm.system_prompt == "You are a strict generator."
    assert cfg.llm.user_prompt == "User {entity_list}"
    assert cfg.llm.feedback_prompt == "Fix {issues}"


def test_load_synthetic_config_missing_prompt_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "engine": "llm",
                "llm": {
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "prompt_file": "missing.txt",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Synthetic system prompt file not found"):
        load_synthetic_config(cfg_path)


def test_load_synthetic_config_yaml_requires_pyyaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("engine: procedural\n", encoding="utf-8")
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object):
        if name == "yaml":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match="PyYAML is required"):
        load_synthetic_config(cfg_path)


def test_generate_text_with_llm_request_delay_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(max_attempts=2, request_delay_seconds=0.01),
            "text_length": {"min_chars": 10, "max_chars": 5000},
        }
    )
    plan = [PlannedEntity("BERT", "model")]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(
        "ner_gold_generator.synthetic_compose.asyncio.sleep",
        fake_sleep,
    )

    class _BadThenGood:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            self.calls += 1
            if self.calls == 1:
                return {"text": "no entities here"}
            return {"text": "We study BERT on GLUE."}

    generate_text_with_llm(
        plan,
        config,
        settings=None,
        mock=False,
        client=_BadThenGood(),
        request_delay_seconds=config.llm.request_delay_seconds,
    )
    assert sleeps == [pytest.approx(0.01)]


def test_validate_llm_text_or_log_writes_failure_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ner_gold_generator import llm_logging
    from ner_gold_generator import synthetic_compose as compose_mod

    log_path = tmp_path / "failures.log"
    monkeypatch.setenv("NER_GOLD_LOG_FILE", str(log_path))
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "0")
    llm_logging._LOGGER.handlers.clear()
    llm_logging._LOGGER = llm_logging.get_logger(llm_logging.__name__)

    plan = [PlannedEntity("BERT", "model")]
    err = compose_mod._validate_llm_text_or_log(
        "BERT and BERT again",
        plan,
        document_id="synthetic-test-00001",
        model_id="test/model",
        attempt=2,
        call_budget=7,
        min_chars=1,
        max_chars=9999,
    )
    assert err is not None
    log_text = log_path.read_text(encoding="utf-8")
    assert "synthetic-test-00001" in log_text
    assert "BERT and BERT again" in log_text
    assert "appears 2 times" in log_text


def test_generate_text_with_llm_respects_validation_retry_budget() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=5, validation_max_attempts=2),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    class _AlwaysInvalid:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            self.calls += 1
            return {"text": "no matching entities"}

    client = _AlwaysInvalid()
    with pytest.raises(ValueError, match="reason=validation"):
        generate_text_with_llm(plan, config, settings=None, mock=False, client=client)
    assert client.calls == 2


def test_invalid_json_counts_as_provider_failure() -> None:
    from ner_gold_generator.llm_client import InvalidLlmResponseError

    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=2, validation_max_attempts=5),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    class _BadJson:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            self.calls += 1
            raise InvalidLlmResponseError(
                "Model 'm' returned invalid JSON: Expecting value",
                raw="not json at all",
            )

    client = _BadJson()
    with pytest.raises(ValueError, match="reason=provider"):
        generate_text_with_llm(plan, config, settings=None, mock=False, client=client)
    assert client.calls == 2


def test_generate_text_with_llm_non_retryable_provider_fails_fast() -> None:
    config = parse_synthetic_config_dict(
        {"engine": "llm", "llm": make_llm_config(provider_max_attempts=5)}
    )
    plan = [PlannedEntity("BERT", "model")]

    class _AuthFail:
        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            raise LlmRequestError("forbidden", retryable=False)

    with pytest.raises(ValueError, match="reason=provider"):
        generate_text_with_llm(plan, config, settings=None, mock=False, client=_AuthFail())


def test_generate_text_with_llm_respects_provider_retry_budget() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(provider_max_attempts=2, validation_max_attempts=5),
        }
    )
    plan = [PlannedEntity("BERT", "model")]

    class _AlwaysRateLimited:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            self.calls += 1
            raise LlmRequestError("rate limit", retryable=True, retry_after_seconds=0.0)

    client = _AlwaysRateLimited()
    with pytest.raises(ValueError, match="reason=provider"):
        generate_text_with_llm(plan, config, settings=None, mock=False, client=client)
    assert client.calls == 2


def test_parse_llm_retry_budgets_fallback_to_max_attempts() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(max_attempts=4),
        }
    )
    assert config.llm.provider_max_attempts == 4
    assert config.llm.validation_max_attempts == 4


def test_generate_text_with_llm_on_status_before_call() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(),
        }
    )
    plan = plan_entities_for_document(random.Random(0), config)
    statuses: list[str] = []

    class _Client:
        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            return {"text": assemble_mock_llm_text(plan, config)}

    generate_text_with_llm(
        plan,
        config,
        settings=None,
        mock=False,
        client=_Client(),
        on_status=statuses.append,
    )
    assert statuses[0] == "calling OpenRouter…"


def test_generate_text_with_llm_reports_elapsed_per_call() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(),
        }
    )
    plan = plan_entities_for_document(random.Random(0), config)
    elapsed: list[tuple[int, float, str]] = []

    class _Client:
        def complete_json(
            self,
            *,
            system: str,
            user: str,
            temperature: float,
            call_label: str | None = None,
        ) -> dict:
            del system, user, temperature, call_label
            return {"text": assemble_mock_llm_text(plan, config)}

    text = generate_text_with_llm(
        plan,
        config,
        settings=None,
        mock=False,
        client=_Client(),
        on_call_elapsed=lambda attempt, seconds, outcome: elapsed.append(
            (attempt, seconds, outcome)
        ),
    )
    assert text
    assert len(elapsed) == 1
    assert elapsed[0][0] == 1
    assert elapsed[0][2] == "ok"


def test_generate_text_with_llm_mock_client() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(),
        }
    )
    plan = plan_entities_for_document(random.Random(0), config)
    client = mock_llm_client_for_plan(plan, config)
    text = generate_text_with_llm(plan, config, settings=None, mock=False, client=client)
    assert not validate_text_contains_plan(text, plan)


def test_sample_entity_count_distributions() -> None:
    config = parse_synthetic_config_dict(
        {
            "entities_per_document": {
                "distribution": "poisson",
                "min": 2,
                "max": 10,
                "lambda": 5.0,
            }
        }
    )
    rng = random.Random(0)
    for _ in range(20):
        n = sample_entity_count(rng, config)
        assert 2 <= n <= 10

    fixed_cfg = parse_synthetic_config_dict(
        {"entities_per_document": {"distribution": "fixed", "fixed": 7}}
    )
    assert sample_entity_count(random.Random(0), fixed_cfg) == 7


def test_load_synthetic_config_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.dump({"seed": 1, "num_documents": 1, "engine": "procedural"}),
        encoding="utf-8",
    )
    cfg = load_synthetic_config(path)
    assert cfg.seed == 1


def test_invalid_config_and_engine() -> None:
    bad = parse_synthetic_config_dict({"label_weights": {"model": 0}})
    with pytest.raises(ValueError, match="label_weights"):
        bad.normalized_label_weights()
    with pytest.raises(ValueError, match="llm.model"):
        validate_synthetic_config(parse_synthetic_config_dict({"engine": "llm"}))
    config = parse_synthetic_config_dict({"engine": "unknown"})
    with pytest.raises(ValueError, match="engine"):
        build_examples_synthetic(config)


def test_cli_synthetic(tmp_path: Path) -> None:
    out = tmp_path / "gold.jsonl"
    code = main(
        [
            "--source",
            "synthetic",
            "--synthetic-config",
            str(_SMOKE_CONFIG),
            "--output",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["id"].startswith("synthetic-")


def test_cli_synthetic_missing_config() -> None:
    assert main(["--source", "synthetic", "--output", "x.jsonl"]) == 2


def test_fill_template_no_placeholders() -> None:
    assert fill_template("Plain sentence.", []) == "Plain sentence."


def test_fill_template_insufficient_entities() -> None:
    assert fill_template("Use {e0} and {e1}.", [PlannedEntity("A", "model")]) is None


def test_generate_surface_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        generate_surface_for_label("not_a_label", random.Random(0), SyntheticConfig(), used=set())


def test_assemble_mock_llm_text() -> None:
    plan = [PlannedEntity("X", "model")]
    config = parse_synthetic_config_dict({})
    text = assemble_mock_llm_text(plan, config)
    assert "X" in text


def test_sample_label() -> None:
    w = {"model": 1.0}
    assert sample_label(random.Random(0), w) == "model"


def test_parse_full_synthetic_config() -> None:
    cfg = parse_synthetic_config_dict(
        {
            "generators": {
                "date": {"formats": ["%Y-%m-%d"], "year_min": 2020, "year_max": 2021},
                "number": {"decimal_separators": [","]},
            },
            "lexicon": {"model": ["TinyLM"]},
            "templates": ["Model {e0} on {e1}."],
            "entities_per_document": {"distribution": "uniform", "min": 2, "max": 2},
        }
    )
    assert cfg.generators["date"].formats == ("%Y-%m-%d",)
    assert cfg.lexicon["model"] == ("TinyLM",)
    assert cfg.templates[0].startswith("Model")


def test_load_config_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_synthetic_config(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_synthetic_config(bad)


def test_num_documents_validation() -> None:
    with pytest.raises(ValueError, match="num_documents"):
        build_examples_synthetic(parse_synthetic_config_dict({"num_documents": 0}))


def test_procedural_truncation_and_reference_line() -> None:
    plan = [PlannedEntity("Z" * 120, "model")]
    config = parse_synthetic_config_dict(
        {"text_length": {"min_chars": 10, "max_chars": 50}, "templates": ("{e0}",)}
    )
    text = assemble_procedural_text(plan, config, random.Random(1))
    assert "Z" * 120 in text or "Reference:" in text


def test_generate_text_with_llm_retries_then_succeeds() -> None:
    config = parse_synthetic_config_dict(
        {
            "engine": "llm",
            "llm": make_llm_config(max_attempts=3),
        }
    )
    plan = [PlannedEntity("BERT", "model"), PlannedEntity("GLUE", "dataset")]
    good = assemble_mock_llm_text(plan, config)
    from ner_gold_generator.llm_client import MockChatClient

    client = MockChatClient(responses=[{"text": ""}, {"text": good}])
    text = generate_text_with_llm(
        plan,
        config,
        settings=None,
        mock=False,
        client=client,
    )
    assert "BERT" in text


def test_generate_text_with_llm_mock_shortcut() -> None:
    from ner_gold_generator.labeling import LlmGoldSettings

    config = parse_synthetic_config_dict({})
    plan = [PlannedEntity("GPT-3", "model")]
    text = generate_text_with_llm(plan, config, settings=LlmGoldSettings(), mock=True)
    assert "GPT-3" in text


def test_llm_engine_with_settings_mock_llm() -> None:
    from ner_gold_generator.labeling import LlmGoldSettings

    config = parse_synthetic_config_dict(
        {
            "num_documents": 1,
            "engine": "llm",
            "llm": make_llm_config(),
            "entities_per_document": {"distribution": "fixed", "fixed": 2},
        }
    )
    examples = build_examples_synthetic(
        config,
        settings=LlmGoldSettings(),
        mock=True,
    )
    assert examples.examples[0]["entities"]


def test_lexical_collision_suffix() -> None:
    config = parse_synthetic_config_dict({"lexicon": {"model": []}})
    used = {"NeuralNet-X", "NeuralNet-X-1", "NeuralNet-X-2"}
    surface = generate_surface_for_label("model", random.Random(0), config, used=used)
    assert surface == "NeuralNet-X-3"


def test_arxiv_and_currency_variants() -> None:
    config = parse_synthetic_config_dict({})
    rng = random.Random(12345)
    seen_arxiv: set[str] = set()
    for _ in range(30):
        seen_arxiv.add(generate_arxiv_id_surface(rng, config))
    assert len(seen_arxiv) > 1
    rng2 = random.Random(54321)
    for _ in range(30):
        generate_currency_surface(rng2, config)


def test_build_user_prompt_and_meta_json() -> None:
    from ner_gold_generator.synthetic_compose import (
        build_synthetic_user_prompt,
        config_summary_json,
    )

    plan = [PlannedEntity("X", "model")]
    cfg = parse_synthetic_config_dict(
        {
            "llm": make_llm_config(),
            "text_length": {"min_chars": 1, "max_chars": 9},
        }
    )
    prompt = build_synthetic_user_prompt(
        plan,
        config=cfg,
        document_type="news",
        style_description="News article style",
    )
    assert "news" in prompt and "X" in prompt

    transcript_prompt = build_synthetic_user_prompt(
        plan,
        config=cfg,
        document_type="transcript",
        style_description="Spoken transcript",
    )
    assert "Transcript-specific rules" in transcript_prompt
    assert "Host:" in transcript_prompt

    spaced_transcript = build_synthetic_user_prompt(
        plan,
        config=cfg,
        document_type=" Transcript ",
        style_description="Spoken transcript",
    )
    assert "Transcript-specific rules" in spaced_transcript
    meta_cfg = parse_synthetic_config_dict({})
    assert '"seed": 42' in config_summary_json(meta_cfg)


def test_uniform_entity_count() -> None:
    config = parse_synthetic_config_dict(
        {"entities_per_document": {"distribution": "uniform", "min": 4, "max": 4}}
    )
    assert sample_entity_count(random.Random(0), config) == 4
