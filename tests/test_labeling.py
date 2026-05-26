"""Tests for LLM gold labeling with multi-sample consensus."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ner_gold_generator.consensus import consensus_entities
from ner_gold_generator.labeling import (
    LlmGoldSettings,
    _parse_run,
    build_chat_client,
    default_min_votes,
    extract_entities_once,
    label_text_with_llm_consensus,
    resolve_openrouter_model_id,
)
from ner_gold_generator.llm_client import (
    MockChatClient,
    OpenRouterChatClient,
    parse_entities_json,
    strip_json_fence,
)
from ner_gold_generator.mock import mock_extract_entities
from ner_gold_generator.types import ProposedEntity
from ner_gold_generator.validate import (
    index_unique_span,
    normalize_entity_label,
    validate_proposed_entities,
)


def test_normalize_entity_label_aliases() -> None:
    assert normalize_entity_label("ORG") == "organization"
    assert normalize_entity_label("misc") == "miscellaneous"
    assert normalize_entity_label("not_a_label") is None


def test_index_unique_span() -> None:
    assert index_unique_span("Alice at OpenAI", "OpenAI") == (9, 15)
    assert index_unique_span("a a", "a") is None


def test_validate_drops_hallucinated_span() -> None:
    text = "GPT-3 on ImageNet."
    proposed = [
        ProposedEntity(text="GPT-3", label="model"),
        ProposedEntity(text="FakeSet", label="dataset"),
    ]
    validated = validate_proposed_entities(text, proposed)
    assert len(validated) == 1
    assert validated[0].text == "GPT-3"


def test_consensus_majority() -> None:
    runs = [
        [
            ProposedEntity("GPT-3", "model"),
            ProposedEntity("ImageNet", "dataset"),
        ],
        [
            ProposedEntity("GPT-3", "model"),
            ProposedEntity("BLEU", "metric"),
        ],
        [
            ProposedEntity("GPT-3", "model"),
            ProposedEntity("ImageNet", "dataset"),
        ],
    ]
    agreed = consensus_entities(runs, min_votes=2)
    keys = {(e.text, e.label) for e in agreed}
    assert ("GPT-3", "model") in keys
    assert ("ImageNet", "dataset") in keys
    assert ("BLEU", "metric") not in keys


def test_consensus_requires_min_votes() -> None:
    runs = [
        [ProposedEntity("CLIP", "model")],
        [ProposedEntity("BERT", "model")],
        [ProposedEntity("CLIP", "model")],
    ]
    assert len(consensus_entities(runs, min_votes=3)) == 0
    assert len(consensus_entities(runs, min_votes=2)) == 1


def test_default_min_votes_majority() -> None:
    assert default_min_votes(3, 0) == 2
    assert default_min_votes(5, 0) == 3
    assert default_min_votes(3, 2) == 2


def test_label_text_mock_consensus() -> None:
    text = "OpenAI released GPT-3 trained on ImageNet in 2020."
    client = MockChatClient()
    result = label_text_with_llm_consensus(
        text,
        settings=LlmGoldSettings(mock_llm=True),
        num_samples=3,
        min_votes=2,
        mock=True,
        client=client,
    )
    assert result.sample_runs == 3
    assert result.min_votes == 2
    assert result.entities
    for ent in result.entities:
        assert text[ent.start : ent.end] == ent.text


def test_mock_extract_entities() -> None:
    ents = mock_extract_entities("GPT-3 and GLUE benchmark")
    labels = {e.label for e in ents}
    assert "model" in labels
    assert "dataset" in labels


def test_label_text_invalid_samples() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        label_text_with_llm_consensus("x", num_samples=0, mock=True)


def test_strip_json_fence() -> None:
    raw = '```json\n{"entities": []}\n```'
    assert strip_json_fence(raw) == '{"entities": []}'


def test_parse_entities_json() -> None:
    parsed = parse_entities_json('{"entities": [{"text": "A", "label": "model"}]}')
    assert parsed == [{"text": "A", "label": "model"}]


def test_parse_entities_json_errors() -> None:
    with pytest.raises(ValueError, match="object"):
        parse_entities_json("[]")
    with pytest.raises(ValueError, match="entities"):
        parse_entities_json("{}")


def test_consensus_skips_empty_and_dup_in_run() -> None:
    runs = [
        [
            ProposedEntity("", "model"),
            ProposedEntity("CLIP", "model"),
            ProposedEntity("CLIP", "model"),
        ],
        [ProposedEntity("CLIP", "model")],
    ]
    assert len(consensus_entities(runs, min_votes=2)) == 1


def test_consensus_invalid_min_votes() -> None:
    with pytest.raises(ValueError, match="min_votes"):
        consensus_entities([], min_votes=0)


def test_validate_duplicate_span_label() -> None:
    text = "GPT-3"
    proposed = [
        ProposedEntity("GPT-3", "model"),
        ProposedEntity("GPT-3", "model"),
    ]
    assert len(validate_proposed_entities(text, proposed)) == 1


def test_resolve_openrouter_model_id_requires_value() -> None:
    with pytest.raises(ValueError, match="LLM model id is required"):
        resolve_openrouter_model_id(LlmGoldSettings())


def test_build_chat_client_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Env vars override LlmGoldSettings() fields; CI may set empty OPENROUTER_API_KEY.
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("LLM_GOLD_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setenv("MOCK_LLM", "false")
    settings = LlmGoldSettings()
    client = build_chat_client(settings)
    assert isinstance(client, OpenRouterChatClient)


def test_openrouter_client_complete_json(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_GOLD_LOG_STDERR", "1")
    monkeypatch.setenv("NER_GOLD_LOG_FILE", "")
    import ner_gold_generator.llm_client as llm_client_mod

    llm_client_mod._LOGGER.handlers.clear()
    llm_client_mod._LOGGER = llm_client_mod.get_logger(llm_client_mod.__name__)
    pytest.importorskip("openai")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"entities": []}'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    with patch("openai.OpenAI", return_value=mock_client):
        client = OpenRouterChatClient(api_key="k", model_id="m")
        out = client.complete_json(
            system="s",
            user="u",
            temperature=0.1,
            call_label="attempt 2/3",
        )
    assert out == {"entities": []}
    err = capsys.readouterr().err
    assert "--- LLM prompt (m, temperature=0.1, attempt 2/3) ---" in err
    assert "SYSTEM:\ns\nUSER:\nu\n" in err
    assert "--- end LLM prompt ---" in err
    assert "--- LLM response (m, attempt 2/3) ---" in err
    assert '{"entities": []}' in err
    assert "--- end LLM response ---" in err


def test_mock_chat_client_custom_responses() -> None:
    client = MockChatClient(responses=[{"entities": [{"text": "X", "label": "model"}]}])
    out = client.complete_json(system="s", user="u", temperature=0.0)
    assert out["entities"][0]["text"] == "X"


def test_extract_entities_once_filters_bad_labels() -> None:
    client = MockChatClient(
        responses=[
            {"entities": [{"text": "GPT-3", "label": "model"}, {"text": "Z", "label": "bogus"}]}
        ]
    )
    ents = extract_entities_once(client, "GPT-3 only", temperature=0.0)
    assert len(ents) == 1
    assert ents[0].label == "model"


def test_index_unique_span_empty_surface() -> None:
    assert index_unique_span("text", "") is None


def test_validate_skips_whitespace_surface() -> None:
    proposed = [ProposedEntity("   ", "model")]
    assert validate_proposed_entities("GPT-3", proposed) == []


def test_mock_extract_skips_duplicate_year_match() -> None:
    ents = mock_extract_entities("Results in 2014 were better than 2014 baselines.")
    years = [e for e in ents if e.label == "year"]
    assert len(years) == 0


def test_mock_extract_adds_unique_year() -> None:
    ents = mock_extract_entities("Published in 2014.")
    assert any(e.text == "2014" and e.label == "year" for e in ents)


def test_consensus_empty_runs() -> None:
    assert consensus_entities([], min_votes=1) == []


def test_parse_entities_json_skips_invalid_items() -> None:
    raw = '{"entities": [{"text": "A", "label": "model"}, "bad", {}]}'
    assert parse_entities_json(raw) == [{"text": "A", "label": "model"}]


def test_mock_chat_client_without_text_marker() -> None:
    client = MockChatClient()
    out = client.complete_json(system="s", user="GPT-3 on ImageNet", temperature=0.0)
    assert out["entities"]


def test_mock_chat_client_reset() -> None:
    client = MockChatClient(
        responses=[
            {"entities": [{"text": "A", "label": "model"}]},
            {"entities": [{"text": "B", "label": "model"}]},
        ]
    )
    client.complete_json(system="s", user="u", temperature=0.0)
    client.reset()
    client._call_index = 0
    out = client.complete_json(system="s", user="u", temperature=0.0)
    assert out["entities"][0]["text"] == "A"


def test_build_chat_client_uses_injected_mock() -> None:
    custom = MockChatClient(responses=[{"entities": []}])
    client = build_chat_client(LlmGoldSettings(), mock_client=custom)
    assert client is custom


def test_parse_run_skips_malformed_entries() -> None:
    assert _parse_run({"entities": "not-a-list"}) == []
    assert _parse_run({"entities": [1, {"text": 1, "label": "model"}]}) == []


def test_label_text_on_sample_callback() -> None:
    seen: list[tuple[int, int]] = []

    def on_sample(current: int, total: int) -> None:
        seen.append((current, total))

    label_text_with_llm_consensus(
        "GPT-3",
        num_samples=2,
        min_votes=1,
        mock=True,
        on_sample=on_sample,
    )
    assert seen == [(1, 2), (2, 2)]
