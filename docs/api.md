# Python API reference

Interactive documentation is generated from **Google-style** module docstrings (`__docformat__ = "google"`). After install:

```bash
uv sync --extra dev --extra llm   # llm extra for Agno/OpenRouter imports in some modules
python -m pydoc ner_gold_generator
python -m pydoc -b http://localhost:8765/   # optional local browser (Ctrl+C to stop)
```

Generate static HTML (writes `ner_gold_generator.html` in the current directory):

```bash
python -m pydoc -w ner_gold_generator
```

## Package overview

| Module | Purpose |
|--------|---------|
| `ner_gold_generator` | Public exports: `LlmGoldSettings`, `label_text_with_llm_consensus`, `mock_extract_entities` |
| `ner_gold_generator.build_gold` | `build-gold` CLI: `main`, `build_examples_manual`, `build_examples_llm` |
| `ner_gold_generator.build_arxiv` | `build-arxiv-gold` wrapper around `build_gold.main` |
| `ner_gold_generator.sources` | `SourceDocument`, `load_documents`, `SUPPORTED_SOURCES` |
| `ner_gold_generator.manual` | Manual span files (YAML/JSON) and built-in arXiv table |
| `ner_gold_generator.manual_entities` | Curated arXiv benchmark spans (`PAPER_ORDER`, `PAPER_ENTITIES`) |
| `ner_gold_generator.labeling` | LLM multi-pass labeling and settings |
| `ner_gold_generator.llm_client` | `OpenRouterChatClient`, `AsyncOpenRouterChatClient`, `MockChatClient` |
| `ner_gold_generator.consensus` | `consensus_entities` majority vote |
| `ner_gold_generator.validate` | Label normalization and `index_unique_span` anchoring |
| `ner_gold_generator.mock` | `mock_extract_entities` for tests |
| `ner_gold_generator.types` | `ProposedEntity`, `ValidatedEntity` |
| `ner_gold_generator.progress` | stderr / Rich progress helper |
| `ner_gold_generator.env` | `load_project_env` |
| `ner_gold_generator.logger` | `get_logger` — structured logs to `gold.log` |
| `ner_gold_generator.llm_logging` | Synthetic LLM trace logging and per-document log files |
| `ner_gold_generator.synthetic_config` | `SyntheticConfig`, `load_synthetic_config` (YAML primary; JSON accepted) |
| `ner_gold_generator.document_profiles` | Genre profiles and `merge_config_for_profile` |
| `ner_gold_generator.synthetic_generators` | `PlannedEntity`, `plan_entities_for_document` |
| `ner_gold_generator.synthetic_compose` | Text assembly, validation, `generate_text_with_llm_async` |
| `ner_gold_generator.synthetic_agno` | Agno per-document sessions (`agent_mode: per_doc`) |
| `ner_gold_generator.synthetic` | `build_examples_synthetic`, `build_examples_from_config_path` |
| `ner_gold_generator.build_stats` | `summarize_gold_build` (includes lexical diversity) |
| `ner_gold_generator.lexical_diversity` | distinct-1/2 reporting helpers |
| `ner_gold_generator.interrupt` | `BuildInterrupted` for graceful Ctrl+C handling |

## Browse by task

| Task | pydoc entry point |
|------|-------------------|
| Label one document with LLM consensus | `ner_gold_generator.labeling.label_text_with_llm_consensus` |
| Load arXiv / YouTube / folder / JSONL | `ner_gold_generator.sources.load_documents` |
| Build synthetic gold from YAML config | `ner_gold_generator.synthetic.build_examples_from_config_path` |
| Parse synthetic YAML | `ner_gold_generator.synthetic_config.load_synthetic_config` |
| Validate planned entity surfaces in text | `ner_gold_generator.synthetic_compose.collect_validation_issues` |
| CLI entry | `ner_gold_generator.build_gold.main` |

## Corpus labeling

```python
from ner_gold_generator import LlmGoldSettings, label_text_with_llm_consensus

result = label_text_with_llm_consensus(
    "OpenAI released GPT-3 on ImageNet.",
    settings=LlmGoldSettings(),
    mock=True,
)
for ent in result.entities:
    print(ent.text, ent.label, ent.start, ent.end)
```

```python
from pathlib import Path
from ner_gold_generator.sources import load_documents

docs = load_documents("arxiv", Path("../text-compressor/data/raw/arxiv"), limit=1)
```

## Synthetic gold (entity-first)

Configs live under `configs/*.yaml`. Spans come from the **plan**, not LLM NER extraction.

```python
from pathlib import Path
from ner_gold_generator.synthetic import build_examples_from_config_path

result = build_examples_from_config_path(
    Path("configs/synthetic_smoke.yaml"),
    mock=True,
)
for row in result.examples:
    print(row["id"], len(row["entities"]), row["synthetic_meta"]["text_engine"])
for skip in result.skipped:
    print("skipped", skip.id, skip.error[:80])
```

Lower-level pieces:

```python
from pathlib import Path
from ner_gold_generator.synthetic_config import load_synthetic_config
from ner_gold_generator.synthetic import build_examples_synthetic

config = load_synthetic_config(Path("configs/synthetic_mixed.yaml"))
result = build_examples_synthetic(config, mock=True, config_path=Path("configs/synthetic_mixed.yaml"))
```

Per-document LLM traces (when `llm.document_log_dir` or `--document-log-dir` is set):

```python
from ner_gold_generator.llm_logging import configure_document_log_dir

configure_document_log_dir("logs_tmp", relative_to=Path("configs"))
```

Full signatures and parameter docs: `python -m pydoc ner_gold_generator.synthetic.build_examples_from_config_path`.
