# Python API reference

Interactive documentation is generated from docstrings via **pydoc**. After `uv sync`, run:

```bash
python -m pydoc ner_gold_generator
```

## Package overview

| Module | Purpose |
|--------|---------|
| `ner_gold_generator` | Public exports: `LlmGoldSettings`, `label_text_with_llm_consensus`, `mock_extract_entities` |
| `ner_gold_generator.build_gold` | `build-gold` CLI: `main`, `build_examples_manual`, `build_examples_llm` |
| `ner_gold_generator.build_arxiv` | `build-arxiv-gold` wrapper around `build_gold.main` |
| `ner_gold_generator.sources` | `SourceDocument`, `load_documents`, `SUPPORTED_SOURCES` |
| `ner_gold_generator.manual` | Manual span files and built-in arXiv table |
| `ner_gold_generator.manual_entities` | Curated arXiv benchmark spans (`PAPER_ORDER`, `PAPER_ENTITIES`) |
| `ner_gold_generator.labeling` | LLM multi-pass labeling and settings |
| `ner_gold_generator.llm_client` | `OpenRouterChatClient`, `MockChatClient`, `ChatClient` protocol |
| `ner_gold_generator.consensus` | `consensus_entities` majority vote |
| `ner_gold_generator.validate` | Label normalization and substring anchoring |
| `ner_gold_generator.mock` | `mock_extract_entities` for tests |
| `ner_gold_generator.types` | `ProposedEntity`, `ValidatedEntity` |
| `ner_gold_generator.progress` | stderr progress helper |
| `ner_gold_generator.env` | `load_project_env` |

## Common calls

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

Full signatures and parameter docs: `python -m pydoc ner_gold_generator.labeling`.
