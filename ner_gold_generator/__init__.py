"""NER benchmark gold JSONL generator.

Build character-offset gold datasets from arXiv abstracts, YouTube transcripts,
document folders, JSONL corpora, or **synthetic** YAML configs. Supports manual
curation and multi-sample LLM labeling with majority consensus.

Public API (labeling)
---------------------
LlmGoldSettings, label_text_with_llm_consensus, mock_extract_entities

Synthetic API
-------------
Use :mod:`ner_gold_generator.synthetic` for config-driven corpus generation::

    from pathlib import Path
    from ner_gold_generator.synthetic import build_examples_from_config_path

    result = build_examples_from_config_path(Path("configs/synthetic_smoke.yaml"), mock=True)

CLI entry points (see :mod:`ner_gold_generator.build_gold`):

- ``build-gold`` — unified corpus builder
- ``build-arxiv-gold`` — arXiv shortcut (:mod:`ner_gold_generator.build_arxiv`)

Browse module docs: ``python -m pydoc ner_gold_generator`` (see ``docs/api.md``).
"""

from ner_gold_generator.labeling import (
    LlmGoldSettings,
    label_text_with_llm_consensus,
    mock_extract_entities,
)

__all__ = [
    "LlmGoldSettings",
    "label_text_with_llm_consensus",
    "mock_extract_entities",
]

__docformat__ = "google"
