"""NER benchmark gold JSONL generator.

Build character-offset gold datasets from arXiv abstracts, YouTube transcripts,
document folders, or JSONL corpora. Supports manual curation and multi-sample
LLM labeling with majority consensus.

Public API
----------
LlmGoldSettings, label_text_with_llm_consensus, mock_extract_entities

CLI entry points (see :mod:`ner_gold_generator.build_gold`):

- ``build-gold`` — unified corpus builder
- ``build-arxiv-gold`` — arXiv shortcut (:mod:`ner_gold_generator.build_arxiv`)
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
