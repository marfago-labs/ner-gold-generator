# CLI reference

## Commands

| Entry point | Implementation |
|-------------|----------------|
| `uv run build-gold` | `ner_gold_generator.build_gold:main` |
| `uv run build-arxiv-gold` | Wrapper: forces `--source arxiv`, default output to ner-detector `arxiv_gold.jsonl`; maps `--arxiv-dir` → `--input` |

```bash
uv run build-gold --help
```

## `build-gold` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `arxiv` | `arxiv`, `youtube`, `folder`, `jsonl` |
| `--input` | *(source default)* | Input directory or JSONL file path |
| `--output` | `../ner-detector/benchmark/datasets/arxiv_gold.jsonl` | Output JSONL path |
| `--mode` | `manual` | `manual` or `llm` |
| `--manual` | — | JSON/YAML manual annotations (required for manual non-arxiv) |
| `--text-field` | — | JSON/JSONL field name for body text |
| `--glob` | `*` | Glob under `--input` for `folder` source |
| `--samples` | env / 3 | LLM passes per document |
| `--min-votes` | env / majority | Minimum agreeing LLM passes |
| `--mock` | false | Use deterministic mock LLM (no API) |
| `--limit` | — | Process only first N documents |
| `--quiet` | false | Suppress progress on stderr |

Progress and per-document status go to **stderr**; the final `Wrote …` line goes to **stdout**.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Usage error (e.g. manual mode without `--manual` on non-arxiv) |
| Non-zero | Uncaught exception (missing input, invalid span, API error, …) |

## Common recipes

```bash
# Default arXiv manual gold for ner-detector
uv run build-arxiv-gold

# arXiv LLM with stricter consensus
uv run build-gold --source arxiv --mode llm --samples 5 --min-votes 3 \
  --output output/arxiv_gold_llm.jsonl

# YouTube, mock, one video
uv run build-gold --source youtube --mode llm --mock --limit 1 \
  --output output/yt.jsonl

# Markdown folder only
uv run build-gold --source folder --input ./content --glob "*.md" \
  --manual annotations.json --mode manual \
  --output output/md_gold.jsonl

# Import JSONL corpus + LLM
uv run build-gold --source jsonl --input corpus.jsonl --mode llm \
  --output output/corpus_gold.jsonl
```

## Python API

For programmatic labeling (without writing JSONL):

```python
from ner_gold_generator import LlmGoldSettings, label_text_with_llm_consensus

result = label_text_with_llm_consensus(
    "OpenAI released GPT-3 on ImageNet.",
    settings=LlmGoldSettings(mock_llm=True),
    num_samples=3,
    min_votes=2,
    mock=True,
)
for ent in result.entities:
    print(ent.text, ent.label, ent.start, ent.end)
```

See `ner_gold_generator.labeling` for `entities_to_gold_dict`, `extract_entities_once`, etc.
