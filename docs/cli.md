# CLI reference

## Commands

| Entry point | Implementation |
|-------------|----------------|
| `uv run build-gold` | `ner_gold_generator.build_gold:main` |
| `uv run build-arxiv-gold` | Wrapper: forces `--source arxiv`, default output to `../ner-dataset/datasets/arxiv_gold.jsonl`; maps `--arxiv-dir` → `--input` |

```bash
uv run build-gold --help
```

For **LLM** modes (`--mode llm` or synthetic with `engine: llm` in YAML), install optional dependencies first:

```bash
uv sync --extra llm
```

If live OpenRouter calls fail with `[Errno 2] No such file or directory` or `cannot import name 'from_json' from 'jiter'`, reinstall the native helper:

```bash
uv pip install --force-reinstall jiter
```

## `build-gold` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `arxiv` | `arxiv`, `youtube`, `folder`, `jsonl`, `synthetic` |
| `--synthetic-config` | — | YAML spec (required for `synthetic`; JSON still accepted; see [synthetic.md](synthetic.md)) |
| `--input` | *(source default)* | Input directory or JSONL file path |
| `--output` | `../ner-dataset/datasets/arxiv_gold.jsonl` | Output JSONL path |
| `--mode` | `manual` | `manual` or `llm` |
| `--manual` | — | JSON/YAML manual annotations (required for manual non-arxiv) |
| `--text-field` | — | JSON/JSONL field name for body text |
| `--glob` | `*` | Glob under `--input` for `folder` source |
| `--samples` | env / 3 | LLM passes per document |
| `--min-votes` | env / majority | Minimum agreeing LLM passes |
| `--mock` | false | Use deterministic mock LLM (no API) |
| `--limit` | — | Process only first N documents |
| `--quiet` | false | Suppress progress on stderr |
| `--verbose` | false | Print detailed step-by-step logs on stderr |
| `--document-log-dir` | — | Per-document LLM logs under `DIR/<document_id>.log` (see also `llm.document_log_dir` in synthetic YAML) |

Progress goes to **stderr** via [Rich](https://github.com/Textualize/rich) on interactive terminals (stable bar + skip lines that do not overwrite each other). Non-TTY runs (pipes, tests) use plain ASCII lines. Synthetic LLM runs also show attempt `1/3` while waiting on OpenRouter. Use `--quiet` to disable stderr progress/logging. The final `Wrote …` line goes to **stdout**; a **statistics** block (counts, label histogram, **lexical diversity** distinct-1/2 quality numbers, synthetic entity-plan totals, elapsed time and throughput) is printed to **stderr** after a successful run (and after Ctrl+C when partial output was written).

For live OpenRouter requests, each LLM call is logged to `gold.log` by default (including retries), with structured fields (timestamp, level, logger, message):

- `--- LLM prompt (model, temperature, attempt X/Y) ---`
- `--- LLM response (model, attempt X/Y) ---`

Logger controls (optional):

- `NER_GOLD_LOG_FILE` (default: `gold.log`) — where structured debug logs are written.
- `NER_GOLD_LOG_LEVEL` (default: `INFO`) — `DEBUG` logs full prompts/responses; `WARNING` keeps failures only.
- `NER_GOLD_LOG_STDERR` (default: `0`) — set to `1` to mirror debug logs to stderr.

Synthetic `engine: llm` with `agent_mode: per_doc` (default) uses **Agno** per-document sessions; install with `uv sync --extra llm` (includes `agno`, `sqlalchemy` for session storage, and the OpenAI client). See [synthetic.md](synthetic.md) for prompt files and document profiles.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Usage error (e.g. manual mode without `--manual` on non-arxiv) |
| `1` | Runtime error (missing API key, unknown model, validation failure, …) |
| `130` | Cancelled by user (Ctrl+C); partial JSONL is written when at least one document finished |
| Non-zero | Other uncaught exceptions |

Press **Ctrl+C** during a build to stop gracefully: in-flight synthetic LLM tasks are cancelled, completed rows are written to `--output` in document order, and the process exits with code `130`.

## Common recipes

```bash
# Default arXiv manual gold → ner-dataset (ner-detector loads it automatically)
uv run build-arxiv-gold

# arXiv LLM with stricter consensus
uv run build-gold --source arxiv --mode llm --samples 5 --min-votes 3 \
  --output ../ner-dataset/datasets/arxiv_gold_llm.jsonl

# YouTube, mock, one video
uv run build-gold --source youtube --mode llm --mock --limit 1 \
  --output ../ner-dataset/datasets/yt.jsonl

# Markdown folder only
uv run build-gold --source folder --input ./content --glob "*.md" \
  --manual docs/examples/manual-sample.yaml --mode manual \
  --output ../ner-dataset/datasets/md_gold.jsonl

# Import JSONL corpus + LLM
uv run build-gold --source jsonl --input corpus.jsonl --mode llm \
  --output ../ner-dataset/datasets/corpus_gold.jsonl

# Synthetic procedural gold (no corpus, no API)
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output ../ner-dataset/datasets/synthetic_gold.jsonl
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

Synthetic builds from YAML:

```python
from pathlib import Path
from ner_gold_generator.synthetic import build_examples_from_config_path

result = build_examples_from_config_path(Path("configs/synthetic_smoke.yaml"), mock=True)
```

See [api.md](api.md) for the full module index and pydoc commands.
