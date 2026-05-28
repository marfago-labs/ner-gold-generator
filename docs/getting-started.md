# Getting started

## Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (recommended)
- For **LLM mode**: OpenRouter API key (`OPENROUTER_API_KEY`)
- For default arXiv/YouTube paths: sibling clone of [text-compressor](https://github.com/marfago-labs/text-compressor) under `marfago-labs/`

## Install

```bash
cd ner-gold-generator
uv sync --extra dev
```

Optional extras:

| Extra | When |
|-------|------|
| `llm` | Live OpenRouter calls (`openai`, `agno`, …) |
| `dev` | pytest, ruff, mypy, pre-commit |

**PyYAML** is a core dependency (synthetic and manual configs use YAML under `configs/`).

```bash
uv sync --extra dev --extra llm
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` if using LLM mode.

## Monorepo layout (defaults)

When this repo sits next to `text-compressor` and `ner-detector`:

```
marfago-labs/
  text-compressor/data/raw/arxiv/     ← default --input for --source arxiv
  text-compressor/data/raw/youtube/   ← default --input for --source youtube
  ner-detector/benchmark/datasets/    ← default --output for build-arxiv-gold
  ner-gold-generator/
```

Override paths with `--input`, `--output`, or env vars (see [Configuration](configuration.md)).

## First runs

### Regenerate committed arXiv gold (manual, no API)

```bash
uv run build-arxiv-gold
# or: uv run build-gold --source arxiv --mode manual
```

Writes `../ner-detector/benchmark/datasets/arxiv_gold.jsonl` by default.

### LLM gold with mock (no API key)

```bash
uv run build-gold --source arxiv --mode llm --mock --limit 1 \
  --output output/test_arxiv_llm.jsonl
```

### YouTube corpus (text-compressor raw JSON)

```bash
uv run build-gold --source youtube --mode llm --mock \
  --output output/youtube_gold.jsonl
```

### Blog or docs in a folder (manual annotations required)

```bash
uv run build-gold --source folder \
  --input ./my-articles \
  --manual docs/examples/manual-sample.yaml \
  --mode manual \
  --output output/blog_gold.jsonl
```

### Synthetic gold (no external corpus)

```bash
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output output/synthetic_gold.jsonl
```

See [Synthetic generation](synthetic.md) for LLM configs, document profiles, and per-document logs.

## Entry points

| Command | Role |
|---------|------|
| `build-gold` | Unified CLI (`--source`, `--input`, …) |
| `build-arxiv-gold` | Shortcut: `--source arxiv`, default ner-detector output path |

## Next steps

- [Input sources](input-sources.md) — file formats per corpus type
- [Labeling modes](labeling-modes.md) — when to use manual vs LLM
- [CLI reference](cli.md) — full flag list
