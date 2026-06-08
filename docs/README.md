# ner-gold-generator documentation

Build **NER benchmark gold datasets** (JSONL with character offsets) from arXiv abstracts, YouTube transcripts, document folders, or custom JSONL corpora. Output is scored by [ner-detector](https://github.com/marfago-labs/ner-detector) and similar eval tooling.

## Guides

| Document | Contents |
|----------|----------|
| [Getting started](getting-started.md) | Install, first run, monorepo layout |
| [Input sources](input-sources.md) | `arxiv`, `youtube`, `folder`, `jsonl` formats and defaults |
| [Synthetic generation](synthetic.md) | Config-driven synthetic gold (`--source synthetic`) |
| [Labeling modes](labeling-modes.md) | Manual curation vs multi-sample LLM consensus |
| [Gold schema](gold-schema.md) | JSONL fields, allowed labels, extra metadata |
| [CLI reference](cli.md) | `build-gold`, `build-arxiv-gold`, flags, exit codes |
| [Configuration](configuration.md) | `.env`, API keys (`OPENROUTER_API_KEY`), env vars |
| [Architecture](architecture.md) | Package layout and data flow |
| [Development](development.md) | Tests, coverage, CI, pre-commit |
| [Python API (pydoc)](api.md) | Module index and `python -m pydoc` usage |
| [For coding agents](for-agents.md) | Offline/LLM tasks, schema links |

## Examples

| File | Purpose |
|------|---------|
| [examples/manual-sample.yaml](examples/manual-sample.yaml) | Template for `--manual` annotations |
| [../configs/synthetic_reference.yaml](../configs/synthetic_reference.yaml) | Annotated synthetic config (all options) |

## Quick start

```bash
uv sync --extra dev
uv run build-arxiv-gold          # arXiv manual gold → ../ner-dataset/datasets/ by default
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output ../ner-dataset/datasets/synthetic_gold.jsonl
uv run build-gold --help
```

See the [root README](../README.md) for a one-page overview.
