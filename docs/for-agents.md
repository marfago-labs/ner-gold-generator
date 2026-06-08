# Guide for coding agents

Upstream **gold-data** repo in the marfago-labs NER stack. Output JSONL is consumed by [ner-dataset](https://github.com/marfago-labs/ner-dataset) and [ner-detector](https://github.com/marfago-labs/ner-detector).

## Package map

See [architecture.md](architecture.md) for module layout and data flows.

## Credentials (`.env`)

Never commit `.env`. Copy `.env.example` → `.env` only for live LLM work.

| Task | Variables |
|------|-----------|
| Procedural synthetic, manual arXiv gold | **None** |
| `engine: llm` or `build-gold --mode llm` | `OPENROUTER_API_KEY`; corpus LLM also needs `LLM_GOLD_MODEL` |

```bash
cp .env.example .env
uv sync --extra dev --extra llm   # llm extra required for API calls
```

Details: [configuration.md](configuration.md).

## Safety

- Never commit `.env` or `OPENROUTER_API_KEY`.
- **Offline gold:** use `engine: procedural` or [configs/synthetic_smoke.yaml](../configs/synthetic_smoke.yaml) (no API).
- **LLM gold:** `uv sync --extra llm` and `OPENROUTER_API_KEY` in `.env`.

## Key contracts

| Artifact | Location |
|----------|----------|
| Gold JSONL schema | [gold-schema.md](gold-schema.md) |
| Annotated synthetic config | [configs/synthetic_reference.yaml](../configs/synthetic_reference.yaml) |
| arXiv manual benchmark | [configs/arxiv_benchmark.yaml](../configs/arxiv_benchmark.yaml) |

Invariant: `text[start:end] == entity.text` for every entity. Offsets are computed in code—never taken from LLM span indices.

## Agent task — procedural smoke gold

```bash
cd ner-gold-generator
uv sync --extra dev
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output ../ner-dataset/datasets/synthetic_smoke.jsonl
```

## Agent task — benchmark (downstream)

After gold exists in `ner-dataset/datasets/`:

```bash
cd ../ner-detector
uv run python scripts/agent_smoke.py
```

See [ner-detector for-agents.md](https://github.com/marfago-labs/ner-detector/blob/master/docs/for-agents.md).

## Sibling docs

- [ner-detector for-agents.md](https://github.com/marfago-labs/ner-detector/blob/master/docs/for-agents.md)
- [ner-dataset for-agents.md](https://github.com/marfago-labs/ner-dataset/blob/master/docs/for-agents.md)
- Repo root [llms.txt](../llms.txt)
