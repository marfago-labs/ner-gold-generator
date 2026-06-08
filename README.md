# ner-gold-generator

Build **NER benchmark gold JSONL** from arXiv abstracts, YouTube transcripts, document folders, JSONL corpora, or **synthetic YAML configs** — with **manual** curation or **multi-sample LLM** labeling.

Example configs: `configs/synthetic_*.yaml` (all options: `configs/synthetic_reference.yaml`), manual annotations: `docs/examples/manual-sample.yaml`.

Output is written to the sibling [ner-dataset](https://github.com/marfago-labs/ner-dataset) repo by default (`datasets/*.jsonl`); [ner-detector](https://github.com/marfago-labs/ner-detector) loads those files for benchmarks (with fallback to shipped `benchmark/datasets/`).

## Technical highlights

- **Entity-first synthetic generation:** plan entities and labels first, then generate text around required surfaces.
- **Deterministic offsets:** compute `start`/`end` from exact unique span anchoring in source text (no trust in model offsets).
- **Consensus labeling for corpora:** multi-sample LLM extraction with majority vote before validation.
- **Self-correcting generation loop:** retry and feedback budgets for provider failures and validation failures.
- **Production safeguards:** strict allow-list labels, anti-hallucination validation, CI checks, and reproducible config-driven runs.

## Quick start

```bash
uv sync --extra dev

# 1. Regenerate arXiv manual gold (the legacy benchmark)
uv run build-arxiv-gold

# 2. Offline smoke gold (3 docs, procedural — no API)
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output /tmp/synthetic_smoke.jsonl

# 3. Full LLM synthetic batch (requires --extra llm + OPENROUTER_API_KEY)
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_mixed.yaml \
  --output ../ner-dataset/datasets/synthetic_mixed.jsonl
```

## API keys and `.env`

Manual arXiv gold and **procedural** synthetic runs need **no API keys**.

| What you run | Keys needed |
|--------------|-------------|
| `build-arxiv-gold`, procedural synthetic (`engine: procedural`) | None |
| Synthetic or corpus **LLM** mode (`engine: llm`, `build-gold --mode llm`) | [`OPENROUTER_API_KEY`](https://openrouter.ai/keys) + `LLM_GOLD_MODEL` (for corpus LLM) + `uv sync --extra llm` |

Local setup:

```bash
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY and LLM_GOLD_MODEL when using live LLM
```

Never commit `.env`. Full variable list: [docs/configuration.md](docs/configuration.md).

## Documentation

Full guides: **[docs/README.md](docs/README.md)**

| Topic | Doc |
|-------|-----|
| Install and first runs | [docs/getting-started.md](docs/getting-started.md) |
| Input formats (arxiv, youtube, folder, jsonl) | [docs/input-sources.md](docs/input-sources.md) |
| Synthetic gold (configurable corpus) | [docs/synthetic.md](docs/synthetic.md) |
| Manual vs LLM labeling | [docs/labeling-modes.md](docs/labeling-modes.md) |
| JSONL schema and labels | [docs/gold-schema.md](docs/gold-schema.md) |
| CLI flags | [docs/cli.md](docs/cli.md) |
| Environment variables | [docs/configuration.md](docs/configuration.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Tests, CI, pre-commit | [docs/development.md](docs/development.md) |
| Python API (pydoc) | [docs/api.md](docs/api.md) |
| Coding agents | [docs/for-agents.md](docs/for-agents.md) |

## Quality

```bash
uv sync --extra dev --extra llm
uv run pytest --cov=ner_gold_generator -q
uv run pre-commit run --all-files
```

Coverage gate: **95%**. **Gitleaks** runs in CI and pre-commit to catch committed secrets (see [docs/development.md](docs/development.md#secret-scanning-gitleaks)).

## License

[MIT](LICENSE) — see [LICENSE](LICENSE) for terms.
