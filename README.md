# ner-gold-generator

Build **NER benchmark gold JSONL** from arXiv abstracts, YouTube transcripts, document folders, JSONL corpora, or **synthetic YAML configs** — with **manual** curation or **multi-sample LLM** labeling.

Example configs: `configs/synthetic_*.yaml`, manual annotations: `docs/examples/manual-sample.yaml`.

Output is consumed by [ner-detector](https://github.com/marfago-labs/ner-detector) (`benchmark/datasets/*.jsonl`).

## Quick start

```bash
uv sync --extra dev
uv run build-arxiv-gold    # regenerate arxiv_gold.jsonl (manual, default paths)
```

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

## Quality

```bash
uv sync --extra dev --extra llm
uv run pytest --cov=ner_gold_generator -q
uv run pre-commit run --all-files
```

Coverage gate: **95%**. **Gitleaks** runs in CI and pre-commit to catch committed secrets (see [docs/development.md](docs/development.md#secret-scanning-gitleaks)).

## License

[MIT](LICENSE) — see [LICENSE](LICENSE) for terms.
