# Development

## Setup

```bash
uv sync --extra dev --extra llm
uv run pre-commit install
```

On Windows, `pre-commit` is only on PATH inside the project venv — always prefix with `uv run` unless you activated `.venv` first.

## Tests and coverage

```bash
uv run pytest tests/ -q --cov=ner_gold_generator --cov-report=term-missing
```

Tests run in **random order** via [`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly) (enabled in dev dependencies). On a git checkout, the seed is derived from the current commit so CI and local runs are stable for the same revision but shuffled vs. plain pytest file order.

If a run fails, pytest prints the seed — replay with:

```bash
uv run pytest tests/ --randomly-seed=1234567890
```

For a new shuffle on every run (local debugging):

```bash
uv run pytest tests/ --randomly-seed=last
```

Policy:

- **95%** minimum line coverage (`pyproject.toml` + CI `--cov-fail-under=95`)
- Unit tests for sources, consensus, validation, LLM mock client, CLI
- Live OpenRouter tests are **not** run in CI (`pytest.importorskip("openai")` only where needed)

## Lint and types

```bash
uv run ruff check ner_gold_generator tests
uv run ruff format ner_gold_generator tests
uv run mypy ner_gold_generator
```

Or run all hooks:

```bash
uv run pre-commit run --all-files
```

`ruff` hook uses `--fix` on commit; CI runs check + format `--check` without modifying files.

## Secret scanning (gitleaks)

Hardcoded API keys and tokens are blocked by **[Gitleaks](https://github.com/gitleaks/gitleaks)**:

| Where | When |
|-------|------|
| **CI** job `secrets` | Every push/PR (open-source `gitleaks detect` CLI, full git history) |
| **pre-commit** hook `gitleaks` | Staged files before each commit |

Config: [`.gitleaks.toml`](../.gitleaks.toml) (extends default rules; allowlists `tests/`, `.env.example`, doc placeholders).

Run manually:

```bash
# After installing gitleaks (or: pre-commit run gitleaks --all-files)
gitleaks detect --source . --verbose --config .gitleaks.toml
```

Keep real keys only in local `.env` (gitignored).

**GitHub org vs license:** `marfago-labs` may be a one-person GitHub organization. CI uses the **free** Gitleaks binary, not `gitleaks-action@v2` (that action can require a paid `GITLEAKS_LICENSE` for org-owned repos). Pre-commit uses the same OSS tool. No extra secrets are needed for secret scanning.

## CI

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

| Job | Command |
|-----|---------|
| lint | `ruff check` + `ruff format --check` |
| typecheck | `mypy ner_gold_generator` |
| secrets | Gitleaks scan (`.gitleaks.toml`) |
| test | `pytest` with coverage ≥ 95% |

`lint` and `typecheck` use `setup-uv` with `save-cache: false` so parallel jobs do not race on the same GitHub Actions cache key (only `test` writes the cache).

Triggers: push and pull request to `main` / `master`.

## Python API docs (pydoc)

All public modules use **Google-style** docstrings (`Args`, `Returns`, `Raises`).
Browse interactively after install:

```bash
uv sync --extra dev
python -m pydoc ner_gold_generator
python -m pydoc ner_gold_generator.labeling.label_text_with_llm_consensus
python -m pydoc ner_gold_generator.build_gold.main
```

Generate static HTML into `docs/_pydoc/` (gitignored if you add it to `.gitignore`):

```bash
python -m pydoc -w ner_gold_generator
# opens ner_gold_generator.html in the current directory
```

See also [docs/api.md](api.md) for a module index.

## Project layout

```
ner-gold-generator/
  ner_gold_generator/   # library + CLI (documented with pydoc)
  tests/
  docs/                 # this documentation
  .github/workflows/
  pyproject.toml
  uv.lock
```

## Adding a new input source

1. Extend `SUPPORTED_SOURCES` and `load_documents` in `sources.py`
2. Document format in [input-sources.md](input-sources.md)
3. Add tests in `tests/test_sources.py`
4. If manual mode should be supported with builtins, extend `manual.py`; otherwise document `--manual` only

## Adding labels

1. Add to `_ALLOWED_LABELS` in `validate.py`
2. Add to `_DEFAULT_LABELS` / prompt in `labeling.py`
3. Document in [gold-schema.md](gold-schema.md)
4. Update ner-detector `label_maps.yaml` if benchmarks should score the new label
