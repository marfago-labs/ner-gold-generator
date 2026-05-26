# Synthetic gold generation

Generate NER benchmark gold JSONL without external corpora. The pipeline is **entity-first** (similar in spirit to [ProgGen](https://arxiv.org/abs/2403.11103)): plan spans and labels, embed them verbatim in text, then export offsets from the plan—not from free-form LLM NER.

## Quick start

**Procedural (no API):**

```bash
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.json \
  --output output/synthetic_gold.jsonl
```

**LLM prose + deterministic entity plan (recommended for realistic text):**

```bash
# Requires OPENROUTER_API_KEY in .env and: uv sync --extra llm
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_llm.json \
  --output output/synthetic_gold_llm.jsonl
```

Flow:

1. **Plan** entities from `seed` + `label_weights` + generators (deterministic surfaces).
2. **Prompt** the LLM: system rules (JSON + verbatim constraints) and a per-document user message listing required spans (domain/length from config).
3. **Validate** the response: each planned surface must appear **exactly once** (counted with a regex on `re.escape(surface)`).
4. **Retry** up to `llm.max_attempts` if any surface is missing or duplicated.
5. **Gold** offsets come from the plan, not from LLM-extracted NER.

Each output row includes `synthetic_meta.entity_usage` with `planned`, `used_exactly_once`, `missing`, `duplicate_in_text`, and per-surface `occurrence_counts`.

Use `--mock` only for CI/tests; it skips the API and uses template assembly even when `engine: llm`.

`engine: llm` requires `OPENROUTER_API_KEY` and a valid `llm.model` — missing config or API errors exit with a clear message (no silent fallback to templates).

## Config file (YAML or JSON)

| Field | Description |
| ------- | ----------- |
| `seed` | Master RNG seed (default `42`) |
| `num_documents` | Rows to emit |
| `engine` | `procedural` (templates, no API) or `llm` (prose glue via OpenRouter) |
| `domain` | Style hint for LLM prompts |
| `text_length.min_chars` / `max_chars` | Target document size |
| `entities_per_document` | `uniform`, `fixed`, or `poisson` with min/max/fixed/lambda |
| `label_weights` | Relative weights per label (normalized); defaults if omitted |
| `lexicon` | Lists of surfaces for lexical labels (`model`, `dataset`, …) |
| `generators` | Per-label format options (`date.formats`, `currency.codes`, …) |
| `templates` | Sentences with `{e0}`, `{e1}`, … placeholders |
| `llm.model` | **Required** when `engine: llm` — OpenRouter model id (use a `:free` variant for zero cost) |
| `llm.max_attempts` / `llm.temperature` | Retry and sampling for LLM engine |
| `llm.timeout_seconds` | Per-request HTTP timeout (default `120`; lower if a run seems stuck) |
| `llm.request_delay_seconds` | Pause before retries (helps free-tier rate limits) |
| `llm.parallelism` | Number of concurrent synthetic LLM document calls (default `3`; use `1` for sequential, no thread pool) |
| `llm.prompt_file` | Path (relative to config file or absolute) to system prompt text file |
| `llm.system_prompt` | Inline system prompt string (used when `prompt_file` is not set) |

With live OpenRouter calls, each request now prints the full prompt and raw model response to **stderr** (`--- LLM prompt (...) ---`, `--- LLM response (...) ---`). This is emitted for every attempt, including retries.

**Stuck on `LLM 2/3`?** The bar only updates between steps. During `calling OpenRouter…` the client is waiting on the API (reasoning models can take minutes). Prefer instruct/chat `:free` models over `*:reasoning:free`, or use `engine: procedural` for offline runs.

Concurrency safety notes:

- `llm.parallelism = 1` runs sequentially (no `ThreadPoolExecutor`).
- Progress updates and output row aggregation are done on the main thread only.
- On interrupt or worker failure, pending futures are cancelled and the executor is shut down cleanly.

Example: `configs/synthetic_smoke.json`.

## Labels

Structured generators: `date`, `year`, `number`, `percentage`, `currency`, `arxiv_id`, `metric`.

Lexicon sampling: `model`, `dataset`, `benchmark`, `organization`, `method`, `person`, `location`, `acronym`, `miscellaneous`.

Output rows include `gold_source: "synthetic"` and `synthetic_meta` (seed, document index, planned count).

## Python API

```python
from pathlib import Path
from ner_gold_generator.synthetic import build_examples_from_config_path

examples = build_examples_from_config_path(
    Path("configs/synthetic_smoke.json"),
    mock=True,
)
```
