# Synthetic gold generation

Generate NER benchmark gold JSONL without external corpora. The pipeline is **entity-first** (similar in spirit to [ProgGen](https://arxiv.org/abs/2403.11103)): plan spans and labels, embed them verbatim in text, then export offsets from the plan—not from free-form LLM NER.

## Quick start

**Procedural (no API):**

```bash
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output output/synthetic_gold.jsonl
```

**LLM prose + deterministic entity plan (recommended for realistic text):**

```bash
# Requires OPENROUTER_API_KEY in .env and: uv sync --extra llm
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_llm.yaml \
  --output output/synthetic_gold_llm.jsonl
```

Flow:

1. **Plan** entities from `seed` + `label_weights` + generators (deterministic surfaces).
2. **Prompt** the LLM: system rules (JSON + verbatim constraints) and a per-document user message listing required spans (domain/length from config).
3. **Validate** the response: each planned surface must appear **exactly once**, and text length must fall within `text_length.min_chars`–`max_chars`.
4. **Retry** with separate budgets: `llm.provider_max_attempts` (API/JSON errors) and `llm.validation_max_attempts` (missing/duplicate surfaces, length). Legacy `llm.max_attempts` sets both when the split knobs are omitted.
5. **Skip** the document if it still fails after all attempts; continue with the rest.
6. **Gold** offsets come from the plan, not from LLM-extracted NER.

Each output row includes `synthetic_meta.entity_usage` with `planned`, `used_exactly_once`, `missing`, `duplicate_in_text`, and per-surface `occurrence_counts`.

Use `--mock` only for CI/tests; it skips the API and uses template assembly even when `engine: llm`.

`engine: llm` requires `OPENROUTER_API_KEY` and a valid `llm.model` — missing config or API errors exit with a clear message (no silent fallback to templates).

## Config directory layout

Paths in synthetic YAML are resolved **relative to the config file** (usually `configs/`):

```
configs/
  synthetic_reference.yaml      # annotated template (all options)
  synthetic_smoke.yaml
  synthetic_llm.yaml
  synthetic_mixed.yaml
  prompts/                          # LLM templates (not inlined in Python)
    synthetic_llm_system_prompt.txt
    synthetic_llm_user_prompt.txt
    synthetic_llm_feedback_prompt.txt
    labeling_system_prompt.txt      # used by build-gold --mode llm (not synthetic)
    labeling_user_prompt.txt
  document_profiles/                # per-genre YAML overrides
    scientific_abstract.yaml
    news_article.yaml
    blog_post.yaml
    transcript.yaml
```

At load time, `load_synthetic_config` reads `llm.prompt_file`, `user_prompt_file`, and `feedback_prompt_file` from disk and stores the text in `llm.system_prompt`, `user_prompt`, and `feedback_prompt`.

## Config file (YAML)

Synthetic batch specs live under `configs/*.yaml`. JSON is still accepted by `load_synthetic_config` for backward compatibility.

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
| `llm.max_attempts` | Legacy retry knob (fallback for both retry classes) |
| `llm.provider_max_attempts` | Retries for provider/API failures (timeouts, 429, transient errors) |
| `llm.validation_max_attempts` | Retries for non-compliant outputs (missing/duplicate planned surfaces, empty text) |
| `llm.max_restarts` | After validation turns are exhausted, replan entities from scratch (new surfaces, same profile hyperparameters) and start a fresh Agno session; default `1` |
| `llm.document_log_dir` | Directory for per-document LLM traces (`logs/synthetic-00000001-00016.log`, etc.); path relative to the synthetic config file. CLI: `--document-log-dir logs` |
| `llm.temperature` | Sampling temperature for LLM engine |
| `llm.timeout_seconds` | Per-request HTTP timeout (default `120`; lower if a run seems stuck) |
| `llm.request_delay_seconds` | Pause before retries (helps free-tier rate limits) |
| *(automatic)* | On HTTP 429, backoff uses OpenRouter `retry_after_seconds`, `Retry-After`, or `X-RateLimit-Reset` (up to 1h wait) before retrying |
| `llm.parallelism` | Number of concurrent synthetic LLM document calls (default `3`; use `1` for sequential, no thread pool) |
| `llm.agent_mode` | `per_doc` (default): Agno session per document with in-session feedback turns; `stateless`: one-shot OpenRouter JSON loop per attempt |
| `llm.agno_db_path` | SQLite path for Agno session history (`:memory:` default; one DB per document build) |
| `llm.prompt_file` | System prompt under `configs/prompts/` (relative to the synthetic config file or absolute) |
| `llm.user_prompt_file` | User prompt template for the first turn (`{document_type}`, `{style_description}`, `{entity_list}`, `{min_chars}`, `{max_chars}`, …) |
| `llm.feedback_prompt_file` | Validation feedback template for retry turns (`{issues}`, `{min_chars}`, `{max_chars}`) |
| `llm.system_prompt` | Inline system prompt (only when `prompt_file` is not set; tests may use inline strings) |
| `document_type` | Single genre id for all documents (e.g. `scientific_abstract`, `news_article`) |
| `document_types` | Weighted mix, e.g. `[{"type": "news_article", "weight": 0.5}, {"type": "blog_post", "weight": 0.5}]` |
| `document_profiles_dir` | Directory of YAML/JSON profiles (default `document_profiles` next to the config) |

`engine: llm` requires all three prompt files under `configs/prompts/` (via `prompt_file`, `user_prompt_file`, `feedback_prompt_file`) or inline equivalents in tests.

### Defaults and sparse configs

Omitted top-level fields use **code defaults** from `SyntheticConfig` (`ner_gold_generator.synthetic_config`), for example:

| Field | Default when omitted |
|-------|----------------------|
| `seed` | `42` |
| `engine` | `procedural` |
| `text_length` | `min_chars=200`, `max_chars=800` |
| `label_weights` | Built-in label mix (normalized at sampling time) |
| `lexicon` / `templates` | Built-in scientific-style lists and sentence templates |
| `llm.temperature` | `0.4` |
| `llm.agent_mode` | `per_doc` |
| `llm.parallelism` | `3` |
| `document_profiles_dir` | `document_profiles` (next to the config file) |

A minimal LLM config such as `configs/synthetic_mixed.yaml` only sets `seed`, `num_documents`, `engine`, `document_types`, `entities_per_document`, and `llm.*` prompt paths. Everything else above still applies until a **document profile** overrides it (see below).

### Document profiles (multi-genre)

Profiles under `configs/document_profiles/` override style, templates, lexicon, and optional `text_length` per genre:

| Profile id | Typical use |
|------------|-------------|
| `scientific_abstract` | Paper-style abstracts |
| `news_article` | News wire tone |
| `blog_post` | Informal blog |
| `transcript` | Spoken/interview style (`Host:`/`Guest:` turn labels; planned names are full names used once, not repeated as speaker prefixes) |

Example configs: `configs/synthetic_news.yaml`, `synthetic_blog.yaml`, `synthetic_transcript.yaml`, `synthetic_mixed.yaml`. Each output row sets `synthetic_meta.document_type` to the profile used for that document.

**Merge rules** (`merge_config_for_profile`): for each document, the sampled profile is merged into the base YAML config:

| Setting | Precedence |
|---------|------------|
| `text_length` | Profile value if present, else base YAML / defaults |
| `label_weights` | Profile if non-empty, else base |
| `lexicon` | Base lexicon, then profile keys merged on top |
| `templates` | Profile templates if any, else base; procedural engine uses these |
| `style_description` | From profile; passed to the LLM user prompt as `{style_description}` |

For **`engine: llm`**, profile `templates` are **not** sent to the model; the LLM writes prose from `style_description` and the shared prompts in `configs/prompts/`. Length and entity sampling still follow the merged config (e.g. `news_article` sets 300–1200 chars while the global default is 200–800).

**Example — `synthetic_mixed.yaml`:** 30 documents with a 40/30/30 mix of `news_article`, `blog_post`, and `transcript`; five entities per document (`entities_per_document.fixed: 5`); Agno `per_doc` sessions; shared system/user/feedback prompts. No `text_length` in the YAML — each document uses the profile’s bounds when the profile defines `text_length` (news: 300–1200), otherwise 200–800.

### Logging (`gold.log`)

With live LLM calls, prompts and responses are logged to **`gold.log`** (structured, grep-friendly):

- Stateless: `--- LLM prompt (engine=stateless, doc=..., attempt X/Y) ---` / matching response blocks.
- Agno `per_doc`: `synthetic session start`, `Agno prompt (turn=N/M, initial|feedback)`, `Agno response`, `synthetic session success`.
- Failures: `synthetic failure` / `synthetic failure (final_validation)` with planned entities and last generated text.

Useful greps: `Agno prompt`, `synthetic session`, `synthetic failure`, `prompt_kind=feedback`.

| Variable | Default | Effect |
|----------|---------|--------|
| `NER_GOLD_LOG_FILE` | `gold.log` | Log file path |
| `NER_GOLD_LOG_LEVEL` | `INFO` | Set `DEBUG` for full prompt bodies; `WARNING` hides DEBUG prompts but keeps failure lines |
| `NER_GOLD_LOG_STDERR` | `0` | Set `1` to mirror logs to stderr |

**Stuck on `LLM 2/3`?** The bar only updates between steps. During `calling OpenRouter…` the client is waiting on the API (reasoning models can take minutes). Prefer instruct/chat `:free` models over `*:reasoning:free`, or use `engine: procedural` for offline runs.

### Dataset quality (lexical diversity)

After a successful `build-gold` run, the **statistics** block on stderr includes **distinct-1** and **distinct-2** scores (per-document mean/min/max and corpus-level). These measure lexical variety in accepted documents only; they are **informational** and do not fail the build. Lower corpus distinct-n is normal when many documents share function words; compare runs with the same length and genre mix.

Concurrency safety notes:

- `llm.parallelism = 1` runs sequentially (no tasks created).
- Progress updates and output row aggregation are done on the main thread only.
- On interrupt, pending asyncio tasks are cancelled.
- Per-document LLM failures after all retries are skipped; other documents still complete.

Example: `configs/synthetic_smoke.yaml`.

## Labels

Structured generators: `date`, `year`, `number`, `percentage`, `currency`, `arxiv_id`, `metric`.

Lexicon sampling: `model`, `dataset`, `benchmark`, `organization`, `method`, `person`, `location`, `acronym`, `miscellaneous`.

Output rows include `gold_source: "synthetic"` and `synthetic_meta` (seed, document index, planned count).

## Python API

```python
from pathlib import Path
from ner_gold_generator.synthetic import build_examples_from_config_path

examples = build_examples_from_config_path(
    Path("configs/synthetic_smoke.yaml"),
    mock=True,
)
```
