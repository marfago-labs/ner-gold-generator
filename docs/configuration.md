# Configuration

Configuration is via **environment variables** and an optional **`.env`** file in the repo root (loaded at CLI startup via `python-dotenv`).

Copy `.env.example` to `.env` and edit locally. Do not commit secrets — CI and pre-commit run [Gitleaks](development.md#secret-scanning-gitleaks) to block accidental key commits.

## OpenRouter (LLM mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(empty)* | API key; required for live LLM mode |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `LLM_GOLD_MODEL` | *(empty)* | OpenRouter model id for `build-gold --mode llm` (required for live LLM) |
| `LLM_GOLD_SAMPLES` | `3` | LLM passes per document |
| `LLM_GOLD_MIN_VOTES` | `0` | Min agreeing passes (`0` = auto majority: ⌊N/2⌋+1) |
| `LLM_GOLD_TEMPERATURE` | `0.4` | Sampling temperature per pass |
| `MOCK_LLM` | `false` | If true, never call the API (same as `--mock`) |
| `LLM_GOLD_SYSTEM_PROMPT_FILE` | *(empty → `configs/prompts/labeling_system_prompt.txt`)* | Path to labeling system prompt |
| `LLM_GOLD_USER_PROMPT_FILE` | *(empty → `configs/prompts/labeling_user_prompt.txt`)* | Path to labeling user prompt template |

CLI flags `--samples`, `--min-votes`, and `--mock` override behavior for that run without changing env defaults for samples/votes/mock.

When `LLM_GOLD_*_PROMPT_FILE` is unset, prompts load from **`configs/prompts/labeling_system_prompt.txt`** and **`configs/prompts/labeling_user_prompt.txt`** (repo root relative). Override with absolute paths or paths relative to the repo root.

Synthetic `engine: llm` uses a separate trio under the same directory (`synthetic_llm_*.txt`), referenced from each `configs/synthetic_*.yaml` via `llm.prompt_file`, `user_prompt_file`, and `feedback_prompt_file` (paths relative to that YAML file). See [synthetic.md](synthetic.md).

## Synthetic / build logging

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_GOLD_LOG_FILE` | `gold.log` | File for LLM prompt/response and synthetic failure logs |
| `NER_GOLD_LOG_LEVEL` | `INFO` | `DEBUG` for full Agno/OpenRouter prompt bodies; `WARNING` for failures only |
| `NER_GOLD_LOG_STDERR` | `0` | Set `1` to echo log lines to stderr |

## Input path overrides

| Variable | Applies to | Description |
|----------|------------|-------------|
| `ARXIV_RAW_DIR` | `--source arxiv` | Directory of `{id}.json` papers |
| `YOUTUBE_INPUT_DIR` | `--source youtube` | Directory of `{video_id}.json` |
| `{SOURCE}_INPUT_DIR` | Any source | e.g. `FOLDER_INPUT_DIR` (uncommon; prefer `--input`) |

If unset, defaults assume:

```
../text-compressor/data/raw/arxiv
../text-compressor/data/raw/youtube
```

(relative to the `ner-gold-generator` repo root).

## Output path

There is no env var for output; use `--output`. Default output path is set in code to:

`../ner-detector/benchmark/datasets/arxiv_gold.jsonl`

Only when using `build-gold` without `--output`. `build-arxiv-gold` always applies the same default.

## Example `.env`

```bash
OPENROUTER_API_KEY=sk-or-...
LLM_GOLD_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_GOLD_SAMPLES=5
LLM_GOLD_MIN_VOTES=3
LLM_GOLD_TEMPERATURE=0.3

ARXIV_RAW_DIR=../text-compressor/data/raw/arxiv
YOUTUBE_INPUT_DIR=../text-compressor/data/raw/youtube
```
