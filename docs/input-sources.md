# Input sources

The `--source` flag selects how documents are loaded. All sources produce the same internal shape: a **document id**, **body text**, and a **source tag** (`arxiv`, `youtube`, `folder`, `jsonl`).

## Summary

| `--source` | `--input` | Default input (sibling monorepo) |
|------------|-----------|--------------------------------|
| `arxiv` | Directory of `*.json` | `../text-compressor/data/raw/arxiv/` |
| `youtube` | Directory of `*.json` | `../text-compressor/data/raw/youtube/` |
| `folder` | Directory of files | *(none — `--input` required)* |
| `jsonl` | Single `.jsonl` file | *(none — `--input` required)* |

Set per-source overrides: `ARXIV_RAW_DIR`, `YOUTUBE_INPUT_DIR`, or `{SOURCE}_INPUT_DIR` (see [Configuration](configuration.md)).

---

## `arxiv` — text-compressor paper JSON

One file per paper: `{arxiv_id}.json`.

```json
{
  "kind": "paper",
  "arxiv_id": "1706.03762",
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models..."
}
```

**Text used for labeling:** `abstract` (override with `--text-field`).

**Document id:** `arxiv-{arxiv_id}` (e.g. `arxiv-1706.03762`).

**Manual mode:** Built-in curated spans in `configs/arxiv_benchmark.yaml` for ten benchmark papers (used when `--manual` is omitted and source is `arxiv`).

---

## `youtube` — text-compressor video JSON

One file per video: `{video_id}.json`.

```json
{
  "kind": "video",
  "video_id": "hTWKbfoikeg",
  "title": "...",
  "description": "...",
  "transcript": "..."
}
```

**Text used for labeling:**

1. `transcript` if non-empty
2. Else `title` + blank line + `description`

Override with `--text-field transcript` (or another field).

**Document id:** `youtube-{video_id}`.

**Manual mode:** Requires `--manual` (no built-in YouTube gold table).

---

## `folder` — blogs, docs, mixed files

Scans `--input` with `--glob` (default `*`).

| Extension | Behavior |
|-----------|----------|
| `.txt`, `.md`, `.markdown` | Entire file is the document body |
| `.json` | Object with a text field (see below) |

**Document id:** `file-{relative_path}` using forward slashes (e.g. `file-posts/intro.md`).

**JSON text fields** (first match wins unless `--text-field` is set):

`text`, `abstract`, `transcript`, `content`, `body`, `markdown`, `post`, `article`

**Manual mode:** Requires `--manual` keyed by document `id`.

**Tips:**

- Use `--glob "*.md"` to limit to Markdown only
- Empty `.txt` files are skipped
- JSON without a recognized text field raises an error at load time

---

## `jsonl` — one record per line

Each line is a JSON object with at least:

- **`text`** — document body (override with `--text-field`)
- **`id`** — document id (fallback: `arxiv_id`, `video_id`, `slug`, `name`, `doc_id`, or `line-{n}`)

Example corpus:

```json
{"id": "blog-001", "text": "OpenAI released GPT-4 in 2023."}
{"id": "blog-002", "text": "Results on GLUE and SuperGLUE benchmarks."}
```

**Manual mode:** Requires `--manual` unless ids match built-in arXiv keys (unusual for this source).

---

## Ordering and limits

- Files are processed in **sorted path order** (folder/arXiv/YouTube) or **file line order** (jsonl)
- `--limit N` keeps only the first **N** documents after loading
- For arXiv **manual** mode without `--manual`, only papers listed in `configs/arxiv_benchmark.yaml` `paper_order` are emitted (built-in benchmark set)
