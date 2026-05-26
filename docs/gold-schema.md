# Gold JSONL schema

Output is **newline-delimited JSON**: one gold example per line. Compatible with [ner-detector](https://github.com/marfago-labs/ner-detector) `load_gold_jsonl` (extra fields are ignored).

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable example id (e.g. `arxiv-1706.03762`, `youtube-abc123`) |
| `text` | string | Full text that was annotated (abstract, transcript, article body, …) |
| `entities` | array | List of gold spans (may be empty) |

Each entity:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Verbatim substring of `text` |
| `label` | string | Unified label (see below) |
| `start` | int | Character offset start (0-based, inclusive) |
| `end` | int | Character offset end (exclusive) |

Invariant: `text[start:end] == entity.text` for every entity.

## Example

```json
{
  "id": "arxiv-1706.03762",
  "text": "We propose a new simple network architecture, the Transformer...",
  "entities": [
    {
      "text": "network architecture, the Transformer",
      "label": "model",
      "start": 263,
      "end": 300
    },
    {
      "text": "28.4 BLEU",
      "label": "metric",
      "start": 582,
      "end": 591
    }
  ]
}
```

## Optional fields

| Field | When present |
|-------|----------------|
| `source` | Corpus type: `arxiv`, `youtube`, `folder`, `jsonl`, `synthetic` |
| `gold_source` | `llm` or `synthetic` when not manual |
| `llm_meta` | LLM run statistics (samples, votes, model id) |
| `synthetic_meta` | Seed, engine, document index (synthetic mode) |

## Allowed labels

Used in manual files and LLM prompts; normalized to lowercase with underscores.

| Label | Typical use |
|-------|-------------|
| `person` | People |
| `organization` | Companies, labs (OpenAI, Google, …) |
| `location` | Places |
| `model` | Model names (Transformer, BERT, GPT-3, …) |
| `dataset` | Datasets (ImageNet, GLUE, …) |
| `benchmark` | Benchmarks / tasks (WMT 2014, SQuAD, …) |
| `metric` | Metrics (BLEU, F1, accuracy phrases) |
| `method` | Methods, training procedures |
| `number` | Numeric results (28.4, …) |
| `percentage` | Percentages (12.5%, …) |
| `currency` | Money amounts ($1.2M, 50 EUR, …) |
| `date` | Calendar dates in varied formats |
| `year` | Years |
| `arxiv_id` | arXiv identifiers in text |
| `acronym` | Acronyms |
| `miscellaneous` | Other |

Aliases accepted when normalizing (e.g. `org` → `organization`, `misc` → `miscellaneous`).

## Scoring in ner-detector

Gold labels are mapped through `benchmark/config/label_maps.yaml` in ner-detector. ML backends (BERT-NER, GLiNER) use different label sets; compare backends only on labels both sides support, or extend the map.

Domain labels (`model`, `dataset`, …) are intended for **pattern** or **custom** NER on scientific text—not for off-the-shelf CoNLL BERT without label alignment.
