"""Curated gold spans for the default arXiv benchmark papers.

Each surface string in :data:`PAPER_ENTITIES` must occur **exactly once** in the
corresponding abstract so :func:`validate.index_unique_span` can assign offsets.

Attributes:
    PAPER_ORDER: arXiv ids (no prefix) in benchmark emission order.
    PAPER_ENTITIES: Map from arXiv id to ``(surface, label)`` pairs.
"""

from __future__ import annotations

PAPER_ORDER: list[str] = [
    "1706.03762",
    "1512.03385",
    "2010.11929",
    "2103.00020",
    "2005.14165",
    "1312.6114",
    "1810.04805",
    "1409.0473",
    "1606.05328",
    "1905.00537",
]

# (surface text, label) — must appear exactly once in abstract for auto-indexing
PAPER_ENTITIES: dict[str, list[tuple[str, str]]] = {
    "1706.03762": [
        ("network architecture, the Transformer", "model"),
        ("28.4 BLEU", "metric"),
        ("WMT 2014 English-to-German", "benchmark"),
        ("WMT 2014 English-to-French", "benchmark"),
        ("41.8", "number"),
    ],
    "1512.03385": [
        ("On the ImageNet dataset", "dataset"),
        ("VGG nets", "model"),
        ("ILSVRC 2015", "benchmark"),
        ("CIFAR-10", "dataset"),
        ("COCO detection", "benchmark"),
        ("3.57%", "number"),
    ],
    "2010.11929": [
        ("Vision Transformer (ViT)", "model"),
        ("CNNs", "model"),
        ("ImageNet", "dataset"),
        ("CIFAR-100", "dataset"),
        ("VTAB", "benchmark"),
        ("a pure transformer", "method"),
    ],
    "2103.00020": [
        ("CLIP", "model"),
        ("ResNet-50", "model"),
        ("ImageNet", "dataset"),
        ("400 million", "number"),
        ("OpenAI", "organization"),
    ],
    "2005.14165": [
        ("GPT-3 achieves", "model"),
        ("175 billion", "number"),
        ("autoregressive language model", "method"),
    ],
    "1312.6114": [
        ("stochastic variational inference", "method"),
        ("stochastic gradient", "method"),
    ],
    "1810.04805": [
        ("Bidirectional Encoder Representations", "model"),
        ("GLUE", "dataset"),
        ("MultiNLI", "dataset"),
        ("SQuAD v1.1", "dataset"),
        ("80.5%", "number"),
    ],
    "1409.0473": [
        ("English-to-French", "benchmark"),
        ("statistical machine translation", "method"),
        ("family of encoder-decoders", "model"),
    ],
    "1606.05328": [
        ("PixelCNN architecture", "model"),
        ("ImageNet database", "dataset"),
        ("PixelRNN", "model"),
    ],
    "1905.00537": [
        ("GLUE benchmark", "dataset"),
        ("present SuperGLUE, a new benchmark", "benchmark"),
    ],
}
