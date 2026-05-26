"""Print OpenRouter models with zero prompt/completion pricing."""

from __future__ import annotations

import json
import urllib.request

URL = "https://openrouter.ai/api/v1/models"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        data = json.load(resp)
    rows: list[tuple[str, int | None, str]] = []
    for model in data.get("data", []):
        pricing = model.get("pricing") or {}
        if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
            continue
        rows.append(
            (
                str(model["id"]),
                model.get("context_length"),
                str(model.get("name", "")),
            )
        )
    rows.sort(key=lambda r: (":free" not in r[0], r[0]))
    print(f"{'Model ID':<58} {'Ctx':>8}  Name")
    print("-" * 90)
    for model_id, ctx, name in rows:
        print(f"{model_id:<58} {str(ctx or '?'):>8}  {name}")
    free_tagged = [r for r in rows if ":free" in r[0]]
    print(f"\nTotal: {len(rows)} ($0 pricing), {len(free_tagged)} with :free suffix")
    print(f"Source: {URL}")


if __name__ == "__main__":
    main()
