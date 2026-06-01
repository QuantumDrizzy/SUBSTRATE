"""
substrate.pipeline.reporter — Result rendering
==============================================
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.lab import SubstrateResult


def render(result: "SubstrateResult", fmt: str = "markdown") -> str:
    if fmt == "json":
        return result.to_json()
    elif fmt == "html":
        return _to_html(result)
    else:
        return _to_markdown(result)


def _to_markdown(result: "SubstrateResult") -> str:
    ts = result.meta.get("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    lines = [
        f"# SUBSTRATE Report — `{result.instrument}` / `{result.task}`",
        f"*Generated: {ts}*",
        "",
    ]

    # If data already contains markdown, embed it directly
    if isinstance(result.data, dict) and "markdown" in result.data:
        lines.append(result.data["markdown"])
    elif isinstance(result.data, dict) and "text" in result.data:
        lines.append(result.data["text"])
    else:
        lines.append("## Result")
        lines.append(f"```\n{str(result.data)[:3000]}\n```")

    # Metadata section
    lines += ["", "## Provenance", "```json"]
    safe_meta = {k: v for k, v in result.meta.items() if k != "markdown"}
    lines.append(json.dumps(safe_meta, indent=2, default=str))
    lines.append("```")

    warnings = result.meta.get("warnings", [])
    if warnings:
        lines += ["", "## ⚠️ Warnings"]
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _to_html(result: "SubstrateResult") -> str:
    md = _to_markdown(result)
    # Minimal HTML wrapper — no external dependencies
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SUBSTRATE — {result.instrument}/{result.task}</title>
<style>
  body {{ font-family: 'JetBrains Mono', monospace; background:#0d0d0d; color:#e0e0e0;
         max-width:900px; margin:40px auto; padding:0 20px; }}
  h1 {{ color:#c0392b; }} h2 {{ color:#7f8c8d; }}
  pre {{ background:#1a1a1a; padding:16px; border-radius:6px; overflow-x:auto; }}
  code {{ color:#e74c3c; }}
</style></head><body>
<pre>{md}</pre>
</body></html>"""
