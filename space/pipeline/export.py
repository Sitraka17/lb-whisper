"""Render transcript blocks as Markdown minutes, SRT subtitles and JSON. Pure Python."""

from __future__ import annotations

import json
from datetime import datetime, timezone


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        seconds, ms = int(seconds) + 1, 0
    return f"{hms(seconds)},{ms:03d}"


def to_markdown(blocks: list[dict], summary_md: str | None, meta: dict | None = None, title: str = "Meeting notes") -> str:
    meta = meta or {}
    lines = [f"# {title}", ""]
    if meta:
        facts = []
        if meta.get("duration"):
            facts.append(f"duration {hms(meta['duration'])}")
        if meta.get("language"):
            facts.append(f"language `{meta['language']}`")
        if meta.get("model"):
            facts.append(f"model `{meta['model']}`")
        facts.append("processed locally, no cloud API")
        lines += ["_" + " · ".join(facts) + "_", ""]
    if summary_md:
        lines += [summary_md.strip(), "", "---", ""]
    lines += ["## Transcript", ""]
    for b in blocks:
        lines.append(f"**[{hms(b['start'])}] {b['speaker']}:** {b['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_srt(blocks: list[dict]) -> str:
    out = []
    for i, b in enumerate(blocks, 1):
        out += [str(i), f"{srt_time(b['start'])} --> {srt_time(b['end'])}", f"{b['speaker']}: {b['text']}", ""]
    return "\n".join(out).rstrip() + "\n"


def to_json(blocks: list[dict], summary: dict | None, meta: dict | None) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta or {},
        "summary": summary or {},
        "blocks": blocks,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
