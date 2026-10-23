"""Attach speakers to transcript segments. Pure Python, unit-tested.

A transcript segment gets the speaker whose turns overlap it the most; segments with no
overlapping turn inherit the nearest turn in time. Consecutive segments of the same speaker
are then joined into one block so the output reads like minutes, not subtitles.
"""

from __future__ import annotations


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments: list[dict], turns: list[dict], default: str = "SPEAKER") -> list[dict]:
    """segments: [{"start","end","text"}], turns: [{"start","end","speaker"}] → segments + "speaker"."""
    out = []
    for seg in segments:
        best, best_overlap = None, 0.0
        per_speaker: dict[str, float] = {}
        for turn in turns:
            ov = _overlap(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > 0:
                per_speaker[turn["speaker"]] = per_speaker.get(turn["speaker"], 0.0) + ov
        if per_speaker:
            best = max(per_speaker, key=per_speaker.get)
        elif turns:
            mid = (seg["start"] + seg["end"]) / 2
            nearest = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)))
            best = nearest["speaker"]
        out.append({**seg, "speaker": best or default})
    return out


def group_turns(segments: list[dict], max_gap: float = 2.0) -> list[dict]:
    """Join consecutive segments of the same speaker (gap below max_gap seconds) into blocks."""
    blocks: list[dict] = []
    for seg in segments:
        if blocks and blocks[-1]["speaker"] == seg["speaker"] and seg["start"] - blocks[-1]["end"] <= max_gap:
            blocks[-1]["end"] = seg["end"]
            blocks[-1]["text"] = (blocks[-1]["text"] + " " + seg["text"]).strip()
        else:
            blocks.append({"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"], "text": seg["text"].strip()})
    return blocks


def relabel_speakers(blocks: list[dict], names: dict[str, str] | None = None) -> list[dict]:
    """SPEAKER_00 → "Speaker 1" (or a user-provided name map), in order of first appearance."""
    names = dict(names or {})
    order: list[str] = []
    for b in blocks:
        if b["speaker"] not in order:
            order.append(b["speaker"])
    for i, label in enumerate(order, 1):
        names.setdefault(label, f"Speaker {i}")
    return [{**b, "speaker": names[b["speaker"]]} for b in blocks]
