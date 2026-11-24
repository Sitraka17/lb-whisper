#!/usr/bin/env python3
"""Turn the export of an online service into the JSONL that score.py reads.

    python bench/import_hypotheses.py --system luxasr --test-date 2026-12-01 --pass 1 \
        --input exports/luxasr/ --set plan=free --set "options=default" --out hyps/luxasr.pass1.jsonl

--input is a file or a folder of .txt, .srt or .vtt files exported by hand from the service
(LuxASR, Sproochmaschinn, ElevenLabs, Google and so on). The name of each export, without the
extension, must be the id of the audio file it transcribes (its path under the audio folder,
as in refs.jsonl); for a folder, sub-folders are part of the id.

The date of the test is mandatory, and so is writing down the settings (--set key=value, as
many as needed): an online service changes without notice, so a transcript without a date and
its settings cannot be compared with anything. With --pass the system is named
"<system>@pass<N>", which keeps the two passes of the protocol apart in the score table.

The text is taken as exported. Nothing is corrected, and the only thing removed is markup that
is not speech: cue numbers, timestamps, VTT tags and headers, and, if asked with
--strip-regex, a pattern such as speaker labels (applied to each line of text, so "^" means
the start of a line in every format).

A subtitle whose text contains an empty line is not valid SRT, but exports do it. The lines
after the gap are attached to the subtitle before them and a warning is printed, so that no
text is lost without a trace.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_TAG = re.compile(r"<[^>]+>")


def _seconds(h, m, s, frac) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)


def _clean(line: str, strip_regex: str | None) -> str:
    line = _TAG.sub("", line).strip()
    if strip_regex:
        line = re.sub(strip_regex, " ", line)
    return re.sub(r"\s+", " ", line).strip()


def parse_cues(content: str, strip_regex: str | None = None, warnings: list | None = None) -> list:
    """Segments of an SRT or WebVTT file: [{"start", "end", "text"}].

    Handles both time formats (comma or dot, hours optional), cue numbers or identifiers, VTT
    headers, NOTE / STYLE / REGION blocks and inline tags such as <v Speaker> or <i>. A block
    without a time line that follows a cue is the rest of that cue (its text had an empty
    line): it is attached to it, and a message is added to `warnings`."""
    segments = []
    blocks = re.split(r"\n\s*\n", content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        for k, line in enumerate(lines):
            match = _CUE.search(line)
            if match:
                g = match.groups()
                text = " ".join(t for t in (_clean(t, strip_regex) for t in lines[k + 1:]) if t)
                segments.append({"start": round(_seconds(*g[:4]), 3), "end": round(_seconds(*g[4:]), 3), "text": text})
                break
        else:
            text = " ".join(t for t in (_clean(t, strip_regex) for t in lines) if t)
            if not text:
                continue
            if segments:
                segments[-1]["text"] = (segments[-1]["text"] + " " + text).strip()
                where = f"attached to the subtitle at {segments[-1]['start']} s"
            else:
                where = "dropped, there is no subtitle before it"
            if warnings is not None:
                warnings.append(f"text without a time line ({text[:40]!r}): {where}")
    return [seg for seg in segments if seg["text"]]


def parse_export(path: Path, strip_regex: str | None = None, warnings: list | None = None) -> tuple:
    """(text, segments or None, format) for one exported file."""
    content = path.read_text(encoding="utf-8-sig")
    fmt = path.suffix.lower().lstrip(".")
    if fmt in ("srt", "vtt"):
        segments = parse_cues(content, strip_regex, warnings)
        return " ".join(seg["text"] for seg in segments), segments, fmt
    if fmt != "txt":
        raise SystemExit(f"[import] {path}: expected .txt, .srt or .vtt")
    return " ".join(t for t in (_clean(t, strip_regex) for t in content.splitlines()) if t), None, fmt


def parse_settings(items: list) -> dict:
    """["plan=free", "options=default"] -> {"plan": "free", "options": "default"}."""
    settings = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"[import] --set expects KEY=VALUE, got {item!r} (for example --set plan=free)")
        settings[key.strip()] = value
    return settings


def build_row(path: Path, uid: str, system: str, test_date: str, settings: dict, pass_number: int | None = None,
              url: str | None = None, strip_regex: str | None = None) -> dict:
    if pass_number not in (None, 1, 2):
        raise SystemExit(f"[import] the protocol has two passes: --pass is 1 or 2, got {pass_number}")
    warnings = []
    text, segments, fmt = parse_export(path, strip_regex, warnings)
    for message in warnings:
        print(f"[import] WARNING {path.name}: {message}", file=sys.stderr)
    return {
        "id": uid,
        "system": f"{system}@pass{pass_number}" if pass_number else system,
        "text": text,
        "segments": segments,
        "online": {
            "service": system,
            "url": url,
            "test_date": test_date,
            "pass": pass_number,
            "settings": settings,
            "export_format": fmt,
            "export_file": path.name,
            "strip_regex": strip_regex,
            "warnings": warnings,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True, help="short name of the service, e.g. luxasr")
    ap.add_argument("--input", type=Path, required=True, help="one export, or a folder of exports")
    ap.add_argument("--id", default=None, help="id of the audio file, for a single export whose name is not the id")
    ap.add_argument("--test-date", required=True, help="day the audio was submitted to the service, YYYY-MM-DD")
    ap.add_argument("--pass", dest="pass_number", type=int, default=None, choices=[1, 2],
                    help="1 or 2: the protocol tests online services twice")
    ap.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE",
                    help="a setting of the service as used for the test (plan, language option, model name shown, ...)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--strip-regex", default=None, help="pattern removed from the text, e.g. '^Speaker \\d+: ' for speaker labels")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        datetime.date.fromisoformat(args.test_date)
    except ValueError:
        raise SystemExit(f"[import] --test-date must be YYYY-MM-DD, got {args.test_date}")
    if not args.settings:
        raise SystemExit("[import] give at least one --set key=value (plan, options, model name shown by the service)")
    settings = parse_settings(args.settings)

    if args.input.is_dir():
        exports = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".txt", ".srt", ".vtt") and not p.name.startswith("._"))
        pairs = [(p.relative_to(args.input).with_suffix("").as_posix(), p) for p in exports]
    else:
        pairs = [(args.id or args.input.stem, args.input)]
    ids = [uid for uid, _ in pairs]
    duplicated = sorted({uid for uid in ids if ids.count(uid) > 1})
    if duplicated:
        raise SystemExit(f"[import] several exports for the same id: {duplicated}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for uid, path in pairs:
            row = build_row(path, uid, args.system, args.test_date, settings, args.pass_number, args.url, args.strip_regex)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[import] {len(pairs)} transcript(s) of {args.system} dated {args.test_date} -> {args.out}")


if __name__ == "__main__":
    main()
