#!/usr/bin/env python3
"""Build the reference file of a bench run from a folder of audio and transcripts.

    python bench/make_refs.py --audio audio/ --out refs.jsonl
    python bench/make_refs.py --audio audio/ --normalised normalised/ --out refs.jsonl

Layout expected: audio/<genre>/<name>.wav next to its transcript, in the first of these forms found
- <name>.csv   header "start_seconds,end_seconds,text" (the layout of the ZLS open data),
- <name>.json  {"fragments": [{"begin": "0.000", "end": "14.700", "lines": ["..."]}]},
- <name>.txt   plain text, no timestamps.
The audio path written in each row is relative to the output file.
The normalised reference, when there is one, has the same name with the suffix "_normalised"
(or no suffix), anywhere under --normalised.

--meta is a CSV file with one row per clip: "id" (as below), then "device" (what recorded it),
"codec" (what it was re-encoded by, empty if nothing) and "consent" (the parts of CONSENT.md
the speaker ticked, separated by spaces: "A1 A1b A2"). These fields are copied into each row;
score.py uses "consent" to score apart the clips that may not be uploaded. With --meta, a clip
without a row is an error.

A transcript without a single word is refused: its error rate would be 0 % whatever a system
wrote.

The id of an utterance is its path under --audio without the extension. run_local.py and
import_hypotheses.py use the same rule, which is what links a hypothesis to its reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from lb_text import normalize  # noqa: E402

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
META_FIELDS = ("device", "codec", "consent")


def list_audio(root: Path) -> list:
    """Audio files under root, sorted, without the "._" files macOS leaves on external disks."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("._"))


def audio_id(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def read_segments(stem: Path) -> tuple:
    """(text, segments or None) from <stem>.csv, .json or .txt; (None, None) if there is none."""
    if stem.with_suffix(".csv").exists():
        with stem.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as fh:
            segments = [{"start": float(r["start_seconds"]), "end": float(r["end_seconds"]), "text": r["text"].strip()}
                        for r in csv.DictReader(fh)]
    elif stem.with_suffix(".json").exists():
        data = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        segments = [{"start": float(f["begin"]), "end": float(f["end"]), "text": " ".join(f["lines"]).strip()}
                    for f in data["fragments"]]
    elif stem.with_suffix(".txt").exists():
        return stem.with_suffix(".txt").read_text(encoding="utf-8-sig").strip(), None
    else:
        return None, None
    return " ".join(s["text"] for s in segments), segments


def duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def recording_of(name: str) -> str:
    """Name of the original recording: the file name without a trailing part number."""
    return re.sub(r"__part\d+$", "", name)


def read_meta(path: Path) -> dict:
    """{id: {"device", "codec", "consent": [parts]}} from the per-clip CSV described above."""
    meta = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in ("id",) + META_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"[make_refs] {path}: missing column(s) {missing}")
        for r in reader:
            uid = (r["id"] or "").strip()
            if uid in meta:
                raise SystemExit(f"[make_refs] {path}: two rows for {uid}")
            meta[uid] = {
                "device": (r["device"] or "").strip() or None,
                "codec": (r["codec"] or "").strip() or None,
                "consent": (r["consent"] or "").replace(",", " ").replace(";", " ").split(),
            }
    return meta


def build_rows(audio: Path, out: Path, normalised_dir: Path | None = None, genre: str | None = None,
               licence: str | None = None, meta: dict | None = None) -> list:
    """One reference row per audio file under `audio`; audio paths relative to the folder of `out`."""
    normalised = {}
    if normalised_dir:
        for p in sorted(normalised_dir.rglob("*")):
            if p.suffix in (".csv", ".json", ".txt") and not p.name.startswith("._"):
                normalised.setdefault(re.sub(r"_normalised$", "", p.stem), p.with_suffix(""))

    rows = []
    for path in list_audio(audio):
        text, segments = read_segments(path.with_suffix(""))
        if text is None:
            raise SystemExit(f"[make_refs] no transcript next to {path}")
        if not normalize(text).split():
            raise SystemExit(f"[make_refs] the transcript of {path} has no word")
        uid = audio_id(path, audio)
        row = {
            "id": uid,
            "audio": os.path.relpath(path.resolve(), out.resolve().parent),
            "text": text,
            "text_normalised": None,
            "genre": genre or path.parent.name,
            "recording": recording_of(path.stem),
            "duration": round(duration_seconds(path), 3),
            "segments": segments,
        }
        if licence:
            row["licence"] = licence
        if meta is not None:
            if uid not in meta:
                raise SystemExit(f"[make_refs] no row for {uid} in the --meta file")
            row.update(meta[uid])
        if path.stem in normalised:
            row["text_normalised"], row["segments_normalised"] = read_segments(normalised[path.stem])
            if not normalize(row["text_normalised"] or "").split():
                raise SystemExit(f"[make_refs] the normalised transcript of {path} has no word")
        rows.append(row)
    if meta is not None:
        unused = sorted(set(meta) - {r["id"] for r in rows})
        if unused:
            raise SystemExit(f"[make_refs] --meta rows without audio: {unused[:5]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--normalised", type=Path, default=None, help="folder with the normalised transcripts")
    ap.add_argument("--genre", default=None, help="one genre for every file; default: the name of the parent folder")
    ap.add_argument("--licence", default=None)
    ap.add_argument("--meta", type=Path, default=None, help="CSV with id, device, codec, consent for each clip")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = build_rows(args.audio, args.out, args.normalised, args.genre, args.licence,
                      read_meta(args.meta) if args.meta else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    minutes = sum(r["duration"] for r in rows) / 60
    print(f"[make_refs] {len(rows)} references, {minutes:.1f} min, genres: {sorted({r['genre'] for r in rows})} -> {args.out}")


if __name__ == "__main__":
    main()
