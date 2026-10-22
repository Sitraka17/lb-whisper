#!/usr/bin/env python3
"""Turn a folder of audio + transcripts (e.g. the ZLS fine-tuning set from data.public.lu)
into the JSONL manifest that prepare_data.py --extra expects.

    python make_manifest.py --root data/zls --out data/zls/manifest.jsonl --source zls --split train

Transcripts are looked up, in this order, for every audio file <name>.<ext>:
  1. <name>.txt next to the audio,
  2. a row of any .tsv / .csv in --root whose first column (or a column named path/file/audio)
     matches the file name and whose text column is named sentence/text/transcript,
  3. a .json / .jsonl in --root with {"audio"/"path"/"file": ..., "text"/"sentence": ...}.
Files without a transcript are listed on stderr and skipped. Check the dataset licence on the
data.public.lu page before training on it; write it in the model card.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
PATH_KEYS = ("path", "file", "audio", "filename", "clip")
TEXT_KEYS = ("sentence", "text", "transcript", "transcription")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    lowered = {k.lower(): v for k, v in row.items() if k}
    for key in keys:
        if key in lowered and lowered[key]:
            return str(lowered[key]).strip()
    return None


def load_tables(root: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for path in list(root.rglob("*.tsv")) + list(root.rglob("*.csv")):
        delim = "\t" if path.suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delim, quoting=csv.QUOTE_NONE if delim == "\t" else csv.QUOTE_MINIMAL)
            for row in reader:
                name = _pick(row, PATH_KEYS) or (list(row.values())[0] if row else None)
                text = _pick(row, TEXT_KEYS)
                if name and text:
                    table[Path(name).stem] = text
    for path in list(root.rglob("*.json")) + list(root.rglob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        raw = path.read_text(encoding="utf-8")
        items = [json.loads(l) for l in raw.splitlines() if l.strip()] if path.suffix == ".jsonl" else json.loads(raw)
        if isinstance(items, dict):
            items = items.get("data") or items.get("items") or list(items.values())
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                name, text = _pick(item, PATH_KEYS), _pick(item, TEXT_KEYS)
                if name and text:
                    table[Path(name).stem] = text
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="zls")
    ap.add_argument("--split", default="train", help="train, validation or test (extra data goes to train by default)")
    args = ap.parse_args()

    table = load_tables(args.root)
    written, missing = 0, []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for audio in sorted(p for p in args.root.rglob("*") if p.suffix.lower() in AUDIO_EXT):
            txt = audio.with_suffix(".txt")
            text = txt.read_text(encoding="utf-8").strip() if txt.exists() else table.get(audio.stem)
            if not text:
                missing.append(audio.name)
                continue
            rel = audio.relative_to(args.out.parent) if args.out.parent in audio.parents else audio.resolve()
            fh.write(json.dumps({"audio": str(rel), "text": text, "split": args.split, "source": args.source}, ensure_ascii=False) + "\n")
            written += 1
    print(f"[manifest] {written} clips → {args.out}")
    if missing:
        print(f"[manifest] {len(missing)} audio files without transcript, e.g. {missing[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
