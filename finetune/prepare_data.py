#!/usr/bin/env python3
"""Build the training dataset: Google FLEURS Luxembourgish (+ optional extra manifests).

FLEURS (google/fleurs, config lb_lu) is read Luxembourgish, CC-BY-4.0: 2,502 / 408 / 934
utterances (8.46 / 1.12 / 2.73 h) in train / dev / test. It is the only open (commercial use allowed)
Luxembourgish speech set we found on 22/10/2026. Mozilla Common Voice has no Luxembourgish data in
any release from 17.0 to 27.0. Sentences are stored with their HTML entities unescaped.

  --source fleurs   download data/lb_lu (TSVs + audio tarballs) from the Hub, no login, no loading
                    script: the TSVs and tarballs are parsed here.
  --extra FILE      a JSONL manifest {"audio": "path.wav", "text": "...", "split": "train"} for
                    additional data whose licence allows it.

Output: an on-disk DatasetDict (train / validation / test, columns: audio, sentence, source).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import tarfile
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets

FLEURS_REPO = "google/fleurs"
CONFIG = "lb_lu"
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}
FLEURS_COLUMNS = ["id", "path", "sentence", "normalized", "chars", "num_samples", "gender"]


def _read_fleurs_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, fieldnames=FLEURS_COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row for row in reader if row.get("sentence")]


# Review note: FLEURS is *read* speech, people reading Wikipedia-style sentences into a good mic.
# Real Luxembourgish is the opposite: radio, meetings, voice notes, half the sentence in French.
# That gap is the next thing to measure, not to guess. Plan: a small hand-checked set of
# spontaneous speech with a clean licence (see ROADMAP.md), plugged in through --extra.
def _dataset_from_rows(rows: list[dict], clips_dir: Path, source: str) -> Dataset:
    audio = [str(clips_dir / row["path"]) for row in rows]
    # FLEURS TSVs carry HTML entities (`&apos;`, `&quot;`, ...): store the plain text.
    text = [html.unescape(row["sentence"]).strip() for row in rows]
    ds = Dataset.from_dict({"audio": audio, "sentence": text, "source": [source] * len(rows)})
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def from_fleurs(cache_dir: Path) -> DatasetDict:
    from huggingface_hub import hf_hub_download

    out = {}
    for fl_split, our_split in SPLIT_MAP.items():
        tsv = Path(hf_hub_download(FLEURS_REPO, f"data/{CONFIG}/{fl_split}.tsv", repo_type="dataset", cache_dir=str(cache_dir)))
        tar = Path(hf_hub_download(FLEURS_REPO, f"data/{CONFIG}/audio/{fl_split}.tar.gz", repo_type="dataset", cache_dir=str(cache_dir)))
        clips = cache_dir / f"fleurs_{CONFIG}" / fl_split
        if not clips.is_dir():
            clips.mkdir(parents=True, exist_ok=True)
            print(f"[prepare] extracting {tar.name} ({fl_split})")
            with tarfile.open(tar) as tf:
                tf.extractall(clips, filter="data")
        # the tarball holds <split>/<file>.wav: point at the folder that contains the wav files
        wavs = next((d for d in [clips / fl_split, clips] if any(d.glob("*.wav"))), clips)
        out[our_split] = _dataset_from_rows(_read_fleurs_tsv(tsv), wavs, "fleurs_lb")
    return DatasetDict(out)


def from_manifest(path: Path) -> DatasetDict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row.get("split", "train"), []).append(row)
    out = {}
    for split, items in by_split.items():
        ds = Dataset.from_dict(
            {
                "audio": [str(Path(path).parent / r["audio"]) if not Path(r["audio"]).is_absolute() else r["audio"] for r in items],
                "sentence": [html.unescape(r["text"]).strip() for r in items],
                "source": [r.get("source", path.stem) for r in items],
            }
        )
        out[split] = ds.cast_column("audio", Audio(sampling_rate=16_000))
    return DatasetDict(out)


def merge(primary: DatasetDict, extras: list[DatasetDict]) -> DatasetDict:
    merged = dict(primary)
    for extra in extras:
        for split, ds in extra.items():
            merged[split] = concatenate_datasets([merged[split], ds]) if split in merged else ds
    return DatasetDict(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fleurs"], default="fleurs")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    ap.add_argument("--extra", type=Path, action="append", default=[], help="JSONL manifest(s) of extra data")
    ap.add_argument("--out", type=Path, default=Path("data/fleurs_lb"))
    args = ap.parse_args()

    primary = from_fleurs(args.cache_dir)
    dsd = merge(primary, [from_manifest(p) for p in args.extra])
    for split, ds in dsd.items():
        print(f"[prepare] {split}: {len(ds)} utterances")
    dsd.save_to_disk(str(args.out))
    stats = {split: len(ds) for split, ds in dsd.items()}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[prepare] saved to {args.out}")


if __name__ == "__main__":
    main()
