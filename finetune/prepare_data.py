#!/usr/bin/env python3
"""Build the training dataset: Common Voice Luxembourgish (+ optional extra manifests).

Three ways to get Common Voice `lb` in, all ending in the same on-disk DatasetDict
(train / validation / test, columns: audio, sentence, source):

  --source hf       download the `lb` subset of mozilla-foundation/common_voice_17_0 from the
                    Hub. Broken since October 2025: Mozilla moved Common Voice to Mozilla Data
                    Collective and the Hub repo is now a pointer (checked 22/10/2026). Use cv-dir.
                    Only the lb tarballs and TSVs are fetched, not the 100+ other locales.
  --source cv-dir   a directory downloaded from Mozilla (Data Collective / commonvoice.mozilla.org):
                    <dir>/clips/*.mp3 + train.tsv / dev.tsv / test.tsv (or validated.tsv).
  --extra FILE      a JSONL manifest {"audio": "path.wav", "text": "...", "split": "train"} for
                    additional open data (e.g. the ZLS fine-tuning set from data.public.lu).

No loading script, no `trust_remote_code`: the TSVs and tarballs are parsed here, so the
pipeline survives `datasets` dropping script support.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets

CV_REPO = "mozilla-foundation/common_voice_17_0"
LOCALE = "lb"
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row for row in reader if row.get("sentence")]


def _dataset_from_rows(rows: list[dict], clips_dir: Path, source: str) -> Dataset:
    audio = [str(clips_dir / row["path"]) for row in rows]
    text = [row["sentence"].strip() for row in rows]
    ds = Dataset.from_dict({"audio": audio, "sentence": text, "source": [source] * len(rows)})
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def from_cv_dir(cv_dir: Path) -> DatasetDict:
    clips = cv_dir / "clips"
    if not clips.is_dir():
        raise SystemExit(f"{clips} not found: expected a Common Voice download (clips/ + *.tsv)")
    out = {}
    for cv_split, our_split in SPLIT_MAP.items():
        tsv = cv_dir / f"{cv_split}.tsv"
        if tsv.exists():
            out[our_split] = _dataset_from_rows(_read_tsv(tsv), clips, "common_voice_lb")
    if not out:
        validated = cv_dir / "validated.tsv"
        if not validated.exists():
            raise SystemExit("no train/dev/test.tsv nor validated.tsv in the Common Voice directory")
        full = _dataset_from_rows(_read_tsv(validated), clips, "common_voice_lb")
        split = full.train_test_split(test_size=0.1, seed=42)
        dev_test = split["test"].train_test_split(test_size=0.5, seed=42)
        out = {"train": split["train"], "validation": dev_test["train"], "test": dev_test["test"]}
    return DatasetDict(out)


def from_hub(cache_dir: Path) -> DatasetDict:
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            CV_REPO,
            repo_type="dataset",
            allow_patterns=[f"audio/{LOCALE}/*", f"transcript/{LOCALE}/*", f"n_shards.json"],
            cache_dir=str(cache_dir),
        )
    )
    clips = cache_dir / f"cv17_{LOCALE}_clips"
    clips.mkdir(parents=True, exist_ok=True)
    out = {}
    for cv_split, our_split in SPLIT_MAP.items():
        tsv = local / "transcript" / LOCALE / f"{cv_split}.tsv"
        if not tsv.exists():
            print(f"[prepare] no {tsv.name} for {LOCALE}, skipping {our_split}")
            continue
        for tar in sorted((local / "audio" / LOCALE / cv_split).glob("*.tar")):
            print(f"[prepare] extracting {tar.name}")
            with tarfile.open(tar) as tf:
                tf.extractall(clips, filter="data")
        out[our_split] = _dataset_from_rows(_read_tsv(tsv), clips, "common_voice_lb")
    if not out:
        raise SystemExit(
            "Nothing downloaded. Did you accept the dataset terms on the Hub and log in? "
            "Alternative: download the lb archive from Mozilla and use --source cv-dir."
        )
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
                "sentence": [r["text"].strip() for r in items],
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
    ap.add_argument("--source", choices=["hf", "cv-dir"], default="cv-dir")
    ap.add_argument("--cv-dir", type=Path, help="Common Voice download directory (with --source cv-dir)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    ap.add_argument("--extra", type=Path, action="append", default=[], help="JSONL manifest(s) of extra data")
    ap.add_argument("--out", type=Path, default=Path("data/cv_lb"))
    args = ap.parse_args()

    primary = from_hub(args.cache_dir) if args.source == "hf" else from_cv_dir(args.cv_dir)
    dsd = merge(primary, [from_manifest(p) for p in args.extra])
    for split, ds in dsd.items():
        print(f"[prepare] {split}: {len(ds)} utterances")
    dsd.save_to_disk(str(args.out))
    stats = {split: len(ds) for split, ds in dsd.items()}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[prepare] saved to {args.out}")


if __name__ == "__main__":
    main()
