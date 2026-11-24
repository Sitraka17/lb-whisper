#!/usr/bin/env python3
"""Build the training dataset: Google FLEURS Luxembourgish (+ optional open data).

FLEURS (google/fleurs, config lb_lu) is read Luxembourgish, CC-BY-4.0: 2,502 / 408 / 934
utterances (8.46 / 1.12 / 2.73 h) in train / dev / test. It was the only open (commercial use allowed)
Luxembourgish speech set we had found on 22/10/2026; --shards adds the open data released since.
Mozilla Common Voice has no Luxembourgish data in any release from 17.0 to 27.0. Sentences are
stored with their HTML entities unescaped.

  --source fleurs   download data/lb_lu (TSVs + audio tarballs) from the Hub, no login, no loading
                    script: the TSVs and tarballs are parsed here.
  --extra FILE      a JSONL manifest {"audio": "path.wav", "text": "...", "split": "train"} for
                    additional data whose licence allows it. An optional "duration" (seconds) is
                    used as given; without it the length is read from the header of the audio
                    file. The v2 configs filter on that column, so it must not be zero.
  --shards DIR      the output folder of build_open_data.py, or a copy of its shard-*.tar files
                    alone (each archive carries the manifest of its own clips). The rows come
                    from DIR/manifest.jsonl when it is there, from the manifests inside the
                    archives otherwise; no other .jsonl is read, so a held-out manifest that sits
                    in the same folder stays out. v2 manifest format, one line per clip:
                    {"audio": "path relative to the manifest", "text": "verbatim transcript",
                     "text_normalised": "normalised transcript or null", "source": "zls-rtl",
                     "licence": "CC0-1.0", "recording": "id", "start": 0.0, "end": 12.3,
                     "duration": 12.3, "split": "train"}
                    The audio is taken from the folder if it is there, from the archives otherwise.
                    Rows whose licence is not in ALLOWED_LICENCES stop the script.
  --text-field      which transcript of the v2 manifest becomes the training label: "text"
                    (verbatim, the default) or "text_normalised".

Output: an on-disk DatasetDict with the columns audio, sentence, reference_norm, source, duration.
`reference_norm` is the normalised transcript of the source itself (the FLEURS "normalized" TSV
column, the ZLS "Normalised" text), empty when the source has none. Splits:

  train            FLEURS train + every open-data row with split "train"
  validation       FLEURS dev, never mixed with anything else
  validation_open  open-data rows with split "validation" (only with --shards)
  test             FLEURS test, never mixed with anything else
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

# `datasets` is imported inside the functions that build a Dataset, so that the manifest and
# archive readers below can be imported (and tested) with the standard library only.

FLEURS_REPO = "google/fleurs"
CONFIG = "lb_lu"
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}
FLEURS_COLUMNS = ["id", "path", "sentence", "normalized", "chars", "num_samples", "gender"]

# Licences that allow a published model. Anything else (NC, ND, unknown) is refused, not skipped:
# a manifest that contains such a row was built wrongly and should be fixed at the source.
ALLOWED_LICENCES = {"CC0-1.0", "CC-BY-4.0"}
TEXT_FIELDS = ("text", "text_normalised")
MANIFEST_KEYS = ("audio", "text", "source", "licence", "duration", "split")
# Open-data splits never touch the FLEURS dev and test splits: they get their own name.
OPEN_SPLITS = {"train": "train", "validation": "validation_open", "test": "test_open"}


def _read_fleurs_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, fieldnames=FLEURS_COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row for row in reader if row.get("sentence")]


# Review note: FLEURS is *read* speech, people reading Wikipedia-style sentences into a good mic.
# Real Luxembourgish is the opposite: radio, meetings, voice notes, half the sentence in French.
# That gap is the next thing to measure, not to guess. Plan: a small hand-checked set of
# spontaneous speech with a clean licence (see ROADMAP.md), plugged in through --extra.
def _dataset_from_rows(rows: list[dict], clips_dir: Path, source: str):
    # FLEURS TSVs carry HTML entities (`&apos;`, `&quot;`, ...): store the plain text.
    return _dataset(
        audio=[str(clips_dir / row["path"]) for row in rows],
        sentence=[html.unescape(row["sentence"]).strip() for row in rows],
        # The "normalized" column is kept exactly as FLEURS ships it: evaluate.py uses it as the
        # reference of the `wer_fleurs_norm` measure.
        reference_norm=[(row.get("normalized") or "").strip() for row in rows],
        source=[source] * len(rows),
        duration=[_fleurs_seconds(row) for row in rows],
    )


def _fleurs_seconds(row: dict) -> float:
    try:
        return round(int(row["num_samples"]) / 16_000, 3)
    except (KeyError, TypeError, ValueError):
        return 0.0


def _dataset(audio: list[str], sentence: list[str], reference_norm: list[str], source: list[str], duration: list[float]):
    from datasets import Audio, Dataset

    ds = Dataset.from_dict(
        {"audio": audio, "sentence": sentence, "reference_norm": reference_norm, "source": source, "duration": duration}
    )
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def from_fleurs(cache_dir: Path):
    from datasets import DatasetDict
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


def from_manifest(path: Path):
    from datasets import DatasetDict

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row.get("split", "train"), []).append(row)
    out = {}
    for split, items in by_split.items():
        out[split] = _dataset(
            audio=[str(Path(path).parent / r["audio"]) if not Path(r["audio"]).is_absolute() else r["audio"] for r in items],
            sentence=[html.unescape(r["text"]).strip() for r in items],
            reference_norm=[(r.get("text_normalised") or "").strip() for r in items],
            source=[r.get("source", path.stem) for r in items],
            duration=[_manifest_seconds(r, Path(path).parent) for r in items],
        )
    return DatasetDict(out)


def _manifest_seconds(row: dict, base: Path) -> float:
    """Duration of an --extra row: the "duration" field when it is there, the audio header otherwise.

    train.py drops clips of zero length when the features are computed per batch (the v2 configs),
    so a manifest without durations must not end up with 0.0 everywhere."""
    if row.get("duration"):
        return float(row["duration"])
    import soundfile as sf

    audio = Path(row["audio"])
    info = sf.info(str(audio if audio.is_absolute() else base / audio))
    return round(info.frames / info.samplerate, 3)


# --- v2 open data: shard-*.tar archives + JSONL manifests (standard library only) -------------


def read_manifest(path: Path) -> list[dict]:
    """Rows of one v2 manifest, checked: required keys, an allowed licence, a known split."""
    with Path(path).open(encoding="utf-8-sig") as fh:
        return parse_manifest(fh, str(path))


def parse_manifest(lines, path: str) -> list[dict]:
    rows = []
    for number, line in enumerate(lines, 1):
        if isinstance(line, bytes):
            line = line.decode("utf-8-sig" if number == 1 else "utf-8")
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in MANIFEST_KEYS if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{path}:{number}: missing {missing}")
        if row["licence"] not in ALLOWED_LICENCES:
            raise ValueError(
                f"{path}:{number}: licence {row['licence']!r} ({row['source']}) is not one of "
                f"{sorted(ALLOWED_LICENCES)}; this data must not be used for a published model"
            )
        if row["split"] not in OPEN_SPLITS:
            raise ValueError(f"{path}:{number}: split {row['split']!r} is not one of {sorted(OPEN_SPLITS)}")
        rows.append(row)
    return rows


def find_shard_inputs(folder: Path) -> tuple[Path | None, list[Path]]:
    """(manifest.jsonl or None, shard-*.tar archives) of a shard folder.

    Only these names are looked at: held-out archives and manifests written next to them by
    build_open_data.py, and macOS `._*` companion files, are never picked up."""
    folder = Path(folder)
    manifest = folder / "manifest.jsonl"
    archives = sorted(p for p in folder.glob("shard-*.tar") if not p.name.startswith("._"))
    if not manifest.is_file() and not archives:
        raise FileNotFoundError(f"neither manifest.jsonl nor shard-*.tar in {folder}")
    return (manifest if manifest.is_file() else None), archives


def _is_shard_manifest(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("manifest-shard-") and base.endswith(".jsonl")


def embedded_manifests(archives: list[Path]) -> list[dict]:
    """Rows of the manifest-shard-NNN.jsonl stored in each archive. build_open_data.py writes it
    as the first member, so one header is read per archive; any other position is found too."""
    rows: list[dict] = []
    for archive in archives:
        before = len(rows)
        with tarfile.open(archive) as tf:
            for position, member in enumerate(tf):
                if member.isfile() and _is_shard_manifest(member.name):
                    rows.extend(parse_manifest(tf.extractfile(member), f"{archive.name}:{member.name}"))
                    if position == 0:
                        break
        if len(rows) == before:
            raise FileNotFoundError(f"{archive} holds no manifest-shard-*.jsonl and there is no manifest.jsonl next to it")
    return rows


def _member_key(name: str) -> str:
    """Archive member name or manifest path as a clean relative POSIX path ('' if unsafe)."""
    parts = [part for part in PurePosixPath(name.replace("\\", "/")).parts if part not in (".", "/")]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def extract_shards(archives: list[Path], wanted: set[str], dest: Path) -> dict[str, Path]:
    """Extract the wanted members (keys as given by _member_key) from the archives into dest.

    One sequential pass per archive, which is what a mounted Google Drive likes: a member whose
    path is exactly a wanted path is written at once. Looser matches are only remembered during
    that pass and used afterwards, for the wanted paths that no archive holds under their exact
    name: first a member stored under an extra leading folder (shard-000/clips/a.flac for
    clips/a.flac), then a member with the same file name when that name is unique among the
    wanted paths. An exact match in a later archive therefore always wins over a namesake in an
    earlier one. Members are written by hand under dest: no absolute path, no "..", no links."""
    by_name: dict[str, list[str]] = {}
    for key in wanted:
        by_name.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    found: dict[str, Path] = {}
    # key -> (rank, archive, member name); rank 0 = same path under a leading folder, 1 = same file name
    loose: dict[str, tuple[int, Path, str]] = {}
    dest = Path(dest)

    def write(tf: tarfile.TarFile, member: tarfile.TarInfo, key: str) -> None:
        target = dest / key
        if not (target.is_file() and target.stat().st_size == member.size):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                shutil.copyfileobj(tf.extractfile(member), fh)
        found[key] = target

    for archive in archives:
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = _member_key(member.name)
                base = name.rsplit("/", 1)[-1]
                if not name or base.startswith("._") or base.endswith(".jsonl"):
                    continue
                if name in wanted:
                    if name not in found:
                        write(tf, member, name)
                    continue
                candidates = by_name.get(base, [])
                key, rank = next((k for k in candidates if name.endswith("/" + k)), None), 0
                if key is None and len(candidates) == 1:
                    key, rank = candidates[0], 1
                if key is not None and (key not in loose or rank < loose[key][0]):
                    loose[key] = (rank, archive, member.name)

    second_pass: dict[Path, dict[str, str]] = {}
    for key, (_, archive, member_name) in loose.items():
        if key not in found:
            second_pass.setdefault(archive, {})[member_name] = key
    for archive, names in second_pass.items():
        with tarfile.open(archive) as tf:
            for member in tf:
                if member.isfile() and member.name in names and names[member.name] not in found:
                    write(tf, member, names[member.name])
    return found


def load_shards(folder: Path, extract_dir: Path, text_field: str = "text") -> tuple[dict[str, list[dict]], dict]:
    """Read a shard folder. Returns ({our split: rows}, report).

    Each returned row has audio (absolute path), sentence, reference_norm, source, licence,
    duration. With text_field="text_normalised", rows that have no normalised transcript (the
    LOD sentences, for instance) fall back to "text"; the report counts them."""
    if text_field not in TEXT_FIELDS:
        raise ValueError(f"text_field must be one of {TEXT_FIELDS}")
    folder = Path(folder)
    manifest, archives = find_shard_inputs(folder)
    pending: list[tuple[dict, str]] = []
    rows_out: dict[str, list[dict]] = {}
    report = {"folder": str(folder), "manifest": manifest.name if manifest else "inside the archives", "archives": len(archives),
              "text_field": text_field, "fallback_to_text": 0, "empty_text": 0, "missing_audio": 0}

    def add(row: dict, audio_path: Path) -> None:
        normalised = (row.get("text_normalised") or "").strip()
        sentence = normalised if text_field == "text_normalised" else ""
        if text_field == "text_normalised" and not sentence:
            report["fallback_to_text"] += 1
        sentence = sentence or html.unescape(str(row["text"])).strip()
        if not sentence:
            report["empty_text"] += 1
            return
        rows_out.setdefault(OPEN_SPLITS[row["split"]], []).append(
            {"audio": str(audio_path), "sentence": sentence, "reference_norm": normalised,
             "source": row["source"], "licence": row["licence"], "duration": float(row["duration"])}
        )

    for row in read_manifest(manifest) if manifest else embedded_manifests(archives):
        local = Path(row["audio"]) if Path(row["audio"]).is_absolute() else folder / row["audio"]
        if local.is_file():
            add(row, local.resolve())
        else:
            pending.append((row, _member_key(row["audio"])))
    if pending:
        found = extract_shards(archives, {key for _, key in pending if key}, extract_dir)
        for row, key in pending:
            if key in found:
                add(row, found[key].resolve())
            else:
                report["missing_audio"] += 1
    return rows_out, report


def summarise(rows_by_split: dict[str, list[dict]]) -> dict:
    """Clips and hours per split, per source and per licence, from the manifest durations."""
    out: dict = {}
    for split, rows in sorted(rows_by_split.items()):
        by_source: dict[str, dict] = {}
        by_licence: dict[str, float] = {}
        for row in rows:
            entry = by_source.setdefault(row["source"], {"clips": 0, "hours": 0.0, "licence": row["licence"]})
            entry["clips"] += 1
            entry["hours"] += row["duration"] / 3600
            by_licence[row["licence"]] = by_licence.get(row["licence"], 0.0) + row["duration"] / 3600
        out[split] = {
            "clips": len(rows),
            "hours": round(sum(row["duration"] for row in rows) / 3600, 2),
            "clips_over_30s": sum(1 for row in rows if row["duration"] > 30.0),
            "by_source": {k: {**v, "hours": round(v["hours"], 2)} for k, v in sorted(by_source.items())},
            "hours_by_licence": {k: round(v, 2) for k, v in sorted(by_licence.items())},
        }
    return out


def from_shards(folder: Path, extract_dir: Path, text_field: str):
    from datasets import DatasetDict

    rows_by_split, report = load_shards(folder, extract_dir, text_field)
    if report["missing_audio"]:
        raise FileNotFoundError(
            f"{report['missing_audio']} manifest rows of {folder} have no audio, neither next to the "
            f"manifest nor in the {report['archives']} shard-*.tar archive(s): the copy is incomplete"
        )
    report["splits"] = summarise(rows_by_split)
    out = {}
    for split, rows in rows_by_split.items():
        out[split] = _dataset(
            audio=[r["audio"] for r in rows],
            sentence=[r["sentence"] for r in rows],
            reference_norm=[r["reference_norm"] for r in rows],
            source=[r["source"] for r in rows],
            duration=[r["duration"] for r in rows],
        )
    return DatasetDict(out), report


def shard_subdir(number: int) -> str:
    """Name of the sub-folder of --extract-dir that receives the clips of the n-th --shards folder."""
    return f"{number:02d}"


def remove_extracted(extract_dir: Path, count: int) -> None:
    """Delete what this script extracted, and nothing else.

    Only the numbered sub-folders it created (00, 01, ...) are removed. --extract-dir itself may be
    a folder the user already had (/content, a Drive folder): it goes only if it is empty afterwards."""
    extract_dir = Path(extract_dir)
    for number in range(count):
        shutil.rmtree(extract_dir / shard_subdir(number), ignore_errors=True)
    try:
        extract_dir.rmdir()
    except OSError:
        pass      # not empty (somebody else's files, or macOS companion files), or already gone


def merge(primary, extras: list):
    from datasets import DatasetDict, concatenate_datasets

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
    ap.add_argument("--shards", type=Path, action="append", default=[],
                    help="folder(s) with shard-*.tar archives and v2 JSONL manifest(s)")
    ap.add_argument("--text-field", choices=TEXT_FIELDS, default="text",
                    help="transcript of the v2 manifests used as the training label")
    ap.add_argument("--extract-dir", type=Path, default=None,
                    help="where the archive members are extracted, in sub-folders 00, 01, ... that are "
                         "deleted afterwards; anything else in that folder is left alone (default: <cache-dir>/shards)")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="keep the extracted clips after the dataset is saved (the saved dataset holds its own copy)")
    ap.add_argument("--no-fleurs", action="store_true",
                    help="dry run of the open data only: no FLEURS download, so no dev and test split")
    ap.add_argument("--out", type=Path, default=Path("data/fleurs_lb"))
    args = ap.parse_args()

    extract_dir = args.extract_dir or args.cache_dir / "shards"
    extras = [from_manifest(p) for p in args.extra]
    reports = []
    for number, folder in enumerate(args.shards):
        dsd_open, report = from_shards(folder, extract_dir / shard_subdir(number), args.text_field)
        extras.append(dsd_open)
        reports.append(report)
        for split, entry in report["splits"].items():
            print(f"[prepare] {folder.name} {split}: {entry['clips']} clips, {entry['hours']} h")
    if args.no_fleurs:
        if not extras:
            ap.error("--no-fleurs needs --shards or --extra")
        dsd = merge(extras[0], extras[1:])
    else:
        dsd = merge(from_fleurs(args.cache_dir), extras)
    for split, ds in dsd.items():
        print(f"[prepare] {split}: {len(ds)} utterances")
    dsd.save_to_disk(str(args.out))
    stats: dict = {split: len(ds) for split, ds in dsd.items()}
    # Hours per split from the stored durations (FLEURS: num_samples of the TSV; open data: manifest).
    stats["hours"] = {split: round(sum(ds["duration"]) / 3600, 2) for split, ds in dsd.items()}
    if reports:
        stats["open_data"] = reports
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.shards and not args.keep_extracted:
        remove_extracted(extract_dir, len(args.shards))   # the saved dataset holds its own copy of the audio
    print(f"[prepare] saved to {args.out}")


if __name__ == "__main__":
    main()
