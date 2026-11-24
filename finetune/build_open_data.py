#!/usr/bin/env python3
"""Build the v2 training set from open Luxembourgish speech data: clips, manifest, shards.

    python build_open_data.py --data /path/to/data --out /path/to/open-data-v2

Expected layout under --data (nothing in it is modified):

    zls-cc0/        Schreifmaschinn-Trainingsdaten CC0 (ZLS, CC0 1.0): Metadata.xlsx,
                    Verbatim/<Source>/*.wav|.csv|.json, Normalised/<Source>_normalised/*_normalised.csv
    zls-ccby/       the Konterbont Magasinn part of the same release (CC BY 4.0), same layout
    lod-fraestemm/  LOD example sentences, female voice (CC0 1.0): iwwersiicht.csv, fraestemm.ogg.zip

What it does, in this order:

1. ZLS sets: the time-stamped fragments of every part are grouped into clips of at most
   --max-clip seconds. A clip never cuts a fragment; a fragment longer than the limit is
   discarded and counted. A fragment that repeats an earlier one (same time stamps, or mostly
   the same audio transcribed twice) is discarded and counted, and so is a fragment whose
   transcript is stuck on one phrase. Where the time stamps of two
   neighbouring fragments overlap, the clip edge is put in the middle of the overlap, so that
   two clips never share audio. Normalised text is paired with verbatim text by time stamps.
2. LOD: only the rows marked "Fraestëmm" in iwwersiicht.csv are used. The male voice is
   CC BY-NC and is never read from disk; its rows of the index (text and duration, no audio)
   are only used to detect the same sentences inside the ZLS set.
3. Duplicates: a fragment of the "ZLS" source that reads a sentence already present in the
   LOD female set is dropped (one copy is enough). Recordings of that source that are mostly
   LOD sentences also published with the male voice are held out in their own manifest
   (see --lod-male-overlap).
4. Leak check: no sentence of the FLEURS lb_lu test (and dev) split may appear word for word
   in a clip; a clip that contains one is dropped.
5. Validation: about --val-minutes minutes, whole recordings only, every source that has
   more than one recording, fixed seed. A broadcast delivered in two halves (__1, __2) is
   one recording.
6. Audio is cut to 16 kHz mono FLAC with ffmpeg; manifest.jsonl, stats.json, tar shards of
   at most --shard-gb GB (clips plus one manifest per shard) and SHA256SUMS are written.

The script can be run again: existing clips, shards and checksums are kept. Clips left over
from a run with other settings are neither in the manifest nor in a shard; --prune deletes them.
Only the standard library and ffmpeg are needed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent))
from lb_text import normalize  # noqa: E402

FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/lb_lu/{split}.tsv"

# Folder (or file prefix) of the ZLS release -> source label of the manifest.
ZLS_SOURCES = {
    "RTL": "zls-rtl",
    "Chambre": "zls-chambre",
    "100komma7": "zls-1007",
    "Speakeasy": "zls-speakeasy",
    "Jemp-Schuster": "zls-jemp-schuster",
    "Happy-Hippo": "zls-happy-hippo",
    "ZLS": "zls-zls",
}
LOD_SOURCE = "lod-f1"
LOD_FEMALE = "Fraestëmm"
# '__part03' and, for a broadcast delivered in two halves, '__1' or '__2' before it.
PART_SUFFIX = re.compile(r"(__\d)?(__part\d+)?$")
# Whitespace other than a plain space (no-break space, line break, tab ...) and zero-width characters.
ODD_SPACE = re.compile(r"[^\S ]|[\u200b\u2060\ufeff]")
ZERO_WIDTH = re.compile(r"[\u200b\u2060\ufeff]")
# A FLEURS lb_lu split has several hundred sentences; fewer than this means the file is not the TSV.
MIN_FLEURS_SENTENCES = 100


class Fragment(NamedTuple):
    begin: float
    end: float
    text: str


# ----------------------------------------------------------------------------- reading


def visible(paths: Iterable[Path]) -> list[Path]:
    """Drop the AppleDouble files (._name) that macOS writes on exFAT disks."""
    return sorted(p for p in paths if not p.name.startswith("._"))


def read_fragments(path: Path) -> list[Fragment]:
    """Read one transcript with time stamps, .csv (start_seconds,end_seconds,text) or .json."""
    out: list[Fragment] = []
    if path.suffix == ".json":
        for item in json.loads(path.read_text(encoding="utf-8-sig")).get("fragments", []):
            out.append(Fragment(float(item["begin"]), float(item["end"]), " ".join(item.get("lines") or []).strip()))
        return out
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out.append(Fragment(float(row["start_seconds"]), float(row["end_seconds"]), (row.get("text") or "").strip()))
    return out


def wav_seconds(path: Path) -> Optional[float]:
    """Length of a PCM .wav file, None if the header cannot be read."""
    try:
        with wave.open(str(path), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate())
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def transcript_path(folder: Path, stem: str) -> Optional[Path]:
    for suffix in (".csv", ".json"):
        path = folder / (stem + suffix)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def recording_id(stem: str) -> str:
    """'RTL__Kloertext__2024-04-18__part03' -> 'RTL__Kloertext__2024-04-18'.

    The two halves of one broadcast are one recording:
    'RTL__Background__2024-03-09__2__part05' -> 'RTL__Background__2024-03-09'.
    """
    return PART_SUFFIX.sub("", stem, count=1)


def read_xlsx_rows(path: Path, sheet: str) -> list[list[str]]:
    """Rows of one sheet of an .xlsx file as lists of strings (standard library only)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as zf:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", ns):
                strings.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
        rels = {
            rel.get("Id"): rel.get("Target")
            for rel in ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        }
        target = None
        for node in ET.fromstring(zf.read("xl/workbook.xml")).iter("{%s}sheet" % ns["m"]):
            if node.get("name") == sheet:
                target = rels[node.get(rel_ns)]
        if target is None:
            raise KeyError(f"no sheet named {sheet!r} in {path}")
        target = target.lstrip("/")
        root = ET.fromstring(zf.read(target if target.startswith("xl/") else "xl/" + target))
    rows: list[list[str]] = []
    for row in root.iter("{%s}row" % ns["m"]):
        cells: list[str] = []
        for cell in row.findall("m:c", ns):
            letters = re.match(r"[A-Z]+", cell.get("r") or "")
            col = 0
            for ch in letters.group(0) if letters else "":
                col = col * 26 + ord(ch) - 64
            col = col - 1 if letters else len(cells)
            value = cell.find("m:v", ns)
            if cell.get("t") == "s" and value is not None:
                text = strings[int(value.text or 0)]
            elif cell.get("t") == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter("{%s}t" % ns["m"]))
            else:
                text = value.text if value is not None and value.text else ""
            while len(cells) < col:
                cells.append("")
            cells.append(text)
        rows.append(cells)
    return rows


def metadata_table(path: Path) -> list[dict]:
    """Sheet 'Individual Files' of Metadata.xlsx as dicts keyed by the header row."""
    rows = read_xlsx_rows(path, "Individual Files")
    header = next((r for r in rows if "File Name" in r), None)
    if header is None:
        return []
    out = []
    for row in rows[rows.index(header) + 1:]:
        item = {name: (row[i] if i < len(row) else "") for i, name in enumerate(header) if name}
        if item.get("File Name"):
            out.append(item)
    return out


def metadata_seconds(text: str) -> float:
    """'03:00,10' (minutes:seconds,hundredths) -> 180.1"""
    minutes, _, rest = text.partition(":")
    return int(minutes) * 60 + float(rest.replace(",", "."))


def year_of(date: str) -> str:
    """'12.10.2021' -> '2021'; 'undated' when the cell holds no year."""
    found = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", date or "")
    return found.group(0) if found else "undated"


WATCH = re.compile(r"\bIMS\b|Everling|\b2026\b|[.\-]2026\b|\b2026-")


def flag_metadata(rows: Sequence[dict]) -> list[dict]:
    """Rows that mention 2026, the IMS or Pit Everling: candidates for a clash with my own test audio."""
    flagged = []
    for row in rows:
        blob = " | ".join(str(v) for k, v in row.items() if "Location" not in k and v)
        if WATCH.search(blob):
            flagged.append({k: row.get(k, "") for k in ("File Name", "Title", "Creator", "Date", "Source")})
    return flagged


# ----------------------------------------------------------------------------- grouping and pairing


def group_fragments(
    fragments: Sequence[Fragment],
    max_len: float = 30.0,
    max_gap: float = 2.0,
    keep: Optional[Sequence[bool]] = None,
) -> tuple[list[list[int]], Counter]:
    """Group consecutive fragments into clips of at most max_len seconds.

    Returns the groups as lists of fragment indices, and counters of what was left out.
    A group is closed when the next fragment would make it longer than max_len, when the
    silence before the next fragment is longer than max_gap, or when a fragment is unusable:
    no word in its text, end before begin, longer than max_len on its own, or keep[i] is False.
    """
    groups: list[list[int]] = []
    stats: Counter = Counter()
    current: list[int] = []

    def close() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for i, frag in enumerate(fragments):
        length = frag.end - frag.begin
        reason = None
        if keep is not None and not keep[i]:
            reason = "excluded"
        elif not normalize(frag.text):  # nothing, or punctuation only
            reason = "empty_text"
        elif length <= 0:
            reason = "bad_times"
        elif length > max_len:
            reason = "too_long"
            stats["too_long_seconds"] += length
        if reason:
            stats[reason] += 1
            close()
            continue
        if current:
            first, last = fragments[current[0]], fragments[current[-1]]
            if frag.end - first.begin > max_len or frag.begin - last.end > max_gap or frag.begin < first.begin:
                close()
        current.append(i)
    close()
    return groups, stats


def drop_repeated(
    fragments: Sequence[Fragment], max_len: float = 30.0, max_overlap: float = 2.0
) -> tuple[list[bool], Counter]:
    """Find the fragments that transcribe again audio that an earlier fragment already covers.

    Returns keep flags and counters. In file order, a fragment is dropped when an earlier
    kept fragment has the same time stamps ("repeated_same_times"), or shares with it more
    than half of the shorter of the two, or more than max_overlap seconds
    ("repeated_overlap"). The earlier one stays: one transcript per stretch of audio.
    Fragments with unusable times or longer than max_len are ignored here; they are
    discarded by group_fragments and must not push out a usable fragment.
    """
    keep = [True] * len(fragments)
    stats: Counter = Counter()
    kept: list[int] = []
    for i, frag in enumerate(fragments):
        length = frag.end - frag.begin
        if length <= 0 or length > max_len:
            continue
        reason = None
        for j in kept:
            other = fragments[j]
            if abs(other.begin - frag.begin) < 0.0005 and abs(other.end - frag.end) < 0.0005:
                reason = "repeated_same_times"
                break
            shared = min(frag.end, other.end) - max(frag.begin, other.begin)
            if shared > max_overlap or shared > 0.5 * min(length, other.end - other.begin):
                reason = "repeated_overlap"
        if reason:
            keep[i] = False
            stats[reason] += 1
            stats[reason + "_seconds"] += length
        else:
            kept.append(i)
    return keep, stats


def is_repetition_loop(text: str, phrase_times: int = 5, word_times: int = 8, max_words: int = 6) -> bool:
    """True for a transcript stuck on a phrase: "den neien, den neien, den neien, ..." .

    A phrase of two to max_words words said phrase_times times in a row, or one word
    word_times times in a row. People do repeat a word ("nee, nee, nee, nee, nee"), hence
    the higher bar for a single word. The text is read twice, as words and as written, so
    that a run of lone punctuation marks (", , , , , , , ,") is found too.
    """
    for words in (normalize(text).split(), text.lower().split()):
        for size in range(1, max_words + 1):
            for i in range(len(words) - size + 1):
                phrase = words[i:i + size]
                times = 1
                while words[i + times * size: i + (times + 1) * size] == phrase:
                    times += 1
                if len(set(phrase)) == 1:  # "an an an an" is one word repeated, not the phrase "an an"
                    if times * size >= word_times:
                        return True
                elif times >= phrase_times:
                    return True
    return False


def repetition_loops(texts: Sequence[str]) -> list[bool]:
    """Flags for the transcripts that are a loop, alone or together with the one before or after.

    A loop often runs over several fragments, each of them below the bar on its own
    ("Et geet net." four times in five fragments in a row): two consecutive transcripts
    that form a loop only when joined are both flagged.
    """
    alone = [is_repetition_loop(t) for t in texts]
    flags = list(alone)
    for i in range(len(texts) - 1):
        if not alone[i] and not alone[i + 1] and is_repetition_loop(f"{texts[i]} {texts[i + 1]}"):
            flags[i] = flags[i + 1] = True
    return flags


def clip_bounds(
    fragments: Sequence[Fragment], group: Sequence[int], neighbours: Iterable[int], max_overlap: float = 2.0
) -> tuple[float, float]:
    """Start and end of the clip made of `group`, without the audio it shares with a neighbour.

    The time stamps of consecutive fragments often overlap by a fraction of a second (the
    segmentation leaves a margin on both sides). Cutting at the raw time stamps would put the
    same audio at the end of one clip and at the start of the next. When a fragment outside
    the group overlaps the edge of the clip by at most max_overlap seconds, the edge moves to
    the middle of the overlap.
    """
    inside = set(group)
    start = fragments[group[0]].begin
    end = max(fragments[i].end for i in group)
    new_start, new_end = start, end
    for j in neighbours:
        if j in inside:
            continue
        other = fragments[j]
        if other.begin <= start < other.end < end and other.end - start <= max_overlap:
            new_start = max(new_start, (start + other.end) / 2)
        elif start < other.begin < end <= other.end and end - other.begin <= max_overlap:
            new_end = min(new_end, (other.begin + end) / 2)
    return new_start, new_end


def pair_normalised(
    verbatim: Sequence[Fragment], normalised: Sequence[Fragment], tolerance: float = 0.05
) -> list[Optional[str]]:
    """For every verbatim fragment, the text of the normalised fragment with the same time stamps.

    None when no normalised fragment begins and ends within `tolerance` seconds of it. A
    normalised fragment is used at most once. An empty normalised text is a valid pairing:
    the normalised version drops fillers.
    """
    order = sorted(range(len(normalised)), key=lambda j: normalised[j].begin)
    begins = [normalised[j].begin for j in order]
    used: set[int] = set()
    out: list[Optional[str]] = []
    for frag in verbatim:
        best, best_cost = None, None
        lo = bisect.bisect_left(begins, frag.begin - tolerance)
        hi = bisect.bisect_right(begins, frag.begin + tolerance)
        for pos in range(lo, hi):
            j = order[pos]
            if j in used or abs(normalised[j].end - frag.end) > tolerance:
                continue
            cost = abs(normalised[j].begin - frag.begin) + abs(normalised[j].end - frag.end)
            if best_cost is None or cost < best_cost:
                best, best_cost = j, cost
        if best is None:
            out.append(None)
        else:
            used.add(best)
            out.append(normalised[best].text)
    return out


def past_end_of_audio(fragments: Sequence[Fragment], audio_seconds: float, tolerance: float = 0.5) -> list[int]:
    """Indices of the fragments that begin after the audio ends, or end more than `tolerance` after it."""
    return [
        i for i, frag in enumerate(fragments)
        if frag.end > frag.begin and (frag.begin >= audio_seconds or frag.end > audio_seconds + tolerance)
    ]


def clean_text(text: str) -> str:
    """One plain space for any run of whitespace (no-break spaces, line breaks), no zero-width characters."""
    return " ".join(ZERO_WIDTH.sub("", text or "").split())


def join_text(parts: Iterable[str]) -> str:
    return " ".join(c for c in (clean_text(p) for p in parts) if c)


# ----------------------------------------------------------------------------- validation split


def choose_validation(
    recordings: dict[str, dict[str, float]],
    target_seconds: float = 3600.0,
    floor_seconds: float = 300.0,
    seed: int = 20261124,
) -> dict[str, list[str]]:
    """Pick whole recordings for validation: {source: [recording, ...]}.

    `recordings` maps source -> recording -> seconds. A recording is never split, and a
    source with a single recording stays entirely in train. Each source aims at a share of
    the target in proportion to its duration, with a floor. A source whose shortest
    recording is longer than that contributes exactly that one recording, so the total can
    exceed the target. The result depends only on the arguments.
    """
    eligible = {s: r for s, r in recordings.items() if len(r) > 1}
    total = sum(sum(r.values()) for r in eligible.values())
    chosen: dict[str, list[str]] = {}
    if not total:
        return chosen

    def smallest(recs: dict[str, float]) -> str:
        return min(sorted(recs), key=lambda name: recs[name])

    for source in sorted(eligible):
        recs = eligible[source]
        goal = max(floor_seconds, target_seconds * sum(recs.values()) / total)
        names = sorted(recs)
        random.Random(f"{seed}:{source}").shuffle(names)
        picked: list[str] = []
        acc = 0.0
        for name in names:
            if acc >= 0.9 * goal or len(picked) == len(names) - 1:
                break
            if acc + recs[name] <= 1.2 * goal:
                picked.append(name)
                acc += recs[name]
        chosen[source] = sorted(picked) if picked else [smallest(recs)]
    return chosen


# ----------------------------------------------------------------------------- leaks and duplicates


def find_leaks(sentences: Iterable[str], texts: Sequence[str]) -> dict[int, list[str]]:
    """Indices of `texts` that contain one of `sentences` word for word (after lb_text.normalize)."""
    wanted = sorted({normalize(s) for s in sentences if normalize(s)})
    by_first: dict[str, list[str]] = defaultdict(list)
    for sentence in wanted:
        by_first[sentence.split(" ", 1)[0]].append(sentence)
    hits: dict[int, list[str]] = {}
    for i, text in enumerate(texts):
        norm = normalize(text or "")
        if not norm:
            continue
        padded = f" {norm} "
        for word in set(norm.split(" ")):
            for sentence in by_first.get(word, ()):
                if f" {sentence} " in padded:
                    hits.setdefault(i, []).append(sentence)
    return hits


def fleurs_sentences(tsv: str, origin: str) -> list[str]:
    """Sentences (raw and normalised column) of a FLEURS TSV.

    Stops the run when the text holds fewer than MIN_FLEURS_SENTENCES distinct sentences: an
    error page or a changed format would otherwise make the leak check pass with nothing searched.
    """
    sentences = []
    for line in tsv.splitlines():
        cols = line.split("\t")
        if len(cols) > 2:
            sentences.append(cols[2])  # raw_transcription
            if len(cols) > 3:
                sentences.append(cols[3])  # transcription
    distinct = len({normalize(s) for s in sentences} - {""})
    if distinct < MIN_FLEURS_SENTENCES:
        raise SystemExit(
            f"FLEURS leak check: {origin} holds {distinct} distinct sentences, at least {MIN_FLEURS_SENTENCES} expected. "
            "It is not the FLEURS TSV (error page, changed format?). Delete it and run again, "
            "or pass --skip-fleurs-check to run without the check."
        )
    return sentences


def fetch_fleurs(split: str, cache: Path) -> list[str]:
    """Sentences of one FLEURS lb_lu split. Only the TSV is downloaded, no audio, no token."""
    path = cache / f"{split}.tsv"
    if not path.is_file() or path.stat().st_size == 0:
        url = FLEURS_URL.format(split=split)
        request = urllib.request.Request(url, headers={"User-Agent": "lb-whisper-data-build"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        fleurs_sentences(body.decode("utf-8", errors="replace"), url)  # checked before it is cached
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return fleurs_sentences(path.read_text(encoding="utf-8"), str(path))


def close_enough(a: float, candidates: Sequence[float], tolerance: float) -> bool:
    return any(abs(a - b) <= tolerance for b in candidates)


# ----------------------------------------------------------------------------- shards


def plan_shards(sizes: Sequence[int], max_bytes: int, reserve: int = 0) -> list[list[int]]:
    """Fill tar shards in order. `sizes` are file sizes; tar adds a 512-byte header and pads to 512.

    `reserve` is kept free in every shard for its manifest.
    """
    shards: list[list[int]] = []
    current: list[int] = []
    used = 0
    budget = max_bytes - reserve - 20 * 512  # end-of-archive blocks and record padding
    for i, size in enumerate(sizes):
        need = 512 + ((size + 511) // 512) * 512
        if current and used + need > budget:
            shards.append(current)
            current, used = [], 0
        current.append(i)
        used += need
    if current:
        shards.append(current)
    return shards


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shards(out: Path, prefix: str, rows: Sequence[dict], max_bytes: int, log: Callable[[str], None]) -> list[str]:
    """Write <prefix>-NNN.tar files holding the clips of `rows` and one manifest each."""
    sizes = [(out / row["audio"]).stat().st_size for row in rows]
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    reserve = 512 + max((sum(len(line.encode("utf-8")) for line in lines) // 512 + 2) * 512, 1024)
    names = []
    for number, members in enumerate(plan_shards(sizes, max_bytes, reserve)):
        name = f"{prefix}-{number:03d}.tar"
        names.append(name)
        target = out / name
        manifest = "".join(lines[i] for i in members).encode("utf-8")
        signature = hashlib.sha256(manifest + b"".join(str(sizes[i]).encode() + b"," for i in members)).hexdigest()
        stamp = out / "shards" / (name + ".signature")
        if target.is_file() and stamp.is_file() and stamp.read_text().strip() == signature:
            log(f"{name}: kept ({target.stat().st_size / 1e9:.2f} GB, {len(members)} clips)")
            continue
        partial = target.with_suffix(".tar.part")
        with tarfile.open(partial, "w", format=tarfile.USTAR_FORMAT if all(len(rows[i]["audio"]) < 100 for i in members) else tarfile.PAX_FORMAT) as tf:
            info = tarfile.TarInfo(f"manifest-{name[:-4]}.jsonl")
            info.size, info.mode, info.mtime = len(manifest), 0o644, 0
            tf.addfile(info, io.BytesIO(manifest))
            for i in members:
                info = tarfile.TarInfo(rows[i]["audio"])
                info.size, info.mode, info.mtime = sizes[i], 0o644, 0
                with (out / rows[i]["audio"]).open("rb") as fh:
                    tf.addfile(info, fh)
        partial.replace(target)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(signature + "\n")
        log(f"{name}: written ({target.stat().st_size / 1e9:.2f} GB, {len(members)} clips)")
    return names


def write_checksums(out: Path, names: Sequence[str]) -> None:
    """SHA256SUMS for `names`; a checksum is computed again only if size or mtime changed."""
    cache_path = out / "shards" / "sha256-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    lines = []
    for name in names:
        stat = (out / name).stat()
        key = f"{stat.st_size}:{stat.st_mtime_ns}"
        if cache.get(name, {}).get("key") != key:
            cache[name] = {"key": key, "sha256": sha256(out / name)}
        lines.append(f"{cache[name]['sha256']}  {name}\n")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    (out / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- audio


def ffmpeg_flac(args_in: list[str], target: Path, data: Optional[bytes] = None) -> None:
    partial = target.with_name(target.stem + ".part.flac")
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args_in, "-map_metadata", "-1",
               "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "flac", str(partial)]
    if data is not None:
        command.remove("-nostdin")
    subprocess.run(command, input=data, check=True, stdout=subprocess.DEVNULL)
    partial.replace(target)


def cut_all(jobs: Sequence[dict], out: Path, lod_zip: Optional[Path], workers: int, log: Callable[[str], None]) -> int:
    """Create the missing clips. A job has 'audio' plus either 'wav','start','end' or 'zip_member'."""
    todo = [j for j in jobs if not ((out / j["audio"]).is_file() and (out / j["audio"]).stat().st_size > 0)]
    if not todo:
        return 0
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to cut the clips")
    for folder in sorted({(out / j["audio"]).parent for j in todo}):
        folder.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(lod_zip) if lod_zip and any("zip_member" in j for j in todo) else None
    lock = threading.Lock()
    done = [0]

    def work(job: dict) -> None:
        target = out / job["audio"]
        if "zip_member" in job:
            with lock:
                data = archive.read(job["zip_member"])  # type: ignore[union-attr]
            ffmpeg_flac(["-i", "pipe:0"], target, data)
        else:
            ffmpeg_flac(["-ss", f"{job['start']:.3f}", "-t", f"{job['end'] - job['start']:.3f}", "-i", str(job["wav"])], target)
        with lock:
            done[0] += 1
            if done[0] % 2000 == 0:
                log(f"clips: {done[0]} / {len(todo)}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    if archive is not None:
        archive.close()
    return len(todo)


# ----------------------------------------------------------------------------- the build


def scan_zls(root: Path, licence: str, fixed_source: Optional[str]) -> tuple[list[dict], Counter]:
    """One dict per part: wav, source, recording, verbatim fragments, paired normalised texts."""
    parts: list[dict] = []
    stats: Counter = Counter()
    verbatim_root = root / "Verbatim"
    for folder in visible(verbatim_root.iterdir()) if verbatim_root.is_dir() else []:
        if not folder.is_dir():
            continue
        source = fixed_source or ZLS_SOURCES.get(folder.name)
        if source is None:
            stats[f"unknown_folder:{folder.name}"] += 1
            continue
        norm_folder = root / "Normalised" / f"{folder.name}_normalised"
        for wav in visible(folder.glob("*.wav")):
            stats["parts"] += 1
            transcript = transcript_path(folder, wav.stem)
            if transcript is None:
                stats["parts_without_transcript"] += 1
                continue
            fragments = read_fragments(transcript)
            norm_path = transcript_path(norm_folder, wav.stem + "_normalised")
            if norm_path is None:
                stats["parts_without_normalised"] += 1
                paired: list[Optional[str]] = [None] * len(fragments)
            else:
                paired = pair_normalised(fragments, read_fragments(norm_path))
            stats["fragments"] += len(fragments)
            stats["fragments_unpaired"] += sum(1 for p in paired if p is None)
            parts.append({
                "wav": wav, "stem": wav.stem, "source": source, "licence": licence,
                "recording": recording_id(wav.stem), "fragments": fragments, "normalised": paired,
                "seconds": wav_seconds(wav),
            })
    return parts, stats


def hours(seconds: float) -> float:
    return round(seconds / 3600, 2)


def says_8_words_twice(text: str) -> bool:
    words = text.lower().split()
    seen: set = set()
    for i in range(len(words) - 7):
        gram = tuple(words[i:i + 8])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def text_conventions(clips: Sequence[dict], odd_spaces: dict) -> dict:
    """What a training script has to know about the texts of the main manifest, and clip lengths."""
    lod = [c for c in clips if c["source"] == LOD_SOURCE]
    zls = [c for c in clips if c["source"] != LOD_SOURCE]
    tags: Counter = Counter(c["source"] for c in zls if re.search(r"\[[^\]]*\]", c["text"]))

    def lengths(rows: Sequence[dict]) -> dict:
        values = sorted(r["duration"] for r in rows)
        if not values:
            return {"clips": 0}
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        return {"clips": len(values), "mean_seconds": round(sum(values) / len(values), 2), "median_seconds": round(median, 2)}

    return {
        "lod_sentences": len(lod),
        "lod_starting_with_a_lower_case_letter": sum(1 for c in lod if c["text"][:1].islower()),
        "lod_ending_with_full_stop_question_or_exclamation_mark": sum(1 for c in lod if c["text"][-1:] in ".?!"),
        "lod_with_round_brackets": sum(1 for c in lod if "(" in c["text"]),
        "zls_clips": len(zls),
        "zls_clips_with_a_square_bracket_tag": dict(sorted(tags.items())),
        "zls_clips_with_a_sequence_of_8_words_twice": sum(1 for c in zls if says_8_words_twice(c["text"])),
        "zls_clips_with_a_digit": sum(1 for c in zls if re.search(r"\d", c["text"])),
        "zls_clips_whose_normalised_text_differs": sum(1 for c in zls if c["text_normalised"] is not None and c["text_normalised"] != c["text"]),
        "zls_clip_length": lengths(zls),
        "lod_clip_length": lengths(lod),
        "odd_whitespace_replaced_by_a_space_or_removed": {s: dict(sorted(c.items())) for s, c in sorted(odd_spaces.items()) if c},
    }


def build(args: argparse.Namespace) -> dict:
    def log(message: str) -> None:
        print(f"[open-data] {message}", flush=True)

    data, out = args.data, args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"settings": {
        "max_clip_seconds": args.max_clip, "max_gap_seconds": args.max_gap, "max_overlap_seconds": args.max_overlap, "val_minutes": args.val_minutes,
        "val_floor_minutes": args.val_floor_minutes, "seed": args.seed, "lod_male_overlap": args.lod_male_overlap,
        "duration_tolerance_seconds": args.duration_tolerance,
        "end_tolerance_seconds": args.end_tolerance, "shard_gb": args.shard_gb,
    }}

    # --- 1. read everything
    parts, scan = scan_zls(data / "zls-cc0", "CC0-1.0", None)
    ccby_parts, ccby_scan = scan_zls(data / "zls-ccby", "CC-BY-4.0", "zls-ccby")
    parts += ccby_parts
    scan.update(ccby_scan)
    log(f"ZLS: {scan['parts']} parts with audio, {scan['fragments']} fragments")

    metadata: dict[str, list[dict]] = {}
    for name in ("zls-cc0", "zls-ccby"):
        path = data / name / "Metadata.xlsx"
        metadata[name] = metadata_table(path) if path.is_file() else []
    listed = {row["File Name"] for rows in metadata.values() for row in rows}
    present = {part["wav"].name for part in parts}
    stats["metadata"] = {
        "files_listed": len(listed),
        "listed_hours": {name: hours(sum(metadata_seconds(r["Duration"]) for r in rows if r.get("Duration"))) for name, rows in metadata.items()},
        "listed_but_missing": sorted(listed - present),
        "present_but_not_listed": sorted(present - listed),
        "flagged_2026_ims_everling": flag_metadata([r for rows in metadata.values() for r in rows]),
        "rtl_by_year": dict(sorted(Counter(year_of(r.get("Date", "")) for r in metadata["zls-cc0"] if r["File Name"].startswith("RTL__")).items())),
    }

    lod_rows: list[dict] = []
    male_index: dict[str, list[float]] = defaultdict(list)
    lod_csv = data / "lod-fraestemm" / "iwwersiicht.csv"
    lod_zip = data / "lod-fraestemm" / "fraestemm.ogg.zip"
    lod_stats: Counter = Counter()
    odd_spaces: dict[str, Counter] = defaultdict(Counter)
    if lod_csv.is_file():
        with lod_csv.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row["voice"] == LOD_FEMALE:
                    lod_rows.append(row)
                else:
                    lod_stats["index_rows_other_voice_not_used"] += 1
                    male_index[normalize(row["text"])].append(float(row["duration"]))
    members = set()
    if lod_zip.is_file():
        with zipfile.ZipFile(lod_zip) as zf:
            members = {n for n in zf.namelist() if n.endswith(".ogg")}
    by_id = {Path(n).stem: n for n in members}
    female_index: dict[str, list[float]] = defaultdict(list)
    lod_clips: list[dict] = []
    for row in lod_rows:
        lod_stats["female_rows"] += 1
        key = normalize(row["text"])
        duration = float(row["duration"])
        if row["example_id"] not in by_id:
            lod_stats["female_rows_without_audio"] += 1
            continue
        if not key or duration <= 0 or duration > args.max_clip:
            lod_stats["female_rows_unusable"] += 1
            continue
        if key in female_index:
            lod_stats["female_same_text_dropped"] += 1
            continue
        female_index[key].append(duration)
        lod_clips.append({
            "audio": f"clips/{LOD_SOURCE}/{row['example_id'][:2]}/{row['example_id']}.flac",
            "text": clean_text(row["text"]), "text_normalised": None, "source": LOD_SOURCE, "licence": "CC0-1.0",
            "recording": f"LOD__{row['entry_id']}", "start": 0.0, "end": duration, "duration": duration,
            "zip_member": by_id[row["example_id"]],
        })
        odd_spaces[LOD_SOURCE].update(f"U+{ord(ch):04X}" for ch in ODD_SPACE.findall(row["text"].strip()))
    lod_stats["zip_members"] = len(members)
    lod_stats["zip_members_not_female_in_index"] = len(set(by_id) - {r["example_id"] for r in lod_rows})
    stats["lod"] = dict(lod_stats)
    log(f"LOD: {len(lod_clips)} female-voice sentences kept of {lod_stats['female_rows']} rows")

    # --- 2. the ZLS source against the LOD
    overlap: Counter = Counter()
    per_recording: dict[str, Counter] = defaultdict(Counter)
    for part in parts:
        if part["source"] != "zls-zls":
            continue
        labels = []
        for frag, norm_text in zip(part["fragments"], part["normalised"]):
            keys = {normalize(frag.text), normalize(norm_text or "")} - {""}
            length = frag.end - frag.begin
            in_f = [k for k in keys if k in female_index]
            in_m = [k for k in keys if k in male_index]
            label = "other"
            if in_f:
                label = "lod_female"
                overlap["same_text_as_lod_female"] += 1
                overlap["same_text_as_lod_female_seconds"] += length
                if any(close_enough(length, female_index[k], args.duration_tolerance) for k in in_f):
                    overlap["same_text_and_close_duration_lod_female"] += 1
                if in_m:
                    overlap["same_text_as_both_voices"] += 1
            elif in_m:
                label = "lod_male"
                overlap["same_text_as_lod_male"] += 1
                overlap["same_text_as_lod_male_seconds"] += length
                if any(close_enough(length, male_index[k], args.duration_tolerance) for k in in_m):
                    overlap["same_text_and_close_duration_lod_male"] += 1
            overlap["fragments"] += 1
            overlap["seconds"] += length
            per_recording[part["recording"]][label] += 1
            per_recording[part["recording"]]["all"] += 1
            labels.append(label)
        part["labels"] = labels
    mostly_lod = sorted(r for r, c in per_recording.items() if c["all"] and (c["lod_male"] + c["lod_female"]) / c["all"] >= 0.5)
    stats["zls_vs_lod"] = {**{k: (round(v, 1) if isinstance(v, float) else v) for k, v in overlap.items()},
                           "per_recording": {r: dict(c) for r, c in per_recording.items()},
                           "recordings_mostly_lod_sentences": mostly_lod}

    # --- 3. group into clips
    clips: list[dict] = []
    held_out: list[dict] = []
    grouping: dict[str, Counter] = defaultdict(Counter)
    for part in parts:
        fragments, paired = part["fragments"], part["normalised"]
        counter = grouping[part["source"]]
        counter["fragment_seconds"] += sum(max(0.0, f.end - f.begin) for f in fragments)
        # The same audio transcribed twice (same time stamps, or one fragment mostly inside
        # another) would repeat a sentence in the text of a clip: only the first one stays.
        keep, repeated = drop_repeated(fragments, args.max_clip, args.max_overlap)
        counter.update(repeated)
        neighbours = [i for i, f in enumerate(fragments) if keep[i] and f.end > f.begin]
        ordered = sorted(neighbours, key=lambda i: fragments[i].begin)
        for a, b in zip(ordered, ordered[1:]):
            shared = fragments[a].end - fragments[b].begin
            counter["neighbour_pairs"] += 1
            counter["neighbour_pairs_overlapping"] += shared > 0
            counter["neighbour_pairs_overlapping_over_half_second"] += shared > 0.5
        # A few transcripts are stuck on one phrase for several fragments. Text like that teaches
        # a model to loop: those fragments are left out.
        loops = [a or b for a, b in zip(repetition_loops([f.text for f in fragments]),
                                        repetition_loops([p or "" for p in paired]))]
        for i, frag in enumerate(fragments):
            if keep[i] and loops[i] and 0 < frag.end - frag.begin <= args.max_clip:
                keep[i] = False
                counter["repetition_loop"] += 1
                counter["repetition_loop_seconds"] += frag.end - frag.begin
        length = part["seconds"]
        if length is None:
            counter["parts_with_unreadable_wav_header"] += 1
        else:
            # A transcript sometimes runs past the end of its audio file. A fragment that ends more
            # than --end-tolerance seconds after the last sample has text without audio: left out.
            for i in past_end_of_audio(fragments, length, args.end_tolerance):
                if not keep[i]:
                    continue
                keep[i] = False
                counter["past_end_of_audio"] += 1
                counter["past_end_of_audio_seconds"] += fragments[i].end - fragments[i].begin
        if "labels" in part:
            for i, label in enumerate(part["labels"]):
                if keep[i] and (label == "lod_female" or (part["recording"] in mostly_lod and args.lod_male_overlap == "drop")):
                    keep[i] = False
                    counter["same_text_as_lod"] += 1
                    counter["same_text_as_lod_seconds"] += fragments[i].end - fragments[i].begin
        groups, counters = group_fragments(fragments, args.max_clip, args.max_gap, keep)
        counters.pop("excluded", None)
        counter.update(counters)
        for group in groups:
            raw_start, raw_end = fragments[group[0]].begin, max(fragments[i].end for i in group)
            start, end = clip_bounds(fragments, group, neighbours, args.max_overlap)
            counter["clip_edges_moved_to_mid_overlap"] += (start > raw_start) + (end < raw_end)
            counter["clip_edges_moved_seconds"] += (start - raw_start) + (raw_end - end)
            if length is not None and end > length:
                end = length
                counter["clips_clamped_to_end_of_audio"] += 1
            norm_parts = [paired[i] for i in group]
            if any(p is None for p in norm_parts):
                text_normalised = None
                counter["clips_without_normalised"] += 1
            else:
                text_normalised = join_text(norm_parts)  # type: ignore[arg-type]
            for i in group:
                odd_spaces[part["source"]].update(
                    f"U+{ord(ch):04X}" for ch in ODD_SPACE.findall(fragments[i].text.strip() + (paired[i] or "").strip()))
            clip = {
                "audio": f"clips/{part['source']}/{part['stem']}_{round(start * 1000):07d}-{round(end * 1000):07d}.flac",
                "text": join_text(fragments[i].text for i in group), "text_normalised": text_normalised,
                "source": part["source"], "licence": part["licence"], "recording": part["recording"],
                "start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3), "wav": part["wav"],
            }
            if part["recording"] in mostly_lod and args.lod_male_overlap == "hold":
                held_out.append(clip)
            else:
                clips.append(clip)
    clips += lod_clips
    stats["grouping"] = {s: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items()} for s, c in sorted(grouping.items())}

    # --- 4. leaks towards FLEURS
    leak_stats: dict = {}
    if args.skip_fleurs_check:
        leak_stats["checked"] = False
        log("FLEURS check skipped on request: the manifest is NOT verified against the test set")
    else:
        for split in ("test", "dev"):
            sentences = fetch_fleurs(split, out / "fleurs")
            for label, rows in (("main", clips), ("held_out", held_out)):
                real: dict[int, list[str]] = {}
                for version in ("text", "text_normalised"):
                    for i, found in find_leaks(sentences, [r[version] or "" for r in rows]).items():
                        real[i] = sorted(set(real.get(i, []) + found))
                leak_stats[f"{split}_{label}"] = {
                    "distinct_sentences": len({normalize(s) for s in sentences}),
                    "clips_with_a_sentence": len(real),
                    "examples": [{"audio": rows[i]["audio"], "sentences": real[i]} for i in sorted(real)[:20]],
                }
                for i in sorted(real, reverse=True):
                    del rows[i]
        leak_stats["checked"] = True
    stats["fleurs_leak_check"] = leak_stats

    # --- 5. validation split
    recordings: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for clip in clips:
        recordings[clip["source"]][clip["recording"]] += clip["duration"]
    chosen = choose_validation({s: dict(r) for s, r in recordings.items()}, args.val_minutes * 60, args.val_floor_minutes * 60, args.seed)
    validation = {(s, r) for s, names in chosen.items() for r in names}
    for clip in clips:
        clip["split"] = "validation" if (clip["source"], clip["recording"]) in validation else "train"
    for clip in held_out:
        clip["split"] = "train"
    clips.sort(key=lambda c: (c["split"] != "validation", c["source"], c["audio"]))
    held_out.sort(key=lambda c: c["audio"])
    train_texts = {normalize(c["text"]) for c in clips if c["split"] == "train"}
    stats["validation"] = {
        "recordings": {s: names for s, names in sorted(chosen.items())},
        "sources_without_validation": sorted(s for s in recordings if s not in chosen),
        "clips_whose_text_is_also_in_train": sum(1 for c in clips if c["split"] == "validation" and normalize(c["text"]) in train_texts),
    }

    stats["text_conventions"] = text_conventions(clips, odd_spaces)

    # --- 6. audio
    if args.skip_audio:
        log("audio skipped on request: manifest and statistics only")
    else:
        made = cut_all(clips + held_out, out, lod_zip, args.workers, log)
        log(f"clips: {made} written, {len(clips) + len(held_out) - made} already there")

    wanted = {c["audio"] for c in clips} | {c["audio"] for c in held_out}
    clip_root = out / "clips"
    orphans = [p for p in clip_root.rglob("*.flac") if p.relative_to(out).as_posix() not in wanted] if clip_root.is_dir() else []
    orphans = [p for p in orphans if not (p.name.startswith("._") and p.with_name(p.name[2:]).relative_to(out).as_posix() in wanted)]
    stats["orphan_files_under_clips"] = len(orphans)
    if orphans and args.prune:
        for path in orphans:
            path.unlink(missing_ok=True)  # macOS removes the ._ companion together with its file
        log(f"clips: {len(orphans)} files that are in no manifest deleted (--prune)")
    elif orphans:
        log(f"clips: {len(orphans)} files under clips/ are in no manifest (left over from another run; --prune deletes them)")

    # --- 7. manifest, statistics, shards
    fields = ("audio", "text", "text_normalised", "source", "licence", "recording", "start", "end", "duration", "split")

    def public(rows: Sequence[dict]) -> list[dict]:
        return [{k: row[k] for k in fields} for row in rows]

    def table(rows: Sequence[dict]) -> dict:
        result: dict = defaultdict(lambda: {"licence": "", "clips": 0, "hours": 0.0, "train_hours": 0.0, "validation_minutes": 0.0, "recordings": set()})
        for row in rows:
            cell = result[row["source"]]
            cell["licence"] = row["licence"]
            cell["clips"] += 1
            cell["hours"] += row["duration"] / 3600
            cell["train_hours" if row["split"] == "train" else "validation_minutes"] += row["duration"] / (3600 if row["split"] == "train" else 60)
            cell["recordings"].add(row["recording"])
        return {s: {**c, "hours": round(c["hours"], 2), "train_hours": round(c["train_hours"], 2),
                    "validation_minutes": round(c["validation_minutes"], 1), "recordings": len(c["recordings"])}
                for s, c in sorted(result.items())}

    stats["sources"] = table(clips)
    stats["held_out"] = table(held_out)
    by_licence: dict = defaultdict(float)
    for clip in clips:
        by_licence[clip["licence"]] += clip["duration"]
    stats["totals"] = {
        "clips": len(clips), "hours": hours(sum(c["duration"] for c in clips)),
        "train_clips": sum(1 for c in clips if c["split"] == "train"),
        "train_hours": hours(sum(c["duration"] for c in clips if c["split"] == "train")),
        "validation_clips": sum(1 for c in clips if c["split"] == "validation"),
        "validation_minutes": round(sum(c["duration"] for c in clips if c["split"] == "validation") / 60, 1),
        "hours_by_licence": {k: hours(v) for k, v in sorted(by_licence.items())},
        "held_out_clips": len(held_out), "held_out_hours": hours(sum(c["duration"] for c in held_out)),
        "clips_without_normalised_text": sum(1 for c in clips if c["text_normalised"] is None and c["source"] != LOD_SOURCE),
    }
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public(clips)), encoding="utf-8")
    outputs = ["manifest.jsonl", "stats.json"]
    held_name = "manifest.held-out-lod-male.jsonl"
    if held_out:
        (out / held_name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public(held_out)), encoding="utf-8")
        outputs.append(held_name)
    elif (out / held_name).is_file():
        (out / held_name).unlink()
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.skip_audio and not args.skip_shards:
        limit = int(args.shard_gb * 1e9)
        outputs += write_shards(out, "shard", public(clips), limit, log)
        if held_out:
            outputs += write_shards(out, "held-out-lod-male", public(held_out), limit, log)
        write_checksums(out, outputs)
        log(f"SHA256SUMS: {len(outputs)} files")
    return stats


def print_summary(stats: dict) -> None:
    print()
    print("| Source | Licence | Recordings | Clips | Hours | Train h | Validation min |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for source, c in stats["sources"].items():
        print(f"| {source} | {c['licence']} | {c['recordings']} | {c['clips']} | {c['hours']:.2f} | {c['train_hours']:.2f} | {c['validation_minutes']:.1f} |")
    t = stats["totals"]
    print(f"| total | | | {t['clips']} | {t['hours']:.2f} | {t['train_hours']:.2f} | {t['validation_minutes']:.1f} |")
    print()
    print(f"hours by licence: {t['hours_by_licence']}")
    if t["held_out_clips"]:
        print(f"held out (ZLS recordings of LOD sentences also published with the male voice): {t['held_out_clips']} clips, {t['held_out_hours']:.2f} h")
    for source, c in stats["grouping"].items():
        print(f"{source}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    z = stats["zls_vs_lod"]
    print("ZLS source against the LOD: " + ", ".join(f"{k}={v}" for k, v in z.items() if k != "per_recording"))
    print(f"LOD: {stats['lod']}")
    print(f"validation: {stats['validation']['clips_whose_text_is_also_in_train']} validation clips share their text with a train clip; "
          f"sources without validation: {stats['validation']['sources_without_validation']}")
    leak = stats["fleurs_leak_check"]
    if leak.get("checked"):
        print("FLEURS: " + ", ".join(f"{k}: {v['clips_with_a_sentence']} clips dropped ({v['distinct_sentences']} sentences searched)" for k, v in leak.items() if k != "checked"))
    m = stats["metadata"]
    print(f"metadata: {m['files_listed']} files listed, {len(m['listed_but_missing'])} missing on disk, "
          f"{len(m['present_but_not_listed'])} on disk but not listed; RTL files by year {m['rtl_by_year']}")
    print(f"metadata rows mentioning 2026, IMS or Everling: {len(m['flagged_2026_ims_everling'])}")
    for row in m["flagged_2026_ims_everling"]:
        print(f"  {row}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder that holds zls-cc0/, zls-ccby/ and lod-fraestemm/")
    ap.add_argument("--out", type=Path, required=True, help="output folder, outside the repository")
    ap.add_argument("--max-clip", type=float, default=30.0, help="longest clip in seconds (Whisper's window)")
    ap.add_argument("--max-gap", type=float, default=2.0, help="a silence longer than this between two fragments closes the clip")
    ap.add_argument("--max-overlap", type=float, default=2.0,
                    help="seconds; neighbouring fragments that overlap by at most this are cut apart in the middle of the "
                         "overlap, a fragment that overlaps an earlier one by more is dropped as a repeat")
    ap.add_argument("--val-minutes", type=float, default=60.0)
    ap.add_argument("--val-floor-minutes", type=float, default=5.0, help="smallest validation share of a source")
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--end-tolerance", type=float, default=0.5,
                    help="seconds; a fragment may end this much after the end of its audio file (it is then clamped)")
    ap.add_argument("--duration-tolerance", type=float, default=0.5, help="seconds; 'close duration' when comparing with LOD sentences")
    ap.add_argument("--lod-male-overlap", choices=["hold", "include", "drop"], default="hold",
                    help="ZLS recordings made of LOD sentences that the LOD also publishes with its male voice (CC BY-NC there, "
                         "CC0 in the ZLS release): hold = separate manifest and shards, include = in the main manifest, drop = not cut")
    ap.add_argument("--shard-gb", type=float, default=1.9, help="largest tar shard, in GB (10^9 bytes)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prune", action="store_true", help="delete files under <out>/clips that are in no manifest")
    ap.add_argument("--skip-audio", action="store_true", help="manifest and statistics only")
    ap.add_argument("--skip-shards", action="store_true")
    ap.add_argument("--skip-fleurs-check", action="store_true", help="offline run; the leak check is then NOT done")
    return ap.parse_args(argv)


def main() -> None:
    print_summary(build(parse_args()))


if __name__ == "__main__":
    main()
