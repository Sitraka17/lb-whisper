#!/usr/bin/env python3
"""Check that the bench references do not overlap the text of known training sets.

    python bench/leak_check.py --refs refs.jsonl --out leak.json \
        --corpus zls-cc0=/data/zls-cc0/Verbatim --corpus lod=/data/lod/iwwersiicht.csv \
        --corpus fleurs=/data/fleurs_lb/manifest.jsonl

A corpus is a folder of .txt files, a .txt file (one sentence per line), a JSONL manifest (field
"text", "text_normalised" or "sentence") or a CSV / TSV file (column "text" if there is one,
otherwise every cell).

For each reference, the share of its words that sit inside a sequence of words also found in
the corpus: any run of --n words (6 by default), or a whole corpus line of 4 words or more when
the line is shorter than that (dictionary example sentences are short). Text goes through the
bench normaliser first.

A reference is flagged on either of two criteria:
- relative: a covered share above --flag-share (0.2 by default);
- absolute: a run of --flag-run consecutive covered words or more (25 by default, about ten
  seconds of speech). A clip can be ten minutes long: one minute lifted from a training set
  is a leak even if it is a small share of the clip. The longest run is written in the report.

Limits, stated in PROTOCOL.md too: this compares text, not audio. It finds a test recording
whose words were in a training set (same broadcast, same read sentences). It cannot find the
same voice or the same programme on another day, and it says nothing about training sets that
are not public.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from lb_text import normalize  # noqa: E402

MIN_LINE_WORDS = 4


def corpus_lines(path: Path):
    """Yield the text lines of a corpus, whatever its form (see the module docstring)."""
    if path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            if not p.name.startswith("._"):
                yield from p.read_text(encoding="utf-8-sig").splitlines()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    for key in ("text", "text_normalised", "sentence"):
                        if row.get(key):
                            yield row[key]
    elif path.suffix in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            header = fh.readline()
            fh.seek(0)
            reader = csv.reader(fh, delimiter="\t" if header.count("\t") > header.count(",") else ",")
            names = next(reader)
            column = names.index("text") if "text" in names else None
            for row in reader:
                if column is None:
                    yield from row
                elif column < len(row):
                    yield row[column]
    else:
        yield from path.read_text(encoding="utf-8-sig").splitlines()


def build_index(lines, n: int) -> dict:
    """{length: set of word sequences of that length}: every n-gram, plus whole short lines."""
    index = {n: set()}
    for line in lines:
        words = normalize(line).split()
        if len(words) >= n:
            index[n].update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
        elif len(words) >= MIN_LINE_WORDS:
            index.setdefault(len(words), set()).add(" ".join(words))
    return index


def coverage(text: str, index: dict) -> tuple:
    """(normalised words, one flag per word: inside a word sequence present in the index)."""
    words = normalize(text).split()
    covered = [False] * len(words)
    for length, grams in index.items():
        if not grams:
            continue
        for i in range(len(words) - length + 1):
            if " ".join(words[i:i + length]) in grams:
                covered[i:i + length] = [True] * length
    return words, covered


def covered_share(text: str, index: dict) -> float:
    """Share of the words of `text` inside a word sequence present in the index."""
    words, covered = coverage(text, index)
    return sum(covered) / len(words) if words else 0.0


def longest_run(text: str, index: dict) -> tuple:
    """(length in words, text) of the longest run of consecutive covered words."""
    words, covered = coverage(text, index)
    best = (0, 0)
    start = None
    for i, flag in enumerate(covered + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[0]:
                best = (i - start, start)
            start = None
    return best[0], " ".join(words[best[1]:best[1] + best[0]])


def check(ref: dict, index: dict) -> dict:
    """Worst case over the verbatim and the normalised text of one reference."""
    out = {"share": 0.0, "covered_words": 0, "longest_run": 0, "longest_run_text": ""}
    for key in ("text", "text_normalised"):
        text = ref.get(key) or ""
        words, covered = coverage(text, index)
        if not words:
            continue
        run, run_text = longest_run(text, index)
        out["share"] = max(out["share"], sum(covered) / len(words))
        out["covered_words"] = max(out["covered_words"], sum(covered))
        if run > out["longest_run"]:
            out["longest_run"], out["longest_run_text"] = run, run_text
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--flag-share", type=float, default=0.2)
    ap.add_argument("--flag-run", type=int, default=25, help="flag a run of this many consecutive covered words")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    refs = [json.loads(line) for line in args.refs.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {"n": args.n, "flag_share": args.flag_share, "flag_run": args.flag_run, "corpora": {}, "flagged": []}
    for item in args.corpus:
        name, path = item.split("=", 1)
        index = build_index(corpus_lines(Path(path)), args.n)
        report["corpora"][name] = {"path": path, "sequences": sum(len(g) for g in index.values())}
        worst = worst_run = 0
        for ref in refs:
            found = check(ref, index)
            worst = max(worst, found["share"])
            worst_run = max(worst_run, found["longest_run"])
            reasons = [reason for reason, hit in (("share", found["share"] > args.flag_share),
                                                  ("run", found["longest_run"] >= args.flag_run)) if hit]
            if reasons:
                report["flagged"].append({"id": ref.get("id") or ref.get("audio"), "corpus": name, "reasons": reasons,
                                          "share": round(found["share"], 3), "covered_words": found["covered_words"],
                                          "longest_run": found["longest_run"], "longest_run_text": found["longest_run_text"]})
        report["corpora"][name].update(highest_share=round(worst, 3), longest_run=worst_run)
        print(f"[leak_check] {name}: {report['corpora'][name]['sequences']} sequences, highest share {worst:.3f}, "
              f"longest run {worst_run} words")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[leak_check] {len(report['flagged'])} flagged reference/corpus pair(s) out of {len(refs)} references -> {args.out}")


if __name__ == "__main__":
    main()
