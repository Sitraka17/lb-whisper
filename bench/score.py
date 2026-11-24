#!/usr/bin/env python3
"""Score every system of the LB-Real bench with one published normalisation.

    python bench/score.py --refs refs.jsonl --hyps out/*.jsonl --out results/

Inputs
- refs.jsonl: one line per audio file, made by make_refs.py (or a manifest in the shared
  format of this repository): {"id", "audio", "text", "text_normalised", "genre",
  "duration", "segments": [{"start", "end", "text"}], "segments_normalised": [...]}.
  Without "genre", the "source" field of the shared manifest is used as the genre.
- one JSONL per system, made by run_local.py (open models) or import_hypotheses.py (online
  services): {"id", "system", "text", ...}.

What it computes, per system, overall and per genre
- WER and CER on text normalised by finetune/lb_text.py (the same function for every system),
- wer_n_rule: the same WER with Eifeler Regel disagreements forgiven (diagnostic only),
- a 95 % confidence interval by bootstrap over utterances (fixed seed),
- substitutions / deletions / insertions,
- skipped passages: runs of consecutive deleted reference words worth more than 5 s of audio.
  The duration comes from the reference timestamps when the reference has them (the union of
  the time spans of the deleted words, so overlapping segments are not counted twice),
  otherwise from the mean speaking rate of the utterance (reference words / audio duration).
  Two runs separated by a single reference word that was not deleted count as one passage:
  one word aligned by chance in the middle of a hole must not hide the hole,
- pairwise WER differences between systems with a paired bootstrap interval.

A file a system did not transcribe is an error unless --allow-missing is given, in which case
it is scored as an empty transcript (every reference word deleted). Silently dropping the
files a system fails on would flatter that system.

Refused, because each would produce a number that means nothing: a reference without a word
(its WER would be 0 % whatever the system wrote), and a system whose rows were not all made
with the same model revision and decoding settings (--allow-mixed-settings turns the second
one into a warning).

--only-consent A2 keeps the clips whose "consent" list has that part; --without-consent A2
keeps the others. That is how clips that may not be uploaded are scored apart (PROTOCOL.md,
section 3).

Standard library only: the tests run without jiwer, numpy or a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from lb_text import normalize, strip_n_rule  # noqa: E402

SKIP_SECONDS = 5.0
BRIDGE_WORDS = 1  # reference words that were not deleted, tolerated between two runs of one passage
N_BOOT = 2000
SEED = 2026


# ---------------------------------------------------------------------------- distances

def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words).

    Bit-parallel (Myers 1999, Hyyrö 2003) on Python integers: a 10-minute recording is about
    9,000 characters, and the textbook table would be 81 million cells in pure Python."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    m = len(b)
    peq = {}
    for i, item in enumerate(b):
        peq[item] = peq.get(item, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for item in a:
        eq = peq.get(item, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def align(ref: list, hyp: list) -> list:
    """Word alignment as a list of operations, one per step of the cheapest path:
    "e" (equal), "s" (substitution), "d" (reference word deleted), "i" (hypothesis word
    inserted). Ties prefer the diagonal, then a deletion, then an insertion, so the result is
    deterministic."""
    # Common prefix and suffix are always part of an optimal path: cut them to keep the table small.
    pre = 0
    while pre < len(ref) and pre < len(hyp) and ref[pre] == hyp[pre]:
        pre += 1
    suf = 0
    while suf < len(ref) - pre and suf < len(hyp) - pre and ref[-1 - suf] == hyp[-1 - suf]:
        suf += 1
    r, h = ref[pre:len(ref) - suf], hyp[pre:len(hyp) - suf]
    n, m = len(r), len(h)
    width = m + 1
    back = bytearray((n + 1) * width)  # 0 = diagonal, 1 = deletion, 2 = insertion
    prev = list(range(m + 1))
    for j in range(1, m + 1):
        back[j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = i * width
        back[row] = 1
        ri = r[i - 1]
        for j in range(1, m + 1):
            best = prev[j - 1] + (ri != h[j - 1])
            move = 0
            if prev[j] + 1 < best:
                best, move = prev[j] + 1, 1
            if cur[j - 1] + 1 < best:
                best, move = cur[j - 1] + 1, 2
            cur[j] = best
            back[row + j] = move
        prev = cur
    ops = []
    i, j = n, m
    while i or j:
        move = back[i * width + j]
        if move == 0:
            ops.append("e" if r[i - 1] == h[j - 1] else "s")
            i, j = i - 1, j - 1
        elif move == 1:
            ops.append("d")
            i -= 1
        else:
            ops.append("i")
            j -= 1
    ops.reverse()
    return ["e"] * pre + ops + ["e"] * suf


# ---------------------------------------------------------------------------- references

def reference_words(text: str, segments: list | None, duration: float | None):
    """Normalised reference words and, for each word, the seconds of audio it stands for.

    With timestamped segments, each segment's duration is shared between its words in
    proportion to their length in characters. A segment that ends before it starts is given a
    duration of zero. Without timestamps, the utterance duration is shared evenly (mean
    speaking rate). Returns (words, seconds or None, starts or None, method)."""
    if segments:
        words, seconds, starts = [], [], []
        for seg in segments:
            seg_words = normalize(seg["text"]).split()
            if not seg_words:
                continue
            begin = float(seg["start"])
            end = max(float(seg["end"]), begin)
            total = sum(len(w) for w in seg_words)
            t = begin
            for w in seg_words:
                d = (end - begin) * len(w) / total
                words.append(w)
                seconds.append(d)
                starts.append(t)
                t += d
        return words, seconds, starts, "reference timestamps"
    words = normalize(text).split()
    if duration and words:
        return words, [float(duration) / len(words)] * len(words), None, "mean speaking rate"
    return words, None, None, "unavailable"


def _span_seconds(spans: list) -> float:
    """Length of the union of [start, end] time spans: audio counted once even where the
    reference segments overlap, which they often do in hand-made timestamps."""
    total, reach = 0.0, None
    for begin, end in sorted(spans):
        if reach is None or begin > reach:
            total += end - begin
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


def skipped_passages(ops: list, words: list, seconds: list | None, starts: list | None,
                     threshold: float = SKIP_SECONDS, bridge: int = BRIDGE_WORDS) -> list:
    """Passages of deleted reference words that stand for more than `threshold` seconds of audio.

    A passage is a run of consecutive deleted words. Two runs separated by at most `bridge`
    reference words that were not deleted (matched or substituted) are one passage; the bridged
    words are listed in the text but their time is not counted. With reference timestamps the
    duration is the union of the time spans of the deleted words; without, their summed share
    of the utterance. Each passage: first word index, deleted word count, bridged word count,
    seconds, start time, text."""
    if seconds is None:
        return []
    deleted = []  # one flag per reference word
    for op in ops:
        if op != "i":
            deleted.append(op == "d")
    runs = []  # [first, end) of each run of deleted words
    i = 0
    while i < len(deleted):
        if deleted[i]:
            j = i
            while j < len(deleted) and deleted[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= bridge:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    out = []
    for first, end in merged:
        lost = [k for k in range(first, end) if deleted[k]]
        if starts:
            length = _span_seconds([(starts[k], starts[k] + max(seconds[k], 0.0)) for k in lost])
        else:
            length = sum(max(seconds[k], 0.0) for k in lost)
        if length > threshold:
            out.append({
                "first_word": first,
                "n_words": len(lost),
                "bridged_words": end - first - len(lost),
                "seconds": round(length, 2),
                "start": round(starts[first], 2) if starts else None,
                "text": " ".join(words[first:end]),
            })
    return out


# ---------------------------------------------------------------------------- bootstrap

def quantile(sorted_values: list, q: float) -> float:
    """Linear-interpolation quantile of an already sorted list."""
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def resamples(n_units: int, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    """n_boot draws of n_units indices with replacement. Same seed, same draws: every system is
    resampled on the same utterances, which is what makes the pairwise interval a paired one."""
    rng = random.Random(seed)
    return [[rng.randrange(n_units) for _ in range(n_units)] for _ in range(n_boot)]


def ratio(pairs: list, draw: list | None = None) -> float:
    """Sum of numerators over sum of denominators, on all units or on one bootstrap draw."""
    if draw is None:
        num, den = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    else:
        num, den = sum(pairs[k][0] for k in draw), sum(pairs[k][1] for k in draw)
    return num / den if den else 0.0


def bootstrap_ci(pairs: list, draws: list) -> list | None:
    """95 % percentile interval of an error rate. `pairs` is one (errors, reference length)
    per unit. None with fewer than two units: one utterance has no sampling variability to show."""
    if len(pairs) < 2:
        return None
    values = sorted(ratio(pairs, d) for d in draws)
    return [quantile(values, 0.025), quantile(values, 0.975)]


def paired_difference(pairs_a: list, pairs_b: list, draws: list) -> dict | None:
    """Difference of error rates a - b with a paired 95 % bootstrap interval."""
    if len(pairs_a) < 2:
        return None
    values = sorted(ratio(pairs_a, d) - ratio(pairs_b, d) for d in draws)
    lo, hi = quantile(values, 0.025), quantile(values, 0.975)
    return {"difference": ratio(pairs_a) - ratio(pairs_b), "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}


# ---------------------------------------------------------------------------- scoring

def score_utterance(ref_words: list, hyp_text: str, seconds: list | None, starts: list | None,
                    threshold: float = SKIP_SECONDS) -> dict:
    """All per-utterance counts for one reference / hypothesis pair."""
    hyp_words = normalize(hyp_text).split()
    ops = align(ref_words, hyp_words)
    ref_norm, hyp_norm = " ".join(ref_words), " ".join(hyp_words)
    return {
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "substitutions": ops.count("s"),
        "deletions": ops.count("d"),
        "insertions": ops.count("i"),
        "errors": len(ops) - ops.count("e"),
        "errors_n_rule": edit_distance(strip_n_rule(ref_norm).split(), strip_n_rule(hyp_norm).split()),
        "ref_chars": len(ref_norm),
        "char_errors": edit_distance(ref_norm, hyp_norm),
        "ref_digit_words": sum(any(c.isdigit() for c in w) for w in ref_words),
        "hyp_digit_words": sum(any(c.isdigit() for c in w) for w in hyp_words),
        "skipped": skipped_passages(ops, ref_words, seconds, starts, threshold),
    }


def _units(rows: list, unit: str) -> list:
    """Group utterance rows into bootstrap units: the utterance itself, or its recording."""
    groups = {}
    for row in rows:
        key = row["recording"] if unit == "recording" and row.get("recording") else row["id"]
        groups.setdefault(key, []).append(row)
    return [groups[k] for k in sorted(groups)]


def _pairs(units: list, num: str, den: str) -> list:
    return [(sum(r[num] for r in u), sum(r[den] for r in u)) for u in units]


def summarise(rows: list, unit: str = "utterance", n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Aggregate per-utterance rows (one system, one group of utterances) into the published block."""
    units = _units(rows, unit)
    draws = resamples(len(units), n_boot, seed) if len(units) >= 2 else []
    wer_pairs = _pairs(units, "errors", "ref_words")
    cer_pairs = _pairs(units, "char_errors", "ref_chars")

    def pct(x):
        return round(100 * x, 2)

    def pct_ci(ci):
        return [pct(ci[0]), pct(ci[1])] if ci else None

    total = {k: sum(r[k] for r in rows) for k in (
        "ref_words", "hyp_words", "substitutions", "deletions", "insertions", "ref_digit_words", "hyp_digit_words")}
    skipped = [s for r in rows for s in r["skipped"]]
    durations = [r["duration"] for r in rows if r.get("duration")]
    return {
        "utterances": len(rows),
        "bootstrap_units": len(units),
        "audio_minutes": round(sum(durations) / 60, 2) if len(durations) == len(rows) else None,
        **total,
        "wer": pct(ratio(wer_pairs)),
        "wer_ci95": pct_ci(bootstrap_ci(wer_pairs, draws)),
        "wer_n_rule": pct(ratio(_pairs(units, "errors_n_rule", "ref_words"))),
        "cer": pct(ratio(cer_pairs)),
        "cer_ci95": pct_ci(bootstrap_ci(cer_pairs, draws)),
        "skipped_passages": len(skipped),
        "skipped_seconds": round(sum(s["seconds"] for s in skipped), 1),
        "missing": sum(1 for r in rows if r.get("missing")),
    }


def score(refs: dict, systems: dict, reference: str = "verbatim", threshold: float = SKIP_SECONDS,
          unit: str = "utterance", n_boot: int = N_BOOT, seed: int = SEED, allow_missing: bool = False) -> dict:
    """refs: id -> reference row. systems: name -> {id -> hypothesis row}. Returns the report."""
    text_key = "text" if reference == "verbatim" else "text_normalised"
    seg_key = "segments" if reference == "verbatim" else "segments_normalised"
    prepared = {}
    for uid, ref in refs.items():
        if ref.get(text_key) is None and not ref.get(seg_key):
            raise SystemExit(f"[score] reference {uid} has no {text_key}: cannot score against the {reference} reference")
        duration = ref.get("duration")
        prepared[uid] = reference_words(ref.get(text_key) or "", ref.get(seg_key), duration)
        if not prepared[uid][0]:
            raise SystemExit(f"[score] the {reference} reference of {uid} has no word once normalised: its error rate "
                             "would be 0 % whatever the system wrote. Fix the transcript or remove the clip.")

    report = {"reference": reference, "systems": {}, "skipped": [], "pairwise_wer": []}
    per_system_rows = {}
    for name in sorted(systems):
        hyps = systems[name]
        missing = sorted(set(refs) - set(hyps))
        if missing and not allow_missing:
            raise SystemExit(f"[score] {name}: no hypothesis for {len(missing)} reference(s), e.g. {missing[0]}. "
                             "Pass --allow-missing to score them as empty transcripts.")
        rows = []
        for uid in sorted(refs):
            words, seconds, starts, method = prepared[uid]
            row = score_utterance(words, (hyps.get(uid) or {}).get("text", ""), seconds, starts, threshold)
            row.update(id=uid, genre=genre_of(refs[uid]), recording=refs[uid].get("recording"),
                       duration=refs[uid].get("duration"), missing=uid not in hyps, timing=method)
            for s in row["skipped"]:
                report["skipped"].append({"system": name, "id": uid, "genre": row["genre"], "timing": method, **s})
            rows.append(row)
        per_system_rows[name] = rows
        genres = sorted({r["genre"] for r in rows})
        report["systems"][name] = {
            "overall": summarise(rows, unit, n_boot, seed),
            "by_genre": {g: summarise([r for r in rows if r["genre"] == g], unit, n_boot, seed) for g in genres},
            "ignored_hypotheses": sorted(set(hyps) - set(refs)),
        }

    names = sorted(per_system_rows)
    if len(names) > 1:
        units = {n: _pairs(_units(per_system_rows[n], unit), "errors", "ref_words") for n in names}
        n_units = len(units[names[0]])
        draws = resamples(n_units, n_boot, seed) if n_units >= 2 else []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diff = paired_difference(units[a], units[b], draws)
                if diff:
                    report["pairwise_wer"].append({
                        "a": a, "b": b, "difference": round(100 * diff["difference"], 2),
                        "ci95": [round(100 * x, 2) for x in diff["ci95"]], "excludes_zero": diff["excludes_zero"]})
    return report


# ---------------------------------------------------------------------------- input / output

def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def utterance_id(row: dict) -> str:
    """The key that links a hypothesis to its reference: "id", else the audio path without extension."""
    if row.get("id"):
        return str(row["id"])
    return str(Path(row["audio"]).with_suffix(""))


def load_refs(path: Path) -> dict:
    refs = {}
    for row in read_jsonl(path):
        uid = utterance_id(row)
        if uid in refs:
            raise SystemExit(f"[score] duplicate reference id {uid} in {path}")
        refs[uid] = row
    return refs


def genre_of(ref: dict) -> str:
    """Genre of a reference: "genre", else the "source" of the shared manifest format."""
    return str(ref.get("genre") or ref.get("source") or "unknown")


def filter_refs(refs: dict, only: str | None = None, without: str | None = None) -> dict:
    """References whose "consent" list has the part `only`, or does not have the part `without`."""
    def parts(ref):
        value = ref.get("consent") or []
        return set(value.replace(",", " ").replace(";", " ").split()) if isinstance(value, str) else set(value)

    kept = {}
    for uid, ref in refs.items():
        if only and only not in parts(ref):
            continue
        if without and without in parts(ref):
            continue
        kept[uid] = ref
    return kept


def settings_signature(row: dict) -> str:
    """What must be identical on every row of one system: the model and decoding blocks written
    by run_local.py, or the service, date, pass and settings written by import_hypotheses.py."""
    online = row.get("online") or {}
    kept = {"model": row.get("model"), "decode": row.get("decode"),
            "online": {k: online.get(k) for k in ("service", "url", "test_date", "pass", "settings", "strip_regex")} if online else None}
    return json.dumps(kept, sort_keys=True, ensure_ascii=False)


def mixed_settings(systems: dict) -> dict:
    """{system: number of distinct settings} for the systems whose rows do not all share one."""
    out = {}
    for name, rows in systems.items():
        distinct = {settings_signature(r) for r in rows.values()}
        if len(distinct) > 1:
            out[name] = len(distinct)
    return out


def load_systems(paths: list) -> dict:
    systems = {}
    for path in paths:
        for row in read_jsonl(Path(path)):
            name = row.get("system") or Path(path).stem
            uid = utterance_id(row)
            if uid in systems.setdefault(name, {}):
                raise SystemExit(f"[score] {name}: two hypotheses for {uid} (second one in {path})")
            systems[name][uid] = row
    return systems


def to_markdown(report: dict) -> str:
    def ci(block, key):
        value = block[key + "_ci95"]
        return f"{block[key]} [{value[0]}, {value[1]}]" if value else f"{block[key]} [n/a]"

    lines = [
        f"Reference: {report['reference']}. Normaliser: finetune/lb_text.py ({report.get('normaliser_sha256', 'n/a')}).",
        "Intervals: 95 % bootstrap over " + report.get("ci_unit", "utterance") + "s, "
        f"{report.get('n_boot', N_BOOT)} draws, seed {report.get('seed', SEED)}.",
        "",
        "| system | genre | utt. | ref. words | WER % [95 % CI] | WER n-rule % | CER % [95 % CI] | S / D / I | skipped passages | skipped s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, blocks in report["systems"].items():
        for genre, b in [("all", blocks["overall"])] + sorted(blocks["by_genre"].items()):
            lines.append(
                f"| {name} | {genre} | {b['utterances']} | {b['ref_words']} | {ci(b, 'wer')} | {b['wer_n_rule']} | "
                f"{ci(b, 'cer')} | {b['substitutions']} / {b['deletions']} / {b['insertions']} | "
                f"{b['skipped_passages']} | {b['skipped_seconds']} |")
    if report["pairwise_wer"]:
        lines += ["", "| WER difference (a minus b, points) | 95 % CI (paired) | interval excludes 0 |", "|---|---:|---|"]
        for p in report["pairwise_wer"]:
            lines.append(f"| {p['a']} minus {p['b']}: {p['difference']} | [{p['ci95'][0]}, {p['ci95'][1]}] | "
                         f"{'yes' if p['excludes_zero'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--hyps", type=Path, nargs="+", required=True, help="one JSONL per system")
    ap.add_argument("--out", type=Path, required=True, help="output folder (keep it outside the repository)")
    ap.add_argument("--reference", choices=["verbatim", "normalised", "both"], default="verbatim")
    ap.add_argument("--skip-seconds", type=float, default=SKIP_SECONDS)
    ap.add_argument("--ci-unit", choices=["utterance", "recording"], default="utterance",
                    help="'recording' resamples whole recordings when the references are clips of longer recordings")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-missing", action="store_true", help="score a missing hypothesis as an empty transcript")
    ap.add_argument("--allow-mixed-settings", action="store_true",
                    help="only warn when the rows of one system were made with different models or settings")
    ap.add_argument("--only-consent", default=None, metavar="PART", help="keep the clips whose consent list has PART, e.g. A2")
    ap.add_argument("--without-consent", default=None, metavar="PART", help="keep the clips whose consent list lacks PART")
    args = ap.parse_args()

    refs = load_refs(args.refs)
    n_all = len(refs)
    refs = filter_refs(refs, args.only_consent, args.without_consent)
    if not refs:
        raise SystemExit("[score] no reference left after the consent filter")
    if len(refs) != n_all:
        print(f"[score] consent filter: {len(refs)} of {n_all} references kept", file=sys.stderr)
    systems = load_systems(args.hyps)
    mixed = mixed_settings(systems)
    if mixed:
        message = "; ".join(f"{name}: {n} different settings" for name, n in sorted(mixed.items()))
        if not args.allow_mixed_settings:
            raise SystemExit(f"[score] rows of one system made with different models or settings ({message}). "
                             "Re-run the system, or pass --allow-mixed-settings to score it anyway.")
        print(f"[score] WARNING, mixed settings: {message}", file=sys.stderr)
    normaliser = hashlib.sha256((Path(__file__).resolve().parents[1] / "finetune" / "lb_text.py").read_bytes()).hexdigest()[:12]
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = ["verbatim", "normalised"] if args.reference == "both" else [args.reference]
    for kind in kinds:
        report = score(refs, systems, kind, args.skip_seconds, args.ci_unit, args.n_boot, args.seed, args.allow_missing)
        report.update(normaliser_sha256=normaliser, ci_unit=args.ci_unit, n_boot=args.n_boot, seed=args.seed,
                      skip_seconds=args.skip_seconds, bridge_words=BRIDGE_WORDS, refs=str(args.refs), n_references=len(refs),
                      only_consent=args.only_consent, without_consent=args.without_consent, mixed_settings=mixed)
        (args.out / f"scores.{kind}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md = to_markdown(report)
        (args.out / f"scores.{kind}.md").write_text(md, encoding="utf-8")
        with (args.out / f"skipped.{kind}.jsonl").open("w", encoding="utf-8") as fh:
            for s in report["skipped"]:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(md)


if __name__ == "__main__":
    main()
