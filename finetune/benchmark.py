#!/usr/bin/env python3
"""Benchmark several models on the same test set, clean and WhatsApp-degraded.

    python benchmark.py --data data/cv_lb --models openai/whisper-small openai/whisper-large-v3-turbo \
        unilux/whisper-small-v1-luxembourgish runs/turbo-lora/merged --max-samples 300

« Degraded » re-encodes every clip through ffmpeg as 16 kbit/s mono Opus, the codec and
bitrate class of a WhatsApp voice note, then decodes it back to 16 kHz PCM. Same text,
worse audio: the gap between the two columns is the robustness of each model.

Output: results/benchmark.md (one row per model × condition) and the JSON behind it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import char_error_rate, transcribe_dataset  # noqa: E402
from lb_text import word_error_rate  # noqa: E402


def degrade_whatsapp(array: np.ndarray, sr: int, bitrate: str = "16k") -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for --degrade (apt-get install ffmpeg)")
    with tempfile.TemporaryDirectory() as tmp:
        src, mid, dst = Path(tmp) / "in.wav", Path(tmp) / "mid.opus", Path(tmp) / "out.wav"
        sf.write(src, array, sr)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-c:a", "libopus", "-b:a", bitrate, mid], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mid, "-ar", str(sr), "-ac", "1", dst], check=True)
        out, _ = sf.read(dst, dtype="float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/cv_lb"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--language", default="lb")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_from_disk(str(args.data))[args.split].select(range(args.max_samples))
    conditions = {"clean": ds}
    if not args.no_degrade:
        conditions["whatsapp-16k"] = ds.map(
            lambda ex: {"audio": {"array": degrade_whatsapp(ex["audio"]["array"], 16_000), "sampling_rate": 16_000, "path": None}},
            desc="degrade",
        )

    rows = []
    for model in args.models:
        for cond, data in conditions.items():
            refs, hyps, audio_s, wall = transcribe_dataset(model, data, args.language, args.batch_size, device, None)
            rows.append(
                {
                    "model": model,
                    "condition": cond,
                    "n": len(refs),
                    "wer": round(100 * word_error_rate(refs, hyps), 2),
                    "cer": round(100 * char_error_rate(refs, hyps), 2),
                    "seconds_per_audio_minute": round(60 * wall / audio_s, 2),
                    "device": device,
                }
            )
            print(rows[-1])

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["| model | condition | n | WER % | CER % | s / audio-min | device |", "|---|---|---:|---:|---:|---:|---|"]
    md += [f"| {r['model']} | {r['condition']} | {r['n']} | {r['wer']} | {r['cer']} | {r['seconds_per_audio_minute']} | {r['device']} |" for r in rows]
    (args.results_dir / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {args.results_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
