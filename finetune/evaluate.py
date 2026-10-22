#!/usr/bin/env python3
"""WER / CER of any Whisper checkpoint (Hub id or local folder) on a split of the dataset.

    python evaluate.py --model openai/whisper-small --data data/fleurs_lb --split test
    python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test

Writes results/<model-name>.<split>.json and appends a row to results/results.md so the
« before / after » table of the model card is generated, not typed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent))
from lb_text import normalize, word_error_rate  # noqa: E402


def char_error_rate(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return float(jiwer.cer([normalize(r) for r in refs], [normalize(h) for h in hyps]))


def transcribe_dataset(model_id: str, ds, language: str, batch_size: int, device: str, max_samples: int | None):
    processor = WhisperProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    refs, hyps, audio_seconds = [], [], 0.0
    t0 = time.perf_counter()
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        audio_seconds += sum(len(a) for a in arrays) / 16_000
        feats = processor.feature_extractor(arrays, sampling_rate=16_000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.to(device, dtype), language=language, task="transcribe", max_new_tokens=225)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch["sentence"])
    wall = time.perf_counter() - t0
    return refs, hyps, audio_seconds, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/fleurs_lb"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--language", default="lb")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_from_disk(str(args.data))[args.split]
    refs, hyps, audio_s, wall = transcribe_dataset(args.model, ds, args.language, args.batch_size, device, args.max_samples)
    wer = word_error_rate(refs, hyps)
    cer = char_error_rate(refs, hyps)
    rtf = wall / audio_s if audio_s else float("nan")
    name = re.sub(r"[^\w.-]+", "_", args.model.strip("/"))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "split": args.split,
        "n": len(refs),
        "audio_minutes": round(audio_s / 60, 1),
        "wer": round(100 * wer, 2),
        "cer": round(100 * cer, 2),
        "rtf": round(rtf, 4),
        "device": device,
        "samples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:20]],
    }
    (args.results_dir / f"{name}.{args.split}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    table = args.results_dir / "results.md"
    if not table.exists():
        table.write_text("| model | split | n | WER % | CER % | RTF | device |\n|---|---|---:|---:|---:|---:|---|\n", encoding="utf-8")
    with table.open("a", encoding="utf-8") as fh:
        fh.write(f"| {args.model} | {args.split} | {len(refs)} | {payload['wer']} | {payload['cer']} | {payload['rtf']} | {device} |\n")
    print(json.dumps({k: v for k, v in payload.items() if k != 'samples'}, indent=2))


if __name__ == "__main__":
    main()
