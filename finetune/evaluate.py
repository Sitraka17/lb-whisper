#!/usr/bin/env python3
"""WER / CER of any Whisper checkpoint (Hub id or local folder) on a split of the dataset.

    python evaluate.py --model openai/whisper-small --data data/fleurs_lb --split test
    python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test

Writes results/<model-name>.<split>.json and appends a row to results/results.md so the
"before / after" table of the model card is generated, not typed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent))
from lb_text import normalize, word_error_rate  # noqa: E402


def char_error_rate(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return float(jiwer.cer([normalize(r) for r in refs], [normalize(h) for h in hyps]))


def is_ctranslate2(model_id: str) -> bool:
    """True for a CTranslate2 export (model.bin, no Transformers weights), local or on the Hub."""
    local = Path(model_id)
    if local.is_dir():
        return (local / "model.bin").exists() and not any(local.glob("*.safetensors"))
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(model_id))
    return "model.bin" in files and not any(f.endswith((".safetensors", "pytorch_model.bin")) for f in files)


def transcribe_ct2(model_id: str, ds, language: str, device: str, max_samples: int | None):
    """Greedy decoding with faster-whisper, to match the Transformers path (beam 1, no VAD).

    Review note: greedy on purpose, every model gets the same decoding so the table is fair.
    The Space decodes with beam 5 and VAD, so what people see in the app is a bit better than
    these numbers. Good problem to have, but say it out loud when quoting a WER."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "int8")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    refs, hyps, audio_seconds = [], [], 0.0
    t0 = time.perf_counter()
    for ex in ds:
        array = np.asarray(ex["audio"]["array"], dtype=np.float32)
        audio_seconds += len(array) / 16_000
        segments, _ = model.transcribe(array, language=language, task="transcribe", beam_size=1,
                                       condition_on_previous_text=False, vad_filter=False)
        hyps.append(" ".join(seg.text.strip() for seg in segments))
        refs.append(ex["sentence"])
    wall = time.perf_counter() - t0
    return refs, hyps, audio_seconds, wall


def transcribe_dataset(model_id: str, ds, language: str, batch_size: int, device: str, max_samples: int | None):
    if is_ctranslate2(model_id):
        return transcribe_ct2(model_id, ds, language, device, max_samples)
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
        # Same transcripts, Eifeler Regel disagreements forgiven. If this is far below "wer",
        # a good part of the "errors" are a missing or extra n, not misheard words.
        "wer_n_rule": round(100 * word_error_rate(refs, hyps, n_rule=True), 2),
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
