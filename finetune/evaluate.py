#!/usr/bin/env python3
"""WER / CER of any Whisper checkpoint (Hub id or local folder) on a split of the dataset.

    python evaluate.py --model openai/whisper-small --data data/fleurs_lb --split test
    python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test

Writes results/<model-name>.<split>.json and appends a row to results/results.md so the
"before / after" table of the model card is generated, not typed.

Three word error rates are written, and they are not interchangeable:

  wer                the reference number of this repository since v1: raw FLEURS sentence and
                     hypothesis, both through lb_text.normalize. Its definition does not change.
  wer_fleurs_norm    for a comparison with BuzzASR/luxembourgish (25.41 % WER, 8.55 % CER on
                     FLEURS according to its model card, read on 24/11/2026). The card only says
                     "normalized CER / WER"; the paper (arXiv:2609.09554, data section) says that
                     FLEURS "provides normalized transcripts, which we use as-is". So the reference
                     is the "normalized" column of the FLEURS TSV, untouched. The paper does not give
                     the hypothesis side for FLEURS; here it goes through lb_text.normalize
                     (lower case, punctuation dropped, apostrophes and hyphens inside words kept,
                     which is what the FLEURS column does). An approximation of their protocol,
                     not a reproduction: their decoding settings differ too.
  wer_whisper_basic  both sides through Whisper's BasicTextNormalizer (the "Whisper normalisation"
                     many model cards use). It splits d'Kanner in two words, see lb_text.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lb_text import _levenshtein, normalize, word_error_rate  # noqa: E402

# numpy, torch, datasets and transformers are imported where they are used, so that the scoring
# functions below can be imported (and tested) without them.


# Generation limit of the Transformers path. 225 is what every published number of this repository
# was measured with (the longest FLEURS test sentence is far below it), so it stays the default.
# The open-data validation clips are up to 30 s of spontaneous speech and a few transcripts are
# longer than 225 tokens: score that split with --max-new-tokens 440, the most Whisper accepts
# (448 decoder positions minus the start tokens, with a small margin).
DEFAULT_MAX_NEW_TOKENS = 225


def char_error_rate(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return float(jiwer.cer([normalize(r) for r in refs], [normalize(h) for h in hyps]))


def _rates(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus-level (WER, CER) of strings that are already normalised. No further processing."""
    word_errors = sum(_levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps))
    words = sum(len(r.split()) for r in refs)
    char_errors = sum(_levenshtein(list(r), list(h)) for r, h in zip(refs, hyps))
    chars = sum(len(r) for r in refs)
    return (word_errors / words if words else 0.0, char_errors / chars if chars else 0.0)


def fleurs_norm_rates(references_norm: list[str], hyps: list[str]) -> tuple[float, float] | None:
    """(WER, CER) against the FLEURS "normalized" column used as-is (see the module docstring).
    None when the dataset has no such column, or when a reference is empty (not a FLEURS split)."""
    if not references_norm or any(not r.strip() for r in references_norm):
        return None
    return _rates([" ".join(r.split()) for r in references_norm], [normalize(h) for h in hyps])


def whisper_basic_rates(refs: list[str], hyps: list[str]) -> tuple[float, float] | None:
    """(WER, CER) with Whisper's BasicTextNormalizer on both sides. None without transformers."""
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    except ImportError:
        return None
    basic = BasicTextNormalizer()
    return _rates([basic(r) for r in refs], [basic(h) for h in hyps])


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
    import numpy as np
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


def transcribe_dataset(model_id: str, ds, language: str, batch_size: int, device: str, max_samples: int | None,
                       max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    if is_ctranslate2(model_id):
        return transcribe_ct2(model_id, ds, language, device, max_samples)
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
            ids = model.generate(feats.to(device, dtype), language=language, task="transcribe", max_new_tokens=max_new_tokens)
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
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="generation limit per clip (Transformers path only). Keep the default for the FLEURS "
                         "test split, the reference number; use 440 for validation_open")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_from_disk(str(args.data))[args.split]
    refs, hyps, audio_s, wall = transcribe_dataset(args.model, ds, args.language, args.batch_size, device, args.max_samples,
                                                 args.max_new_tokens)
    wer = word_error_rate(refs, hyps)
    cer = char_error_rate(refs, hyps)
    # Only on a pure FLEURS split: the open-data rows carry the ZLS normalised text in that column.
    has_fleurs_norm = "reference_norm" in ds.column_names and all(str(s).startswith("fleurs") for s in ds["source"])
    refs_norm = list(ds["reference_norm"][: len(refs)]) if has_fleurs_norm else []
    fleurs_norm = fleurs_norm_rates(refs_norm, hyps)
    whisper_basic = whisper_basic_rates(refs, hyps)
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
        # Secondary measures for comparisons with other model cards; definitions in the module docstring.
        "wer_fleurs_norm": round(100 * fleurs_norm[0], 2) if fleurs_norm else None,
        "cer_fleurs_norm": round(100 * fleurs_norm[1], 2) if fleurs_norm else None,
        "wer_whisper_basic": round(100 * whisper_basic[0], 2) if whisper_basic else None,
        "cer_whisper_basic": round(100 * whisper_basic[1], 2) if whisper_basic else None,
        "max_new_tokens": args.max_new_tokens,
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
