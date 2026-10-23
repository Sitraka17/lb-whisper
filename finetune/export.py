#!/usr/bin/env python3
"""Merge the LoRA adapter (if any), save a plain Transformers model, convert it to
CTranslate2 int8 for faster-whisper, and optionally push both to the Hub.

    python export.py --run runs/turbo-lora
    python export.py --run runs/turbo-lora --push SitrakaForler/whisper-large-v3-turbo-lb
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def merge(run: Path) -> Path:
    cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    final, merged = run / "final", run / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    processor = WhisperProcessor.from_pretrained(final)
    if cfg.get("lora", {}).get("enabled"):
        from peft import PeftModel

        base = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
        model = PeftModel.from_pretrained(base, final).merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(final)
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(merged, safe_serialization=True)
    processor.save_pretrained(merged)
    # Recent transformers write a single processor_config.json; CTranslate2 and faster-whisper still
    # expect the feature extractor in its own preprocessor_config.json, so write it explicitly.
    processor.feature_extractor.save_pretrained(merged)
    processor.tokenizer.save_pretrained(merged)
    print(f"[export] merged model in {merged}")
    return merged


def to_ct2(merged: Path, quantization: str) -> Path:
    from ctranslate2.converters import TransformersConverter

    out = merged.parent / f"ct2-{quantization}"
    if out.exists():
        shutil.rmtree(out)
    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if (merged / f).exists()]
    TransformersConverter(str(merged), copy_files=copy).convert(
        str(out), quantization=quantization
    )
    print(f"[export] CTranslate2 model in {out}")
    return out


def push(folder: Path, repo_id: str, card: Path | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    if card and card.exists():
        shutil.copy(card, folder / "README.md")
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, commit_message=f"upload {folder.name}")
    print(f"[export] pushed {folder} to https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--quantization", default="int8", help="ct2 quantization (int8, int8_float16, float16)")
    ap.add_argument("--push", metavar="REPO_ID", help="push the merged model to this Hub repo")
    ap.add_argument("--push-ct2", metavar="REPO_ID", help="push the CTranslate2 model to this Hub repo")
    ap.add_argument("--card", type=Path, help="filled-in model card to upload as README.md")
    args = ap.parse_args()

    merged = merge(args.run)
    ct2 = to_ct2(merged, args.quantization)
    if args.push:
        push(merged, args.push, args.card)
    if args.push_ct2:
        push(ct2, args.push_ct2, args.card)


if __name__ == "__main__":
    main()
