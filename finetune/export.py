#!/usr/bin/env python3
"""Merge the LoRA adapter (if any), save a plain Transformers model, convert it to
CTranslate2 int8 for faster-whisper, and optionally push both to the Hub.

    python export.py --run runs/turbo-lora
    python export.py --run runs/turbo-lora --card results/model_card.md \
        --push SitrakaForler/whisper-large-v3-turbo-lb --push-ct2 SitrakaForler/whisper-large-v3-turbo-lb-ct2

The -ct2 repo gets a card derived from the main one: library_name ctranslate2, no model-index
(its WER was not measured separately) and a faster-whisper usage section.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

CT2_USAGE = """## Usage

This repository holds the CTranslate2 int8 export of [{main}](https://huggingface.co/{main}), for
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It does not load with Transformers.

```python
from faster_whisper import WhisperModel

model = WhisperModel("{ct2}", device="cpu", compute_type="int8")
segments, info = model.transcribe("meeting.wav", language="lb", vad_filter=True)
for s in segments:
    print(f"[{{s.start:.1f}}-{{s.end:.1f}}] {{s.text}}")
```

The WER of this int8 export was not measured separately: the figures on this card were measured
with the Transformers model {main}. VAD was not used in the benchmark.

"""


def ct2_card(card: str, main_repo: str, ct2_repo: str) -> str:
    """Model card for the CTranslate2 repo, derived from the main (Transformers) card."""
    if not card.startswith("---\n"):
        raise ValueError("model card has no YAML front matter")
    _, front, body = card.split("---\n", 2)
    front = front.replace("library_name: transformers", "library_name: ctranslate2")
    # The model-index results were measured with the Transformers model, not with this export.
    front = re.sub(r"(?ms)^model-index:\n(?:[ \t]+.*\n|\n)*", "", front)
    front = front.replace("tags:\n", "tags:\n  - ctranslate2\n", 1)
    body = re.sub(r"(?ms)^## Usage\n.*?(?=^## )", CT2_USAGE.format(main=main_repo, ct2=ct2_repo), body)
    body = body.replace(f"# {main_repo.split('/')[-1]}\n", f"# {ct2_repo.split('/')[-1]}\n", 1)
    return f"---\n{front}---\n{body}"


def merge(run: Path) -> Path:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def push(folder: Path, repo_id: str, card_text: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    if card_text:
        (folder / "README.md").write_text(card_text, encoding="utf-8")
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

    card = args.card.read_text(encoding="utf-8") if args.card else None
    if args.push_ct2 and card and not args.push:
        raise SystemExit("--push-ct2 with --card needs --push too: the ct2 card links to the main repo")
    merged = merge(args.run)
    ct2 = to_ct2(merged, args.quantization)
    if args.push:
        push(merged, args.push, card)
    if args.push_ct2:
        push(ct2, args.push_ct2, ct2_card(card, args.push, args.push_ct2) if card else None)


if __name__ == "__main__":
    main()
