#!/usr/bin/env python3
"""Fill model_card_template.md with the numbers produced by benchmark.py and train.py.

    python fill_card.py --run runs/turbo-lora --data data/fleurs_lb --out results/model_card.md

Reads results/benchmark.json (one row per model x condition), the trainer state of the run and the
dataset on disk. Every placeholder left unfilled is reported, so nothing ships with __X__ in it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def wer_of(rows: list[dict], model_prefix: str, condition: str) -> str:
    for r in rows:
        if r["model"].startswith(model_prefix) and r["condition"] == condition:
            return f"{r['wer']:.1f}"
    return "n/a"


def speed_of(rows: list[dict], model_prefix: str) -> str:
    for r in rows:
        if r["model"].startswith(model_prefix) and r["condition"] == "clean":
            return f"{r['seconds_per_audio_minute']:.1f}"
    return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/turbo-lora"))
    ap.add_argument("--config", type=Path, default=Path("configs/turbo-lora.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/fleurs_lb"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--template", type=Path, default=Path("model_card_template.md"))
    ap.add_argument("--out", type=Path, default=Path("results/model_card.md"))
    args = ap.parse_args()

    rows = json.loads((args.results / "benchmark.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ours = str(args.run / "merged")

    from datasets import load_from_disk

    dsd = load_from_disk(str(args.data))
    train_rows = len(dsd["train"]) + len(dsd["validation"])
    seconds = 0.0
    for split in ("train", "validation"):
        for ex in dsd[split].select_columns(["audio"]):
            seconds += len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]

    gpu, runtime_h = "GPU", None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0).replace("Tesla ", "")
    except ImportError:
        pass
    states = sorted(args.run.glob("**/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    for state in reversed(states):
        for entry in reversed(json.loads(state.read_text())["log_history"]):
            if "train_runtime" in entry:
                runtime_h = entry["train_runtime"] / 3600
                break
        if runtime_h:
            break

    n_test = next((r["n"] for r in rows if r["model"] == ours and r["condition"] == "clean"), "n/a")
    values = {
        "__WER_BEFORE__": wer_of(rows, "openai/whisper-large-v3-turbo", "clean"),
        "__WER_BEFORE_DEG__": wer_of(rows, "openai/whisper-large-v3-turbo", "whatsapp-16k"),
        "__WER_AFTER__": wer_of(rows, ours, "clean"),
        "__WER_AFTER_DEG__": wer_of(rows, ours, "whatsapp-16k"),
        "__WER_UNILUX__": wer_of(rows, "unilux/", "clean"),
        "__WER_UNILUX_DEG__": wer_of(rows, "unilux/", "whatsapp-16k"),
        "__WER_ZLS__": wer_of(rows, "ZLSCompLing/", "clean"),
        "__WER_ZLS_DEG__": wer_of(rows, "ZLSCompLing/", "whatsapp-16k"),
        "__SPEED__": speed_of(rows, ours),
        "__N_TEST__": str(n_test),
        "__N_TRAIN__": str(train_rows),
        "__H_TRAIN__": f"{seconds / 3600:.1f}",
        "__EPOCHS__": str(cfg["train"]["num_train_epochs"]),
        "__GPU__": gpu,
        "__GPU_HOURS__": f"{runtime_h:.1f}" if runtime_h else "n/a",
    }
    text = args.template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(key, value)
    left = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps(values, indent=2))
    print(f"[card] wrote {args.out}" + (f"; unfilled: {left}" if left else "; every placeholder filled"))


if __name__ == "__main__":
    main()
