#!/usr/bin/env python3
"""Fill model_card_template.md with the numbers produced by train.py, evaluate.py and benchmark.py.

    python fill_card.py --run runs/turbo-lora --data data/fleurs_lb --out results/model_card.md

Inputs:
- <run>/train_metrics.json            written by train.py (runtime, GPU, best checkpoint, dev WER history)
- results/<merged path>.test.json     written by evaluate.py on the full test split
- results/benchmark.json              written by benchmark.py (one row per model x condition)
- the dataset on disk                 train / dev sizes, train hours, distinct sentences in the subset

Nothing is typed by hand. The script stops with an error if an input is missing or if any
__PLACEHOLDER__ is left in the card.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = "openai/whisper-large-v3-turbo"
UNILUX = "unilux/whisper-medium-v1-luxembourgish"
ZLS = "ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish"
# Rows of the comparison table, in order. None = this model (the merged run folder).
TABLE_MODELS = [
    (BASE, "openai/whisper-large-v3-turbo (zero-shot)"),
    ("openai/whisper-small", "openai/whisper-small (zero-shot)"),
    (None, "this model"),
    (UNILUX, UNILUX),
    (ZLS, ZLS),
]
REQUIRED = [BASE, None, UNILUX, ZLS]
CLEAN, OPUS = "clean", "whatsapp-16k"
PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")


class CardError(RuntimeError):
    pass


def result_name(model: str) -> str:
    """File stem used by evaluate.py for a model id or path (keep in sync with evaluate.py)."""
    return re.sub(r"[^\w.-]+", "_", model.strip("/"))


def _read_json(path: Path, what: str):
    if not path.exists():
        raise CardError(f"missing {what}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row(rows: list[dict], model: str, condition: str) -> dict | None:
    return next((r for r in rows if r["model"] == model and r["condition"] == condition), None)


def _backend(row: dict) -> str:
    if row.get("backend"):
        return "faster-whisper" if row["backend"] == "faster-whisper" else "Transformers"
    return "faster-whisper" if row["model"] == ZLS else "Transformers"


def bench_table(rows: list[dict], ours: str) -> tuple[str, dict]:
    """Markdown comparison table and the clean WER of each required model."""
    gpu = next((r.get("gpu") for r in rows if r.get("gpu")), None)
    speed_head = f"s / audio-min ({_short_gpu(gpu)})" if gpu else "s / audio-min (GPU)"
    lines = [
        f"| model | backend | WER % clean | WER % Opus 16 kbit/s | CER % clean | {speed_head} |",
        "|---|---|---:|---:|---:|---:|",
    ]
    clean_wer = {}
    for model, label in TABLE_MODELS:
        key = ours if model is None else model
        clean, opus = _row(rows, key, CLEAN), _row(rows, key, OPUS)
        if clean is None:
            if model in REQUIRED:
                raise CardError(f"benchmark.json has no clean row for {key}")
            continue
        if opus is None and model in REQUIRED:
            raise CardError(f"benchmark.json has no {OPUS} row for {key}")
        clean_wer[model] = clean["wer"]
        opus_wer = f"{opus['wer']:.1f}" if opus else "n/a"
        lines.append(
            f"| {label} | {_backend(clean)} | {clean['wer']:.1f} | {opus_wer} | "
            f"{clean['cer']:.1f} | {clean['seconds_per_audio_minute']:.1f} |"
        )
    return "\n".join(lines), clean_wer


def comparison_sentence(clean_wer: dict, n: int) -> str:
    ours, uni, zls = clean_wer[None], clean_wer[UNILUX], clean_wer[ZLS]
    where = f"On the {n}-utterance comparison subset below"
    if ours > uni and ours > zls:
        return (f"{where} it does not beat the uni.lu or ZLS models: both are more accurate "
                f"(WER {uni:.1f} and {zls:.1f} against {ours:.1f}), after much more training audio.")
    if ours < uni and ours < zls:
        return f"{where} it has a lower WER than the uni.lu and ZLS models ({ours:.1f} against {uni:.1f} and {zls:.1f})."
    better, worse = ("ZLS", "uni.lu") if ours < zls else ("uni.lu", "ZLS")
    return f"{where} it has a lower WER than the {better} model and a higher or equal WER than the {worse} model."


def dev_history(log_history: list[dict]) -> str:
    evals = [e for e in log_history if "eval_wer" in e]
    if not evals:
        raise CardError("train_metrics.json has no eval_wer entry in log_history")
    lines = ["| step | epoch | dev WER % |", "|---:|---:|---:|"]
    lines += [f"| {e['step']} | {e.get('epoch', 0):.1f} | {e['eval_wer']:.1f} |" for e in evals]
    return "\n".join(lines)


def _short_gpu(name: str | None) -> str:
    return (name or "GPU").replace("Tesla ", "").replace("NVIDIA ", "")


def build_values(*, train_metrics: dict, test_result: dict, rows: list[dict], ours: str,
                 stats: dict, max_eval_samples: int | None) -> dict:
    if test_result.get("split") != "test" or test_result.get("n") != stats["n_test"]:
        raise CardError(
            f"test result covers {test_result.get('n')} {test_result.get('split')} utterances, "
            f"expected the full test split ({stats['n_test']}): rerun evaluate.py without --max-samples"
        )
    table, clean_wer = bench_table(rows, ours)
    n_subset = _row(rows, ours, CLEAN)["n"]
    if n_subset != stats["n_subset"]:
        raise CardError(f"subset sizes disagree: benchmark n={n_subset}, dataset stats n={stats['n_subset']}")
    runtime = train_metrics.get("train_runtime")
    if not runtime:
        raise CardError("train_metrics.json has no train_runtime")
    n_dev = stats["n_dev"]
    return {
        "__WER_TEST__": f"{test_result['wer']:.2f}",
        "__CER_TEST__": f"{test_result['cer']:.2f}",
        "__N_TEST__": str(stats["n_test"]),
        "__N_SUBSET__": str(n_subset),
        "__N_SUBSET_DISTINCT__": str(stats["n_subset_distinct"]),
        "__BENCH_TABLE__": table,
        "__COMPARISON__": comparison_sentence(clean_wer, n_subset),
        "__N_TRAIN__": f"{stats['n_train']:,}",
        "__H_TRAIN__": f"{stats['h_train']:.1f}",
        "__N_DEV__": f"{n_dev:,}",
        "__N_DEV_EVAL__": str(min(max_eval_samples, n_dev) if max_eval_samples else n_dev),
        "__EPOCHS__": f"{train_metrics['epochs']:g}",
        "__STEPS__": str(train_metrics["global_step"]),
        "__GPU__": _short_gpu(train_metrics.get("gpu")),
        "__GPU_HOURS__": f"{runtime / 3600:.1f}",
        "__BEST_CKPT__": Path(train_metrics.get("best_model_checkpoint") or "n/a").name,
        "__DEV_HISTORY__": dev_history(train_metrics.get("log_history", [])),
    }


def render(template: str, values: dict) -> str:
    text = template
    for key, value in values.items():
        text = text.replace(key, value)
    left = sorted(set(PLACEHOLDER.findall(text)))
    if left:
        raise CardError(f"unfilled placeholders: {left}")
    return text


def dataset_stats(data_dir: Path, n_subset: int) -> dict:
    from datasets import load_from_disk

    dsd = load_from_disk(str(data_dir))
    seconds = 0.0
    for ex in dsd["train"].select_columns(["audio"]):
        seconds += len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
    test_sentences = dsd["test"]["sentence"]
    return {
        "n_train": len(dsd["train"]),
        "h_train": seconds / 3600,
        "n_dev": len(dsd["validation"]),
        "n_test": len(dsd["test"]),
        "n_subset": n_subset,
        "n_subset_distinct": len(set(test_sentences[:n_subset])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/turbo-lora"))
    ap.add_argument("--config", type=Path, default=Path("configs/turbo-lora.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/fleurs_lb"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--template", type=Path, default=Path("model_card_template.md"))
    ap.add_argument("--out", type=Path, default=Path("results/model_card.md"))
    args = ap.parse_args()

    import yaml

    ours = str(args.run / "merged")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = _read_json(args.results / "benchmark.json", "benchmark results (run benchmark.py)")
    test_result = _read_json(args.results / f"{result_name(ours)}.test.json",
                             f"full test result (run evaluate.py --model {ours} --split test)")
    train_metrics = _read_json(args.run / "train_metrics.json", "training metrics (run train.py)")
    ours_clean = _row(rows, ours, CLEAN)
    if ours_clean is None:
        raise CardError(f"benchmark.json has no clean row for {ours}")
    stats = dataset_stats(args.data, ours_clean["n"])
    values = build_values(train_metrics=train_metrics, test_result=test_result, rows=rows, ours=ours,
                          stats=stats, max_eval_samples=cfg["train"].get("max_eval_samples"))
    text = render(args.template.read_text(encoding="utf-8"), values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in values.items() if "\n" not in v}, indent=2))
    print(f"[card] wrote {args.out}; every placeholder filled")


if __name__ == "__main__":
    try:
        main()
    except CardError as err:
        raise SystemExit(f"[card] error: {err}")
