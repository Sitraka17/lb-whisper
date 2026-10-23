import json
from pathlib import Path

import pytest

import fill_card
from fill_card import BASE, UNILUX, ZLS, CardError, build_values, render, result_name

TEMPLATE = Path(__file__).resolve().parents[1] / "finetune" / "model_card_template.md"
OURS = "runs/turbo-lora/merged"
STATS = {"n_train": 2502, "h_train": 8.46, "n_dev": 408, "n_test": 934, "n_subset": 300, "n_subset_distinct": 113}


def _rows(ours_wer: float) -> list[dict]:
    rows = []
    for model, wer in [(BASE, 90.1), (UNILUX, 26.5), (ZLS, 28.9), (OURS, ours_wer)]:
        for cond, delta in [("clean", 0.0), ("whatsapp-16k", 5.0)]:
            rows.append({"model": model, "condition": cond, "n": 300, "wer": wer + delta, "cer": wer / 3,
                         "seconds_per_audio_minute": 1.5, "device": "cuda", "gpu": "Tesla T4",
                         "backend": "faster-whisper" if model == ZLS else "transformers"})
    return rows


def _fake_results(tmp_path: Path, ours_wer: float = 31.0) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmark.json").write_text(json.dumps(_rows(ours_wer)))
    (results / f"{result_name(OURS)}.test.json").write_text(json.dumps({"model": OURS, "split": "test", "n": 934, "wer": 30.0, "cer": 9.0}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "train_metrics.json").write_text(json.dumps({
        "train_runtime": 7560.0, "global_step": 468, "epochs": 3.0, "gpu": "Tesla T4",
        "best_model_checkpoint": "runs/turbo-lora/checkpoint-468", "best_metric": 30.0,
        "log_history": [{"step": 156, "epoch": 1.0, "eval_wer": 40.0}, {"step": 468, "epoch": 3.0, "eval_wer": 30.0}],
    }))
    return tmp_path


def _values(tmp_path: Path, **kw) -> dict:
    root = _fake_results(tmp_path, **kw)
    return build_values(
        train_metrics=json.loads((root / "run" / "train_metrics.json").read_text()),
        test_result=json.loads((root / "results" / f"{result_name(OURS)}.test.json").read_text()),
        rows=json.loads((root / "results" / "benchmark.json").read_text()),
        ours=OURS, stats=STATS, max_eval_samples=400,
    )


def test_result_name_matches_evaluate_py():
    assert result_name("runs/turbo-lora/merged") == "runs_turbo-lora_merged"


def test_card_is_fully_filled(tmp_path):
    card = render(TEMPLATE.read_text(encoding="utf-8"), _values(tmp_path))
    assert "__" not in card.replace("__init__", "")
    assert "value: 30.00" in card and "value: 9.00" in card
    assert "does not beat the uni.lu or ZLS models" in card
    assert "first 300 utterances" in card and "113 distinct sentences" in card
    assert "| this model | Transformers | 31.0 | 36.0 |" in card
    assert "| openai/whisper-small" not in card  # optional row absent from the fake benchmark
    assert "2.1 h of training" in card and "one T4" in card
    assert "**" not in card.split("## Results")[1].split("## Training data")[0]


def test_comparison_sentence_follows_the_numbers(tmp_path):
    card = render(TEMPLATE.read_text(encoding="utf-8"), _values(tmp_path, ours_wer=20.0))
    assert "lower WER than the uni.lu and ZLS models" in card
    assert "does not beat" not in card


def test_leftover_placeholder_fails():
    with pytest.raises(CardError, match="__NEW_NUMBER__"):
        render("WER __WER_TEST__ and __NEW_NUMBER__", {"__WER_TEST__": "30.0"})


def test_partial_test_result_fails(tmp_path):
    root = _fake_results(tmp_path)
    with pytest.raises(CardError, match="full test split"):
        build_values(
            train_metrics=json.loads((root / "run" / "train_metrics.json").read_text()),
            test_result={"split": "test", "n": 300, "wer": 30.0, "cer": 9.0},
            rows=json.loads((root / "results" / "benchmark.json").read_text()),
            ours=OURS, stats=STATS, max_eval_samples=400,
        )


def test_missing_benchmark_row_fails(tmp_path):
    rows = [r for r in _rows(31.0) if r["model"] != ZLS]
    with pytest.raises(CardError, match="ZLSCompLing"):
        fill_card.bench_table(rows, OURS)
