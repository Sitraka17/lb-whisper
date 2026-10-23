#!/usr/bin/env python3
"""Fine-tune Whisper on the dataset produced by prepare_data.py.

    python train.py --config configs/turbo-lora.yaml
    python train.py --config configs/small-full.yaml --max-train-samples 2000   # smoke test

Everything that matters is in the YAML: base model, LoRA or not, batch sizes, steps.
Resumable (`--resume`), logs to TensorBoard, writes the runtime, the GPU name and the dev WER
history to <output_dir>/train_metrics.json (read by fill_card.py). The adapter (or the full model) lands in
<output_dir>/final; export.py merges and converts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent))
from lb_text import normalize, word_error_rate  # noqa: E402


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # The tokenizer prepends <|startoftranscript|>; the model adds it itself at decode time.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prepare_dataset(dsd, processor: WhisperProcessor, cfg: dict, max_train: int | None):
    filters = cfg["filters"]
    max_seconds = float(filters["max_audio_seconds"])
    max_tokens = int(filters["max_label_tokens"])

    def featurize(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    # Review note: 8.5 h is tiny, and FLEURS has few speakers per language. The risk is a model that
    # learns their voices rather than the language. Cheap thing to try next run, before hunting data:
    # model.config.apply_spec_augment = True with mask_time_prob around 0.05. One line, one run.
    def keep(length, labels):
        return length <= max_seconds and len(labels) <= max_tokens

    out = {}
    for split, ds in dsd.items():
        if split == "train" and max_train:
            ds = ds.shuffle(seed=42).select(range(min(max_train, len(ds))))
        if split == "validation" and cfg["train"].get("max_eval_samples"):
            ds = ds.shuffle(seed=42).select(range(min(int(cfg["train"]["max_eval_samples"]), len(ds))))
        ds = ds.map(featurize, remove_columns=ds.column_names, num_proc=1, desc=f"featurize {split}")
        ds = ds.filter(keep, input_columns=["input_length", "labels"])
        out[split] = ds
    return out


def build_model(cfg: dict):
    processor = WhisperProcessor.from_pretrained(cfg["base_model"], language=cfg["language"], task=cfg["task"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora = cfg.get("lora", {})
    if lora.get("enabled"):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias="none",
            ),
        )
        model.print_trainable_parameters()
    return processor, model


class CastingSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that feeds the encoder in the dtype of its weights during evaluation.

    Recent transformers load whisper-large-v3-turbo in fp16. Training runs under autocast, but
    `generate` during evaluation does not, so float32 log-mel features reach an fp16 conv layer
    ("Input type (float) and bias type (c10::Half) should be the same").
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        features = inputs.get("input_features")
        if features is not None:
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            dtype = base.model.encoder.conv1.weight.dtype
            if features.dtype != dtype:
                inputs = {**inputs, "input_features": features.to(dtype)}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-train-samples", type=int, default=None, help="smoke-test on a subset")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model = build_model(cfg)
    dsd = load_from_disk(cfg["data_dir"])
    data = prepare_dataset(dsd, processor, cfg, args.max_train_samples)

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100.0 * word_error_rate(refs, hyps)}

    eval_generate = bool(tcfg.get("eval_generate", True))
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=int(tcfg["warmup_steps"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        fp16=bool(tcfg.get("fp16", False)),
        bf16=bool(tcfg.get("bf16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=tcfg.get("eval_strategy", "steps"),
        eval_steps=int(tcfg["eval_steps"]),
        save_strategy=tcfg.get("eval_strategy", "steps"),
        save_steps=int(tcfg["save_steps"]),
        logging_steps=int(tcfg["logging_steps"]),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        predict_with_generate=eval_generate,
        generation_max_length=int(tcfg.get("generation_max_length", 225)),
        load_best_model_at_end=True,
        metric_for_best_model="wer" if eval_generate else "eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(tcfg.get("seed", 42)),
        dataloader_num_workers=2,
    )

    trainer = CastingSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data.get("validation"),
        data_collator=collator,
        compute_metrics=compute_metrics if eval_generate else None,
        processing_class=processor.feature_extractor,
    )
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    state = trainer.state
    metrics = {
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss", result.training_loss),
        "global_step": state.global_step,
        "epochs": state.epoch,
        "best_model_checkpoint": state.best_model_checkpoint,
        "best_metric": state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "log_history": state.log_history,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    final = out_dir / "final"
    trainer.save_model(str(final))      # adapter only when LoRA, full weights otherwise
    processor.save_pretrained(str(final))
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"[train] done, weights in {final}")


if __name__ == "__main__":
    main()
