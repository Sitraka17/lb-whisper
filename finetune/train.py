#!/usr/bin/env python3
"""Fine-tune Whisper on the dataset produced by prepare_data.py.

    python train.py --config configs/turbo-lora.yaml
    python train.py --config configs/small-full.yaml --max-train-samples 2000   # smoke test

Everything that matters is in the YAML: base model, LoRA or not, batch sizes, steps.
Resumable (`--resume`, or `--auto-resume` which also accepts a first start), logs to TensorBoard, writes the runtime, the GPU name and the dev WER
history to <output_dir>/train_metrics.json (read by fill_card.py). The adapter (or the full model) lands in
<output_dir>/final; export.py merges and converts it.
"""

from __future__ import annotations

import argparse
import io
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


def decode_audio(audio: dict):
    """16 kHz mono float32 samples of an undecoded `Audio` cell ({"bytes": ..., "path": ...})."""
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"], dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16_000:
        import librosa

        samples = librosa.resample(samples, orig_sr=rate, target_sr=16_000)
    return samples


@dataclass
class OnTheFlyCollator(DataCollatorSpeechSeq2SeqWithPadding):
    """Same batch as the collator above, but the log-mel features are computed here, per batch.

    Whisper pads every clip to 30 s, so a stored feature matrix is 128 x 3000 float32 = 1.5 MB per
    clip whatever its length. Fine for the 2,502 FLEURS clips, not for tens of thousands of short
    open-data clips on a Colab disk. With `data.on_the_fly_features: true` the dataset keeps the
    compressed audio and the token ids only. The attention mask is returned because SpecAugment
    needs it to keep its time masks inside the real audio."""

    return_attention_mask: bool = False

    def __call__(self, features):
        arrays = [decode_audio(f["audio"]) for f in features]
        extracted = self.processor.feature_extractor(
            arrays, sampling_rate=16_000, return_tensors="pt", return_attention_mask=self.return_attention_mask
        )
        batch = {"input_features": extracted.input_features}
        if self.return_attention_mask:
            batch["attention_mask"] = extracted.attention_mask
        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def last_complete_checkpoint(out_dir: Path) -> str | None:
    """Newest checkpoint-N folder that was written to the end, or None.

    The Trainer writes trainer_state.json last, so a folder without it is a checkpoint that a
    disconnect cut in the middle: it is skipped and the one before is used."""
    numbered = []
    for path in Path(out_dir).glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            numbered.append((int(suffix), path))
    for _, path in sorted(numbered, reverse=True):
        if (path / "trainer_state.json").is_file():
            return str(path)
    return None


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

    data_cfg = cfg.get("data", {})
    if data_cfg.get("on_the_fly_features"):
        return prepare_dataset_on_the_fly(dsd, processor, cfg, max_train)

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


def prepare_dataset_on_the_fly(dsd, processor: WhisperProcessor, cfg: dict, max_train: int | None):
    """Token ids now, log-mel features in the collator. Only the splits training needs are touched:
    train, validation (FLEURS dev) and, when the dataset has one, the open-data validation split."""
    from datasets import Audio

    filters = cfg["filters"]
    max_seconds = float(filters["max_audio_seconds"])
    max_tokens = int(filters["max_label_tokens"])
    data_cfg = cfg.get("data", {})
    open_split = data_cfg.get("open_validation_split", "validation_open")
    limits = {
        "train": max_train,
        "validation": cfg["train"].get("max_eval_samples"),
        open_split: data_cfg.get("max_eval_samples_open"),
    }

    def tokenize(batch):
        return {"labels": processor.tokenizer(batch["sentence"]).input_ids}

    out = {}
    for split, limit in limits.items():
        if split not in dsd:
            continue
        ds = dsd[split]
        if "duration" not in ds.column_names:
            raise SystemExit(f"[train] split {split} has no duration column: rebuild the dataset with prepare_data.py")
        without_length = sum(1 for seconds in ds["duration"] if not seconds or seconds <= 0)
        if without_length:
            # The length filter below reads this column; a zero would silently remove the clip.
            raise SystemExit(
                f"[train] split {split}: {without_length} of {len(ds)} clips have no duration. Rebuild the dataset "
                "with the current prepare_data.py (it reads the length of --extra audio that has no \"duration\" field)"
            )
        if limit:
            ds = ds.shuffle(seed=42).select(range(min(int(limit), len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds.column_names if c not in ("audio", "duration")],
                    desc=f"tokenize {split}")
        before = len(ds)
        # Whisper cuts the audio at 30 s: a longer clip would be trained against a transcript it cannot contain.
        ds = ds.filter(lambda seconds, labels: seconds <= max_seconds and len(labels) <= max_tokens,
                       input_columns=["duration", "labels"])
        print(f"[train] {split}: {len(ds)} clips, {sum(ds['duration']) / 3600:.2f} h ({before - len(ds)} dropped by the filters)")
        out[split] = ds
    return out


def build_model(cfg: dict):
    processor = WhisperProcessor.from_pretrained(cfg["base_model"], language=cfg["language"], task=cfg["task"])
    lora = cfg.get("lora", {})
    if lora.get("enabled"):
        model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    else:
        # Full fine-tuning: every weight is trained, so the weights must be float32 even when the
        # checkpoint is stored in half precision (large-v3-turbo is). Mixed precision then comes
        # from fp16 / bf16 in the training arguments, not from the weights.
        model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"], torch_dtype=torch.float32)
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    augment = cfg.get("spec_augment", {})
    if augment.get("enabled"):
        # SpecAugment (Park et al., 2019) as implemented in transformers: active in training mode only.
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = float(augment.get("mask_time_prob", 0.05))
        model.config.mask_time_length = int(augment.get("mask_time_length", 10))
        model.config.mask_feature_prob = float(augment.get("mask_feature_prob", 0.0))
        model.config.mask_feature_length = int(augment.get("mask_feature_length", 10))

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
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume from the last complete checkpoint if there is one, start from scratch otherwise")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model = build_model(cfg)
    dsd = load_from_disk(cfg["data_dir"])
    data = prepare_dataset(dsd, processor, cfg, args.max_train_samples)

    data_cfg = cfg.get("data", {})
    if data_cfg.get("on_the_fly_features"):
        collator = OnTheFlyCollator(processor, model.config.decoder_start_token_id,
                                    return_attention_mask=bool(cfg.get("spec_augment", {}).get("enabled")))
    else:
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

    # FLEURS dev and the open-data validation split are evaluated separately, never pooled: the
    # Trainer then reports eval_fleurs_wer and eval_open_wer, and the best checkpoint is chosen on
    # FLEURS dev so that v1 and v2 are selected the same way.
    open_split = data_cfg.get("open_validation_split", "validation_open")
    eval_dataset = data.get("validation")
    metric_prefix = ""
    if data_cfg.get("on_the_fly_features") and eval_dataset is not None and open_split in data:
        eval_dataset = {"fleurs": data["validation"], "open": data[open_split]}
        metric_prefix = "eval_fleurs_"

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
        metric_for_best_model=(metric_prefix + ("wer" if eval_generate else "loss")) if metric_prefix
        else ("wer" if eval_generate else "eval_loss"),
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(tcfg.get("seed", 42)),
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "linear"),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
    )

    trainer = CastingSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_generate else None,
        processing_class=processor.feature_extractor,
    )
    resume: bool | str | None = args.resume or None
    if args.auto_resume:
        resume = last_complete_checkpoint(out_dir)
        print(f"[train] resuming from {resume}" if resume else "[train] no checkpoint yet, starting from the base model")
    result = trainer.train(resume_from_checkpoint=resume)
    state = trainer.state
    metrics = {
        # After a resume, train_runtime is the time of the last session only, not of the whole run.
        "resumed_from": resume if isinstance(resume, str) else bool(resume),
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss", result.training_loss),
        "global_step": state.global_step,
        "epochs": state.epoch,
        # Clips per split after the filters: with the eval_*_runtime rows of log_history, enough to
        # separate training time from evaluation time (the v2 notebook does it for its estimate).
        "train_clips": len(data["train"]),
        "eval_clips": {name: len(ds) for name, ds in eval_dataset.items()} if isinstance(eval_dataset, dict)
        else (len(eval_dataset) if eval_dataset is not None else 0),
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
