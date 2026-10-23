# finetune/: the recipe

Runs on a free Colab T4 (16 GB) or Kaggle (P100 / T4 x2). `colab_finetune.ipynb` runs the steps
below in order. Commands are run from this folder.

## 1. Data

```bash
# Google FLEURS, Luxembourgish (lb_lu), CC-BY-4.0, no login needed:
python prepare_data.py --out data/fleurs_lb
# Extra data only if its licence allows a published model (see make_manifest.py for the JSONL format):
# python prepare_data.py --extra data/other/manifest.jsonl --out data/fleurs_lb
# Put extra rows in split "train" (the default) so dev and test stay pure FLEURS.
```

`prepare_data.py` parses the TSVs and tarballs itself (no dataset loading script), unescapes the HTML
entities of the FLEURS sentences, and writes a `DatasetDict` with `train` (2,502 utterances, 8.46 h),
`validation` (408, 1.12 h) and `test` (934, 2.73 h) plus a `stats.json`.

## 2. Baseline first

Measure zero-shot Whisper before training, otherwise "after" means nothing:

```bash
python evaluate.py --model openai/whisper-large-v3-turbo --data data/fleurs_lb --split test
python evaluate.py --model openai/whisper-small           --data data/fleurs_lb --split test
```

## 3. Train

```bash
python train.py --config configs/turbo-lora.yaml --max-train-samples 300   # smoke test
python train.py --config configs/turbo-lora.yaml                           # the real run
tensorboard --logdir runs
```

Checkpoints are saved and evaluated on dev once per epoch (156 steps) in `runs/<name>/`; restart
with `--resume` after a disconnect. Keep `runs/` on Google Drive (the notebook links it) or lose it
with the VM. `train.py` writes `runs/<name>/train_metrics.json` (runtime, GPU, dev WER history).

`configs/small-full.yaml` (whisper-small, full fine-tune) is provided but has never been trained here.

## 4. Export, evaluate, benchmark

```bash
python export.py --run runs/turbo-lora                     # merged/ + ct2-int8/
python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test
python benchmark.py --data data/fleurs_lb --max-samples 300 --models \
    openai/whisper-large-v3-turbo openai/whisper-small \
    unilux/whisper-medium-v1-luxembourgish ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish \
    runs/turbo-lora/merged
```

The benchmark uses the first 300 test utterances, clean and re-encoded as 16 kbit/s mono Opus.
The ZLS model is CTranslate2 only and runs through faster-whisper (float16, greedy, no VAD).
Then fill the model card from the result files and publish:

```bash
python fill_card.py --run runs/turbo-lora --data data/fleurs_lb --out results/model_card.md
python export.py --run runs/turbo-lora --card results/model_card.md \
    --push SitrakaForler/whisper-large-v3-turbo-lb --push-ct2 SitrakaForler/whisper-large-v3-turbo-lb-ct2
```

`fill_card.py` stops with an error if a result file is missing or a placeholder is left.

## Decisions

| Choice | Why | Alternative considered |
|---|---|---|
| large-v3-turbo + LoRA | fits a 16 GB T4 in fp16; 4-layer decoder | full fine-tune of medium (more memory) |
| own normaliser instead of Whisper's | Whisper's splits `d'Kanner` in two tokens and inflates WER on correct output | jiwer's default transforms (same problem) |
| Opus 16 kbit/s test condition | voice notes and calls are compressed; codec loss is closer to that than added noise | adding noise |
| FLEURS only, test split untouched | the only open Luxembourgish speech set we found (22/10/2026) whose licence allows a published model; the test split is public | ZLS dataset (licence conflicting: MIT on data.public.lu, CC-BY-NC-ND-4.0 and RTL.lu copyright on the Hugging Face card); RTL / uni.lu (research only) |
| CTranslate2 int8 export | runs with faster-whisper on CPU, as the Space does | ONNX |

Mozilla Common Voice has no Luxembourgish data in any release from 17.0 to 27.0.

## Budget

| Step | Time |
|---|---|
| data preparation (download + extract about 12 h of FLEURS wav) | not measured, CPU only |
| turbo-lora, 3 epochs (468 steps) | about 2.1 h, measured on a Colab T4 |
| small-full | not measured |
| benchmark, 5 models x 2 conditions x 300 clips | not measured |

A free Colab session fits the turbo-lora run.
