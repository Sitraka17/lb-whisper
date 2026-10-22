# finetune/: the recipe

Runs on a free Colab T4 (16 GB) or Kaggle (P100 / T4 ×2). Nothing here touches your laptop
except the final `git push` of results.

## 1. Data

```bash
# Google FLEURS, Luxembourgish (lb_lu), CC-BY-4.0, no login needed:
python prepare_data.py --out data/fleurs_lb
# Extra data only if its licence allows a published model (see make_manifest.py for the JSONL format):
# python prepare_data.py --extra data/other/manifest.jsonl --out data/fleurs_lb
# Test and validation stay pure FLEURS, so the WER is comparable with everybody else's.
```

`prepare_data.py` parses the TSVs and tarballs itself (no dataset loading script), writes a
`DatasetDict` with `train` / `validation` / `test` and a `stats.json`.

## 2. Baseline first

Measure zero-shot Whisper **before** training, otherwise « after » means nothing:

```bash
python evaluate.py --model openai/whisper-large-v3-turbo --data data/fleurs_lb --split test
python evaluate.py --model openai/whisper-small           --data data/fleurs_lb --split test
```

## 3. Train

```bash
python train.py --config configs/turbo-lora.yaml --max-train-samples 500   # 10-minute smoke test
python train.py --config configs/turbo-lora.yaml                           # the real run
python train.py --config configs/small-full.yaml                           # the CPU candidate
tensorboard --logdir runs
```

Colab disconnects: checkpoints land every 250 steps in `runs/<name>/`, restart with `--resume`.
Keep `runs/` on Google Drive (`--output_dir` in the YAML) or lose it with the VM.

## 4. Evaluate, benchmark, export

```bash
python export.py --run runs/turbo-lora                     # merged/ + ct2-int8/
python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test
python benchmark.py --data data/fleurs_lb --max-samples 300 --models \
    openai/whisper-large-v3-turbo openai/whisper-small \
    unilux/whisper-medium-v1-luxembourgish ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish \
    runs/turbo-lora/merged runs/small-full/merged
```

`results/benchmark.md` is the table of the model card. Fill `model_card_template.md`
(placeholders `__LIKE_THIS__`), then:

```bash
python export.py --run runs/turbo-lora --card results/model_card.md \
    --push SitrakaForler/whisper-large-v3-turbo-lb --push-ct2 SitrakaForler/whisper-large-v3-turbo-lb-ct2
```

## Decisions worth being able to explain

| Choice | Why | Alternative considered |
|---|---|---|
| large-v3-turbo + LoRA as the main run | best accuracy per GPU-hour; 4-layer decoder is fast at inference | full fine-tune of medium (2× the memory, slower on CPU) |
| whisper-small full fine-tune as second run | the one that runs in real time on a laptop CPU | small + LoRA (less capacity to shift the language prior) |
| own normaliser instead of Whisper's | Whisper's splits `d'Kanner` in two tokens and inflates WER on correct output | jiwer's default transforms (same problem) |
| WhatsApp-degraded test condition | the real input of a Luxembourg SME is a voice note, not a studio clip | adding noise (less representative than codec loss) |
| FLEURS only, test split untouched | the only open Luxembourgish speech set with a licence that allows a published model; the test split is public, anyone can compare | ZLS / RTL sets (CC-BY-NC-ND or research-only: unusable weights) |
| no RTL / uni.lu audio | licence is research-only; the published model must be usable commercially | more data, unusable weights |
| CTranslate2 int8 export | 3-4× faster on CPU than Transformers fp32, same WER within noise | ONNX (more work, no gain for Whisper) |

## Budget

| Step | T4 time |
|---|---|
| data preparation (download + decode ~30 h of mp3) | 20-30 min, CPU only |
| baseline evaluation, 2 models | 15 min |
| turbo-lora, 3 epochs | 6-9 h (stop early if dev WER is flat for 2 evaluations) |
| small-full, 5 epochs | 3-4 h |
| benchmark, 6 models × 2 conditions × 300 clips | 45 min |

Free Colab gives ~3-4 h of T4 per session: run turbo-lora over two sessions with `--resume`,
or use Kaggle (30 GPU-hours / week) for the long run.
