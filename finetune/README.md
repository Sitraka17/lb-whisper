# finetune/: the recipe

Runs on a free Colab T4 (16 GB) or Kaggle (P100 / T4 x2). `colab_finetune.ipynb` runs the steps
below in order. Commands are run from this folder.

Sections 1 to 4 and the two tables after them describe v1 (FLEURS only), the model that is
published. [v2](#v2-fleurs--open-data) adds open data and has its own notebook, `colab_finetune_v2.ipynb`;
it is prepared and tested here, not trained yet.

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
| v1: FLEURS only, test split untouched | the only open Luxembourgish speech set I had found on 22/10/2026 whose licence allows a published model (the ZLS released its training data under CC0 on 11/11/2026, see DATA.md); the test split is public | ZLS dataset (licence conflicting: MIT on data.public.lu, CC-BY-NC-ND-4.0 and RTL.lu copyright on the Hugging Face card); RTL / uni.lu (research only) |
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

## v2: FLEURS + open data

Status on 24/11/2026: the code path is written and tested, the data preparation was run on the real
archives, **no v2 model has been trained**. Every WER in this section is either v1 or somebody
else's; there is no v2 number yet.

### Data

`build_open_data.py` writes `shard-NNN.tar` archives (16 kHz mono FLAC clips of at most 30 s, each
archive starting with the manifest of its own clips). `prepare_data.py --shards` reads them:

```bash
python prepare_data.py --shards /path/to/open-data-v2 --out data/open_v2
python prepare_data.py --shards /path/to/open-data-v2 --text-field text_normalised --out data/open_v2_norm
```

- `--text-field text` (default) trains on the verbatim transcripts; `text_normalised` on the ZLS
  standard-spelling version. Rows without a normalised text (the LOD sentences) fall back to the
  verbatim one; `stats.json` counts them.
- Only `manifest.jsonl` and `shard-*.tar` are read. The held-out manifest and archives that
  `build_open_data.py` writes in the same folder are never picked up.
- A row whose licence is not CC0-1.0 or CC-BY-4.0 stops the script.
- Splits: `train` = FLEURS train + open train; `validation` and `test` = FLEURS dev and test,
  untouched; `validation_open` = the open-data validation clips, on their own.

Dry run of 24/11/2026 on the three archives as they were at 16:00 that day (4.23 GB), read as a
folder that holds only the archives, then FLEURS from the Hub:

| Split | Clips | Hours | From |
|---|---:|---:|---|
| `train` | 24,069 | 68.40 | FLEURS train (2,502 / 8.46 h) + open data (21,567 / 59.94 h) |
| `validation` | 408 | 1.12 | FLEURS dev |
| `validation_open` | 504 | 2.11 | open data, whole recordings held out by `build_open_data.py` |
| `test` | 934 | 2.73 | FLEURS test |

Every figure of this section belongs to these archives. If `build_open_data.py` is run again, they
change; compare before trusting the tables (first 12 characters of `shasum -a 256`):

| File | Bytes | SHA-256 |
|---|---:|---|
| `shard-000.tar` | 1,895,065,600 | `806a134b9d8a` |
| `shard-001.tar` | 1,893,447,680 | `dded959626f5` |
| `shard-002.tar` | 440,760,320 | `ee92e8a602e8` |

Open train hours by licence in that run: 58.64 h CC0-1.0, 1.30 h CC-BY-4.0. The raw sources add
up to about 85 h (ZLS CC0 57.86 h and CC BY 1.46 h listed in their metadata, LOD female voice
17.29 h in its index, FLEURS train 8.46 h); 68.40 h reach `train` because `build_open_data.py`
drops fragments longer than 30 s, keeps 2.11 h for `validation_open` and holds out 9.65 h of ZLS
recordings of LOD sentences (its `stats.json` and `DATA.md` have the detail). The saved
dataset stores the FLAC bytes itself (6.6 GB on disk for the run above), so the extracted
clips are deleted afterwards unless `--keep-extracted` is given. Only the numbered sub-folders
that the script creates under `--extract-dir` are deleted; anything else in that folder stays.

### Training

```bash
python train.py --config configs/turbo-lora-v2.yaml --auto-resume    # (a) LoRA, same adapter as v1
python train.py --config configs/turbo-full-v2.yaml --auto-resume    # (b) every layer
```

| | `turbo-lora-v2` | `turbo-full-v2` |
|---|---|---|
| What is trained | LoRA r=32 on attention and MLP, 27,852,800 parameters | all 808,878,080 parameters |
| Learning rate | 1e-4 (as v1) | 1e-5 |
| Warm-up, evaluation interval | 500 steps, every 250 steps (v1: 100 steps, once per epoch of 156 steps): scaled to the longer run | 500 steps, every 500 steps |
| SpecAugment | off (same adapter and learning rate as v1) | on, time masks only (`mask_time_prob` 0.05); `spec_augment.enabled: false` turns it off |
| One checkpoint (weights + two AdamW moments, float32, computed from the parameter counts) | about 0.33 GB | about 9.7 GB |

What changed in `train.py` for v2, all behind config keys so that `turbo-lora.yaml` still runs as before:

- `data.on_the_fly_features: true`: the log-mel features are computed per batch. Whisper pads every
  clip to 30 s, so a stored feature matrix is 128 x 3000 float32 = 1.5 MB per clip; for 24,069
  clips that is about 37 GB of cache, against 6.6 GB for the compressed audio.
- FLEURS dev and `validation_open` are evaluated separately and logged as `eval_fleurs_wer` and
  `eval_open_wer`. The best checkpoint is the one with the lowest `eval_fleurs_wer`, so v1 and v2
  are selected on the same data.
- `--auto-resume`: starts from the base model if `runs/<name>/` has no checkpoint, from the newest
  complete one otherwise. A checkpoint without `trainer_state.json` (cut while being written) is skipped.
- Full fine-tuning loads the weights in float32 (the turbo checkpoint is stored in half precision).
- `generation_max_length: 448` in both v2 configs (225 in v1). Counted with the Whisper tokenizer on
  the manifest of 24/11/2026, special tokens included: 6 of the 504 `validation_open` transcripts are
  longer than 225 tokens (the longest has 244), so a limit of 225 would cut them and inflate
  `eval_open_wer`. No transcript exceeds `max_label_tokens: 448` (longest in train: 293), so that
  filter drops nothing.
- A clip without a duration stops `train.py` with a message instead of being filtered out silently.
  `prepare_data.py --extra` now reads the length from the audio file when the manifest has no
  `duration` field.

### Evaluation: one reference number, two comparison numbers

`evaluate.py` writes three word error rates. They are not interchangeable.

| Field | Reference | Hypothesis | Use |
|---|---|---|---|
| `wer`, `cer` | raw FLEURS sentence through `lb_text.normalize` | same | the reference number of this repository since v1 (30.15 % / 8.82 % on the 934 test utterances). Unchanged. |
| `wer_fleurs_norm`, `cer_fleurs_norm` | the `normalized` column of the FLEURS TSV, as-is | `lb_text.normalize` | comparison with BuzzASR |
| `wer_whisper_basic`, `cer_whisper_basic` | Whisper `BasicTextNormalizer` | same | comparison with cards that use the Whisper normaliser |

About the BuzzASR comparison. The model card of `BuzzASR/luxembourgish` (read on 24/11/2026)
reports 25.41 % WER and 8.55 % CER on FLEURS under the heading "Results (normalized CER / WER, %)"
and does not say what the normalisation is. The paper behind it (arXiv:2609.09554, data section)
says: "FLEURS provides normalized transcripts, which we use as-is." That is what `wer_fleurs_norm`
does on the reference side. The paper does not describe the hypothesis side for FLEURS, so the
hypothesis goes through this repository's normaliser, which keeps what the FLEURS column keeps
(lower case, apostrophes and hyphens inside words). Three caveats before putting the two numbers
in one sentence:

1. It is an approximation of their protocol, not a reproduction. Their usage snippet also decodes
   with `no_repeat_ngram_size=3` and `repetition_penalty=1.2`; `evaluate.py` decodes greedily without either.
2. The FLEURS column is not a clean target for a model that writes normal text: on the 934 test
   references, 818 are identical to `lb_text.normalize` of the raw sentence, and a hypothesis equal
   to the raw sentence still scores 1.53 % WER and 0.43 % CER (times such as `22.00`, leftover entities).
3. The v1 model has not been scored with `wer_fleurs_norm` yet: it needs one `evaluate.py` run on a
   GPU with a dataset rebuilt by the current `prepare_data.py` (older datasets lack the column, and
   the field is then `null`).

`evaluate.py` generates at most 225 new tokens per clip, as for every number published so far; that
default is not changed. For the open-data split pass `--max-new-tokens 440` (the notebook does),
otherwise the few transcripts longer than 225 tokens are cut. The value used is written in the
result file (`max_new_tokens`).

`BasicTextNormalizer` splits on the apostrophe (`D'Kanner` becomes `d kanner`): the 934 test
references count 20,508 words with it against 20,107 with `lb_text.normalize`, which is why it is
not the reference number here.

### v2 budget

Nothing here is measured on v2. The method, so that the numbers can be redone:

1. Whisper pads every clip to 30 s. The cost of a step follows the number of clips, not the hours.
2. v1 measured about 2.1 h on a Colab T4 for 3 epochs of 2,502 clips, dev evaluations included:
   2.1 x 3600 / 7,506 = about 1.0 s per clip.
3. v2 train is 24,069 clips. Three epochs = 72,207 clip passes = 4,515 steps of 16, so 72,207 s = 20.06 h.

| Run | GPU | Estimate | Based on |
|---|---|---|---|
| `turbo-lora-v2`, 3 epochs | T4 16 GB | about 20 h of training | 1.0 s per clip from v1. More evaluations than v1 (18 instead of 3, each on 400 + 300 clips) and longer transcripts are not in the figure: count 20 to 25 h |
| `turbo-lora-v2`, 3 epochs | L4 / A100 | not measured | run the smoke test below on that GPU |
| `turbo-full-v2`, 3 epochs | T4 16 GB | not measured, memory not verified | the float32 weights and AdamW moments alone are about 9.7 GB before any activation; plan for an L4 (24 GB) or an A100 |
| `turbo-full-v2`, 3 epochs | L4 / A100 | not measured | smoke test |

The free version of Colab runs a notebook "for at most 12 hours, depending on availability and your
usage patterns" (Colab FAQ, read on 24/11/2026), and GPU access is not guaranteed. So the LoRA run
on a free T4 is at least two sessions, more likely three, with a resume between them. A paid GPU
shortens it by a factor that I have not measured.

To replace the estimate by a measurement on the GPU you actually get: set `SMOKE_TEST = True` in
`colab_finetune_v2.ipynb`. It trains one epoch of 300 clips in its own folder (`runs/<config>-smoke`,
never resumed by the real run) with one evaluation in the middle, and the cell under the training
cell prints the estimate for the full run. Training and evaluation are timed apart, because their
proportions are not those of the real run: the smoke test evaluates 700 clips (400 FLEURS dev + 300
open) for 300 trained, the real LoRA run evaluates 700 for every 4,000 trained.

    hours = clip passes x (train_runtime - evaluation time) / clips trained
          + number of evaluations x time of one evaluation

`train_runtime` is measured inside `trainer.train()`, so model loading is not in it. What is left
in it is the start of the data loader and of the GPU, which a few hundred clips do not amortise:
the result stays on the high side. Then set `SMOKE_TEST = False` and run the training cell again.

Disk on the VM: the dry run needed 4.6 GB for the FLEURS download and extraction, about the size of
the archives for the clips extracted from them (deleted once the dataset is saved) and 6.6 GB for
the saved dataset. It took 4 min 33 s on a laptop with the archives on an external SSD; reading
them through a mounted Drive will be slower and is not measured.

Disk on Drive, where the notebook keeps the archives, `runs/` and `results/`. The sizes below are
computed from the parameter counts (4 bytes per float32 parameter), not measured:

| On Drive | `turbo-lora-v2` | `turbo-full-v2` |
|---|---:|---:|
| the three archives | 4.23 GB | 4.23 GB |
| two checkpoints (`save_total_limit: 2`, the best and the latest) | 0.7 GB | 19.4 GB |
| `final/` (adapter, or full float32 weights) | 0.1 GB | 3.2 GB |
| `merged/` written by `export.py` (1.6 GB if saved in half precision, 3.2 GB in float32) | 1.6 to 3.2 GB | 3.2 GB |
| `ct2-int8/` written by `export.py` (one byte per weight) | 0.8 GB | 0.8 GB |
| Total | 7.4 to 9.0 GB | about 31 GB |

The LoRA run fits the 15 GB of a free Google account only if little else is stored there. The
full fine-tune does not fit: it needs a paid Drive plan, or `runs/` on another disk. Once the
export is checked, the two `checkpoint-N` folders can be deleted by hand.

### If the Colab session is cut

1. Reconnect, check that a GPU is attached, run all cells of `colab_finetune_v2.ipynb` again. Nothing to edit.
2. The dataset is rebuilt on the new VM from the archives on Drive (the VM disk does not survive).
3. Training prints `resuming from runs/<name>/checkpoint-N` and continues with the same optimiser
   state, learning-rate schedule and best-checkpoint record. At most `save_steps` steps are lost
   (250 for LoRA, 500 for the full fine-tune).
4. After a resume, `train_metrics.json` gives the runtime of the last session only (`resumed_from`
   says where it started). Add the sessions up by hand before quoting a training time.
5. If saving fails because Drive is full, free space and run again; do not lower `save_total_limit`
   below 2 (the best and the latest).
