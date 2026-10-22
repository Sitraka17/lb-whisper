# lb-whisper: Whisper fine-tuned for Luxembourgish, and a sovereign meeting-notes demo

Two deliverables, one repository:

| Folder | What it is | Where it ends up |
|---|---|---|
| `finetune/` | Reproducible recipe to fine-tune Whisper on **open** Luxembourgish speech data, evaluate it, and publish it with an honest model card | A model repo on the Hugging Face Hub |
| `space/` | « Meeting notes souverain » : audio in, transcript + speakers + structured summary out, **no cloud API anywhere in the pipeline** | A Gradio Space on the Hub, and a Docker image you can run on your own machine |
| `tests/` | Pure-Python tests for the parts that do not need a GPU (text normalisation, speaker merge, exports, extractive summary) | GitHub Actions |

Nothing in this project runs on the author's laptop: training runs on Colab / Kaggle (free GPU tiers),
inference runs on a CPU Space or in Docker.

## Why this project is honest about what exists

Luxembourgish ASR is **not** empty ground. Before writing a line of code, know the prior art:

- **LuxASR (University of Luxembourg, Peter Gilles et al.)**: Whisper fine-tuned on 150 h+ of annotated
  Luxembourgish, tiny/base/small/medium open-sourced as `unilux/whisper-*-v1-luxembourgish`; the large
  flagship stays closed. Papers: *LUX-ASR* (wav2vec2, 35 h, WER 14.5) and *ASRLUX* (ICPhS 2023, 67 h,
  WER 9.5 wav2vec2 / 12.1 Whisper).
- **ZLS (Zenter fir d'Lëtzebuerger Sprooch)**: `ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish`, and the
  open dataset *« Dataset for finetuning a Luxembourgish Automatic Speech Recognition »* on data.public.lu.
- Individual fine-tunes: `pgilles/whisper-medium-v2-lb_cased_01` (no longer public: 401 on 22/10/2026), `Tun-Wellens/whisper-medium-lb-*`,
  `Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h`.
- **Google FLEURS** (`google/fleurs`, config `lb_lu`): read Luxembourgish, CC-BY-4.0, about 2,500 / 400 / 930
  utterances in train / dev / test. It is the open training and test data of this project.
- **Mozilla Common Voice has no Luxembourgish data**: `lb` is not in any scripted-speech release from 17.0 to
  27.0 (checked on 22/10/2026 in the official release statistics).
- Other Hugging Face sets labelled Luxembourgish are CC-BY-NC-ND (no derivatives) or have no licence: not used.

So the claim is **not** « first Luxembourgish Whisper ». The claim, and what the model card measures, is:

1. **A fully open, reproducible recipe**: only CC0 / CC-BY data, every number regenerable from
   `finetune/` on a free GPU tier, every hyper-parameter in a YAML file.
2. **A small model that runs on CPU**: `whisper-small` full fine-tune and `large-v3-turbo` + LoRA,
   exported to CTranslate2 int8 for `faster-whisper`. Prior work targets GPU-class medium/large models.
3. **A benchmark nobody has published**: base Whisper vs. uni.lu vs. ZLS vs. this model, on the FLEURS
   `lb_lu` test split *and* on degraded WhatsApp-style audio, with WER **and** processing time per minute of audio.
4. **A deployable, privacy-first application on top**: the Space.

That is the story you can defend in an interview: what you chose, what you measured, what you learnt.

## Quick start

```bash
# 1. Fine-tune (Colab / Kaggle, see finetune/README.md)
pip install -r finetune/requirements.txt
python finetune/prepare_data.py --out data/fleurs_lb   # FLEURS lb_lu from the Hub, no login
python finetune/train.py --config finetune/configs/turbo-lora.yaml
python finetune/evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test
python finetune/export.py --run runs/turbo-lora --push <hf-user>/whisper-large-v3-turbo-lb

# 2. Run the sovereign meeting-notes app locally (CPU is fine)
pip install -r space/requirements.txt
python space/app.py                      # http://127.0.0.1:7860
# or
docker build -t meeting-notes-lu space && docker run -p 7860:7860 meeting-notes-lu

# 3. Tests (no GPU, no model download)
pip install pytest pyyaml && pytest tests/
```

## Data licences

| Source | Licence | Use here |
|---|---|---|
| Google FLEURS `lb_lu` | CC-BY-4.0 | train / dev / test |
| ZLS fine-tuning dataset (data.public.lu) | **Conflicting** (checked 22/10/2026): data.public.lu says MIT (licence field « Other (Open) »), the linked Hugging Face card `Lemswasabi/luxembourgish-asr-rtl-lu` says CC-BY-NC-ND-4.0, and the audio is RTL broadcast material fetched from Google Drive | **not used** for the published weights until ZLS confirms the licence in writing; `finetune/make_manifest.py` stays ready |
| RTL / uni.lu corpora | copyright, research only | **not used** |

The model card lists exactly which splits trained the published weights.

## Publishing

- Model: `huggingface.co/SitrakaForler/whisper-large-v3-turbo-lb` (+ `-ct2`), weights Apache-2.0 like upstream Whisper.
- Space: `huggingface.co/spaces/SitrakaForler/meeting-notes-lu` (this `space/` folder pushed as-is).
- This code: MIT.
