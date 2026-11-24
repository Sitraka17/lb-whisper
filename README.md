# lb-whisper: an open recipe to fine-tune Whisper for Luxembourgish, and a meeting-notes app that runs locally

Two deliverables, one repository:

| Folder | What it is | Where it ends up |
|---|---|---|
| `finetune/` | Reproducible recipe to fine-tune Whisper on open Luxembourgish speech data, evaluate it, and publish it with a model card filled from the measured results | A model repo on the Hugging Face Hub |
| `space/` | Meeting notes LU: audio in, transcript, speakers and structured summary out, with open models and no cloud API in the pipeline | A Gradio Space on the Hub, and a Docker image you can run on your own machine |
| `bench/` | LB-Real: the protocol, consent form and scoring code of a bench on real-life audio. The test set is not recorded yet | [bench/PROTOCOL.md](bench/PROTOCOL.md) |
| `docs/` | A survey of who lists Luxembourgish for transcription, read in the vendors' own documentation | [docs/landscape.md](docs/landscape.md) |
| `tests/` | Pure-Python tests for the parts that do not need a GPU (text normalisation, model card, speaker merge, exports, extractive summary, open-data build, bench scoring) | `.github/workflows/tests.yml` |

Training runs on Colab or Kaggle (free GPU tiers); inference runs on a CPU Space or in Docker.

## Where this stands

Status on 24/11/2026.

**lb-whisper is not the most accurate Luxembourgish recogniser.** The published model (v1) was
trained on 8.46 h of read speech and scores 30.15 % WER and 8.82 % CER on the full FLEURS `lb_lu`
test split (934 utterances). Others do better, and anyone who needs a transcript today should
try them before this one:

| System | What its authors publish | Where |
|---|---|---|
| LuxASR, University of Luxembourg | trained on 150+ hours, word error rate below 10 % | [luxasr.uni.lu](https://luxasr.uni.lu/), read on 24/11/2026 |
| Sproochmaschinn (Schreifmaschinn), ZLS | no accuracy figure in the application; its open model `ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish` was trained on about 75 h according to its model card | [sproochmaschinn.lu](https://sproochmaschinn.lu/), read on 24/11/2026 |
| BuzzASR | 25.41 % WER and 8.55 % CER on FLEURS, "normalized" | model card of [BuzzASR/luxembourgish](https://huggingface.co/BuzzASR/luxembourgish), read on 24/11/2026 |

These figures are self-reported, on different test sets or with different text normalisation, so
they cannot be ranked against each other or against the 30.15 % above. The one comparison I ran
myself is on the same 300 FLEURS test utterances with one normaliser: the open uni.lu and ZLS
models have a lower WER there than this model (table on the model card). BuzzASR is not in that
table yet; `finetune/evaluate.py` now writes a `wer_fleurs_norm` field that approximates its
protocol, and v1 has not been scored with it.

What this repository offers instead:

- **Open weights, code and data.** The weights are MIT, the code is MIT, and every training set
  is public under a licence that allows a published model (table below).
- **Reproducible.** Data preparation, training, evaluation and the model card are scripts in this
  repository; no number is typed by hand.
- **Runs where the audio is.** With faster-whisper on a laptop CPU, in Docker, or in the browser:
  the recording is not sent to a server.

Who else transcribes Luxembourgish, and which meeting tools do not list the language:
[docs/landscape.md](docs/landscape.md), a survey of official documentation (no audio was sent to
anyone). How I intend to compare the systems on real-life audio, including this one, and why a
reader should distrust a bench run by the author of a contestant:
[bench/PROTOCOL.md](bench/PROTOCOL.md). No comparative result exists yet.

**v2: the data is ready, the training has not started.** `finetune/build_open_data.py` turns the
open releases of November 2026 into 22,071 clips, 62.04 h: 59.94 h for training and 126.4 min of
whole recordings kept for validation. With the FLEURS train split (8.46 h) that is 68.40 h of
training audio against 8.46 h for v1: 58.64 h under CC0 1.0 and 9.76 h under CC BY 4.0 (FLEURS
included). Another 9.65 h (ZLS recordings of LOD example sentences that also exist with the LOD
male voice, itself CC BY-NC) are held out until the ZLS confirms their licence; with them the
total would be 78.05 h.
The detail and what was left out are in [finetune/DATA.md](finetune/DATA.md), the training plan in
[finetune/README.md](finetune/README.md#v2-fleurs--open-data). There is no v2 model and no v2
number.

Thanks to the Zenter fir d'Lëtzebuerger Sprooch (ZLS), which published the training data of its
Schreifmaschinn on 11/11/2026 and the audio of the LOD example sentences on 05/11/2026. Without
those releases there is no v2. Attribution:

- Zenter fir d'Lëtzebuerger Sprooch (ZLS), *Schreifmaschinn-Trainingsdaten - CC0*, 2026, CC0 1.0,
  <https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc0/>. Sources named in the
  release: RTL Lëtzebuerg, Chambre des Députés, radio 100,7, Speakeasy, Jemp Schuster, Happy
  Hippo, ZLS.
- Zenter fir d'Lëtzebuerger Sprooch (ZLS), *Schreifmaschinn-Trainingsdaten (CC-BY)*, 2026,
  <https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc-by/>:
  "© Konterbont Magasinn – Autisme Luxembourg a.s.b.l., CC BY 4.0"
  (<https://creativecommons.org/licenses/by/4.0/>).
- Zenter fir d'Lëtzebuerger Sprooch (ZLS), *Audioopname vun den LOD-Beispillssätz (Fraestëmm)*,
  Lëtzebuerger Online Dictionnaire (lod.lu), 2026, CC0 1.0.

## Prior work

Luxembourgish speech recognition already exists. The main open models and data:

- **University of Luxembourg (Peter Gilles et al.)**: Whisper fine-tuned on about 150 h of Luxembourgish
  according to its model card, tiny/base/small/medium published as `unilux/whisper-*-v1-luxembourgish`
  (licence open-mdw). Papers: *LUX-ASR* (wav2vec2) and *ASRLux: Automatic Speech Recognition for the
  Low-Resource Language Luxembourgish* (ICPhS 2023).
- **ZLS (Zenter fir d'Lëtzebuerger Sprooch)**: `ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish`,
  published in CTranslate2 format only, MIT, trained on about 75 h according to its model card; the
  dataset *Dataset for finetuning a Luxembourgish Automatic Speech Recognition* on data.public.lu;
  and, since November 2026, the training data of its Schreifmaschinn under CC0 1.0 and CC BY 4.0
  (see [finetune/DATA.md](finetune/DATA.md)).
- **BuzzASR**: `BuzzASR/luxembourgish` on the Hugging Face Hub, with a paper (arXiv:2609.09554).
- Individual fine-tunes: `Tun-Wellens/whisper-medium-lb-*`,
  `Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h`. `pgilles/whisper-medium-v2-lb_cased_01`
  is no longer public (checked 22/10/2026).
- **Mozilla Common Voice** has no Luxembourgish data in any release from 17.0 to 27.0
  (checked 22/10/2026 in the official release statistics).
- Other Hugging Face sets labelled Luxembourgish are CC-BY-NC-ND (no derivatives) or have no licence: not used.

Data used by the published model (v1): **Google FLEURS** (`google/fleurs`, config `lb_lu`), read
Luxembourgish, CC-BY-4.0: train 2,502 utterances / 8.46 h, dev 408 / 1.12 h, test 934 / 2.73 h.
The FLEURS dev and test splits stay the reference for v2.

## What this repository adds

1. **A reproducible fine-tuning recipe on CC-BY-4.0 data only**: Whisper large-v3-turbo + LoRA on
   FLEURS `lb_lu`, every hyper-parameter in a YAML file. The 3-epoch run took about 2.1 h on a Colab T4.
2. **A CTranslate2 int8 export** of the fine-tuned model, for `faster-whisper` on CPU.
3. **A side-by-side benchmark** of base Whisper, the uni.lu and ZLS models and this model on the same
   300 FLEURS test utterances, on clean audio and re-encoded as 16 kbit/s Opus (WhatsApp-like), with
   the same references and normaliser. The uni.lu and ZLS models are more accurate on this benchmark.
4. **A meeting-notes demo** (`space/`) that runs offline in Docker.
5. **A build script for the open data of November 2026** (`finetune/build_open_data.py`): clips of
   at most 30 s, one manifest with source and licence per clip, whole recordings held out for
   validation. Prepared for v2, not trained on yet.
6. **A bench protocol written before the test set exists** (`bench/`), with the scoring code.

## Results

The results (full-test WER and CER, the benchmark table, training time) are on the model card,
`finetune/model_card_template.md` filled by `finetune/fill_card.py`. The numbers are regenerated by
`finetune/benchmark.py` and `finetune/evaluate.py`; none is typed by hand.

Both scripts also report `wer_n_rule`, the same WER with Eifeler Regel (n-rule) disagreements
forgiven. It is a diagnostic, not the headline number: it tells how much of the error is a missing
or extra final n rather than a misheard word.

What comes next, with a target number per step: [ROADMAP.md](ROADMAP.md).

## Quick start

```bash
# 1. Fine-tune (Colab / Kaggle, see finetune/README.md)
pip install -r finetune/requirements.txt
cd finetune
python prepare_data.py --out data/fleurs_lb   # FLEURS lb_lu from the Hub, no login
python train.py --config configs/turbo-lora.yaml
python export.py --run runs/turbo-lora
python evaluate.py --model runs/turbo-lora/merged --data data/fleurs_lb --split test
cd ..

# 2. Run the meeting-notes app locally (CPU is fine)
pip install -r space/requirements.txt
python space/app.py                      # http://127.0.0.1:7860
# or
docker build -t meeting-notes-lu space && docker run -p 7860:7860 meeting-notes-lu

# 3. Tests (no GPU, no model download)
pip install pytest pyyaml && pytest -q tests
```

## Data licences

| Source | Licence | Use here |
|---|---|---|
| Google FLEURS `lb_lu` | CC-BY-4.0 | v1 and v2: train (8.46 h) / dev / test |
| ZLS, *Schreifmaschinn-Trainingsdaten - CC0* (11/11/2026) | CC0 1.0 | v2: 41.61 h train, 105.8 min validation |
| ZLS, *Schreifmaschinn-Trainingsdaten (CC-BY)* (11/11/2026) | CC BY 4.0, "© Konterbont Magasinn – Autisme Luxembourg a.s.b.l., CC BY 4.0" | v2: 1.30 h train, 5.5 min validation |
| ZLS, LOD example sentences, female voice (05/11/2026) | CC0 1.0 | v2: 17.02 h train, 15.1 min validation |
| ZLS recordings of LOD example sentences inside the CC0 set | CC0 1.0 in the ZLS release; the same sentences with the LOD male voice are CC BY-NC 4.0 on data.public.lu | held out (9.65 h) until the ZLS confirms the licence |
| LOD example sentences, male voice | CC BY-NC 4.0 | not used, audio never downloaded |
| ZLS fine-tuning dataset (older set on data.public.lu) | Conflicting (checked 22/10/2026): MIT on data.public.lu; CC-BY-NC-ND-4.0 and RTL.lu copyright on the linked Hugging Face card | not used |
| RTL / uni.lu corpora | research only | not used |

The v2 hours are those of `finetune/build_open_data.py` (run of 24/11/2026). v2 is not trained yet;
the published model has seen FLEURS only.

## Publishing

- Model: `SitrakaForler/whisper-large-v3-turbo-lb` and the CTranslate2 export
  `SitrakaForler/whisper-large-v3-turbo-lb-ct2` on the Hugging Face Hub, published on 23/10/2026.
  Weights MIT, like the base model `openai/whisper-large-v3-turbo`.
- Browser demo: [huggingface.co/spaces/SitrakaForler/lb-whisper-demo](https://huggingface.co/spaces/SitrakaForler/lb-whisper-demo),
  `web/index.html` on a static Space. The model runs on WebGPU in the visitor's browser and the audio is
  never uploaded. Weights: `SitrakaForler/whisper-large-v3-turbo-lb-onnx`, made by `finetune/export_onnx.py`.
- Meeting-notes app: not hosted. Since October 2026 the Hugging Face Hub only hosts Gradio and Docker
  Spaces for PRO accounts, so the app runs where it was meant to run anyway: on your machine, with Docker
  or `python space/app.py` (see Quick start).
- This code: MIT.
