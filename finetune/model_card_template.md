---
language:
  - lb
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-large-v3-turbo
datasets:
  - google/fleurs
metrics:
  - wer
  - cer
tags:
  - whisper
  - luxembourgish
  - letzebuergesch
  - lora
model-index:
  - name: whisper-large-v3-turbo-lb
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          name: FLEURS (lb_lu, test)
          type: google/fleurs
          config: lb_lu
          split: test
        metrics:
          - type: wer
            value: __WER_AFTER__
            name: WER
---

# whisper-large-v3-turbo-lb

Whisper large-v3-turbo fine-tuned on Luxembourgish (Lëtzebuergesch) with LoRA, on **open data only**,
with a reproducible recipe: <https://github.com/Sitraka17/lb-whisper>.

## Results

Word error rate on FLEURS `lb_lu` test (__N_TEST__ utterances), text normalised with the
project's Luxembourgish normaliser (apostrophes inside words kept, case and punctuation dropped):

| model | WER % clean | WER % WhatsApp-degraded | s / audio-min (__GPU__, Transformers) |
|---|---:|---:|---:|
| openai/whisper-large-v3-turbo (zero-shot) | __WER_BEFORE__ | __WER_BEFORE_DEG__ | n/a |
| **this model** | **__WER_AFTER__** | __WER_AFTER_DEG__ | __SPEED__ |
| unilux/whisper-medium-v1-luxembourgish | __WER_UNILUX__ | __WER_UNILUX_DEG__ | n/a |
| ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish | __WER_ZLS__ | __WER_ZLS_DEG__ | n/a |

Every number above comes from `finetune/benchmark.py`; rerun it to reproduce.

## Training data

| source | licence | split | utterances | hours |
|---|---|---|---:|---:|
| Google FLEURS `lb_lu` | CC-BY-4.0 | train + dev | __N_TRAIN__ | __H_TRAIN__ |

No RTL, no uni.lu, no ZLS, no scraped audio. The FLEURS test split was never seen in training. The uni.lu and ZLS models in the table may have seen FLEURS sentences during their own training: their numbers are indicative.

## Training

- LoRA r=32, alpha=64, dropout 0.05 on q/k/v/out projections and both MLP layers of every block
- __EPOCHS__ epochs, lr 1e-4, effective batch 16, fp16, gradient checkpointing, on one __GPU__ (__GPU_HOURS__ GPU-hours)
- full configuration: `finetune/configs/turbo-lora.yaml`

## Usage

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="SitrakaForler/whisper-large-v3-turbo-lb")
asr("meeting.wav", generate_kwargs={"language": "luxembourgish", "task": "transcribe"})
```

CPU with faster-whisper (CTranslate2 int8 export in `SitrakaForler/whisper-large-v3-turbo-lb-ct2`):

```python
from faster_whisper import WhisperModel
model = WhisperModel("SitrakaForler/whisper-large-v3-turbo-lb-ct2", device="cpu", compute_type="int8")
segments, info = model.transcribe("meeting.wav", language="lb", vad_filter=True)
```

## Limitations

- FLEURS is read speech (Wikipedia sentences), one sentence at a time, and small (a few hours): spontaneous conversation, overlapping speakers
  and code-switching (Luxembourgish ↔ French ↔ German in the same sentence) are under-represented.
- Dialectal variation and the *Eifeler Regel* (n-deletion) are learnt only as far as the data shows them.
- Numbers, dates and proper nouns follow the written conventions of the FLEURS sentences.
- Evaluated on one test set; a model that wins here can lose on your audio. Run `benchmark.py` on yours.

## Citation

If you use this model, cite FLEURS (Conneau et al., 2022) and link this card.
