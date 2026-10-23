---
title: Meeting notes LU (local)
emoji: 🎙️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
short_description: Meeting transcript, speakers, summary; open models
---

# Meeting notes LU (local)

Audio in → transcript with speakers → structured summary → `.md` / `.srt` / `.json`.
Open models only, inference on CPU, **no call to any cloud API** at any step.

Intended for recordings in Luxembourgish (meetings, calls, voice notes), and also usable in
French, German and English with the zero-shot models. The only measured figure is the FLEURS WER of
the fine-tuned model, on read speech; see its model card
(`SitrakaForler/whisper-large-v3-turbo-lb`). Accuracy on real meetings, overlapping speakers and
code-switching has not been measured.

| Step | Component | Runs where |
|---|---|---|
| Transcription | faster-whisper with a Whisper fine-tuned for Luxembourgish (see the `finetune/` folder of the repo) | this container |
| Speakers | pyannote `speaker-diarization-community-1` (open weights, gated download) | this container |
| Summary | Qwen2.5-1.5B-Instruct (GGUF, llama.cpp, in-process); Ollama if configured; extractive fallback with no LLM if the model cannot load | this container |

## Run it on your own machine

```bash
docker build -t meeting-notes-lu .
docker run -p 7860:7860 -v $HOME/models:/models meeting-notes-lu
# with speakers and a local LLM:
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx -e OLLAMA_MODEL=qwen2.5:3b -e OLLAMA_HOST=http://host.docker.internal:11434 meeting-notes-lu
```

Or without Docker: `pip install -r requirements.txt && python app.py`.

## Configuration

| Variable | Effect |
|---|---|
| `HF_TOKEN` | enables diarization (accept the pyannote model terms on the Hub first) |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | summary through a local Ollama server |
| `LB_GGUF_REPO`, `LB_GGUF_FILE` or `LB_GGUF_PATH` | another GGUF than the default Qwen2.5-1.5B-Instruct Q4_K_M |
| `LB_LLM_MAX_TOKENS` | length cap of the summary (default 500) |
| `LB_SUMMARY_BACKEND` | force `extractive`, `ollama` or `llama_cpp` |
| `LB_MODEL_CACHE` | where converted CTranslate2 models are kept |

Models offered in the dropdown are listed in `models.yaml`.

## On the Space itself

The hosted demo processes your file in the Space's container. Uploaded audio and generated files
are deleted within an hour. For sensitive recordings, run the Docker image on your own hardware.
