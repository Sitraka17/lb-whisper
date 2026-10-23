"""Meeting notes LU: audio in, minutes out, with open models and no cloud API.

    python app.py                      # http://127.0.0.1:7860
    HF_TOKEN=hf_... python app.py      # + speaker diarization (pyannote, gated weights)
    OLLAMA_MODEL=qwen2.5:3b python app.py   # + LLM summary via a local Ollama

The same file is the Hugging Face Space. Uploaded audio and generated files are deleted within an
hour (Gradio cache and OUT_ROOT below); for sensitive recordings run it on your own hardware (Docker).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import diarize, export, merge, summarize, transcribe  # noqa: E402

HERE = Path(__file__).parent
MODELS = yaml.safe_load((HERE / "models.yaml").read_text(encoding="utf-8"))["models"]
MODEL_BY_LABEL = {m["label"]: m for m in MODELS}
LANGS = {"auto": None, "lb": "lb", "fr": "fr", "de": "de", "en": "en"}
DIARIZATION_ON = diarize.available()
SUMMARY_BACKEND = summarize.backend()
# Generated files live here and are purged with the same policy as Gradio's cache (delete_cache below):
# checked every 10 minutes, anything older than an hour is removed.
OUT_ROOT = Path(tempfile.gettempdir()) / "meeting-notes"
PURGE_EVERY_S, MAX_AGE_S = 600, 3600
PRIVACY = (
    "Uploaded audio and generated files are deleted within an hour. "
    "For sensitive recordings, run the Docker image on your own hardware."
)

INTRO = f"""
### Meeting notes LU
Upload a recording (meeting, WhatsApp voice note, interview). You get a transcript with speakers,
a structured summary, and files to download. Everything is computed in this container with open
models; no API call to any cloud provider. {PRIVACY}

- Transcription: Whisper fine-tuned for Luxembourgish (`lb`); zero-shot Whisper models in the list for other languages
- Speakers: pyannote community-1: {"**on**" if DIARIZATION_ON else "off on this instance (needs a Hugging Face token for the gated weights)"}
- Summary: `{SUMMARY_BACKEND}` back-end {"(no LLM; key sentences and keyword-spotted actions)" if SUMMARY_BACKEND == "extractive" else "(local LLM, in this container)"}
"""


def purge_old_outputs(root: Path = OUT_ROOT, max_age_s: float = MAX_AGE_S) -> None:
    """Remove per-request output folders older than max_age_s."""
    if not root.is_dir():
        return
    cutoff = time.time() - max_age_s
    for folder in root.iterdir():
        try:
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except FileNotFoundError:
            pass


def _purge_loop() -> None:
    while True:
        time.sleep(PURGE_EVERY_S)
        purge_old_outputs()


threading.Thread(target=_purge_loop, name="purge-outputs", daemon=True).start()


def run(audio_path, model_label, language, want_speakers, num_speakers, want_summary, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")
    model = MODEL_BY_LABEL[model_label]
    progress(0.1, desc="Transcribing…")
    segments, meta = transcribe.transcribe(audio_path, model["repo"], language=LANGS.get(language))
    meta["model"] = model["repo"]
    seg_dicts = [s.as_dict() for s in segments]

    turns = []
    if want_speakers and DIARIZATION_ON:
        progress(0.6, desc="Finding speakers…")
        turns = diarize.diarize(audio_path, int(num_speakers) if num_speakers else None)
    blocks = merge.relabel_speakers(merge.group_turns(merge.assign_speakers(seg_dicts, turns)))

    summary, summary_md = {}, None
    if want_summary and blocks:
        progress(0.8, desc="Summarising…")
        plain = "\n".join(f"{b['speaker']}: {b['text']}" for b in blocks)
        summary = summarize.summarize(plain)
        summary_md = summarize.to_markdown(summary, meta.get("language") or "en")

    md = export.to_markdown(blocks, summary_md, meta)
    purge_old_outputs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="notes-", dir=OUT_ROOT))
    files = {
        out_dir / "meeting-notes.md": md,
        out_dir / "meeting-notes.srt": export.to_srt(blocks),
        out_dir / "meeting-notes.json": export.to_json(blocks, summary, meta),
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    transcript_md = "\n\n".join(f"**[{export.hms(b['start'])}] {b['speaker']}** {b['text']}" for b in blocks) or "_No speech detected._"
    info = f"Language `{meta.get('language')}` (p={meta.get('language_probability')}), {export.hms(meta.get('duration', 0))}, model `{model['repo']}`"
    return transcript_md, summary_md or "_Summary off._", info, [str(p) for p in files]


with gr.Blocks(title="Meeting notes LU", analytics_enabled=False, delete_cache=(PURGE_EVERY_S, MAX_AGE_S)) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            model_dd = gr.Dropdown(choices=list(MODEL_BY_LABEL), value=MODELS[0]["label"], label="Model")
            lang_dd = gr.Dropdown(choices=list(LANGS), value="lb", label="Language")
            with gr.Row():
                speakers_cb = gr.Checkbox(value=DIARIZATION_ON, interactive=DIARIZATION_ON, label="Identify speakers")
                n_speakers = gr.Number(value=None, precision=0, label="Number of speakers (optional)")
            summary_cb = gr.Checkbox(value=True, label="Structured summary")
            go = gr.Button("Transcribe", variant="primary")
        with gr.Column(scale=2):
            info_md = gr.Markdown()
            summary_out = gr.Markdown(label="Summary")
            transcript_out = gr.Markdown(label="Transcript")
            downloads = gr.File(label="Download (.md / .srt / .json)", file_count="multiple")
    go.click(run, [audio, model_dd, lang_dd, speakers_cb, n_speakers, summary_cb], [transcript_out, summary_out, info_md, downloads])
    gr.Markdown(f"Privacy: {PRIVACY} `docker run -p 7860:7860 meeting-notes-lu`")

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)))
