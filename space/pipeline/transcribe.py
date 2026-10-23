"""Transcription with faster-whisper (CTranslate2). Runs on CPU; GPU if present.

Models are declared in ../models.yaml. A Hub repo that is already a CTranslate2 model is
loaded as-is; a plain Transformers Whisper repo is converted to int8 on first use and cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(os.environ.get("LB_MODEL_CACHE", Path.home() / ".cache" / "lb-whisper"))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


def _is_ct2(folder: Path) -> bool:
    return (folder / "model.bin").exists()


def resolve_model(repo_or_path: str) -> str:
    """Return a path (or Hub id) that faster-whisper can load, converting if needed."""
    local = Path(repo_or_path)
    if local.is_dir():
        return str(local) if _is_ct2(local) else _convert(str(local), CACHE / f"ct2-{local.name}")
    from huggingface_hub import list_repo_files

    try:
        files = set(list_repo_files(repo_or_path))
    except Exception:
        return repo_or_path  # let faster-whisper raise a clear error
    if "model.bin" in files:
        return repo_or_path
    return _convert(repo_or_path, CACHE / ("ct2-" + repo_or_path.replace("/", "--")))


def _convert(src: str, dst: Path) -> str:
    if _is_ct2(dst):
        return str(dst)
    from ctranslate2.converters import TransformersConverter

    dst.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(src, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(str(dst), quantization="int8")
    return str(dst)


_MODELS: dict[str, object] = {}


def load(repo_or_path: str, device: str | None = None):
    from faster_whisper import WhisperModel

    key = repo_or_path
    if key not in _MODELS:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _MODELS[key] = WhisperModel(resolve_model(repo_or_path), device=device, compute_type=compute)
    return _MODELS[key]


# Review note: language=None lets Whisper guess, and on Luxembourgish it sometimes guesses "de"
# and starts writing German. In a meeting in Luxembourg people also switch to French mid-sentence.
# Forcing "lb" fixes the first problem and makes the second worse. Idea to test: detect per
# VAD chunk, keep lb unless fr/de wins clearly (probability > 0.8), and log every switch.
def transcribe(audio_path: str, model_id: str, language: str | None = None, beam_size: int = 5) -> tuple[list[Segment], dict]:
    model = load(model_id)
    raw, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = [
        Segment(
            start=float(s.start),
            end=float(s.end),
            text=s.text.strip(),
            words=[{"start": float(w.start), "end": float(w.end), "word": w.word} for w in (s.words or [])],
        )
        for s in raw
        if s.text.strip()
    ]
    meta = {"language": info.language, "language_probability": round(float(info.language_probability), 3), "duration": float(info.duration)}
    return segments, meta
