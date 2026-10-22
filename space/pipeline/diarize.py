"""Speaker diarization with pyannote (community-1 pipeline, open weights, gated download).

Needs: `pip install pyannote.audio>=4`, the model terms accepted on the Hub, and a token in
HF_TOKEN (read access is enough). Without a token the function returns an empty list and the
app degrades to a single-speaker transcript. All inference is local.
"""

from __future__ import annotations

import os

PIPELINE_ID = os.environ.get("LB_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
_PIPELINE = None


def available() -> bool:
    if not os.environ.get("HF_TOKEN"):
        return False
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


def _pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from pyannote.audio import Pipeline

        _PIPELINE = Pipeline.from_pretrained(PIPELINE_ID, token=os.environ["HF_TOKEN"])
        try:
            import torch

            if torch.cuda.is_available():
                _PIPELINE.to(torch.device("cuda"))
        except ImportError:
            pass
    return _PIPELINE


def diarize(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """Return [{"start", "end", "speaker"}] sorted by start, or [] when diarization is off."""
    if not available():
        return []
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = _pipeline()(audio_path, **kwargs)
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda t: t["start"])
