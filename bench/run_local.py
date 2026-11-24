#!/usr/bin/env python3
"""Run the open models of systems.yaml on the bench audio, one JSONL per system and VAD setting.

    python bench/run_local.py --refs refs.jsonl --out hyps/ --vad both
    python bench/run_local.py --audio audio/ --out hyps/ --systems lb-whisper-turbo zls-turbo --vad off

Every model gets the same audio, the same decoding settings and the same compute type (int8 on
CPU by default). Three engines, chosen per system in systems.yaml:
- faster-whisper: the Hub repository is already a CTranslate2 export;
- faster-whisper-converted: Transformers weights, converted locally to CTranslate2 with the
  bench compute type (needs torch and transformers for the conversion only);
- transformers: models that cannot be converted (replaced tokenizer); run 30 s window by window.

Voice activity detection can be switched on, off, or both. With "both" each system is run twice
and gets two output files and two system names ("zls-turbo@vad-on", "zls-turbo@vad-off"), so the
effect of the voice filter is a row in the table and not a hidden setting. --vad-preset app uses
the parameters of the app in space/ (min_silence_duration_ms = 500) instead of the defaults of
faster-whisper; the parameters actually used are written in every row.

Decoding is deterministic: temperature 0 and no temperature fallback.

Each output row carries the text, the segment timestamps, the audio duration, the compute time,
the exact model revision and the decoding settings actually used (beam, VAD parameters, device).
Rows already present in an output file are kept and skipped, but only if they were made with the
same model and settings: otherwise the run stops, unless --overwrite is given, which starts
that file again. Nothing is sent anywhere: models are downloaded once from the Hub, without a token.

The window-by-window engine prints a warning, and counts it in the row, each time a window of
more than 3 s comes back empty: that is a failure of the decoding to look at before blaming
the model.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_refs import audio_id, list_audio  # noqa: E402

WINDOW_SECONDS = 30.0
SAMPLE_RATE = 16_000
EMPTY_WINDOW_SECONDS = 3.0
TEMPERATURE = 0.0
VAD_PRESETS = {
    "default": {},                               # faster-whisper defaults
    "app": {"min_silence_duration_ms": 500},     # space/pipeline/transcribe.py
}


def load_systems(path: Path, names: list | None) -> list:
    import yaml

    systems = yaml.safe_load(path.read_text(encoding="utf-8"))["open"]
    if names:
        known = {s["name"] for s in systems}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"[run_local] not in {path.name}: {unknown}. Known: {sorted(known)}")
        systems = [s for s in systems if s["name"] in names]
    return systems


def system_label(name: str, vad: bool) -> str:
    return f"{name}@vad-{'on' if vad else 'off'}"


def windows(n_samples: int, speech: list | None, max_seconds: float = WINDOW_SECONDS, sr: int = SAMPLE_RATE) -> list:
    """Sample ranges of at most max_seconds for the window-by-window engine.

    Without VAD: consecutive fixed windows. With VAD (speech = [{"start", "end"}] in samples):
    neighbouring speech chunks are packed into one window as long as it stays under the limit,
    so cuts fall in silences."""
    limit = int(max_seconds * sr)
    if speech is None:
        return [(start, min(start + limit, n_samples)) for start in range(0, n_samples, limit)]
    out = []
    for chunk in speech:
        start, end = int(chunk["start"]), int(chunk["end"])
        while end - start > limit:  # a single stretch of speech longer than a window
            out.append((start, start + limit))
            start += limit
        if out and end - out[-1][0] <= limit and start >= out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def vad_parameters(preset: str, engine: str) -> dict:
    """The complete set of VAD parameters used, defaults included, so that it can be written down."""
    import dataclasses

    from faster_whisper.vad import VadOptions

    params = dict(VAD_PRESETS[preset])
    if engine == "transformers":
        params["max_speech_duration_s"] = WINDOW_SECONDS  # a chunk must fit in one window
    # Infinity is not JSON: an unset limit is written as null and left out again when the options are rebuilt.
    return {k: (None if v == float("inf") else v) for k, v in dataclasses.asdict(VadOptions(**params)).items()}


def _set(params: dict | None) -> dict:
    return {k: v for k, v in (params or {}).items() if v is not None}


def resolve_device(device: str, engine: str) -> str:
    """The device a model really runs on: "auto" is resolved here and never written as such."""
    if device != "auto":
        return device
    if engine == "transformers":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def decode_settings(system: dict, vad: bool, args) -> dict:
    """The decoding block written in every row: what was really used, not what was asked."""
    engine = system["engine"]
    generate = system.get("generate")
    return {
        "vad": vad,
        "vad_preset": args.vad_preset if vad else None,
        "vad_parameters": vad_parameters(args.vad_preset, engine) if vad else None,
        "language": system.get("language", args.language),
        "beam_size": (generate or {}).get("num_beams", args.beam_size) if engine == "transformers" else args.beam_size,
        "generate": generate,
        "temperature": TEMPERATURE,
        "condition_on_previous_text": False,
        "long_form": "fixed 30 s windows, or VAD chunks packed into 30 s windows" if engine == "transformers"
                     else "faster-whisper",
        "compute_type": "float32" if engine == "transformers" else args.compute_type,
        "device": resolve_device(args.device, engine),
        "threads": args.threads,
    }


def model_block(system: dict) -> dict:
    return {k: system.get(k) for k in ("repo", "revision", "format", "engine", "licence")}


def check_resume(rows: list, model: dict, decode: dict, path) -> None:
    """Stop if rows already in an output file were made with another model or other settings."""
    for row in rows:
        for name, now in (("model", model), ("decode", decode)):
            before = row.get(name)
            if before != now:
                keys = sorted(k for k in set(before or {}) | set(now) if (before or {}).get(k) != now.get(k))
                raise SystemExit(
                    f"[run_local] {path} already has rows made with other settings ({name}: {', '.join(keys)}). "
                    "Use another --out folder, or --overwrite to start this file again.")


def fetch(system: dict) -> Path:
    """Download exactly the pinned revision. token=False: no local token is read or sent."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(system["repo"], revision=system["revision"], token=False))


def convert_to_ct2(system: dict, snapshot: Path, cache: Path, compute_type: str) -> Path:
    """Transformers weights -> CTranslate2, once, in the cache folder."""
    out = cache / f"{system['name']}-{system['revision'][:8]}-ct2-{compute_type}"
    if (out / "model.bin").exists():
        return out
    from ctranslate2.converters import TransformersConverter

    copy = [f for f in ("preprocessor_config.json", "tokenizer.json") if (snapshot / f).exists()]
    TransformersConverter(str(snapshot), copy_files=copy).convert(str(out), quantization=compute_type, force=True)
    if not (out / "tokenizer.json").exists():
        # faster-whisper needs the fast tokenizer file; older repositories only ship vocab.json + merges.txt.
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(str(snapshot)).backend_tokenizer.save(str(out / "tokenizer.json"))
    return out


class FasterWhisperEngine:
    def __init__(self, model_dir: Path, args):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(model_dir), device=resolve_device(args.device, "faster-whisper"),
                                  compute_type=args.compute_type, cpu_threads=args.threads)
        self.notes = {}

    def transcribe(self, path: Path, system: dict, vad: bool, decode: dict) -> dict:
        segments, info = self.model.transcribe(
            str(path), language=decode["language"], task="transcribe",
            beam_size=decode["beam_size"], temperature=TEMPERATURE, condition_on_previous_text=False,
            vad_filter=vad, vad_parameters=_set(decode["vad_parameters"]) if vad else None)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        return {"segments": segments, "duration": round(info.duration, 3),
                "duration_after_vad": round(info.duration_after_vad, 3) if vad else None}


class TransformersEngine:
    """Window-by-window decoding for models that only exist as Transformers weights with their own
    tokenizer. No long-form logic: with VAD off the windows are cut every 30 s wherever that
    falls, even in the middle of a word, and each window is decoded on its own.

    The model is called as its card documents: no language or task argument, the model predicts
    its own prompt tokens. Checked on 24/11/2026 for BuzzASR/luxembourgish with Transformers
    4.57.6 (the version its generation_config.json names) and 5.24.0: neither applies the
    "language" and "task" entries of that file, and both give the same tokens. Forcing those
    entries was tried and made things worse (the language id in the file is not the one the
    model emits, and a window of clear speech came back as invented text), so nothing is forced.
    A window can still come back empty: the model's first token is then its no-speech token."""

    def __init__(self, snapshot: Path, args):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        if args.threads:
            torch.set_num_threads(args.threads)
        self.device = resolve_device(args.device, "transformers")
        self.dtype = torch.float32  # the same precision whatever the checkpoint stores
        self.processor = WhisperProcessor.from_pretrained(str(snapshot))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(snapshot), torch_dtype=self.dtype).to(self.device).eval()
        self.notes = {"prompt": "predicted by the model, nothing forced (model card usage)"}

    def transcribe(self, path: Path, system: dict, vad: bool, decode: dict) -> dict:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
        speech = get_speech_timestamps(audio, VadOptions(**_set(decode["vad_parameters"]))) if vad else None
        kwargs = dict(decode["generate"] or {})
        kwargs["num_beams"] = decode["beam_size"]
        kwargs["do_sample"] = False
        if decode["language"]:
            kwargs.update(language=decode["language"], task="transcribe")
        segments = []
        empty = []
        for start, end in windows(len(audio), speech):
            feats = self.processor.feature_extractor(audio[start:end], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            with self.torch.no_grad():
                ids = self.model.generate(feats.to(self.device, self.dtype), max_new_tokens=225, **kwargs)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            begin, stop = round(start / SAMPLE_RATE, 2), round(end / SAMPLE_RATE, 2)
            if not text and stop - begin > EMPTY_WINDOW_SECONDS:
                empty.append([begin, stop])
                print(f"[run_local] WARNING {path.name}: the window {begin} to {stop} s came back empty")
            segments.append({"start": begin, "end": stop, "text": text})
        kept = sum(c["end"] - c["start"] for c in speech) / SAMPLE_RATE if vad else None
        return {"segments": segments, "duration": round(len(audio) / SAMPLE_RATE, 3),
                "duration_after_vad": round(kept, 3) if vad else None, "empty_windows": empty}


def build_engine(system: dict, args):
    snapshot = fetch(system)
    if system["engine"] == "faster-whisper":
        return FasterWhisperEngine(snapshot, args)
    if system["engine"] == "faster-whisper-converted":
        return FasterWhisperEngine(convert_to_ct2(system, snapshot, args.cache, args.compute_type), args)
    if system["engine"] == "transformers":
        return TransformersEngine(snapshot, args)
    raise SystemExit(f"[run_local] {system['name']}: unknown engine {system['engine']}")


def versions() -> dict:
    """Versions of what produced the transcripts, read from the package metadata (nothing is imported)."""
    from importlib import metadata

    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for package in ("faster-whisper", "ctranslate2", "transformers", "torch"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=Path, help="folder of audio files")
    src.add_argument("--refs", type=Path, help="refs.jsonl from make_refs.py (ids and audio paths are taken from it)")
    ap.add_argument("--out", type=Path, required=True, help="output folder (keep it outside the repository)")
    ap.add_argument("--config", type=Path, default=HERE / "systems.yaml")
    ap.add_argument("--systems", nargs="+", default=None, help="names from systems.yaml; default: all open systems")
    ap.add_argument("--vad", choices=["on", "off", "both"], default="both")
    ap.add_argument("--vad-preset", choices=sorted(VAD_PRESETS), default="default",
                    help="'app' = the VAD parameters of the app in space/ (min_silence_duration_ms = 500)")
    ap.add_argument("--overwrite", action="store_true",
                    help="start again an output file whose rows were made with another model or other settings")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--compute-type", default="int8", help="CTranslate2 compute type, the same for every model")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--language", default="lb")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads, 0 = library default")
    ap.add_argument("--cache", type=Path, default=None, help="where converted models go; default: <out>/converted")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    args.cache = args.cache or args.out / "converted"
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up at once in a log file

    if args.refs:
        base = args.refs.parent
        files = []
        for line in args.refs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                path = Path(row["audio"])
                files.append((row.get("id") or path.with_suffix("").as_posix(), path if path.is_absolute() else base / path))
    else:
        files = [(audio_id(p, args.audio), p) for p in list_audio(args.audio)]
    files = files[: args.max_files] if args.max_files else files
    if not files:
        raise SystemExit("[run_local] no audio file found")

    args.out.mkdir(parents=True, exist_ok=True)
    env = versions()
    for system in load_systems(args.config, args.systems):
        engine = None
        for vad in {"on": [True], "off": [False], "both": [True, False]}[args.vad]:
            label = system_label(system["name"], vad)
            out_path = args.out / f"{label.replace('@', '.')}.jsonl"
            decode = decode_settings(system, vad, args)
            model = model_block(system)
            done = set()
            if out_path.exists() and args.overwrite:
                out_path.unlink()
            if out_path.exists():
                before = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                check_resume(before, model, decode, out_path)
                done = {row["id"] for row in before}
            todo = [(uid, p) for uid, p in files if uid not in done]
            if not todo:
                print(f"[run_local] {label}: {len(done)} file(s) already done, skipped")
                continue
            if engine is None:
                t0 = time.perf_counter()
                engine = build_engine(system, args)
                print(f"[run_local] {system['name']}: loaded in {time.perf_counter() - t0:.1f} s")
            audio_s = wall = 0.0
            with out_path.open("a", encoding="utf-8") as fh:
                for uid, path in todo:
                    t0 = time.perf_counter()
                    result = engine.transcribe(path, system, vad, decode)
                    seconds = time.perf_counter() - t0
                    row = {
                        "id": uid,
                        "audio": str(path),
                        "system": label,
                        "text": " ".join(seg["text"] for seg in result["segments"] if seg["text"]),
                        "segments": result["segments"],
                        "duration": result["duration"],
                        "duration_after_vad": result["duration_after_vad"],
                        "empty_windows": result.get("empty_windows"),
                        "compute_seconds": round(seconds, 2),
                        "model": model,
                        "decode": decode,
                        "notes": engine.notes,
                        "run": {"date": datetime.date.today().isoformat(), **env},
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    audio_s += result["duration"]
                    wall += seconds
            print(f"[run_local] {label}: {len(todo)} file(s), {audio_s / 60:.1f} min of audio in {wall:.0f} s "
                  f"(real-time factor {wall / audio_s:.2f}) -> {out_path}")


if __name__ == "__main__":
    main()
