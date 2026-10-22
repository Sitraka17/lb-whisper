"""Structured summary of a transcript, three local back-ends, zero cloud:

- "ollama"     : a model served by Ollama on localhost (http://127.0.0.1:11434), e.g. qwen2.5:3b
- "llama_cpp"  : a GGUF model loaded in-process with llama-cpp-python (works on a CPU Space)
- "extractive" : no LLM at all; frequency-scored key sentences + keyword-spotted actions

The output is always the same dict: {"summary": [...], "decisions": [...], "actions": [...]}
so the renderer does not care which back-end produced it.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

PROMPT = """You are writing the minutes of a meeting. The transcript below may be in Luxembourgish, French, German or English; answer in the language of the transcript.
Return ONLY a JSON object with three keys:
"summary": 3 to 6 short bullet points of what was discussed,
"decisions": the decisions that were taken (empty list if none),
"actions": action items as "who: what (when)" strings (empty list if none).
Transcript:
<<<
{transcript}
>>>"""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Short function words in lb / fr / de / en that should not drive sentence scoring.
_STOP = set(
    """an a de den der dem die das d et ech du hien hatt mir dir si sie sinn ass hunn huet hat net nach mat op fir vun vu wéi wat wou wann awer och
    le la les un une des du de et est sont a ont pas ne que qui dans pour sur avec ce cette ces il elle nous vous ils elles on
    der die das ein eine und ist sind hat haben nicht auch mit auf für von zu wie was wo wenn aber
    the a an and is are was were be been of to in on for with that this it we you they he she not or as at by""".split()
)
_ACTION = re.compile(
    r"\b(muss|musse|mussen|soll|sollen|sollt|däerf|wäert|wäerte|bis\s+\w+|deadline|to do|todo|action|"
    r"il faut|on doit|doit|devra|devons|à faire|d'ici|avant le|"
    r"müssen|sollte|sollten|erledigen|"
    r"needs? to|has to|have to|will|should|by (monday|tuesday|wednesday|thursday|friday|next week|end of))\b",
    re.IGNORECASE,
)
_DECISION = re.compile(
    r"\b(decidéiert|décidé|décision|entschieden|entscheidung|beschloss|beschlossen|decided|decision|agreed|accord|d'accord|eens|einverstanden|validé|validated)\b",
    re.IGNORECASE,
)


# Default in-process model: small enough for a CPU Space (≈1 GB), good in FR / DE / EN.
DEFAULT_GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def _llama_cpp_installed() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def backend() -> str:
    """Pick the richest back-end available: forced → Ollama → llama.cpp (installed) → extractive."""
    forced = os.environ.get("LB_SUMMARY_BACKEND")
    if forced:
        return forced
    if os.environ.get("OLLAMA_MODEL"):
        return "ollama"
    if os.environ.get("LB_GGUF_PATH") or os.environ.get("LB_GGUF_REPO") or _llama_cpp_installed():
        return "llama_cpp"
    return "extractive"


def summarize(transcript: str, mode: str | None = None) -> dict:
    """Never fails the request: an LLM back-end that errors falls back to the extractive one."""
    mode = mode or backend()
    try:
        if mode == "ollama":
            return _ollama(transcript)
        if mode == "llama_cpp":
            return _llama_cpp(transcript)
    except Exception as exc:  # model missing, OOM, malformed JSON…
        out = extractive(transcript)
        out["backend"] = f"extractive (fallback, {mode} failed: {type(exc).__name__})"
        return out
    return extractive(transcript)


def extractive(transcript: str, max_points: int = 5) -> dict:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(transcript) if len(s.strip()) > 20]
    if not sentences:
        return {"summary": [], "decisions": [], "actions": [], "backend": "extractive"}
    freq = Counter(w.lower() for w in _WORD.findall(transcript) if w.lower() not in _STOP and len(w) > 2)
    scored = []
    for idx, s in enumerate(sentences):
        words = [w.lower() for w in _WORD.findall(s)]
        score = sum(freq[w] for w in words if w not in _STOP) / (len(words) + 5)
        scored.append((score, idx, s))
    top = sorted(sorted(scored, reverse=True)[:max_points], key=lambda t: t[1])
    return {
        "summary": [s for _, _, s in top],
        "decisions": [s for s in sentences if _DECISION.search(s)][:5],
        "actions": [s for s in sentences if _ACTION.search(s)][:8],
        "backend": "extractive",
    }


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group(0) if match else text)
    return {
        "summary": [str(x) for x in data.get("summary", [])],
        "decisions": [str(x) for x in data.get("decisions", [])],
        "actions": [str(x) for x in data.get("actions", [])],
    }


def _ollama(transcript: str) -> dict:
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    body = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"), "prompt": PROMPT.format(transcript=transcript), "stream": False, "format": "json"}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())["response"]
    return {**_parse_json(out), "backend": "ollama"}


_LLM = None


def _llama_cpp(transcript: str) -> dict:
    global _LLM
    from llama_cpp import Llama

    if _LLM is None:
        path = os.environ.get("LB_GGUF_PATH")
        if not path:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                os.environ.get("LB_GGUF_REPO", DEFAULT_GGUF_REPO), os.environ.get("LB_GGUF_FILE", DEFAULT_GGUF_FILE)
            )
        _LLM = Llama(model_path=path, n_ctx=int(os.environ.get("LB_LLM_CTX", "8192")), n_threads=os.cpu_count(), verbose=False)
    out = _LLM.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT.format(transcript=transcript[:24000])}],
        temperature=0.2,
        max_tokens=int(os.environ.get("LB_LLM_MAX_TOKENS", "500")),
    )["choices"][0]["message"]["content"]
    return {**_parse_json(out), "backend": "llama_cpp"}


def to_markdown(summary: dict, lang: str = "en") -> str:
    labels = {
        "en": ("Summary", "Decisions", "Action items"),
        "fr": ("Résumé", "Décisions", "Actions"),
        "de": ("Zusammenfassung", "Entscheidungen", "Aufgaben"),
        "lb": ("Resumé", "Decisiounen", "Aufgaben"),
    }.get(lang, ("Summary", "Decisions", "Action items"))
    parts = []
    for title, key in zip(labels, ("summary", "decisions", "actions")):
        items = summary.get(key) or []
        parts.append(f"## {title}")
        parts.extend(f"- {item}" for item in items) if items else parts.append("- (aucune)")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
