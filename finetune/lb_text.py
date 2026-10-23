"""Text normalisation for Luxembourgish ASR evaluation.

Whisper's BasicTextNormalizer is too aggressive for Lëtzebuergesch: it splits on the
apostrophe, so « d'Kanner » becomes « d kanner » and « n'Zäit » « n zait »-ish. WER
computed that way punishes a model for correct contractions. This normaliser:

- unescapes HTML entities (FLEURS sentences contain `&apos;`, `&quot;`, ...),
- NFC-normalises and lower-cases,
- unifies typographic apostrophes and hyphens,
- keeps apostrophes and hyphens that sit *between* letters (d'Kanner, Sonndeg-Owend),
- drops every other punctuation mark,
- keeps digits and diacritics (ë é ä ö ü …) untouched,
- collapses whitespace.

Pure Python, no dependency, importable from the Space, the training code and the tests.
"""

from __future__ import annotations

import html
import re
import unicodedata

_APOSTROPHES = "’‘ʼ´`′"
_HYPHENS = "‐‑‒–—−"

# Anything that is not a letter, a digit, a space, an apostrophe or a hyphen.
_JUNK = re.compile(r"[^\w\s'\-]", flags=re.UNICODE)
# Apostrophe / hyphen not surrounded by word characters → space.
_LONE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
_LONE_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise a Luxembourgish (or French/German) transcript for WER computation."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    for ch in _HYPHENS:
        text = text.replace(ch, "-")
    text = text.lower()
    text = _JUNK.sub(" ", text)
    text = _LONE_APOSTROPHE.sub(" ", text)
    text = _LONE_HYPHEN.sub(" ", text)
    # Underscore is matched by \w but is never a word character in speech.
    text = text.replace("_", " ")
    return _SPACES.sub(" ", text).strip()


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER on normalised text. Uses jiwer if installed, else a small
    Levenshtein so the tests do not need the dependency."""
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    try:
        import jiwer  # type: ignore

        return float(jiwer.wer(refs, hyps))
    except ImportError:
        errors = 0
        words = 0
        for r, h in zip(refs, hyps):
            rw, hw = r.split(), h.split()
            errors += _levenshtein(rw, hw)
            words += len(rw)
        return errors / words if words else 0.0


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]
