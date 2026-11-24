#!/usr/bin/env python3
"""Count the words that contain a digit in a folder of transcripts.

    python bench/digit_words.py /data/zls-cc0/Verbatim

PROTOCOL.md (section 4) asks transcribers to write numbers in letters and quotes how rarely the
ZLS verbatim transcripts use digits. This is the script behind that figure. Every .txt file
under the folder is read and passed through the bench normaliser (finetune/lb_text.py).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from lb_text import normalize  # noqa: E402


def count(texts) -> dict:
    """Words, words with a digit, words made of digits only, and the digit-bearing words by frequency."""
    words = with_digit = pure = 0
    forms = Counter()
    for text in texts:
        for w in normalize(text).split():
            words += 1
            if any(c.isdigit() for c in w):
                with_digit += 1
                forms[w] += 1
                pure += w.isdigit()
    return {"words": words, "with_digit": with_digit, "pure_numbers": pure, "most_common": forms.most_common(10)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    files = sorted(p for p in args.folder.rglob("*.txt") if not p.name.startswith("._"))
    result = count(p.read_text(encoding="utf-8-sig") for p in files)
    print(f"{len(files)} files, {result['words']} words, {result['with_digit']} with a digit, "
          f"{result['pure_numbers']} of them pure numbers")
    print("most frequent:", ", ".join(f"{w} ({n})" for w, n in result["most_common"]))


if __name__ == "__main__":
    main()
