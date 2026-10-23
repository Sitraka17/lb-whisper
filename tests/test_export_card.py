from pathlib import Path

import yaml

from export import ct2_card

TEMPLATE = Path(__file__).resolve().parents[1] / "finetune" / "model_card_template.md"
MAIN, CT2 = "SitrakaForler/whisper-large-v3-turbo-lb", "SitrakaForler/whisper-large-v3-turbo-lb-ct2"


def test_ct2_card_front_matter_and_usage():
    card = ct2_card(TEMPLATE.read_text(encoding="utf-8"), MAIN, CT2)
    front = yaml.safe_load(card.split("---\n")[1])
    assert front["library_name"] == "ctranslate2"
    assert "model-index" not in front
    assert "ctranslate2" in front["tags"]
    assert front["license"] == "mit"
    body = card.split("---\n", 2)[2]
    assert body.startswith("\n# whisper-large-v3-turbo-lb-ct2\n")
    usage = body.split("## Usage\n")[1].split("\n## ")[0]
    assert "from transformers" not in usage and "pipeline(" not in usage
    assert f'WhisperModel("{CT2}"' in usage and "not measured separately" in usage
    assert "## Limitations" in body and "## Training data" in body
