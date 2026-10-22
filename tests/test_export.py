import json

from pipeline.export import hms, srt_time, to_json, to_markdown, to_srt

BLOCKS = [
    {"start": 0.0, "end": 2.5, "speaker": "Speaker 1", "text": "Moien alleguer."},
    {"start": 3661.0, "end": 3662.999, "speaker": "Speaker 2", "text": "Merci."},
]


def test_time_formatting():
    assert hms(0) == "00:00:00"
    assert hms(3661.4) == "01:01:01"
    assert srt_time(2.5) == "00:00:02,500"
    assert srt_time(3662.9996) == "01:01:03,000"


def test_markdown_has_summary_then_transcript():
    md = to_markdown(BLOCKS, "## Summary\n- point", {"duration": 3663, "language": "lb", "model": "x/y"})
    assert md.startswith("# Meeting notes\n")
    assert "no cloud API" in md
    assert md.index("## Summary") < md.index("## Transcript")
    assert "**[01:01:01] Speaker 2:** Merci." in md


def test_srt_blocks():
    srt = to_srt(BLOCKS)
    assert srt.split("\n")[:3] == ["1", "00:00:00,000 --> 00:00:02,500", "Speaker 1: Moien alleguer."]


def test_json_roundtrip():
    data = json.loads(to_json(BLOCKS, {"summary": ["a"]}, {"language": "lb"}))
    assert data["blocks"] == BLOCKS and data["summary"]["summary"] == ["a"] and data["meta"]["language"] == "lb"
