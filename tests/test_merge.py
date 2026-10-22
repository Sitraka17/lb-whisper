from pipeline.merge import assign_speakers, group_turns, relabel_speakers

SEGS = [
    {"start": 0.0, "end": 2.0, "text": "Moien alleguer."},
    {"start": 2.2, "end": 4.0, "text": "Fänke mer un."},
    {"start": 4.5, "end": 6.0, "text": "Jo, d'accord."},
    {"start": 9.0, "end": 10.0, "text": "Merci."},
]
TURNS = [
    {"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"},
    {"start": 4.3, "end": 6.5, "speaker": "SPEAKER_01"},
]


def test_assign_by_max_overlap_and_nearest_for_orphans():
    out = assign_speakers(SEGS, TURNS)
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


def test_assign_without_turns_uses_default():
    out = assign_speakers(SEGS, [])
    assert {s["speaker"] for s in out} == {"SPEAKER"}


def test_overlap_wins_over_shorter_competing_turn():
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "A"},
        {"start": 1.2, "end": 2.0, "speaker": "B"},
    ]
    assert assign_speakers(SEGS[:1], turns)[0]["speaker"] == "A"


def test_group_turns_joins_same_speaker_within_gap():
    blocks = group_turns(assign_speakers(SEGS, TURNS), max_gap=2.0)
    assert len(blocks) == 3
    assert blocks[0]["text"] == "Moien alleguer. Fänke mer un."
    assert blocks[0]["end"] == 4.0
    assert blocks[1]["speaker"] == "SPEAKER_01" and blocks[2]["speaker"] == "SPEAKER_01"  # gap 3 s > max_gap


def test_relabel_in_order_of_first_appearance_and_custom_names():
    blocks = [{"start": 0, "end": 1, "speaker": "SPEAKER_03", "text": "a"}, {"start": 1, "end": 2, "speaker": "SPEAKER_00", "text": "b"}]
    assert [b["speaker"] for b in relabel_speakers(blocks)] == ["Speaker 1", "Speaker 2"]
    assert relabel_speakers(blocks, {"SPEAKER_00": "Pedro"})[1]["speaker"] == "Pedro"
