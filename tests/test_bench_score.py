"""LB-Real bench: scoring, confidence intervals, skipped passages, imports. Every expected value
below is worked out by hand in the comment next to it; no model and no audio are needed."""

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))

import digit_words  # noqa: E402
import import_hypotheses  # noqa: E402
import leak_check  # noqa: E402
import make_refs  # noqa: E402
import run_local  # noqa: E402
import score  # noqa: E402


def _reference_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ distances and alignment

def test_edit_distance_known_values():
    assert score.edit_distance("kitten", "sitting") == 3  # k>s, e>i, +g
    assert score.edit_distance("", "abc") == 3
    assert score.edit_distance("abc", "") == 3
    assert score.edit_distance("moien", "moien") == 0
    assert score.edit_distance("een zwee dräi véier".split(), "een zwou dräi".split()) == 2


def test_edit_distance_matches_the_textbook_table():
    import random

    rng = random.Random(7)
    for _ in range(300):
        a = [rng.choice("abc") for _ in range(rng.randrange(0, 80))]
        b = [rng.choice("abc") for _ in range(rng.randrange(0, 80))]
        expected = _reference_distance(a, b)
        assert score.edit_distance(a, b) == expected
        ops = score.align(a, b)
        assert len(ops) - ops.count("e") == expected
        assert sum(op != "i" for op in ops) == len(a)  # every reference word is consumed once
        assert sum(op != "d" for op in ops) == len(b)  # every hypothesis word too


def test_align_operations():
    assert score.align("a b c d".split(), "a x c".split()) == ["e", "s", "e", "d"]
    assert score.align("a b".split(), "a b c".split()) == ["e", "e", "i"]
    assert score.align([], "a b".split()) == ["i", "i"]
    assert score.align("a b".split(), []) == ["d", "d"]


# ------------------------------------------------------------------ per-utterance counts

def test_score_utterance_counts_by_hand():
    # Reference: 4 words. "zwee" > "zwou" is 1 substitution, "véier" is missing: 1 deletion.
    row = score.score_utterance("een zwee dräi véier".split(), "Een zwou, dräi.", None, None)
    assert (row["substitutions"], row["deletions"], row["insertions"]) == (1, 1, 0)
    assert row["errors"] == 2 and row["ref_words"] == 4
    # Characters: "een zwee dräi véier" (19) against "een zwou dräi" (13):
    # "ee" > "ou" is 2 substitutions, " véier" is 6 deletions.
    assert row["ref_chars"] == 19 and row["char_errors"] == 8


def test_score_utterance_insertion_and_normalisation():
    row = score.score_utterance(["d'kanner", "spillen"], "D’Kanner spillen haut!", None, None)
    assert (row["substitutions"], row["deletions"], row["insertions"]) == (0, 0, 1)


def test_n_rule_variant_forgives_only_the_final_n():
    # "kënnen mir" against "kënne mir": one strict error, none once the n-rule is applied to both sides.
    row = score.score_utterance("kënnen mir goen".split(), "kënne mir goen", None, None)
    assert row["errors"] == 1 and row["errors_n_rule"] == 0
    # A different word is still an error under the n-rule.
    row = score.score_utterance("kënnen mir goen".split(), "kënne dir goen", None, None)
    assert row["errors"] == 2 and row["errors_n_rule"] == 1


def test_digit_words_are_counted():
    row = score.score_utterance("zwanzeg leit".split(), "20 Leit", None, None)
    assert row["ref_digit_words"] == 0 and row["hyp_digit_words"] == 1 and row["errors"] == 1


# ------------------------------------------------------------------ word timing and skipped passages

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "a b"},       # 1 s per word
    {"start": 2.0, "end": 10.0, "text": "c d e f"},  # 2 s per word
    {"start": 10.0, "end": 12.0, "text": "g h"},     # 1 s per word
]


def test_reference_words_share_a_segment_by_word_length():
    words, seconds, starts, method = score.reference_words("", [{"start": 0.0, "end": 6.0, "text": "Ab, abcd!"}], None)
    assert words == ["ab", "abcd"] and method == "reference timestamps"
    assert seconds == pytest.approx([2.0, 4.0])  # 6 s shared 2:4
    assert starts == pytest.approx([0.0, 2.0])


def test_reference_words_mean_rate_and_unavailable():
    words, seconds, starts, method = score.reference_words("een zwee dräi véier", None, 10.0)
    assert seconds == [2.5] * 4 and starts is None and method == "mean speaking rate"
    assert score.reference_words("een zwee", None, None)[1:] == (None, None, "unavailable")


def test_skipped_passage_with_reference_timestamps():
    words, seconds, starts, _ = score.reference_words("", SEGMENTS, None)
    # The whole middle segment is missing: 4 words, 8 s, starting at 2 s.
    row = score.score_utterance(words, "a b g h", seconds, starts)
    assert row["skipped"] == [{"first_word": 2, "n_words": 4, "bridged_words": 0, "seconds": 8.0, "start": 2.0,
                               "text": "c d e f"}]
    # Only "e f" missing: 4 s, under the 5 s threshold.
    assert score.score_utterance(words, "a b c d g h", seconds, starts)["skipped"] == []
    # "d e f" missing: 6 s, starting at 4 s.
    row = score.score_utterance(words, "a b c g h", seconds, starts)
    assert [(s["n_words"], s["seconds"], s["start"]) for s in row["skipped"]] == [(3, 6.0, 4.0)]


def test_one_stray_word_does_not_hide_a_skipped_passage():
    words, seconds, starts, _ = score.reference_words("", SEGMENTS, None)
    # "c d" (4 s) and "f" (2 s) are missing, with only "e" matched in between: one passage. The time
    # counted is that of the deleted words, 2-6 s and 8-10 s = 6 s; "e" is listed but not counted.
    row = score.score_utterance(words, "a b e g h", seconds, starts)
    assert row["deletions"] == 3
    assert row["skipped"] == [{"first_word": 2, "n_words": 3, "bridged_words": 1, "seconds": 6.0, "start": 2.0,
                               "text": "c d e f"}]
    # An invented word instead of the four: the alignment pairs it with the last one ("f" > "x") and
    # deletes "c d e", 2-8 s = 6 s in one run.
    row = score.score_utterance(words, "a b x g h", seconds, starts)
    assert [(k["n_words"], k["bridged_words"], k["seconds"], k["text"]) for k in row["skipped"]] == [(3, 0, 6.0, "c d e")]
    # A substituted word between two runs is bridged like a matched one: ops d d s d on "c d e f".
    assert [(k["n_words"], k["bridged_words"], k["seconds"]) for k in
            score.skipped_passages(["e", "e", "d", "d", "s", "d", "e", "e"], words, seconds, starts)] == [(3, 1, 6.0)]


def test_two_recognised_words_keep_the_runs_apart():
    segments = [{"start": 0.0, "end": 16.0, "text": "a b c d e f g h"}]  # 2 s per word
    words, seconds, starts, _ = score.reference_words("", segments, None)
    # "b c" (4 s) and "f g" (4 s) are missing, "d e" recognised in between: two runs, none over 5 s.
    row = score.score_utterance(words, "a d e h", seconds, starts)
    assert row["deletions"] == 4 and row["skipped"] == []
    # With bridge=0 the stray-word tolerance is off: "c d" and "f" above stay apart.
    words, seconds, starts, _ = score.reference_words("", SEGMENTS, None)
    ops = score.align(words, "a b e g h".split())
    assert score.skipped_passages(ops, words, seconds, starts, bridge=0) == []


def test_known_limit_a_passage_replaced_by_invented_words():
    # 9 words, 2 s each. "x1 x2 x3 x4 x5" (10 s) is replaced by 3 invented words: the alignment
    # gives 3 substitutions and 2 deletions (4 s), so nothing is listed. Documented in PROTOCOL.md.
    segments = [{"start": 0.0, "end": 18.0, "text": "aa bb x1 x2 x3 x4 x5 cc dd"}]
    words, seconds, starts, _ = score.reference_words("", segments, None)
    row = score.score_utterance(words, "aa bb y1 y2 y3 cc dd", seconds, starts)
    assert (row["substitutions"], row["deletions"]) == (3, 2) and row["skipped"] == []


def test_overlapping_and_empty_segments_are_not_counted_twice():
    # Hand-made timestamps overlap: 0-5 s and 3-8 s cover 8 s of audio, not 10. A zero-length
    # segment and one that ends before it starts count for nothing.
    segments = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 3.0, "end": 8.0, "text": "b"},
        {"start": 8.0, "end": 8.0, "text": "c"},
        {"start": 9.0, "end": 7.0, "text": "d"},
        {"start": 9.0, "end": 10.0, "text": "e"},
    ]
    words, seconds, starts, _ = score.reference_words("", segments, None)
    assert seconds == [5.0, 5.0, 0.0, 0.0, 1.0] and min(seconds) >= 0
    row = score.score_utterance(words, "e", seconds, starts)
    assert [(k["n_words"], k["seconds"], k["start"]) for k in row["skipped"]] == [(4, 8.0, 0.0)]
    # 0-3 s and 2-5 s: the union is 5 s, not "more than 5 s", although the durations add up to 6.
    close = [{"start": 0.0, "end": 3.0, "text": "a"}, {"start": 2.0, "end": 5.0, "text": "b"}, {"start": 5.0, "end": 6.0, "text": "c"}]
    words, seconds, starts, _ = score.reference_words("", close, None)
    assert score.score_utterance(words, "c", seconds, starts)["skipped"] == []
    assert score._span_seconds([(0, 1), (5, 6), (0.5, 2)]) == 3.0  # 0-2 and 5-6


def test_skipped_threshold_is_strict_and_reaches_the_end():
    exact = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 1.0, "end": 6.0, "text": "b c"}]
    words, seconds, starts, _ = score.reference_words("", exact, None)
    assert score.score_utterance(words, "a", seconds, starts)["skipped"] == []  # exactly 5 s is not "more than 5 s"
    longer = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 1.0, "end": 6.5, "text": "b c"}]
    words, seconds, starts, _ = score.reference_words("", longer, None)
    row = score.score_utterance(words, "a", seconds, starts)  # the run ends with the utterance
    assert [(s["first_word"], s["n_words"], s["seconds"]) for s in row["skipped"]] == [(1, 2, 5.5)]


def test_skipped_passage_from_mean_rate():
    # 10 words in 20 s: 2 s per word. Three consecutive words missing = 6 s, two = 4 s.
    text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
    words, seconds, starts, _ = score.reference_words(text, None, 20.0)
    row = score.score_utterance(words, "w0 w1 w2 w6 w7 w8 w9", seconds, starts)
    assert [(s["first_word"], s["n_words"], s["seconds"], s["start"]) for s in row["skipped"]] == [(3, 3, 6.0, None)]
    assert score.score_utterance(words, "w0 w1 w2 w3 w6 w7 w8 w9", seconds, starts)["skipped"] == []
    # A realistic rate: 20 words skipped with one word matched in the middle, 0.4 s per word.
    # One passage of 20 deleted words = 8 s, where two runs of 10 words (4 s each) would be missed.
    text = " ".join(f"w{k}" for k in range(25))
    words, seconds, starts, _ = score.reference_words(text, None, 10.0)
    row = score.score_utterance(words, "w0 w1 w12 w23 w24", seconds, starts)
    assert [(s["first_word"], s["n_words"], s["bridged_words"], s["seconds"]) for s in row["skipped"]] == [(2, 20, 1, 8.0)]


def test_no_duration_means_no_skipped_detection():
    words, seconds, starts, _ = score.reference_words("a b c d e f g h", None, None)
    assert score.score_utterance(words, "a", seconds, starts)["skipped"] == []


# ------------------------------------------------------------------ bootstrap

def test_quantile_interpolates():
    assert score.quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert score.quantile([1, 2, 3, 4, 5], 0.025) == pytest.approx(1.1)  # position 0.1 between 1 and 2
    assert score.quantile([1, 2, 3, 4, 5], 1.0) == 5
    assert score.quantile([7], 0.3) == 7


def test_resamples_are_deterministic_and_in_range():
    a = score.resamples(5, n_boot=50, seed=3)
    assert a == score.resamples(5, n_boot=50, seed=3)
    assert a != score.resamples(5, n_boot=50, seed=4)
    assert len(a) == 50 and all(len(d) == 5 and all(0 <= k < 5 for k in d) for d in a)


def test_bootstrap_ci_degenerate_cases():
    draws = score.resamples(4)
    # Every utterance has 1 error for 4 words: any resample gives 25 %, so the interval has no width.
    assert score.bootstrap_ci([(1, 4)] * 4, draws) == [0.25, 0.25]
    assert score.bootstrap_ci([(1, 4)], []) is None  # one utterance: no interval


def test_bootstrap_ci_two_units_by_hand():
    # Units: 0 errors / 1 word and 1 error / 1 word. A resample of two is (0,0), (0,1), (1,0) or (1,1):
    # rate 0, 0.5, 0.5 or 1 with probabilities 1/4, 1/2, 1/4. Both tails are far above 2.5 %,
    # so the 95 % percentile interval is [0, 1] around the point estimate 0.5.
    pairs = [(0, 1), (1, 1)]
    assert score.ratio(pairs) == 0.5
    assert score.bootstrap_ci(pairs, score.resamples(2)) == [0.0, 1.0]


def test_bootstrap_ci_brackets_the_estimate():
    pairs = [(1, 10), (4, 12), (0, 8), (3, 9), (2, 11), (6, 10)]
    lo, hi = score.bootstrap_ci(pairs, score.resamples(len(pairs)))
    assert lo < score.ratio(pairs) < hi
    assert 0.0 <= lo and hi <= 0.6  # no resample can exceed the worst utterance (6/10)


def test_paired_difference():
    draws = score.resamples(5)
    a, b = [(2, 10)] * 5, [(1, 10)] * 5
    diff = score.paired_difference(a, b, draws)  # always 20 % against 10 %
    assert diff["difference"] == pytest.approx(0.1) and diff["ci95"] == pytest.approx([0.1, 0.1]) and diff["excludes_zero"]
    same = score.paired_difference(a, a, draws)
    assert same["difference"] == 0 and not same["excludes_zero"]


# ------------------------------------------------------------------ whole report

REFS = {
    "radio/r1": {"id": "radio/r1", "text": "een zwee dräi véier", "genre": "radio", "duration": 8.0},
    "radio/r2": {"id": "radio/r2", "text": "moien alleguer", "genre": "radio", "duration": 2.0},
    "meeting/m1": {"id": "meeting/m1", "text": "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9", "genre": "meeting", "duration": 20.0,
                   "text_normalised": "w0 w1 w2 w3 w4 w5 w6 w7 w8"},
}
GOOD = {"radio/r1": {"text": "een zwee dräi véier"}, "radio/r2": {"text": "moien alleguer"},
        "meeting/m1": {"text": "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"}}
BAD = {"radio/r1": {"text": "een zwou dräi"}, "radio/r2": {"text": "moien alleguer"},
       "meeting/m1": {"text": "w0 w1 w2 w6 w7 w8 w9"}, "extra/x": {"text": "not in the references"}}


def test_report_by_system_and_genre():
    report = score.score(REFS, {"good": GOOD, "bad": BAD}, n_boot=200)
    good, bad = report["systems"]["good"], report["systems"]["bad"]
    assert good["overall"]["wer"] == 0.0 and good["overall"]["cer"] == 0.0
    # bad: radio = 2 errors / 6 words, meeting = 3 / 10, all = 5 / 16.
    assert bad["by_genre"]["radio"]["wer"] == 33.33
    assert bad["by_genre"]["meeting"]["wer"] == 30.0
    assert bad["overall"]["wer"] == 31.25
    assert (bad["overall"]["substitutions"], bad["overall"]["deletions"], bad["overall"]["insertions"]) == (1, 4, 0)
    assert bad["overall"]["utterances"] == 3 and bad["overall"]["ref_words"] == 16
    assert bad["overall"]["audio_minutes"] == 0.5
    lo, hi = bad["overall"]["wer_ci95"]
    assert lo <= 31.25 <= hi
    assert bad["by_genre"]["meeting"]["wer_ci95"] is None  # a single utterance in that genre
    assert bad["ignored_hypotheses"] == ["extra/x"]
    # The three missing words of the meeting are 6 s at 2 s per word: one skipped passage, listed.
    assert bad["overall"]["skipped_passages"] == 1 and bad["overall"]["skipped_seconds"] == 6.0
    assert report["skipped"] == [{"system": "bad", "id": "meeting/m1", "genre": "meeting", "timing": "mean speaking rate",
                                  "first_word": 3, "n_words": 3, "bridged_words": 0, "seconds": 6.0, "start": None,
                                  "text": "w3 w4 w5"}]
    pair = report["pairwise_wer"][0]
    assert (pair["a"], pair["b"], pair["difference"]) == ("bad", "good", 31.25)
    assert "| bad | radio | 2 | 6 | 33.33" in score.to_markdown(report)


def test_report_is_reproducible():
    one = score.score(REFS, {"bad": BAD}, n_boot=300, seed=11)
    two = score.score(REFS, {"bad": BAD}, n_boot=300, seed=11)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


def test_missing_hypothesis_is_an_error_unless_allowed():
    partial = {"radio/r1": {"text": "een zwee dräi véier"}}
    with pytest.raises(SystemExit):
        score.score(REFS, {"partial": partial})
    report = score.score(REFS, {"partial": partial}, allow_missing=True, n_boot=50)
    overall = report["systems"]["partial"]["overall"]
    # The two missing files are scored as empty: 2 + 10 deleted words out of 16.
    assert overall["missing"] == 2 and overall["deletions"] == 12 and overall["wer"] == 75.0


def test_normalised_reference_needs_every_row():
    with pytest.raises(SystemExit):
        score.score(REFS, {"good": GOOD}, reference="normalised")
    refs = {"meeting/m1": REFS["meeting/m1"]}
    report = score.score(refs, {"good": {"meeting/m1": GOOD["meeting/m1"]}}, reference="normalised", n_boot=50)
    assert report["systems"]["good"]["overall"]["insertions"] == 1  # "w9" is not in the normalised reference


def test_recording_as_bootstrap_unit():
    refs = {f"c{k}": {"id": f"c{k}", "text": "a b", "recording": "rec1" if k < 2 else "rec2"} for k in range(4)}
    hyps = {f"c{k}": {"text": "a b"} for k in range(4)}
    by_clip = score.score(refs, {"s": hyps}, n_boot=20)["systems"]["s"]["overall"]
    by_recording = score.score(refs, {"s": hyps}, unit="recording", n_boot=20)["systems"]["s"]["overall"]
    assert by_clip["bootstrap_units"] == 4 and by_recording["bootstrap_units"] == 2


def test_empty_reference_is_refused():
    # Without this, a reference with no word scores 0 % whatever the system wrote.
    with pytest.raises(SystemExit):
        score.score({"u": {"text": "", "duration": 5}}, {"s": {"u": {"text": "hallo du"}}})
    with pytest.raises(SystemExit):
        score.score({"u": {"text": " ... ", "duration": 5}}, {"s": {"u": {"text": ""}}})


def test_shared_manifest_source_is_the_genre():
    refs = {"clips/a": {"audio": "clips/a.wav", "text": "moien alleguer", "source": "zls-rtl", "recording": "r1", "duration": 2.0},
            "clips/b": {"audio": "clips/b.wav", "text": "wéi geet et", "source": "lod-f1", "recording": "r2", "duration": 2.0}}
    hyps = {"clips/a": {"text": "moien"}, "clips/b": {"text": "wéi geet et"}}
    report = score.score(refs, {"s": hyps}, n_boot=20)
    assert sorted(report["systems"]["s"]["by_genre"]) == ["lod-f1", "zls-rtl"]
    assert report["systems"]["s"]["by_genre"]["zls-rtl"]["wer"] == 50.0
    assert score.genre_of({"genre": "read", "source": "x"}) == "read" and score.genre_of({}) == "unknown"


def test_consent_filter():
    refs = {"a": {"consent": ["A1", "A1b", "A2"]}, "b": {"consent": ["A1"]}, "c": {"consent": "A1 A2"}, "d": {}}
    assert sorted(score.filter_refs(refs, only="A2")) == ["a", "c"]
    assert sorted(score.filter_refs(refs, without="A2")) == ["b", "d"]
    assert sorted(score.filter_refs(refs, only="A1")) == ["a", "b", "c"]  # "A1" does not match "A1b" by accident
    assert sorted(score.filter_refs(refs, only="A1b")) == ["a"]
    assert sorted(score.filter_refs(refs)) == ["a", "b", "c", "d"]


def test_mixed_settings_are_detected():
    model = {"repo": "x/y", "revision": "abc"}
    same = {"u1": {"model": model, "decode": {"beam_size": 5}}, "u2": {"model": model, "decode": {"beam_size": 5}}}
    beam = {"u1": {"model": model, "decode": {"beam_size": 5}}, "u2": {"model": model, "decode": {"beam_size": 1}}}
    revision = {"u1": {"model": model, "decode": {}}, "u2": {"model": {"repo": "x/y", "revision": "def"}, "decode": {}}}
    online = {"u1": {"online": {"service": "s", "test_date": "2026-12-01", "settings": {"plan": "free"}, "export_file": "u1.srt"}},
              "u2": {"online": {"service": "s", "test_date": "2026-12-01", "settings": {"plan": "free"}, "export_file": "u2.srt"}}}
    dates = {"u1": {"online": {"service": "s", "test_date": "2026-12-01"}}, "u2": {"online": {"service": "s", "test_date": "2026-12-09"}}}
    found = score.mixed_settings({"same": same, "beam": beam, "revision": revision, "online": online, "dates": dates})
    assert found == {"beam": 2, "revision": 2, "dates": 2}  # the export file name is allowed to differ


def test_utterance_id_and_duplicates(tmp_path):
    assert score.utterance_id({"audio": "radio/r1.wav"}) == "radio/r1"
    assert score.utterance_id({"id": "x", "audio": "radio/r1.wav"}) == "x"
    path = tmp_path / "sys.jsonl"
    path.write_text('{"id": "a", "system": "s", "text": "x"}\n{"id": "a", "system": "s", "text": "y"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        score.load_systems([path])


# ------------------------------------------------------------------ imports of online services

SRT = """1
00:00:00,000 --> 00:00:02,500
Moien alleguer,

2
00:01:02,250 --> 00:01:04,000
wéi geet
et?
"""

VTT = """WEBVTT - export

NOTE made by hand

cue-1
00:00.000 --> 00:02.500
<v Speaker 1>Moien alleguer,</v>

01:02.250 --> 01:04.000 align:start
<i>wéi geet et?</i>
"""


def test_parse_srt_and_vtt_give_the_same_segments():
    expected = [{"start": 0.0, "end": 2.5, "text": "Moien alleguer,"}, {"start": 62.25, "end": 64.0, "text": "wéi geet et?"}]
    assert import_hypotheses.parse_cues(SRT) == expected
    assert import_hypotheses.parse_cues(VTT) == expected
    assert import_hypotheses.parse_cues(SRT.replace("\n", "\r\n")) == expected


def test_build_row_from_exports(tmp_path):
    srt = tmp_path / "r1.srt"
    srt.write_text(SRT, encoding="utf-8")
    row = import_hypotheses.build_row(srt, "radio/r1", "luxasr", "2026-12-01", {"plan": "free"}, pass_number=2)
    assert row["system"] == "luxasr@pass2" and row["id"] == "radio/r1"
    assert row["text"] == "Moien alleguer, wéi geet et?" and len(row["segments"]) == 2
    assert row["online"]["test_date"] == "2026-12-01" and row["online"]["settings"] == {"plan": "free"}
    txt = tmp_path / "r1.txt"
    txt.write_text("\ufeffSpeaker 1: Moien alleguer,\nSpeaker 2: wéi geet et?\n", encoding="utf-8")
    row = import_hypotheses.build_row(txt, "radio/r1", "other", "2026-12-01", {"plan": "free"}, strip_regex=r"^Speaker \d+: ")
    assert row["system"] == "other" and row["text"] == "Moien alleguer, wéi geet et?" and row["segments"] is None


def test_srt_text_with_an_empty_line_is_kept_and_reported():
    # Not valid SRT, but exports do it: "alleguer" used to vanish without a trace.
    content = "1\n00:00:00,000 --> 00:00:02,000\nMoien\n\nalleguer\n\n2\n00:00:03,000 --> 00:00:04,000\nwéi geet et\n"
    warnings = []
    segments = import_hypotheses.parse_cues(content, warnings=warnings)
    assert [seg["text"] for seg in segments] == ["Moien alleguer", "wéi geet et"]
    assert len(warnings) == 1 and "alleguer" in warnings[0]
    assert import_hypotheses.parse_cues(SRT, warnings=warnings) and len(warnings) == 1  # a clean file adds none


def test_strip_regex_applies_to_each_subtitle_line(tmp_path):
    srt = tmp_path / "r1.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nSpeaker 1: wei geet et\nSpeaker 2: gutt\n", encoding="utf-8")
    row = import_hypotheses.build_row(srt, "r1", "svc", "2026-12-01", {"plan": "free"}, strip_regex=r"^Speaker \d+: ")
    assert row["text"] == "wei geet et gutt" and row["online"]["warnings"] == []


def test_pass_number_and_settings_are_checked(tmp_path, monkeypatch):
    txt = tmp_path / "r1.txt"
    txt.write_text("Moien", encoding="utf-8")
    for bad in (0, 3):
        with pytest.raises(SystemExit):
            import_hypotheses.build_row(txt, "r1", "svc", "2026-12-01", {"plan": "free"}, pass_number=bad)
    assert import_hypotheses.parse_settings(["plan=free", "options=a=b"]) == {"plan": "free", "options": "a=b"}
    for bad in ("plan", "=free"):
        with pytest.raises(SystemExit) as error:
            import_hypotheses.parse_settings([bad])
        assert "KEY=VALUE" in str(error.value)
    argv = ["import_hypotheses.py", "--system", "svc", "--input", str(txt), "--test-date", "2026-12-01",
            "--set", "plan=free", "--out", str(tmp_path / "o.jsonl"), "--pass", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        import_hypotheses.main()


# ------------------------------------------------------------------ references

def _wav(path, seconds=1.0, rate=16000):
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def test_make_refs_reads_csv_json_txt_and_normalised(tmp_path):
    audio, norm = tmp_path / "audio", tmp_path / "norm"
    _wav(audio / "radio" / "show__part2.wav", 2.0)
    (audio / "radio" / "show__part2.csv").write_text(
        "\ufeffstart_seconds,end_seconds,text\n0.000,1.000,Moien alleguer\n1.000,2.000,\"wéi, geet et\"\n", encoding="utf-8")
    _wav(audio / "meeting" / "m1.wav", 1.5)
    (audio / "meeting" / "m1.json").write_text(json.dumps(
        {"fragments": [{"begin": "0.000", "end": "1.500", "lines": ["Gudde", "Moien"]}]}), encoding="utf-8")
    _wav(audio / "note" / "n1.wav", 1.0)
    (audio / "note" / "n1.txt").write_text("\ufeffÄddi\n", encoding="utf-8")
    (norm / "radio_normalised").mkdir(parents=True)
    (norm / "radio_normalised" / "show__part2_normalised.csv").write_text(
        "\ufeffstart_seconds,end_seconds,text\n0.000,2.000,Moien alleguer wéi geet et\n", encoding="utf-8")

    rows = {r["id"]: r for r in make_refs.build_rows(audio, tmp_path / "work" / "refs.jsonl", norm)}
    assert sorted(rows) == ["meeting/m1", "note/n1", "radio/show__part2"]
    radio = rows["radio/show__part2"]
    assert radio["text"] == "Moien alleguer wéi, geet et" and radio["genre"] == "radio"
    assert radio["segments"] == [{"start": 0.0, "end": 1.0, "text": "Moien alleguer"}, {"start": 1.0, "end": 2.0, "text": "wéi, geet et"}]
    assert radio["recording"] == "show" and radio["duration"] == 2.0
    assert radio["audio"] == "../audio/radio/show__part2.wav"  # relative to the folder of refs.jsonl
    assert radio["text_normalised"] == "Moien alleguer wéi geet et" and len(radio["segments_normalised"]) == 1
    assert rows["meeting/m1"]["text"] == "Gudde Moien" and rows["meeting/m1"]["segments"][0]["end"] == 1.5
    assert rows["meeting/m1"]["text_normalised"] is None
    assert rows["note/n1"]["text"] == "Äddi" and rows["note/n1"]["segments"] is None  # the BOM is not part of the text


def test_make_refs_meta_and_empty_transcript(tmp_path):
    audio = tmp_path / "audio"
    _wav(audio / "note" / "n1.wav")
    (audio / "note" / "n1.txt").write_text("Moien", encoding="utf-8")
    meta = tmp_path / "meta.csv"
    meta.write_text("\ufeffid,device,codec,consent\nnote/n1,phone,opus,A1 A1b A2\n", encoding="utf-8")
    row = make_refs.build_rows(audio, tmp_path / "refs.jsonl", meta=make_refs.read_meta(meta))[0]
    assert (row["device"], row["codec"], row["consent"]) == ("phone", "opus", ["A1", "A1b", "A2"])
    with pytest.raises(SystemExit):  # a clip without a row in the meta file
        make_refs.build_rows(audio, tmp_path / "refs.jsonl", meta={})
    with pytest.raises(SystemExit):  # a meta row without a clip
        make_refs.build_rows(audio, tmp_path / "refs.jsonl", meta={**make_refs.read_meta(meta), "note/ghost": {}})
    (audio / "note" / "n1.txt").write_text(" ... \n", encoding="utf-8")
    with pytest.raises(SystemExit):  # no word in the transcript
        make_refs.build_rows(audio, tmp_path / "refs.jsonl")


def test_digit_words_count():
    # 7 words; "co2", "covid-19" and "2020" carry a digit, only "2020" is a pure number.
    result = digit_words.count(["Den CO2 an d'Covid-19,", "am Joer 2020."])
    assert (result["words"], result["with_digit"], result["pure_numbers"]) == (7, 3, 1)


# ------------------------------------------------------------------ run_local helpers and configuration

def test_windows_without_and_with_vad():
    sr = run_local.SAMPLE_RATE
    assert run_local.windows(70 * sr, None) == [(0, 30 * sr), (30 * sr, 60 * sr), (60 * sr, 70 * sr)]
    speech = [{"start": 0, "end": 10 * sr}, {"start": 12 * sr, "end": 25 * sr}, {"start": 26 * sr, "end": 40 * sr}]
    # The first two chunks fit in 30 s together; adding the third would make 40 s.
    assert run_local.windows(40 * sr, speech) == [(0, 25 * sr), (26 * sr, 40 * sr)]
    # One stretch of 70 s of speech is cut every 30 s.
    assert run_local.windows(70 * sr, [{"start": 0, "end": 70 * sr}]) == [(0, 30 * sr), (30 * sr, 60 * sr), (60 * sr, 70 * sr)]
    assert run_local.system_label("zls-turbo", True) == "zls-turbo@vad-on"
    assert run_local.system_label("zls-turbo", False) == "zls-turbo@vad-off"


def test_systems_file_lists_pinned_open_models():
    systems = {s["repo"]: s for s in run_local.load_systems(BENCH / "systems.yaml", None)}
    for repo in ("SitrakaForler/whisper-large-v3-turbo-lb-ct2", "ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish",
                 "unilux/whisper-medium-v1-luxembourgish", "BuzzASR/luxembourgish"):
        assert repo in systems
    names = [s["name"] for s in systems.values()]
    assert len(names) == len(set(names))
    for s in systems.values():
        assert len(s["revision"]) == 40 and set(s["revision"]) <= set("0123456789abcdef")
        assert s["engine"] in ("faster-whisper", "faster-whisper-converted", "transformers")
        assert s["licence"]
    with pytest.raises(SystemExit):
        run_local.load_systems(BENCH / "systems.yaml", ["no-such-system"])


def _args(**changes):
    import argparse

    values = dict(vad_preset="default", language="lb", beam_size=5, compute_type="int8", device="cpu", threads=4)
    values.update(changes)
    return argparse.Namespace(**values)


def test_decode_settings_say_what_is_really_used():
    systems = {s["name"]: s for s in run_local.load_systems(BENCH / "systems.yaml", None)}
    turbo = run_local.decode_settings(systems["zls-turbo"], False, _args(beam_size=5))
    assert turbo["beam_size"] == 5 and turbo["temperature"] == 0.0 and turbo["device"] == "cpu"
    assert turbo["vad"] is False and turbo["vad_parameters"] is None and turbo["compute_type"] == "int8"
    # BuzzASR runs with the decoding of its card: beam 1 whatever --beam-size says.
    buzz = run_local.decode_settings(systems["buzzasr-lb"], False, _args(beam_size=5))
    assert buzz["beam_size"] == 1 and buzz["generate"]["num_beams"] == 1 and buzz["compute_type"] == "float32"
    assert run_local.VAD_PRESETS["app"] == {"min_silence_duration_ms": 500}  # as in space/pipeline/transcribe.py


def test_resume_refuses_other_settings():
    systems = {s["name"]: s for s in run_local.load_systems(BENCH / "systems.yaml", None)}
    model = run_local.model_block(systems["zls-turbo"])
    decode = run_local.decode_settings(systems["zls-turbo"], False, _args(beam_size=5))
    rows = [json.loads(json.dumps({"id": "u1", "model": model, "decode": decode}))]  # as read back from the file
    run_local.check_resume(rows, model, decode, "x.jsonl")  # same settings: accepted
    with pytest.raises(SystemExit) as error:
        run_local.check_resume(rows, model, run_local.decode_settings(systems["zls-turbo"], False, _args(beam_size=1)), "x.jsonl")
    assert "beam_size" in str(error.value)
    with pytest.raises(SystemExit) as error:
        run_local.check_resume(rows, {**model, "revision": "0" * 40}, decode, "x.jsonl")
    assert "revision" in str(error.value)
    with pytest.raises(SystemExit):  # a row written before the settings were recorded
        run_local.check_resume([{"id": "u1"}], model, decode, "x.jsonl")


# ------------------------------------------------------------------ leak check

def test_leak_check_coverage(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Dat ass e ganz laange Saz aus dem Trainingsdatesaz, deen ee kennt.\nhal d'Aen op an do!\n", encoding="utf-8")
    index = leak_check.build_index(leak_check.corpus_lines(corpus), n=6)
    # Same sentence, other punctuation and case: every word is covered.
    assert leak_check.covered_share("dat ass e ganz laange saz aus dem trainingsdatesaz deen ee kennt", index) == 1.0
    # Unrelated sentence: nothing.
    assert leak_check.covered_share("Haut reent et zu Esch an et ass kal dobaussen", index) == 0.0
    # 5-word corpus line (shorter than n) found whole inside a 10-word reference: 5 / 10.
    assert leak_check.covered_share("an dunn sot hien hal d'aen op an do elo", index) == 0.5
    # A 6-word overlap in a 12-word reference: 6 / 12.
    assert leak_check.covered_share("hie sot dat ass e ganz laange saz an dunn war roueg", index) == 0.5
    tsv = tmp_path / "lod.tsv"
    tsv.write_text("\ufeffexample_id\ttext\tduration\nx1\thal d'Aen op an do!\t1.2\n", encoding="utf-8")
    assert list(leak_check.corpus_lines(tsv)) == ["hal d'Aen op an do!"]


def test_leak_check_flags_a_long_run_in_a_long_clip():
    # A 10-minute clip: 1,500 words, 200 of them lifted in one piece from a training set.
    passage = [f"t{k}" for k in range(200)]
    index = leak_check.build_index([" ".join(passage)], n=6)
    ref = {"text": " ".join([f"a{k}" for k in range(650)] + passage + [f"b{k}" for k in range(650)])}
    found = leak_check.check(ref, index)
    assert round(found["share"], 3) == 0.133  # 200 / 1500: under the 20 % criterion
    assert found["longest_run"] == 200 and found["covered_words"] == 200
    assert found["longest_run_text"].split()[0] == "t0" and found["longest_run_text"].split()[-1] == "t199"
    # Two separate 6-word overlaps: 12 covered words, longest run 6.
    index = leak_check.build_index(["een zwee dräi véier fënnef sechs", "siwen aacht néng zéng eelef zwielef"], n=6)
    text = "x een zwee dräi véier fënnef sechs y z siwen aacht néng zéng eelef zwielef w"
    assert leak_check.longest_run(text, index) == (6, "een zwee dräi véier fënnef sechs")
    assert leak_check.check({"text": text, "text_normalised": None}, index)["covered_words"] == 12
    assert leak_check.longest_run("näischt dovun", index) == (0, "")
