"""Grouping, pairing, validation split, leak search and shard planning of build_open_data.py,
on small synthetic data (no network), then the whole build on a small synthetic tree."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import wave
import zipfile

import pytest

from build_open_data import (
    Fragment,
    build,
    choose_validation,
    clean_text,
    clip_bounds,
    drop_repeated,
    fetch_fleurs,
    find_leaks,
    flag_metadata,
    fleurs_sentences,
    group_fragments,
    is_repetition_loop,
    join_text,
    metadata_seconds,
    pair_normalised,
    parse_args,
    past_end_of_audio,
    plan_shards,
    read_fragments,
    recording_id,
    repetition_loops,
    year_of,
)


def frags(*spans):
    return [Fragment(b, e, f"saz {i}") for i, (b, e) in enumerate(spans)]


# --- grouping


def test_groups_never_exceed_the_limit_and_never_cut_a_fragment():
    fragments = frags((0, 9), (9, 18), (18, 27), (27, 36), (36, 44), (44, 57), (57, 60))
    groups, stats = group_fragments(fragments, max_len=30, max_gap=2)
    assert groups == [[0, 1, 2], [3, 4, 5], [6]]
    for group in groups:
        assert fragments[group[-1]].end - fragments[group[0]].begin <= 30
    assert sorted(i for g in groups for i in g) == list(range(7))
    assert not stats


def test_a_fragment_longer_than_the_limit_is_discarded_counted_and_splits_the_run():
    fragments = frags((0, 5), (5, 40.5), (40.5, 45), (45, 50))
    groups, stats = group_fragments(fragments, max_len=30)
    assert groups == [[0], [2, 3]]
    assert stats["too_long"] == 1
    assert stats["too_long_seconds"] == 35.5


def test_a_fragment_of_exactly_the_limit_is_kept():
    groups, stats = group_fragments(frags((0, 30), (30, 31)), max_len=30)
    assert groups == [[0], [1]]
    assert stats["too_long"] == 0


def test_a_long_silence_closes_the_clip():
    groups, _ = group_fragments(frags((0, 4), (4.5, 8), (12, 15), (15, 17)), max_len=30, max_gap=2)
    assert groups == [[0, 1], [2, 3]]


def test_overlapping_fragments_stay_together():
    groups, _ = group_fragments(frags((0, 10), (9.8, 20)), max_len=30)
    assert groups == [[0, 1]]


def test_empty_text_bad_times_and_excluded_fragments_are_counted_and_split_the_run():
    fragments = [Fragment(0, 2, "a"), Fragment(2, 4, "  "), Fragment(4, 6, "b"), Fragment(7, 6.5, "c"), Fragment(8, 9, "d"), Fragment(9, 10, "e")]
    groups, stats = group_fragments(fragments, keep=[True, True, True, True, False, True])
    assert groups == [[0], [2], [5]]
    assert (stats["empty_text"], stats["bad_times"], stats["excluded"]) == (1, 1, 1)


def test_a_fragment_of_punctuation_only_counts_as_empty():
    fragments = [Fragment(0, 2, "Moien."), Fragment(2, 3, ". , , , , ,"), Fragment(3, 5, "Äddi.")]
    groups, stats = group_fragments(fragments)
    assert groups == [[0], [2]]
    assert stats["empty_text"] == 1


def test_no_fragments_no_groups():
    assert group_fragments([]) == ([], {})


def test_fragments_that_run_past_the_end_of_the_audio_are_found():
    fragments = frags((0, 10), (10, 20.3), (20.3, 26), (30.1, 30.4), (31, 29))
    assert past_end_of_audio(fragments, audio_seconds=20.0, tolerance=0.5) == [2, 3]
    assert past_end_of_audio(fragments, audio_seconds=20.0, tolerance=0.1) == [1, 2, 3]
    assert past_end_of_audio(fragments, audio_seconds=40.0) == []


# --- the same audio transcribed twice, overlapping time stamps, loops


def test_a_fragment_with_the_time_stamps_of_an_earlier_one_is_dropped():
    fragments = [Fragment(0, 8, "a"), Fragment(267.72, 276.09, "Jo. Just wat."), Fragment(267.72, 276.09, "Jo. Just wat."), Fragment(276.09, 280, "b")]
    keep, stats = drop_repeated(fragments)
    assert keep == [True, True, False, True]
    assert stats["repeated_same_times"] == 1
    assert round(stats["repeated_same_times_seconds"], 2) == 8.37
    groups, _ = group_fragments(fragments, keep=keep)
    assert join_text(fragments[i].text for g in groups for i in g).count("Jo. Just wat.") == 1


def test_a_second_transcript_of_mostly_the_same_audio_is_dropped_even_later_in_the_file():
    fragments = [
        Fragment(267.72, 276.09, "first transcript"), Fragment(276.09, 284.0, "next"),
        Fragment(267.52, 276.04, "second transcript of the same passage"),   # not next to the first one
        Fragment(283.0, 283.8, "short one inside the fragment before"),
        Fragment(284.0, 290.0, "fine"),
    ]
    keep, stats = drop_repeated(fragments)
    assert keep == [True, True, False, False, True]
    assert stats["repeated_overlap"] == 2 and stats["repeated_same_times"] == 0


def test_a_long_fragment_that_swallows_an_earlier_short_one_is_the_one_dropped():
    keep, _ = drop_repeated([Fragment(10, 12, "short"), Fragment(5, 25, "long, later in the file")])
    assert keep == [True, False]


def test_a_margin_between_neighbours_is_not_a_repeat_but_a_long_overlap_is():
    fragments = frags((0, 10), (9.5, 20), (17.5, 30))   # 0.5 s shared, then 2.5 s
    assert drop_repeated(fragments, max_overlap=2.0)[0] == [True, True, False]
    assert drop_repeated(fragments, max_overlap=3.0)[0] == [True, True, True]


def test_unusable_fragments_never_push_out_a_usable_one():
    fragments = [Fragment(0, 45, "longer than a clip, discarded by the grouping"), Fragment(5, 15, "usable"), Fragment(9, 8, "bad times")]
    keep, stats = drop_repeated(fragments, max_len=30)
    assert keep == [True, True, True] and not stats


def test_clip_edges_are_cut_in_the_middle_of_the_overlap_with_a_neighbour():
    fragments = frags((0, 10.4), (10.0, 20.2), (20.0, 29.0), (28.6, 40.0))
    everyone = range(len(fragments))
    assert clip_bounds(fragments, [0, 1], everyone) == (0, 20.1)
    start, end = clip_bounds(fragments, [2], everyone)
    assert (round(start, 3), round(end, 3)) == (20.1, 28.8)
    assert clip_bounds(fragments, [3], everyone)[0] == end   # no audio on both sides, none lost
    assert clip_bounds(fragments, [0, 1, 2, 3], everyone) == (0, 40.0)


def test_clip_edges_stay_where_they_are_without_overlap_or_when_it_is_too_long():
    fragments = frags((0, 10), (10, 20), (16, 30))
    assert clip_bounds(fragments, [0], range(3)) == (0, 10)
    assert clip_bounds(fragments, [1], range(3), max_overlap=2.0) == (10, 20)
    assert clip_bounds(fragments, [1], range(3), max_overlap=5.0) == (10, 18)
    assert clip_bounds(fragments, [1], [0, 1]) == (10, 20)   # fragment 2 was dropped: not a neighbour


def test_a_transcript_stuck_on_a_phrase_is_a_loop_and_a_repeated_word_is_not():
    assert is_repetition_loop("Ech ginn der eeben den neien, den neien, den neien, den neien, den neien, den")
    assert is_repetition_loop("déi an d’nächst Jor, " * 5)
    assert is_repetition_loop("ë, " * 8)
    assert is_repetition_loop(". , , , , , , , , , Mir ginn elo op d'Schlass.")
    assert not is_repetition_loop("Solle mer déi da lo vun uewen no ënnen... Nee, nee, nee, nee, nee.")
    assert not is_repetition_loop("Et gëtt ganz, ganz, ganz, ganz, ganz vill verschidden Opfassungen.")
    assert not is_repetition_loop("Erlieft. Erlieft. Erlieft. Erlieft. A verzielt. A verzielt. A verzielt. A verzielt.")
    assert not is_repetition_loop("")


def test_a_loop_that_runs_over_two_fragments_flags_both():
    texts = ["Dat ass ganz wichteg.", "Et geet net. Et geet net. Et geet net.", "Et geet net. Et geet net. Et geet net.", "Sou, elo weider."]
    assert repetition_loops(texts) == [False, True, True, False]
    assert repetition_loops(["Nee, nee, nee.", "Nee, nee."]) == [False, False]
    assert repetition_loops([]) == []


def test_text_is_joined_with_plain_single_spaces():
    assert clean_text("Wat\u00a0?  Dëse Kreeslaf \u200bgouf\nduerch\tdat\u202f!") == "Wat ? Dëse Kreeslaf gouf duerch dat !"
    assert join_text(["  Moien,\u00a0", "", "\u200b", "wéi\ngeet et? "]) == "Moien, wéi geet et?"


# --- pairing verbatim and normalised text


def test_pairs_by_time_stamps_not_by_position():
    verbatim = [Fragment(0, 3.2, "ëëë jo"), Fragment(3.2, 7, "wi geet et"), Fragment(7, 9, "nek schlecht")]
    normalised = [Fragment(7.0, 9.0, "net schlecht"), Fragment(0.0, 3.2, "jo"), Fragment(3.21, 6.99, "wéi geet et")]
    assert pair_normalised(verbatim, normalised) == ["jo", "wéi geet et", "net schlecht"]


def test_a_missing_or_shifted_normalised_fragment_gives_none():
    verbatim = [Fragment(0, 3, "a"), Fragment(3, 6, "b"), Fragment(6, 9, "c")]
    normalised = [Fragment(0, 3, "A"), Fragment(6.4, 9, "C")]
    assert pair_normalised(verbatim, normalised) == ["A", None, None]


def test_a_normalised_fragment_is_used_once_and_empty_text_is_a_valid_pairing():
    verbatim = [Fragment(0, 1, "ëëë"), Fragment(0, 1, "ëëë")]
    assert pair_normalised(verbatim, [Fragment(0, 1, "")]) == ["", None]


def test_reads_csv_with_bom_and_json(tmp_path):
    csv_path = tmp_path / "a.csv"
    csv_path.write_bytes('﻿start_seconds,end_seconds,text\n0.000,9.850,"Moien, wéi geet et?"\n10.150,20.490,Gutt.\n'.encode("utf-8"))
    json_path = tmp_path / "a.json"
    json_path.write_text(json.dumps({"fragments": [{"begin": "0.000", "end": "9.850", "lines": ["Moien, wéi geet et?"]},
                                                   {"begin": "10.150", "end": "20.490", "lines": ["Gutt."]}]}), encoding="utf-8")
    expected = [Fragment(0.0, 9.85, "Moien, wéi geet et?"), Fragment(10.15, 20.49, "Gutt.")]
    assert read_fragments(csv_path) == expected
    assert read_fragments(json_path) == expected


# --- validation split by recording


def test_recording_id_drops_only_the_part_suffix():
    assert recording_id("RTL__Kloertext__2024-04-18__part03") == "RTL__Kloertext__2024-04-18"
    # one broadcast delivered in two halves is one recording
    assert recording_id("RTL__Background-am-Gespreich-Martine-Deprez__2024-03-09__1__part06") == "RTL__Background-am-Gespreich-Martine-Deprez__2024-03-09"
    assert recording_id("RTL__Background-am-Gespreich-Martine-Deprez__2024-03-09__2__part01") == "RTL__Background-am-Gespreich-Martine-Deprez__2024-03-09"
    assert recording_id("RTL__Vun-Alabama-bis-Wyoming-50-Staaten-1-Land__2024-10-26__2") == "RTL__Vun-Alabama-bis-Wyoming-50-Staaten-1-Land__2024-10-26"
    assert recording_id("RTL__Vun-Alabama-bis-Wyoming-50-Staaten-1-Land__2024-10-18") == "RTL__Vun-Alabama-bis-Wyoming-50-Staaten-1-Land__2024-10-18"
    assert recording_id("Happy-Hippo__122-Happy-Hippo-Podcast-Marry-Dirk__2023-02-01__part10") == "Happy-Hippo__122-Happy-Hippo-Podcast-Marry-Dirk__2023-02-01"
    assert recording_id("RTL__5-Froen-un-Ali-Ruckert__2024-05-27") == "RTL__5-Froen-un-Ali-Ruckert__2024-05-27"
    assert recording_id("ZLS__LOD-Beispillssatz__part12") == "ZLS__LOD-Beispillssatz"


SOURCES = {
    "radio": {f"radio-{i:02d}": 120.0 + 10 * i for i in range(40)},       # many short recordings
    "parliament": {"session-a": 2700.0, "session-b": 9000.0},            # shortest is 45 minutes
    "book": {f"chapter-{i}": 700.0 for i in range(6)},
    "single": {"one-long-recording": 30000.0},
}


def test_validation_takes_whole_recordings_from_every_source_that_has_several():
    chosen = choose_validation(SOURCES, target_seconds=3600, floor_seconds=300, seed=7)
    assert set(chosen) == {"radio", "parliament", "book"}
    for source, names in chosen.items():
        assert names and set(names) <= set(SOURCES[source])
        assert len(names) < len(SOURCES[source])  # train keeps at least one recording
    assert "single" not in chosen


def test_a_source_whose_shortest_recording_exceeds_its_share_gives_exactly_that_one():
    chosen = choose_validation(SOURCES, target_seconds=3600, floor_seconds=300, seed=7)
    assert chosen["parliament"] == ["session-a"]
    assert chosen["book"] == ["chapter-0"]


def test_validation_split_is_reproducible_and_depends_on_the_seed():
    a = choose_validation(SOURCES, seed=7)
    b = choose_validation({s: dict(reversed(list(r.items()))) for s, r in reversed(list(SOURCES.items()))}, seed=7)
    assert a == b
    assert any(choose_validation(SOURCES, seed=s)["radio"] != a["radio"] for s in range(8, 14))


def test_validation_stays_close_to_the_target_when_recordings_are_short():
    sources = {"a": {f"a{i}": 60.0 for i in range(300)}, "b": {f"b{i}": 90.0 for i in range(100)}}
    chosen = choose_validation(sources, target_seconds=3600, floor_seconds=300, seed=1)
    total = sum(sources[s][name] for s, names in chosen.items() for name in names)
    assert 0.85 * 3600 <= total <= 1.2 * 3600


def test_no_recording_is_on_both_sides():
    chosen = choose_validation(SOURCES, seed=3)
    validation = {(s, name) for s, names in chosen.items() for name in names}
    parts = [(s, name, part) for s, recs in SOURCES.items() for name in recs for part in range(3)]
    train = {(s, name) for s, name, _ in parts if (s, name) not in validation}
    assert not train & validation
    assert {(s, name) for s, name, _ in parts} == train | validation


# --- leaks, metadata, shards


def test_finds_a_test_sentence_word_for_word_whatever_the_case_and_punctuation():
    sentences = ["D'Kanner spillen an der Stad.", "Moien"]
    texts = ["Gëschter: d’Kanner spillen an der Stad, sot hien.", "D'Kanner spillen net an der Stad.", "Moiens ass et kal.", ""]
    assert find_leaks(sentences, texts) == {0: ["d'kanner spillen an der stad"]}


def test_metadata_helpers():
    assert metadata_seconds("03:00,10") == 180.1
    assert metadata_seconds("45:15,76") == 2715.76
    assert year_of("12.10.2021") == "2021"
    assert year_of("N/A") == "undated"
    rows = [
        {"File Name": "RTL__a.wav", "Title": "Reportage iwwer den IMS", "Date": "01.02.2024", "AUDIO Location (Hugging Face)": "https://x/2026/a.wav"},
        {"File Name": "RTL__b.wav", "Title": "Interview mam Pit Everling", "Date": "01.02.2024"},
        {"File Name": "RTL__c.wav", "Title": "Journal", "Date": "13.02.2026"},
        {"File Name": "RTL__d.wav", "Title": "Kloertext: Steiersystem", "Date": "28.09.2023", "AUDIO Location (Hugging Face)": "https://x/2026/d.wav"},
    ]
    assert [r["File Name"] for r in flag_metadata(rows)] == ["RTL__a.wav", "RTL__b.wav", "RTL__c.wav"]


def test_shards_keep_the_order_and_stay_under_the_limit():
    sizes = [700, 300, 1000, 5000, 200, 200]
    shards = plan_shards(sizes, max_bytes=20 * 512 + 4096)
    assert [i for shard in shards for i in shard] == list(range(len(sizes)))
    assert shards == [[0, 1, 2], [3], [4, 5]]  # a file larger than a shard still gets one
    assert plan_shards([], 1000) == []


# --- the FLEURS file must really be the FLEURS file


def fleurs_tsv(rows: int) -> str:
    return "".join(f"{i}\t{i}.wav\tDe Saz Nummer {i} ass hei.\tde saz nummer {i} ass hei\tx\t10\tFEMALE\n" for i in range(rows))


def test_a_fleurs_file_without_sentences_stops_the_run():
    assert len(fleurs_sentences(fleurs_tsv(120), "test.tsv")) == 240
    for broken in ("", "<html><body>503 Service Unavailable</body></html>", fleurs_tsv(12)):
        with pytest.raises(SystemExit, match="distinct sentences"):
            fleurs_sentences(broken, "test.tsv")


def test_a_broken_cached_fleurs_file_is_not_trusted(tmp_path):
    (tmp_path / "test.tsv").write_text("<html>Entry not found</html>", encoding="utf-8")
    with pytest.raises(SystemExit):
        fetch_fleurs("test", tmp_path)   # the file exists: nothing is downloaded
    (tmp_path / "dev.tsv").write_text(fleurs_tsv(150), encoding="utf-8")
    assert len(fetch_fleurs("dev", tmp_path)) == 300


# --- the whole build on a small tree: LOD duplicates, male-voice recordings, repeats, resume

FEMALE = [("f1", "KAZ1", "d'Kaz schléift um Canapé", 2.0), ("f2", "HOND1", "den Hond billt", 1.5),
          ("f3", "SCHLOFEN1", "D'Kaz schléift um Canapé.", 2.1), ("f4", "BUCH1", "ech liesen e Buch", 1.8)]
MALE = [("m1", "AUTO1", "den Auto ass rout", 2.0), ("m2", "HAUS1", "d'Haus ass grouss", 2.0),
        ("m3", "REEN1", "et reent haut", 2.0), ("m4", "KAFFI1", "de Kaffi ass waarm", 2.0)]


def write_wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))


def write_part(root, folder, stem, seconds, rows, normalised=None):
    """rows: (begin, end, verbatim text). normalised: texts in the same order, default = verbatim."""
    write_wav(root / "Verbatim" / folder / f"{stem}.wav", seconds)
    for version, texts in (("Verbatim", [r[2] for r in rows]), ("Normalised", normalised or [r[2] for r in rows])):
        suffix = "" if version == "Verbatim" else "_normalised"
        path = root / version / (folder + suffix) / f"{stem}{suffix}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(f'{b:.3f},{e:.3f},"{text}"\n' for (b, e, _), text in zip(rows, texts))
        path.write_bytes(("\ufeffstart_seconds,end_seconds,text\n" + body).encode("utf-8"))


@pytest.fixture()
def tree(tmp_path):
    data = tmp_path / "data"
    cc0 = data / "zls-cc0"
    # one broadcast in two halves, plus two other broadcasts
    write_part(cc0, "RTL", "RTL__Show__2024-01-01__1__part01", 60.0, [
        (0.0, 10.4, "Gudde Moien, ë, alleguer."), (10.0, 20.2, "Haut\u00a0schwätze mer iwwer d'Wieder."),
        (10.0, 20.2, "Haut\u00a0schwätze mer iwwer d'Wieder."),            # same time stamps: a repeat
        (20.0, 29.0, "Et reent zanter dräi Deeg."), (28.6, 40.0, "Mar gëtt et besser."),
        (40.0, 45.0, "den neien, den neien, den neien, den neien, den neien, den"),   # a loop
        (58.0, 63.0, "Dëse Saz geet iwwer d'Enn vum Fichier eraus."),        # past the end of the audio
    ], normalised=["Gudde Moien alleguer.", "Haut schwätze mer iwwer d'Wieder.", "Haut schwätze mer iwwer d'Wieder.",
                   "Et reent zanter dräi Deeg.", "Muer gëtt et besser.", "den neien", "Dëse Saz geet eraus."])
    write_part(cc0, "RTL", "RTL__Show__2024-01-01__2__part01", 30.0, [(0.0, 12.0, "Zweeten Deel vun der Emissioun."), (12.0, 25.0, "An nach e Saz.")])
    write_part(cc0, "RTL", "RTL__Other__2024-02-02", 20.0, [(0.0, 15.0, "Eng aner Emissioun.")])
    write_part(cc0, "RTL", "RTL__Third__2024-03-03", 20.0, [(1.0, 16.0, "Eng drëtt Emissioun.")])
    # the "ZLS" source: one recording of LOD sentences (mostly male voice), one of other speech
    write_part(cc0, "ZLS", "ZLS__LOD-Beispillssatz__part01", 30.0, [
        (0.0, 2.0, "Den Auto ass rout."), (2.5, 4.5, "D'Haus ass grouss."), (5.0, 7.0, "Et reent haut."),
        (7.5, 9.5, "De Kaffi ass waarm."), (10.0, 12.0, "D'Kaz schléift um Canapé."),
    ])
    write_part(cc0, "ZLS", "ZLS__LOD-Trial", 30.0, [(0.0, 5.0, "Eng fräi Ried iwwer d'Sprooch."), (5.0, 6.5, "Den Hond billt."), (7.0, 12.0, "An nach eng Ausso.")])
    write_part(data / "zls-ccby", "Konterbont", "Konterbont__Artikel-A__2025-05-05", 20.0, [(0.0, 9.0, "Een Artikel.")])
    write_part(data / "zls-ccby", "Konterbont", "Konterbont__Artikel-B__2025-06-06", 20.0, [(0.0, 8.0, "Nach een Artikel.")])
    lod = data / "lod-fraestemm"
    lod.mkdir(parents=True)
    header = "example_id\tentry_id\ttext\tattributes\tvoice\tspeaker\tpath_ogg\tpath_m4a\tduration\n"
    lines = [f"{i}\t{entry}\t{text}\tEGS\tFraestëmm\tF1\tx\tx\t{seconds}\n" for i, entry, text, seconds in FEMALE]
    lines += [f"{i}\t{entry}\t{text}\tEGS\tMännerstëmm\tM1\tx\tx\t{seconds}\n" for i, entry, text, seconds in MALE]
    (lod / "iwwersiicht.csv").write_bytes(("\ufeff" + header + "".join(lines)).encode("utf-8"))
    ogg = b"not audio"
    if shutil.which("ffmpeg"):
        made = tmp_path / "tone.ogg"
        run = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=16000:duration=2",
                              "-c:a", "flac", "-f", "ogg", str(made)])
        if run.returncode == 0:
            ogg = made.read_bytes()
    with zipfile.ZipFile(lod / "fraestemm.ogg.zip", "w") as zf:
        for i, _, _, _ in FEMALE:
            zf.writestr(f"fraestemm.ogg/{i}.ogg", ogg)
    return tmp_path


def run_build(tree, *options):
    out = tree / ("out" + "".join(options).replace("--", "-").replace(" ", ""))
    stats = build(parse_args(["--data", str(tree / "data"), "--out", str(out), "--skip-fleurs-check", "--val-floor-minutes", "0.1", *options]))
    read = lambda name: [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines()]  # noqa: E731
    held = read("manifest.held-out-lod-male.jsonl") if (out / "manifest.held-out-lod-male.jsonl").is_file() else None
    return out, stats, read("manifest.jsonl"), held


def test_build_holds_the_male_voice_recording_and_keeps_one_copy_of_lod_sentences(tree):
    out, stats, rows, held = run_build(tree, "--skip-audio")
    assert set(rows[0]) == {"audio", "text", "text_normalised", "source", "licence", "recording", "start", "end", "duration", "split"}
    # LOD: the female voice only, and the sentence listed twice is kept once
    lod = [r for r in rows if r["source"] == "lod-f1"]
    assert sorted(r["text"] for r in lod) == ["d'Kaz schléift um Canapé", "den Hond billt", "ech liesen e Buch"]
    assert stats["lod"]["female_same_text_dropped"] == 1 and stats["lod"]["index_rows_other_voice_not_used"] == 4
    assert all(r["text_normalised"] is None and r["licence"] == "CC0-1.0" for r in lod)
    # ZLS source: the recording of male-voice sentences is in its own manifest, all train
    assert stats["zls_vs_lod"]["recordings_mostly_lod_sentences"] == ["ZLS__LOD-Beispillssatz"]
    assert stats["zls_vs_lod"]["same_text_as_lod_male"] == 4 and stats["zls_vs_lod"]["same_text_as_lod_female"] == 2
    assert not [r for r in rows if r["recording"] == "ZLS__LOD-Beispillssatz"]
    assert held and {r["recording"] for r in held} == {"ZLS__LOD-Beispillssatz"} and {r["split"] for r in held} == {"train"}
    # a ZLS fragment that reads a female-voice LOD sentence is in no ZLS clip
    zls_text = " ".join(r["text"] for r in rows + held if r["source"] == "zls-zls")
    assert "Kaz" not in zls_text and "Hond" not in zls_text
    assert "Den Auto ass rout." in zls_text and "Eng fräi Ried" in zls_text
    assert stats["grouping"]["zls-zls"]["same_text_as_lod"] == 2
    assert stats["totals"]["held_out_clips"] == len(held)
    assert {r["licence"] for r in rows if r["source"] == "zls-ccby"} == {"CC-BY-4.0"}


def test_build_can_include_or_drop_the_male_voice_recording(tree):
    _, stats, rows, held = run_build(tree, "--skip-audio", "--lod-male-overlap", "include")
    assert held is None and stats["totals"]["held_out_clips"] == 0
    assert "Den Auto ass rout." in " ".join(r["text"] for r in rows if r["recording"] == "ZLS__LOD-Beispillssatz")
    assert "Kaz" not in " ".join(r["text"] for r in rows if r["source"] == "zls-zls")
    _, stats, rows, held = run_build(tree, "--skip-audio", "--lod-male-overlap", "drop")
    assert held is None and not [r for r in rows if r["recording"] == "ZLS__LOD-Beispillssatz"]
    assert stats["grouping"]["zls-zls"]["same_text_as_lod"] == 6   # 4 male + 1 female in that recording, 1 female in the other


def test_build_drops_repeats_loops_and_text_without_audio_and_clips_share_no_audio(tree):
    _, stats, rows, _ = run_build(tree, "--skip-audio")
    first = sorted((r for r in rows if "__1__part01" in r["audio"]), key=lambda r: r["start"])
    counters = stats["grouping"]["zls-rtl"]
    assert counters["repeated_same_times"] == 1 and counters["repetition_loop"] == 1 and counters["past_end_of_audio"] == 1
    text = " ".join(r["text"] for r in first)
    assert text == "Gudde Moien, ë, alleguer. Haut schwätze mer iwwer d'Wieder. Et reent zanter dräi Deeg. Mar gëtt et besser."
    assert " ".join(r["text_normalised"] for r in first) == "Gudde Moien alleguer. Haut schwätze mer iwwer d'Wieder. Et reent zanter dräi Deeg. Muer gëtt et besser."
    assert stats["text_conventions"]["odd_whitespace_replaced_by_a_space_or_removed"]["zls-rtl"] == {"U+00A0": 1}
    # 30 s limit: the four fragments (0 to 40 s) make two clips that meet in the middle of an overlap
    assert len(first) == 2 and first[0]["end"] == first[1]["start"]
    assert all(r["duration"] <= 30 and r["duration"] == round(r["end"] - r["start"], 3) for r in rows)


def test_build_keeps_the_two_halves_of_a_broadcast_on_the_same_side(tree):
    for seed in range(6):
        _, stats, rows, _ = run_build(tree, "--skip-audio", "--seed", str(seed))
        sides = {}
        for r in rows:
            sides.setdefault((r["source"], r["recording"]), set()).add(r["split"])
        assert all(len(s) == 1 for s in sides.values())
        assert ("zls-rtl", "RTL__Show__2024-01-01") in sides and stats["sources"]["zls-rtl"]["recordings"] == 3
        assert {r["split"] for r in rows if r["source"] == "zls-rtl"} == {"train", "validation"}
        assert stats["validation"]["sources_without_validation"] == ["zls-zls"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to cut clips")
def test_build_writes_clips_and_shards_and_a_second_run_redoes_nothing(tree, capsys):
    out, stats, rows, held = run_build(tree)
    for r in rows + held:
        assert (out / r["audio"]).stat().st_size > 0
    with tarfile.open(out / "shard-000.tar") as tf:
        names = tf.getnames()
        inner = [json.loads(line) for line in tf.extractfile("manifest-shard-000.jsonl").read().decode("utf-8").splitlines()]
    assert inner == rows and sorted(names) == sorted(["manifest-shard-000.jsonl"] + [r["audio"] for r in rows])
    with tarfile.open(out / "held-out-lod-male-000.tar") as tf:
        assert sorted(tf.getnames()) == sorted(["manifest-held-out-lod-male-000.jsonl"] + [r["audio"] for r in held])
    sums = (out / "SHA256SUMS").read_text()
    assert [line.split("  ")[1] for line in sums.splitlines()] == [
        "manifest.jsonl", "stats.json", "manifest.held-out-lod-male.jsonl", "shard-000.tar", "held-out-lod-male-000.tar"]
    stamp = (out / "shard-000.tar").stat().st_mtime_ns
    capsys.readouterr()
    run_build(tree)
    printed = capsys.readouterr().out
    assert f"clips: 0 written, {len(rows) + len(held)} already there" in printed
    assert "shard-000.tar: kept" in printed and "held-out-lod-male-000.tar: kept" in printed
    assert (out / "shard-000.tar").stat().st_mtime_ns == stamp and (out / "SHA256SUMS").read_text() == sums
