"""Reading the v2 open data: shard-*.tar archives + JSONL manifests in the common format.

The first tests need the standard library only. The ones that build a `datasets` object or
import train.py are skipped when the library is not installed (CI installs pytest and pyyaml)."""

from __future__ import annotations

import io
import json
import math
import struct
import tarfile
import wave
from pathlib import Path

import pytest

import prepare_data
from evaluate import _rates, fleurs_norm_rates
from prepare_data import (
    embedded_manifests,
    extract_shards,
    find_shard_inputs,
    load_shards,
    read_manifest,
    remove_extracted,
    shard_subdir,
    summarise,
)


def wav_bytes(seconds: float, freq: float = 220.0) -> bytes:
    """A 16 kHz mono 16-bit sine wave."""
    frames = int(seconds * 16_000)
    samples = (int(8000 * math.sin(2 * math.pi * freq * i / 16_000)) for i in range(frames))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16_000)
        fh.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buffer.getvalue()


def row(audio: str, text: str, normalised, source: str, licence: str, seconds: float, split: str) -> dict:
    return {"audio": audio, "text": text, "text_normalised": normalised, "source": source, "licence": licence,
            "recording": audio.rsplit("/", 1)[-1].split("_")[0], "start": 0.0, "end": seconds,
            "duration": seconds, "split": split}


ROWS = [
    row("clips/rtl_0001.wav", "Moien, ëëë wi geet et?", "Moien, wéi geet et?", "zls-rtl", "CC0-1.0", 1.0, "train"),
    row("clips/rtl_0002.wav", "D'Kanner si nek doheem.", "D'Kanner si net doheem.", "zls-rtl", "CC0-1.0", 1.5, "train"),
    row("clips/lod_0001.wav", "hal d'Aen op!", None, "lod-f1", "CC0-1.0", 0.5, "train"),
    row("clips/ccby_0001.wav", "Et ass schéi Wieder.", "Et ass schéi Wieder.", "zls-ccby", "CC-BY-4.0", 0.75, "train"),
    row("clips/chambre_0001.wav", "Här President, merci.", "Här President, merci.", "zls-chambre", "CC0-1.0", 2.0, "validation"),
]


HELD_OUT = row("clips/zls-zls/held_0001.wav", "säi jäizeg gréngen Auto", None, "zls-zls", "CC0-1.0", 0.25, "train")


def jsonl(rows: list) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows)


def add_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


@pytest.fixture()
def shard_dir(tmp_path: Path) -> Path:
    """What build_open_data.py leaves in its output folder: manifest.jsonl, shard-NNN.tar files
    that each start with the manifest of their own clips, and a held-out manifest and archive
    that must never be read. The second archive stores its clips under a folder, and macOS
    `._` companion files are scattered around."""
    folder = tmp_path / "open-data"
    folder.mkdir()
    (folder / "manifest.jsonl").write_text(jsonl(ROWS), encoding="utf-8")
    with tarfile.open(folder / "shard-000.tar", "w") as tf:
        add_member(tf, "manifest-shard-000.jsonl", jsonl(ROWS[:3]).encode("utf-8"))
        for item in ROWS[:3]:
            add_member(tf, item["audio"], wav_bytes(item["duration"]))
        add_member(tf, "clips/._rtl_0001.wav", b"not audio")
        add_member(tf, "../outside.wav", wav_bytes(0.1))
    with tarfile.open(folder / "shard-001.tar", "w") as tf:
        add_member(tf, "manifest-shard-001.jsonl", jsonl(ROWS[3:]).encode("utf-8"))
        for item in ROWS[3:]:
            add_member(tf, "shard-001/" + item["audio"], wav_bytes(item["duration"]))
    (folder / "._manifest.jsonl").write_bytes(b"\x00\x05\x16\x07")
    (folder / "._shard-000.tar").write_bytes(b"\x00\x05\x16\x07")
    (folder / "manifest.held-out-lod-male.jsonl").write_text(jsonl([HELD_OUT]), encoding="utf-8")
    with tarfile.open(folder / "held-out-lod-male-000.tar", "w") as tf:
        add_member(tf, "manifest-held-out-lod-male-000.jsonl", jsonl([HELD_OUT]).encode("utf-8"))
        add_member(tf, HELD_OUT["audio"], wav_bytes(HELD_OUT["duration"]))
    return folder


def test_find_inputs_ignores_held_out_and_macos_companion_files(shard_dir):
    manifest, archives = find_shard_inputs(shard_dir)
    assert manifest.name == "manifest.jsonl"
    assert [a.name for a in archives] == ["shard-000.tar", "shard-001.tar"]


def test_archives_alone_are_enough(shard_dir, tmp_path):
    """Only the shard-*.tar files were copied to Drive: the rows come from inside the archives."""
    (shard_dir / "manifest.jsonl").unlink()
    assert [r["audio"] for r in embedded_manifests(find_shard_inputs(shard_dir)[1])] == [r["audio"] for r in ROWS]
    rows, report = load_shards(shard_dir, tmp_path / "extracted")
    assert report["manifest"] == "inside the archives" and report["missing_audio"] == 0
    assert len(rows["train"]) == 4 and len(rows["validation_open"]) == 1
    assert not any("held" in r["audio"] or r["source"] == "zls-zls" for r in rows["train"])


def test_end_to_end_verbatim(shard_dir, tmp_path):
    rows, report = load_shards(shard_dir, tmp_path / "extracted", text_field="text")
    assert sorted(rows) == ["train", "validation_open"]
    assert [r["sentence"] for r in rows["train"]] == [r["text"] for r in ROWS[:4]]
    assert rows["validation_open"][0]["source"] == "zls-chambre"
    assert report["missing_audio"] == 0 and report["fallback_to_text"] == 0 and report["archives"] == 2
    for item in rows["train"] + rows["validation_open"]:
        with wave.open(item["audio"], "rb") as fh:
            assert fh.getframerate() == 16_000
            assert fh.getnframes() / 16_000 == pytest.approx(item["duration"])
    extracted = sorted(p.name for p in (tmp_path / "extracted").rglob("*") if p.is_file())
    assert extracted == sorted(Path(r["audio"]).name for r in ROWS)      # no ._ file, no held-out clip, nothing outside
    assert not (tmp_path / "outside.wav").exists()


def test_normalised_text_with_fallback(shard_dir, tmp_path):
    rows, report = load_shards(shard_dir, tmp_path / "extracted", text_field="text_normalised")
    sentences = [r["sentence"] for r in rows["train"]]
    assert sentences[0] == "Moien, wéi geet et?"
    assert sentences[2] == "hal d'Aen op!"            # the LOD row has no normalised text
    assert report["fallback_to_text"] == 1
    assert rows["train"][0]["reference_norm"] == "Moien, wéi geet et?"
    assert rows["train"][2]["reference_norm"] == ""


def test_summary_counts_hours_per_source_and_licence(shard_dir, tmp_path):
    rows, _ = load_shards(shard_dir, tmp_path / "extracted")
    summary = summarise(rows)
    assert summary["train"]["clips"] == 4
    assert summary["train"]["by_source"]["zls-rtl"]["clips"] == 2
    assert summary["train"]["hours_by_licence"] == {"CC-BY-4.0": round(0.75 / 3600, 2), "CC0-1.0": round(3.0 / 3600, 2)}
    assert summary["validation_open"]["clips_over_30s"] == 0


def test_audio_next_to_the_manifest_needs_no_archive(tmp_path):
    folder = tmp_path / "flat"
    (folder / "clips").mkdir(parents=True)
    (folder / "clips" / "rtl_0001.wav").write_bytes(wav_bytes(1.0))
    (folder / "manifest.jsonl").write_text(jsonl(ROWS[:1]), encoding="utf-8")
    rows, report = load_shards(folder, tmp_path / "extracted")
    assert report["archives"] == 0 and len(rows["train"]) == 1
    assert not (tmp_path / "extracted").exists()


def test_missing_audio_is_counted_not_hidden(shard_dir, tmp_path):
    (shard_dir / "shard-001.tar").unlink()
    rows, report = load_shards(shard_dir, tmp_path / "extracted")
    assert report["missing_audio"] == 2
    assert "validation_open" not in rows


def test_non_commercial_licence_is_refused(tmp_path):
    bad = row("clips/m1.wav", "hal d'Aen op!", None, "lod-m1", "CC-BY-NC-4.0", 1.1, "train")
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CC-BY-NC-4.0"):
        read_manifest(path)


def test_unknown_split_and_missing_key_are_refused(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({**ROWS[0], "split": "dev"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        read_manifest(path)
    path.write_text(json.dumps({k: v for k, v in ROWS[0].items() if k != "licence"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="licence"):
        read_manifest(path)


def test_same_file_name_in_two_folders_is_matched_on_the_full_path(tmp_path):
    with tarfile.open(tmp_path / "shard-00000.tar", "w") as tf:
        add_member(tf, "rtl/0001.wav", wav_bytes(0.2, 220))
        add_member(tf, "chambre/0001.wav", wav_bytes(0.4, 440))
    found = extract_shards([tmp_path / "shard-00000.tar"], {"rtl/0001.wav", "chambre/0001.wav"}, tmp_path / "out")
    assert found["rtl/0001.wav"].stat().st_size < found["chambre/0001.wav"].stat().st_size


def test_exact_path_in_a_later_archive_wins_over_a_namesake_in_an_earlier_one(tmp_path):
    """The file-name fallback must not grab another source's clip that happens to share the name."""
    with tarfile.open(tmp_path / "shard-000.tar", "w") as tf:
        add_member(tf, "clips/zls-rtl/part01_0000-0100.flac", b"WRONG-AUDIO")
    with tarfile.open(tmp_path / "shard-001.tar", "w") as tf:
        add_member(tf, "clips/zls-chambre/part01_0000-0100.flac", b"RIGHT")
    archives = [tmp_path / "shard-000.tar", tmp_path / "shard-001.tar"]
    key = "clips/zls-chambre/part01_0000-0100.flac"
    assert extract_shards(archives, {key}, tmp_path / "out")[key].read_bytes() == b"RIGHT"
    # same check for a member stored under a leading folder in the first archive
    with tarfile.open(tmp_path / "shard-000.tar", "w") as tf:
        add_member(tf, "old/" + key, b"WRONG-AUDIO")
    assert extract_shards(archives, {key}, tmp_path / "out2")[key].read_bytes() == b"RIGHT"


def test_loose_matches_are_used_only_when_no_archive_has_the_exact_path(tmp_path):
    with tarfile.open(tmp_path / "shard-000.tar", "w") as tf:
        add_member(tf, "elsewhere/a.flac", b"BY-NAME")             # same file name only
        add_member(tf, "shard-000/clips/b.flac", b"BY-SUFFIX")      # same path under a leading folder
        add_member(tf, "other/b.flac", b"NAMESAKE")
        add_member(tf, "x/c.flac", b"AMBIGUOUS")                    # two wanted paths end in c.flac
    wanted = {"clips/a.flac", "clips/b.flac", "one/c.flac", "two/c.flac"}
    found = extract_shards([tmp_path / "shard-000.tar"], wanted, tmp_path / "out")
    assert found["clips/a.flac"].read_bytes() == b"BY-NAME"
    assert found["clips/b.flac"].read_bytes() == b"BY-SUFFIX"
    assert sorted(found) == ["clips/a.flac", "clips/b.flac"]         # an ambiguous name is never guessed


def test_cleanup_removes_only_what_the_script_extracted(shard_dir, tmp_path):
    """--extract-dir may be a folder the user already had: its other content must survive."""
    extract_dir = tmp_path / "keep"
    (extract_dir / "mine").mkdir(parents=True)
    (extract_dir / "unrelated.txt").write_text("do not delete", encoding="utf-8")
    (extract_dir / "mine" / "notes.txt").write_text("nor this", encoding="utf-8")
    (shard_dir / "manifest.jsonl").unlink()
    load_shards(shard_dir, extract_dir / shard_subdir(0))
    assert any((extract_dir / "00").rglob("*.wav"))
    remove_extracted(extract_dir, 1)
    assert not (extract_dir / "00").exists()
    assert (extract_dir / "unrelated.txt").read_text(encoding="utf-8") == "do not delete"
    assert (extract_dir / "mine" / "notes.txt").is_file()
    # a folder that held nothing else goes away entirely
    own = tmp_path / "own"
    load_shards(shard_dir, own / shard_subdir(0))
    remove_extracted(own, 1)
    assert not own.exists()
    remove_extracted(own, 1)      # already gone: no error


def test_command_line_leaves_the_extract_folder_of_the_user_alone(shard_dir, tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "unrelated.txt").write_text("do not delete", encoding="utf-8")
    (shard_dir / "manifest.jsonl").unlink()
    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--no-fleurs", "--shards", str(shard_dir),
                                     "--extract-dir", str(keep), "--out", str(tmp_path / "out")])
    prepare_data.main()
    assert [p.name for p in keep.iterdir()] == ["unrelated.txt"]
    stats = json.loads((tmp_path / "out" / "stats.json").read_text(encoding="utf-8"))
    assert stats["train"] == 4 and stats["validation_open"] == 1


def test_extra_manifest_without_duration_gets_it_from_the_audio(tmp_path):
    """The v2 configs filter on the duration column: an --extra manifest must not yield zeros."""
    pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    (tmp_path / "a.wav").write_bytes(wav_bytes(1.25))
    (tmp_path / "b.wav").write_bytes(wav_bytes(0.5))
    manifest = tmp_path / "extra.jsonl"
    manifest.write_text(jsonl([{"audio": "a.wav", "text": "Moien", "split": "train"},
                               {"audio": str(tmp_path / "b.wav"), "text": "Äddi", "split": "train", "duration": 0.5}]),
                        encoding="utf-8")
    dsd = prepare_data.from_manifest(manifest)
    assert dsd["train"]["duration"] == [pytest.approx(1.25), pytest.approx(0.5)]


def test_fleurs_norm_measure_uses_the_reference_as_is():
    # Reference: FLEURS "normalized" column, untouched. Hypothesis: lower-cased, punctuation dropped.
    wer, cer = fleurs_norm_rates(["d'kanner si doheem"], ["D'Kanner si doheem."])
    assert wer == 0.0 and cer == 0.0
    wer, _ = fleurs_norm_rates(["d'kanner si doheem"], ["D'Kanner sinn doheem."])
    assert wer == pytest.approx(1 / 3)
    assert fleurs_norm_rates([], ["x"]) is None            # dataset built before the column existed
    assert fleurs_norm_rates(["", "a"], ["x", "a"]) is None  # not a FLEURS split
    assert _rates(["a b c d"], ["a b"]) == (0.5, pytest.approx(4 / 7))


def test_dataset_keeps_fleurs_and_open_validation_apart(shard_dir, tmp_path):
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    dsd_open, report = prepare_data.from_shards(shard_dir, tmp_path / "extracted", "text")
    assert report["splits"]["train"]["clips"] == 4
    fleurs_like = datasets.DatasetDict({
        split: prepare_data._dataset([str(next((tmp_path / "extracted").rglob("rtl_0001.wav")))],
                                     ["Eng FLEURS Saz."], ["eng fleurs saz"], ["fleurs_lb"], [1.0])
        for split in ("train", "validation", "test")
    })
    merged = prepare_data.merge(fleurs_like, [dsd_open])
    assert {split: len(ds) for split, ds in merged.items()} == {"train": 5, "validation": 1, "test": 1, "validation_open": 1}
    assert set(merged["validation"]["source"]) == {"fleurs_lb"}
    assert set(merged["test"]["source"]) == {"fleurs_lb"}
    assert merged["train"].column_names == ["audio", "sentence", "reference_norm", "source", "duration"]


def test_auto_resume_skips_a_checkpoint_cut_by_a_disconnect(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from train import last_complete_checkpoint

    assert last_complete_checkpoint(tmp_path) is None
    for step, complete in ((250, True), (500, True), (750, False)):
        folder = tmp_path / f"checkpoint-{step}"
        folder.mkdir()
        if complete:
            (folder / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-final").mkdir()
    assert last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-500")
