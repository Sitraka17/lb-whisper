# finetune/DATA.md: the open data behind v2

Version 1 was trained on FLEURS `lb_lu` only (8.46 h of read speech). In November 2026 the ZLS
published the training data of its Schreifmaschinn and the audio of the LOD example sentences
under open licences. `build_open_data.py` turns those releases into clips of at most 30 seconds,
one JSONL manifest and tar shards. The numbers on this page are printed by that script (run of
24/11/2026, written to `stats.json` next to the manifest) or read on the source pages the same day.
Three checks are one-off measurements outside the script and are marked as such: reading every
clip back, the signal levels, and the pitch estimate.

No audio is in this repository. The script reads the downloads in place and writes elsewhere.

## Sources

| Set | Publisher | Licence | Published | What it is |
|---|---|---|---|---|
| [Schreifmaschinn-Trainingsdaten - CC0](https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc0/) | ZLS | CC0 1.0 | 11/11/2026 | 554 files, 57.86 h listed in `Metadata.xlsx`: RTL, ZLS, Chambre des Députés, radio 100,7, the Speakeasy and Happy Hippo podcasts, an audiobook by Jemp Schuster. Verbatim and normalised transcripts with time stamps. |
| [Schreifmaschinn-Trainingsdaten (CC-BY)](https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc-by/) | ZLS | CC BY 4.0 | 11/11/2026 | 13 files, 1.46 h listed: articles of Konterbont Magasinn read aloud. Same layout. |
| [Audio recordings of the LOD's example sentences (female voice)](https://data.public.lu/fr/datasets/audioopname-vun-den-lod-beispillssatz-fraestemm-audio-recordings-of-the-lods-example-sentences-female-voice/) | ZLS | CC0 1.0 | 05/11/2026 | 15,205 sentences, 17.29 h, one speaker, one file per sentence. |
| Google FLEURS `lb_lu` | Google | CC BY 4.0 | 2022 | Not rebuilt here: `prepare_data.py` already handles it. Its dev and test splits stay the reference. |

Not used: the male voice of the LOD example sentences. Its page states CC BY-NC 4.0 (read on
24/11/2026), which does not allow a model I publish under MIT. The script never opens that audio.
It only reads the male rows of `iwwersiicht.csv`, the index shipped with the female set (text and
duration), to recognise the same sentences elsewhere. See "Held out" below.

## What comes out

Settings of the run: clips of at most 30 s, a silence longer than 2 s closes a clip, neighbouring
fragments that overlap are cut apart in the middle of the overlap, seed 20261124.

| Source | Licence | Recordings | Clips | Hours | Train h | Validation min |
|---|---|---:|---:|---:|---:|---:|
| lod-f1 | CC0-1.0 | 11,067 | 15,182 | 17.27 | 17.02 | 15.1 |
| zls-1007 | CC0-1.0 | 33 | 879 | 5.38 | 5.30 | 4.7 |
| zls-ccby | CC-BY-4.0 | 13 | 211 | 1.39 | 1.30 | 5.5 |
| zls-chambre | CC0-1.0 | 4 | 1,301 | 8.70 | 7.96 | 44.0 |
| zls-happy-hippo | CC0-1.0 | 2 | 163 | 0.98 | 0.82 | 9.8 |
| zls-jemp-schuster | CC0-1.0 | 13 | 510 | 3.68 | 3.52 | 9.7 |
| zls-rtl | CC0-1.0 | 93 | 3,087 | 20.27 | 19.89 | 22.9 |
| zls-speakeasy | CC0-1.0 | 11 | 728 | 4.30 | 4.05 | 14.7 |
| zls-zls | CC0-1.0 | 1 | 10 | 0.06 | 0.06 | 0.0 |
| **total** | | | **22,071** | **62.04** | **59.94** | **126.4** |

By licence: 60.65 h CC0 1.0 and 1.39 h CC BY 4.0. Train split: 59.94 h. With the FLEURS train split
(8.46 h, CC BY 4.0) that is 68.40 h of training audio, against 8.46 h for version 1.

Held out, in its own manifest and shards: 1,363 clips, 9.65 h (source `zls-zls`, CC0 1.0 in the
ZLS release). If they are added, the total is 71.69 h of clips and 78.05 h of training audio
with FLEURS.

So the honest headline today is "from 8.5 h to about 68 h", and "about 78 h" only if the ZLS
confirms the licence of the held-out recording. The releases list 76.61 h (57.86 + 1.46 + 17.29);
"What was left out" explains the difference. No two clips share audio, so no second is counted twice.

A "recording" is one broadcast, session, episode or chapter: the file name without its
`__partNN` suffix, and without the `__1` or `__2` that marks the two halves of a broadcast
delivered in two files (two RTL broadcasts). For the LOD it is the dictionary entry a sentence
illustrates (`LOD__<entry_id>`), so the example sentences of a headword stay together. The ZLS
clips are 23.4 s long on average (median 25.0 s), the LOD sentences 4.1 s.

### Manifest

`manifest.jsonl`, one line per clip, paths relative to the manifest:

```json
{"audio": "clips/zls-rtl/RTL__..._0000000-0028120.flac", "text": "verbatim transcript",
 "text_normalised": "normalised transcript or null", "source": "zls-rtl", "licence": "CC0-1.0",
 "recording": "RTL__...__2024-05-27", "start": 0.0, "end": 28.12, "duration": 28.12, "split": "train"}
```

`text` is the ZLS verbatim transcript (dialect forms, allegro forms and fillers as spoken),
`text_normalised` the ZLS normalised one (standard spelling, fillers removed). The two versions
are paired fragment by fragment on their time stamps. 6,285 of the 6,889 ZLS clips differ between
the two versions. No ZLS clip has `text_normalised: null` in this run: the only fragment without a
counterpart in the normalised file is one of the repetition loops left out below. LOD sentences
have a single version, so `text_normalised` is null for them.

Audio is FLAC, 16 kHz, mono, 16 bit. `start` and `end` are seconds in the original file.

### Where a clip starts and ends

The time stamps of the ZLS fragments leave a margin on both sides, so consecutive fragments
overlap: 18,331 of the 32,139 pairs of neighbouring fragments do, 2,708 of them by more than half
a second (Chambre 994, RTL 899). It is mostly a margin of the segmentation, not two people
speaking at once: the pattern is the same where a single person reads, 7,621 of 11,318 pairs in the LOD sentences read one by
one and 929 of 2,156 in the audiobook.

Inside a clip this does not matter. At the edge of a clip it does: cut at the raw time stamps,
the end of one clip holds the first words of the next one without their text. So where the
neighbour overlaps the edge by at most 2 s (`--max-overlap`), the edge is moved to the middle of
the overlap: 7,717 clip edges, 19.7 minutes in total. After that no two clips of a file share
audio (checked on both manifests: 0 pairs). The cut falls inside the margin, but nothing
guarantees that it never clips the first or last sound of a word; I have not listened to these
edges yet.

### Things to know before training

The words and the punctuation of the transcripts are written to the manifest as published. One
thing is changed: every run of whitespace becomes one plain space and zero-width characters are
removed. The transcripts held 250 no-break spaces (Chambre 111, 100,7 73, RTL 46, Konterbont 12,
Speakeasy 6, audiobook 2), 8 line breaks, 2 zero-width spaces and 1 thin space.

Three conventions differ between the sources and are a choice for the training script, not for
this one:

- **LOD sentences are dictionary examples.** 14,950 of the 15,182 start with a lower-case letter
  and only 2,737 end with a full stop, question mark or exclamation mark. 208 contain round
  brackets around an optional word, as in "(et ass) schued, datt ...".
- **Verbatim ZLS text marks noises in square brackets.** 170 clips contain a tag such as
  `[HAHAHA]` or `[mhm]` (RTL 97, Happy Hippo 29, 100,7 24, Speakeasy 11, Chambre 6, Konterbont 3).
  The normalised text has none.
- **Numbers are mostly spelled out.** Only 65 of the 6,889 ZLS clips contain a digit.

Two more things, about the audio and about what is left of repeated text:

- **Levels are those of the sources.** Measured once on all clips, outside the script: 156 LOD
  clips (10.2 minutes, 4 of them in validation) peak below 1 % of full scale, that is under
  -40 dBFS. They are quiet, not silent (no clip is all zeros). At the other end 273 ZLS clips
  reach full scale (Chambre 188, audiobook 54, 100,7 12, RTL 11, Speakeasy 4, Happy Hippo 3,
  Konterbont 1). The audio is not changed; Whisper's feature extractor normalises each window.
- **Some repeated text remains.** 9 ZLS clips still contain the same sequence of 8 words twice
  (RTL 7, Speakeasy 2). Read one by one, they are speakers repeating themselves, an enumeration,
  and a few transcripts that repeat a short passage, less than the loops left out below. With
  the held-out recording, one more: a LOD sentence read twice.

## What was left out, and why

| What | Count | Audio | Why |
|---|---:|---:|---|
| Fragments longer than 30 s | 218 | 2.74 h | Whisper takes 30 s at most and a fragment cannot be cut without a word-level alignment. Mostly 100,7 (108 fragments, 1.34 h) and Speakeasy (95, 1.17 h); also Chambre 5, Konterbont 5, RTL 3, Happy Hippo 2. |
| Fragments that run past the end of their audio file | 12 | 2.6 min | The transcript continues after the last sample: text without audio. A fragment that begins after the end of the file, or ends more than 0.5 s after it, is left out (Chambre 7, RTL 2, 100,7 1, Konterbont 1, Happy Hippo 1). 23 clips whose last fragment overshoots by less are shortened to the end of the file. |
| Fragments whose end is not after their start, or without a word | 3 | | One fragment with unusable time stamps (Chambre), two with no text (Happy Hippo). |
| Fragments with the time stamps of an earlier fragment | 9 | 51.5 s | The same line twice in the transcript, all in RTL. Kept once. |
| Fragments that mostly cover audio an earlier fragment already covers | 58 | 6.5 min | A second transcript of the same passage, or a short fragment inside a longer one: more than half of the shorter fragment is shared, or more than 2 s. The first one is kept (RTL 35, Happy Hippo 11, Chambre 9, 100,7 3). Kept as published, these gave clips whose text said a sentence twice for one stretch of audio. |
| Fragments whose transcript is stuck in a loop | 16 | 81.8 s | A phrase of two words or more five times in a row, one word eight times, or a run of lone commas, sometimes over several fragments: "den neien, den neien, den neien, ..." (RTL 14, 100,7 2). It is the pattern of a recogniser stuck on a phrase; I have not listened to them. Text like this teaches a model to loop. |
| ZLS fragments reading a sentence that is also in the LOD female set | 216 | 12.8 min | Same normalised text. One copy is enough and the LOD one is a clean single file. 128 of the 216 also have a duration within 0.5 s of the LOD clip. |
| LOD sentences whose normalised text is already in the set | 23 | 79.5 s | The same sentence listed under two dictionary entries, differing only in case, brackets or punctuation. |
| Clips containing a FLEURS test or dev sentence | 0 | | See "Leak check". |

The ZLS sets list 59.32 h and the clips made from them hold 54.42 h (44.77 h in the manifest and
9.65 h held out). Apart from the rows above, the difference is audio outside any transcribed
fragment: jingles, music and pauses longer than 2 s. The fragments themselves add up to 58.77 h,
but that sum counts every overlap between neighbours twice; the clips count it once.

### Held out: the ZLS recording of LOD sentences

The source "ZLS" of the CC0 release (9 h 55 min in its README, "mainly example sentences from the
LOD, read aloud") is two recordings. The script compares every fragment with the LOD index:

- `ZLS__LOD-Beispillssatz` (65 parts, 11,347 fragments): 10,118 fragments have exactly the text of
  a sentence that the LOD publishes with its **male** voice, and 213 the text of a female-voice
  sentence. Over the whole source, 9,120 of the 10,120 male matches also have a duration within
  0.5 s of the LOD male file. A pitch estimate on three parts gives a median of 113 Hz, against
  272 Hz on 60 files of the LOD female set (a rough one-off measurement, not part of the script:
  the order of magnitude matters, not the exact values).
- `ZLS__LOD-Trial` (1 part, 37 fragments): 2 male matches, 3 female ones. Kept: 10 clips, 3.9 minutes.

So the overlap with the LOD female set that I expected is small (216 fragments), and the large
overlap is with the male set: the same sentences, most probably the same recordings, under CC0 in
one release and CC BY-NC 4.0 in the other, both published by the ZLS. I have not compared the
audio itself, because I do not download the CC BY-NC set.

Until the ZLS confirms which licence applies, `ZLS__LOD-Beispillssatz` stays out of
`manifest.jsonl`. Its clips are cut and listed in `manifest.held-out-lod-male.jsonl` (all `train`)
and packed in `held-out-lod-male-*.tar`. `--lod-male-overlap include` puts them in the main
manifest, `--lod-male-overlap drop` does not cut them at all.

## Validation split

Whole recordings only: no two parts of the same broadcast, session or chapter are on both sides
of the split (the two halves of a broadcast delivered in two files count as one recording), and no validation clip has the same normalised text as a train clip (0 found).
Each source aims at its share of 60 minutes, with a floor of 5 minutes, and the pick is a seeded
shuffle (two runs give a byte-identical manifest).

The split separates recordings, not speakers: the metadata has no speaker labels. The LOD is one
voice, the audiobook one reader, and 4 of the 6 RTL validation broadcasts are episodes of a
series (Vun Alabama bis Wyoming) whose other episodes are in train. Validation WER on those
sources is therefore optimistic; FLEURS test stays the number to report.

The result is 126.4 minutes, not 60. Four sources have no short recording: the shortest session of
the Chambre is 44.0 minutes of clips, the shortest Speakeasy episode 14.7, Happy Hippo 9.8 and
the shortest chapter of the audiobook 9.7. Leaving them out of validation would hide the domains
that are closest to a meeting; cutting a session in two would leak speakers. A training script
that wants a shorter evaluation can sample the validation rows per source.

| Source | Validation recordings |
|---|---|
| lod-f1 | 166 dictionary entries |
| zls-rtl | 6 broadcasts |
| zls-chambre | Séance publique N° 01, 12/10/2021 |
| zls-speakeasy | Philo Pol 2, Thomas Hobbes |
| zls-happy-hippo | Happy Hippo Podcast X, 16/05/2018 |
| zls-jemp-schuster | Bluttsëffer, chapter "Freideg, de 27. Juni 1930" |
| zls-ccby | Legenden ëm den Hellege Willibrord, 13/02/2026 |
| zls-1007 | 2 broadcasts |
| zls-zls | none: a single recording is left, it stays in train |

The FLEURS dev and test splits are not touched and remain the numbers that are compared with
version 1 and with other models.

## Leak check

- **FLEURS.** The script downloads `data/lb_lu/test.tsv` and `dev.tsv` of `google/fleurs` (text
  only, no token) and searches every sentence, raw and normalised column, word for word after
  `lb_text.normalize`, in the verbatim and the normalised text of every clip, held-out clips
  included. 350 distinct test strings and 149 distinct dev strings were searched: 0 clips
  contain one. A clip that did would be dropped and counted. The script stops if a FLEURS file
  holds fewer than 100 distinct sentences, so that an error page in place of the TSV cannot
  make the check pass with nothing searched.
- **My own test audio.** I also test on recent radio reports that are in no training set. The
  script searches `Metadata.xlsx` for the year 2026 and for the keywords of the report I used in
  October. In the CC0 set: no match. The RTL files are dated 2003 (12), 2023 (40), 2024 (147) and
  2025 (3: Invité vun der Redaktioun of 09/04/2025 in two parts, and Vun Alabama bis Wyoming of
  05/10/2025); one has no date (a football programme, Lëtzebuerg against Slovakia). In the CC BY
  set: one recording of 2026, the Konterbont article of 13/02/2026 listed above, which is not
  one of my test files.

## Shards

Written next to the manifest, at most 1.9 GB each (10^9 bytes) so that they upload to Google Drive
and unpack on Colab without trouble:

| File | Size | Clips |
|---|---:|---:|
| `shard-000.tar` | 1.90 GB | 17,524 |
| `shard-001.tar` | 1.89 GB | 3,582 |
| `shard-002.tar` | 0.44 GB | 965 |
| `held-out-lod-male-000.tar` | 0.61 GB | 1,363 |

Each tar holds `clips/...` and its own manifest (`manifest-shard-000.jsonl`, same format, same
relative paths). Unpack all of them in one folder and concatenate the manifests, or copy
`manifest.jsonl` next to them: the result is the same. Validation clips come first, so
`shard-000.tar` alone already holds the whole validation split. `SHA256SUMS` covers the shards,
the manifests and `stats.json` (`shasum -a 256 -c SHA256SUMS`).

All 23,434 clips of this run were read back (a one-off check outside the script): 16 kHz, mono,
16-bit FLAC, none longer than 30 s, and the largest difference between the real duration and the
manifest is 5 ms.

## Reproduce

Download the three sets (the two ZLS sets are Hugging Face buckets linked from the data.public.lu
pages; the normalised `.wav` files are copies of the verbatim ones and are not needed), then:

```bash
# data/
#   zls-cc0/        Metadata.xlsx, Verbatim/<Source>/, Normalised/<Source>_normalised/
#   zls-ccby/       same layout
#   lod-fraestemm/  iwwersiicht.csv, fraestemm.ogg.zip
python build_open_data.py --data /path/to/data --out /path/to/open-data-v2
python build_open_data.py --data /path/to/data --out /path/to/open-data-v2 --skip-audio   # numbers only
pytest -q ../tests/test_build_open_data.py
```

Python 3.9 or later, standard library only, plus `ffmpeg` on the path. The script can be run
again: clips, shards and checksums that exist are kept. After a change of settings, `--prune`
deletes the clips that are no longer in a manifest. It needs the network once, for the two
FLEURS TSV files (`--skip-fleurs-check` runs offline and says that the check was not done).

## Attribution

To cite in the model card of any model trained on this data:

- **ZLS, CC0 set.** Zenter fir d'Lëtzebuerger Sprooch (ZLS), *Schreifmaschinn-Trainingsdaten -
  CC0*, 2026, CC0 1.0, https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc0/.
  CC0 asks for no attribution; the sources are named anyway: RTL Lëtzebuerg, Chambre des Députés,
  radio 100,7, Speakeasy, Jemp Schuster, Happy Hippo, ZLS. The release was funded by the
  Ministry for Digitalisation (Data4Gov 2024).
- **ZLS, CC BY set.** Attribution is required, in the wording asked by the `LICENSE` file:
  "© Konterbont Magasinn – Autisme Luxembourg a.s.b.l., CC BY 4.0", with a link to
  https://creativecommons.org/licenses/by/4.0/. Recordings by the ZLS, *Schreifmaschinn-Trainingsdaten
  (CC-BY)*, 2026, https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc-by/.
- **LOD.** Zenter fir d'Lëtzebuerger Sprooch (ZLS), *Audioopname vun den LOD-Beispillssätz
  (Fraestëmm)*, Lëtzebuerger Online Dictionnaire (lod.lu), 2026, CC0 1.0.
- **FLEURS.** Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations
  of Speech*, arXiv:2205.12446, 2022, CC BY 4.0, https://huggingface.co/datasets/google/fleurs.
