# LB-Real: protocol of the bench

LB-Real compares, on the same real-life audio, the systems that transcribe Luxembourgish: open
models that run locally and online services tested by hand. This file is the protocol. It is
written before the test set exists, so that the rules are fixed before anyone sees a result.

Status on 24/11/2026: the code in this folder runs end to end. The scoring code, the imports,
the reference builder and the leak check are covered by `tests/test_bench_score.py`; the two
decoding engines of `run_local.py` are not unit-tested, they were only run on a few recordings.
The test set itself is not recorded yet, and no comparative result has been produced or
published.

## 1. Conflict of interest

lb-whisper, the model of this repository, is one of the systems tested. I am therefore both
the author of the bench and the author of a contestant. A reader should assume that this biases
every choice I get to make, and check that the protocol leaves me few of them:

- **Rules first.** This protocol, the normaliser (`finetune/lb_text.py`) and the scoring code
  are committed and tagged before the first system is run on the test set. The scores file
  records the SHA-256 of the normaliser it used. A later change of rule means a new tag and a
  full re-run of every system, stated in the results.
- **References I do not write.** The reference transcripts are made by two native speakers
  other than me, who never see the output of any system (section 4).
- **No tuning on the test set.** The same decoding settings apply to every open model
  (section 7). lb-whisper gets no setting, prompt or post-processing that the others do not get,
  and nothing is adjusted after looking at test results. Development uses other audio.
- **No training on the test set, ever.** Test recordings are covered by the "test only" consent
  (CONSENT.md, part A) and are not used to train lb-whisper, now or later.
- **Everything recomputable.** For each system I publish the raw transcripts, the per-utterance
  rows, the list of skipped passages and the exact versions. Where the consent allows it
  (CONSENT.md, A.3), the audio and references are published too. Where the speaker only agreed
  to A.1b, a tested party that wants to verify its results can receive the recording and must
  delete it afterwards. Without A.1b or A.3, the audio goes to nobody: the tested parties
  receive the reference transcript, their own transcript and the scores, nothing else.
- **Right of reply before publication** (section 9), and the reply is published unedited.
- **Results are published whatever the ranking.** The FLEURS comparison of this repository
  already reports the uni.lu and ZLS models as more accurate than lb-whisper (README, "What
  this repository adds").
- lb-whisper is marked `author_of_this_bench: true` in `systems.yaml` and is flagged as such in
  every published table.

What the protocol cannot remove: I choose the genres, I recruit the volunteers, and I decide
when the test set is "done". Anyone is welcome to re-run the bench on their own audio; the
code needs nothing from me.

## 2. Audio

Target, to be confirmed once volunteers are found: six genres, at least 30 minutes each, at
least 10 different speakers overall, clips of 10 minutes at most.

| Genre | What it is | Why |
|---|---|---|
| `read` | prepared text read aloud (news style) | the condition most training data covers |
| `public-meeting` | council or association meeting, one speaker at a time, room microphone | formal spontaneous speech |
| `conversation` | two or three people talking freely, podcast-like | overlaps, allegro forms, particles |
| `work-meeting` | laptop microphone on a table, speakers at 1 to 3 metres | the use case of the meeting-notes app |
| `voice-note` | phone recording, re-encoded by a messaging app | compression and handling noise |
| `mixed` | Luxembourgish with French and German passages | code-switching, the everyday reality |

Each clip records, in `refs.jsonl`: genre, recording device, the codec it went through if any,
the consent parts given, and the id of the original recording (clips of the same recording are
resampled together, `--ci-unit recording`). Device, codec and consent come from a CSV file kept
with the audio (`id,device,codec,consent`) that `make_refs.py --meta` copies into every row; a
clip without a row in that file stops the build.

Passages that the two transcribers both find unintelligible are cut out when the clips are
made, before the set is frozen. Nothing is masked or excluded after a system has been run.

## 3. Rights and consent

- Only audio recorded for this bench with written consent (CONSENT.md), or audio whose licence
  allows this use and redistribution, with the licence written in the manifest.
- Consent has two separate parts: A, use as test material; B, reuse for training open models.
  B is optional and never assumed. A recording given under A only never becomes training
  material. It is kept as long as the test set is in use and at most until the date written on
  the form, then deleted.
- Inside part A, three things are separate boxes, each optional: handing the recording to a
  tested party that wants to verify its results (A.1b), uploading it to online services (A.2),
  and publishing it (A.3).
- Testing online services means uploading audio to third parties, some of which process it
  outside the European Union. The consent form says so, names the services and where each one
  processes the audio. A volunteer can accept the local test and refuse the upload. The results
  are then published as two tables: every system on the clips with A.2
  (`score.py --only-consent A2`), and the open models alone on the clips without it
  (`--without-consent A2`). An online service is never scored on a clip it did not receive.
- No speaker identification: no names in file names, manifests or transcripts unless the person
  is a public figure speaking in public. Speakers are numbered per recording.
- Withdrawal: a volunteer can withdraw at any time. The recording is deleted, the set gets a new
  version number, and results on the old version are marked as such.
- Before testing an online service I read its terms of use. If they forbid benchmarking or
  publishing results, I ask for permission; without it the service is not tested, and the
  report says so.

## 4. Reference transcripts

Two references per clip, following the two versions the ZLS defines in the README of its open
training data (read on 24/11/2026):

- **verbatim**: faithful to the audio. Dialectal variants, allegro forms and discourse
  particles are written as spoken.
- **normalised**: the same recording with those phenomena brought to the official standard
  spelling, and particles removed where that makes sense.

Process:

1. Transcriber 1 and transcriber 2, both native speakers, transcribe each clip independently
   from the audio alone. No system output is shown to them, not even as a draft: a draft from
   one system biases the reference towards that system.
2. Their agreement is measured with `score.py` (transcriber 2 scored against transcriber 1).
   That number is published: it is the floor below which differences between systems mean
   nothing.
3. They resolve their differences together by listening again. The adjudicated transcript is
   the reference. The two original transcripts are kept.
4. Each reference carries timestamps per sentence or breath group, as in the ZLS data
   (`start_seconds,end_seconds,text`), so that skipped passages can be located.

Conventions, identical for both transcribers:

- **French and German words** are written in the spelling of their language, as spoken
  (`à usage unique`, not a Luxembourgish respelling). Integrated loanwords follow the LOD.
  Whole sentences in French or German are transcribed in that language and stay in the
  reference: a system that drops or translates them is making an error a user would see.
- **Numbers** are written in letters as spoken (`zweedausendzwanzeg`), in both references. The
  normaliser keeps digits as they are, so a system that writes `2020` gets an error. This
  penalises a formatting habit rather than hearing; `score.py` therefore reports how many
  digit-bearing words each system produced, so the reader can size the effect. The ZLS verbatim
  transcripts mostly follow the same convention: in the complete Verbatim folder of the CC0 set
  (554 files, counted on 24/11/2026 with `python bench/digit_words.py <folder>/Verbatim`), 106
  words out of 584,707 contain a digit, 45 of them pure numbers. Most of the others are forms
  of CO2 (`co2`, `co2-neutral`, `co2-steier`) and `covid-19`.
- **Proper nouns** use the spelling the person or institution uses publicly. Case is ignored by
  the normaliser; a misspelt name is an error like any other.
- **Eifeler Regel**: the transcribers write what they hear. Because speakers and writers apply
  the n-rule unevenly, the bench also reports `wer_n_rule`, where n-rule differences are
  forgiven on both sides. It is a diagnostic, never the headline.
- **Hesitations and false starts** are in the verbatim reference and not in the normalised
  one. Both scores are published side by side: a system trained on normalised text is not
  wrong to drop `ëëë`.

## 5. Scoring

One normalisation, published, applied to references and to every hypothesis:
`finetune/lb_text.py` (lower case, punctuation dropped, apostrophes and hyphens inside words
kept, digits and diacritics untouched). `score.py` then reports, per system, overall and per
genre:

- WER and CER, with substitutions, deletions and insertions;
- `wer_n_rule` (section 4);
- a 95 % confidence interval by bootstrap over utterances, or over recordings with
  `--ci-unit recording` (2,000 draws, seed 2026, percentile method). With a single unit no
  interval is given;
- pairwise WER differences between systems with a paired bootstrap interval. **A ranking
  between two systems is only stated when that interval excludes zero**;
- skipped passages: runs of consecutive deleted reference words that stand for more than
  5 seconds of audio, counted and listed with their position and text. The duration comes from
  the reference timestamps when they exist: a segment's time is shared between its words in
  proportion to their length, and the passage lasts the union of the time spans of its deleted
  words, so that overlapping segments are not counted twice (in the ZLS open data, 18,025 of
  the 32,207 segments of the Verbatim folder start before the previous one ends, 256 of them by
  more than a second; counted on 24/11/2026). Without timestamps, the duration comes from the
  mean speaking rate of the clip. Two runs separated by a single reference word that was not
  deleted are one passage: one word aligned by chance inside a hole must not hide the hole.
  WER hides this failure: a system that silently drops 30 seconds of a meeting can still post
  a decent average.

A clip a system fails to transcribe is scored as an empty transcript (`--allow-missing`),
never dropped. Two things are refused instead of scored: a reference without a single word
(its error rate would be 0 % whatever the system wrote), and a system whose rows were not all
produced with the same model revision and the same settings.

## 6. Leak control

A test clip that a system saw in training measures memory. Three checks:

1. **Provenance.** Test audio is recorded for the bench and has never been online, which rules
   out web-scraped training sets as well as the open ones.
2. **Text overlap** with the open training sets, by `leak_check.py`: the ZLS open data (CC0 and
   CC BY sets), FLEURS `lb_lu` (all splits) and the LOD example sentences. A reference is
   removed or re-recorded on either of two criteria: more than 20 % of its words inside a
   sequence found in one of them, or a run of 25 consecutive such words or more (about ten
   seconds of speech), whatever the length of the clip. The second criterion exists because a
   clip can last ten minutes: a full minute lifted from a training set is a small share of it.
   The longest run found is written in `leak.json`. This matters most for the `read` genre:
   volunteers must not read LOD example sentences or FLEURS sentences.
3. **Declared training data.** `systems.yaml` quotes what each model card says about training
   data. If a tested party tells me that a clip overlaps its training data, the clip is
   removed for everyone.

The ZLS open data is, by the ZLS's own description, what its Schreifmaschinn was trained on. It
is therefore never test material here. I only used a few of its recordings to check that the
code of this folder runs; those numbers compare nothing and are not published.

Limit: the overlap check compares text, and the training data of the closed systems and of
models whose card does not itemise its data is unknown to me.

## 7. Versions and settings

- Open models: `systems.yaml` pins the Hub revision of each model. `run_local.py` downloads
  exactly that commit and writes it in every output row, with the versions of faster-whisper,
  CTranslate2, Transformers and PyTorch, the device and the compute type.
- Same decoding for every open model: language `lb`, beam 5, temperature 0 with no temperature
  fallback (so that a run is deterministic), no conditioning on previous text, CTranslate2
  `int8` on CPU. Exception: a model whose card documents its own generation settings and cannot
  be converted runs with those (today: BuzzASR, beam 1), and the row says so.
- Every row records what was really used, not what was asked: the beam of that model, the
  complete list of VAD parameters, the device after `auto` is resolved, the compute type.
- Voice activity detection is run **on and off** for every open model and both are reported.
  The app in `space/` decodes with VAD and the FLEURS benchmark of this repository without; the
  two rows show how much of a gap between systems is the voice filter and how much is the model.
  The VAD row uses the default parameters of faster-whisper; `--vad-preset app` uses those of
  the app (`min_silence_duration_ms = 500`), which is the run to make when the question is
  what the app does.
- A run that was interrupted can be resumed, but `run_local.py` refuses to add rows to a file
  made with another revision or other settings, and `score.py` refuses a system whose rows
  differ in either.
- BuzzASR is called as its card documents, with no language or task argument. The card says
  the prompt is in the generation configuration; I checked on 24/11/2026 with Transformers
  4.57.6 (the version that file names) and 5.24.0 that neither applies its `language` and
  `task` entries, and that the model predicts its own prompt tokens, identically in both.
  On a 75-second extract cut at fixed 30-second marks, the last window (15 s of speech)
  came back empty in both versions: the first token the model produced was its no-speech
  token. The same speech cut 0.46 s later, in a silence, was transcribed. I also tried forcing
  the entries of the file: the empty window was then transcribed, but another window of clear
  speech came back as invented text, and the language id in the file is not the one the model
  emits. `run_local.py` therefore forces nothing. This is on the list of points sent to the
  authors with the right of reply (section 9).
- Online services: `import_hypotheses.py` refuses an export without a test date and at least
  one setting. Recorded: date, plan, options chosen, model name if the service shows one,
  export format, file name.

## 8. Online services: two passes

Every online service is tested twice, at least seven days apart, on the same files, with the
same settings. Both passes are scored and published (`<service>@pass1`, `<service>@pass2`).
Neither is picked as "the" result. If the two differ, the difference is itself a finding about
the stability of the service.

## 9. Right of reply

Before anything is published, each tested party (uni.lu / LuxASR, ZLS, BuzzASR authors, and
the company behind each online service) receives: this protocol, the settings used for its
system, its transcripts, its scores with intervals, and its list of skipped passages. They have
14 days to answer.

- A factual error on my side (wrong model, wrong setting, a newer version available) leads to a
  re-run of that system, with the change noted.
- A disagreement on method is published as their reply, unedited, next to the results.
- No answer after 14 days is reported as "no answer", not as agreement.

## 10. Known limits

- A few hours of audio from a small group of volunteers is a small sample of how Luxembourgish
  is spoken. The intervals say how small.
- `int8` on CPU is not the setting the authors of each model evaluated. Converting Transformers
  weights to CTranslate2 and quantising them can move the numbers slightly.
- Temperature 0 without fallback makes a run repeatable, but it also switches off the retry
  that faster-whisper does by default when a window fails its quality checks. Someone using
  the default settings may get a different transcript, and possibly fewer skipped passages,
  than the bench reports.
- Models that only run through the window-by-window engine (today: BuzzASR) have no long-form
  decoding. With VAD off the audio is cut every 30 seconds wherever that falls, each window is
  decoded on its own, and a whole window can come back empty, not just a cut word (section 7
  describes one: 15 seconds lost). `run_local.py` prints a warning and lists the window in the
  row when a window of more than 3 seconds returns nothing. For these models the VAD-on row,
  where cuts fall in silences, is the one to read as a result; the VAD-off row measures the
  fixed cut as much as the model and is labelled as a diagnostic.
- Skipped passages are found through the word alignment, which has two blind spots. A passage
  replaced by invented words is counted as substitutions, not as a skip. And although a single
  stray word no longer splits a passage, two or more reference words matched inside a hole
  still do, and each half can fall under the 5-second threshold.
- WER and CER ignore punctuation, capitalisation, speaker attribution and readability, which
  matter to users.
- The number convention (section 4) and the single normaliser are choices; other reasonable
  choices would move every system, not necessarily by the same amount.
- `wer_n_rule` can come out slightly above the strict WER on a poor transcript: the rule looks
  at the following word, so a wrong neighbour can turn a matching word into a mismatch.
- Durations of skipped passages are estimates, exact only to the precision of the reference
  timestamps. A reference segment whose end is not after its start counts for zero seconds.
- The VAD comparison only says something on audio that has silences. On continuous speech the
  filter removes almost nothing and the two rows are the same run twice.
- Online services change without notice; their scores describe two dates, not a product.

## 11. Running it

```bash
# References from a folder audio/<genre>/<name>.wav + <name>.csv (start_seconds,end_seconds,text)
# meta.csv: id,device,codec,consent (one row per clip, consent as "A1 A1b A2")
python bench/make_refs.py --audio audio/ --normalised normalised/ --meta audio/meta.csv --out work/refs.jsonl

# Leak control against the open training sets (share above 20 % or a run of 25 words)
python bench/leak_check.py --refs work/refs.jsonl --out work/leak.json \
    --corpus zls-cc0=/data/zls-cc0/Verbatim --corpus lod=/data/lod/iwwersiicht.csv

# Open models, VAD on and off (pip install faster-whisper pyyaml; torch and transformers
# only for the models that are not already CTranslate2)
python bench/run_local.py --refs work/refs.jsonl --out work/hyps --vad both
# The same with the VAD parameters of the app, in its own folder
python bench/run_local.py --refs work/refs.jsonl --out work/hyps-app-vad --vad on --vad-preset app

# Online services, exported by hand
python bench/import_hypotheses.py --system luxasr --test-date 2026-12-01 --pass 1 \
    --set plan=free --input exports/luxasr/ --out work/hyps/luxasr.pass1.jsonl

# Scores, intervals, skipped passages
python bench/score.py --refs work/refs.jsonl --hyps work/hyps/*.jsonl --reference both --out work/scores

# Two published tables when some clips may not be uploaded: every system on the clips with A.2,
# the open models alone on the others
python bench/score.py --refs work/refs.jsonl --hyps work/hyps/*.jsonl --only-consent A2 --out work/scores-a2
python bench/score.py --refs work/refs.jsonl --hyps work/hyps/*vad-*.jsonl --without-consent A2 --out work/scores-local-only

# The figure quoted in section 4
python bench/digit_words.py /data/zls-cc0/Verbatim
```

Audio, transcripts of volunteers and result files stay outside the repository until the right
of reply has run its course.
