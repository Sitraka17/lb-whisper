# Roadmap

Where this goes next. Every step ends with a number in a table, not with "it feels better",
and each step below names its target. Steps 4 and 5 are the ones in progress; step 3 waits for
volunteers.

## Where we are (24 November 2026)

- Published model (v1): Whisper large-v3-turbo + LoRA on FLEURS `lb_lu` (8.5 h of read speech,
  CC-BY-4.0). Full test set: WER 30.15 %, CER 8.82 %.
- It is not the most accurate system for Luxembourgish. On the 300-utterance comparison subset
  the uni.lu and ZLS models have a lower WER; ours has the lowest CER. Translation: we spell the
  sounds right more often than we cut the words right. Contractions, compounds and the n-rule
  are where the words go wrong. LuxASR, the ZLS Sproochmaschinn and BuzzASR publish lower error
  rates on their own test sets (README, "Where this stands").
- Published on the Hub, with a CTranslate2 int8 export that runs on a laptop CPU, and a browser
  demo.
- v2 data is built: 68.40 h of training audio with FLEURS, against 8.46 h. v2 is not trained.
- The LB-Real bench has a protocol and scoring code. Its test set is not recorded.

## 1. Measure what we are actually bad at (no GPU needed)

Target: one table of the 50 most frequent confusions, for three models.

- [x] `wer_n_rule`: a second WER that forgives Eifeler Regel disagreements. If it drops a lot,
      part of the gap is spelling convention, not hearing.
- [ ] Error breakdown on the 934 test sentences: substitutions vs insertions vs deletions, and
      the 50 most frequent confusions. My bet: `d'`/`de`, `an`/`a`, compounds split in two.
- [ ] Same breakdown for uni.lu and ZLS. Knowing *where* they beat us is worth more than the
      headline gap.

## 2. Cheap training wins (one Colab evening each)

Target: each variant is kept only if it lowers the full-test WER by at least 1 point against
the same run without it.

- [ ] SpecAugment on (`apply_spec_augment`, `mask_time_prob` around 0.05). The switch exists
      since the v2 work (`spec_augment` in the configs); no run has measured it.
- [ ] LoRA rank 16 and 64 next to 32. One table, three rows.
- [ ] 5 epochs instead of 3, keeping the best checkpoint on dev. Only if the dev curve is
      still going down at epoch 3.

## 3. LB-Real: real-life audio, not only read sentences

FLEURS is read Wikipedia sentences. People speak Luxembourgish on the radio, in meetings and in
voice notes. `bench/PROTOCOL.md` fixes the rules before any result exists.

Target: six genres, at least 30 minutes each, at least 10 speakers overall, two reference
transcripts by native speakers other than me, and one table with a confidence interval per
system. The number I commit to for lb-whisper: none in advance. The bench is published whatever
the ranking.

- [x] Protocol, consent form, scoring code with skipped-passage detection, leak check against
      the training data. Tested on scoring; the decoders were only run on a few recordings.
- [ ] Have the consent form checked by someone qualified in data protection.
- [ ] Recruit volunteers and two native transcribers; record, including audio with real
      silences (meeting, voice note).
- [ ] Commit and tag the protocol and the normaliser before the first run on the test set.
- [ ] Run the open models; test the online services by hand, twice, a week apart.
- [ ] Right of reply (uni.lu, ZLS, BuzzASR authors, vendors), 14 days, then publish.
- [ ] `radio` and `meeting` conditions on FLEURS (telephone band with a music bed; room reverb
      with light babble), as a cheap complement to the 16 kbit/s Opus condition.

## 4. More data, with clean licences only

Target: about 85 h of open training audio with a licence per clip. Reached so far: 68.40 h, and
78.05 h if the held-out recordings below are confirmed. The raw releases add up to about 85 h;
what does not reach the training split is listed in `finetune/DATA.md`.

- [x] The licence question for the ZLS data is settled, and not by an email of mine: on
      11/11/2026 the ZLS published the training data of its Schreifmaschinn under CC0 1.0
      (about 58 h) and CC BY 4.0 (about 90 minutes), after the LOD example sentences on
      05/11/2026 (female voice, CC0 1.0).
- [x] `finetune/build_open_data.py`: 22,071 clips, 62.04 h, of which 59.94 h for training and
      126.4 min of whole recordings for validation. Every clip has its source and licence in
      the manifest, and `prepare_data.py --shards` stops on any other licence. See
      `finetune/DATA.md`.
- [ ] Ask the ZLS to confirm that CC0 covers its recordings of LOD example sentences (9.65 h,
      held out: the same sentences with the LOD male voice are CC BY-NC). One email; with
      them, 78.05 h of training audio.
- [ ] Recover the 218 fragments longer than 30 s (2.74 h) with a word-level alignment.
- [ ] Talk to Luxembourgish media and institutions that own spoken archives. The offer: an
      open evaluation on their audio (their data never leaves their hands), and the fine-tuned
      model back for free if they want it.

## 5. v2: train on the open data

Target: **under 26 % WER on the full FLEURS test split** (934 utterances, the `wer` field of
`evaluate.py`, same normaliser as the 30.15 % of v1). If v2 does not get there, it is published
with its numbers but not put forward, and v1 stays the default.

- [x] Training path written and tested on a small model: `prepare_data.py --shards`,
      `configs/turbo-lora-v2.yaml`, `configs/turbo-full-v2.yaml`, `colab_finetune_v2.ipynb`,
      resumable across Colab sessions.
- [ ] Smoke test on the real GPU, to replace the estimate (about 20 to 25 h on a T4 for the
      LoRA run, computed from the v1 speed, not measured) by a measurement.
- [ ] LoRA run, 3 epochs. Report FLEURS test WER and CER, and the WER on the open-data
      validation split separately.
- [ ] Score v1 and v2 with `wer_fleurs_norm`, the closest I can get to the protocol behind the
      25.41 % that BuzzASR reports on FLEURS. Second target: under 25.41 % on that field, stated
      as an approximate comparison.
- [ ] Verbatim or normalised transcripts as training text: one run each if the budget allows,
      otherwise verbatim.
- [ ] Full fine-tune only if the LoRA run misses the target and a 24 GB GPU is available.
- [ ] Model card from `fill_card.py`, with the v2 data table and the attributions.

## 6. The meeting-notes app and the browser demo

Target: a measured WER for the weights the browser actually runs.

- [ ] Code-switching: detect the language per VAD chunk, keep `lb` unless French or German
      wins clearly, and show the switches in the transcript.
- [x] Hosted demo without PRO: static Space, the model runs in the browser (WebGPU). 25/10/2026.
- [ ] Browser demo: score the 4-bit weights on FLEURS, and fix the words lost at 30 s window boundaries.
- [ ] A "send me your worst recording" page, opt-in, under the consent of `bench/CONSENT.md`.

## Things I will not do

- Train on data with a non-commercial or no-derivatives licence, even "just to see".
- Claim we beat uni.lu, ZLS, LuxASR or BuzzASR until a table in this repository says so.
- Publish a bench result before the tested parties have had their right of reply.
