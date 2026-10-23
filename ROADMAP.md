# Roadmap

Where this goes next, in the order I would do it. Every step ends with a number in a table,
not with "it feels better".

## Where we are (October 2026)

- Whisper large-v3-turbo + LoRA on FLEURS `lb_lu` (8.5 h of read speech, CC-BY-4.0).
- Full test set: WER 30.15 %, CER 8.82 %. On the 300-utterance comparison subset the uni.lu
  and ZLS models have a lower WER; ours has the lowest CER. Translation: we spell the sounds
  right more often than we cut the words right. Contractions, compounds and the n-rule are
  where the words go wrong.
- Published on the Hub, with a CTranslate2 int8 export that runs on a laptop CPU.

## 1. Measure what we are actually bad at (no GPU needed)

- [x] `wer_n_rule`: a second WER that forgives Eifeler Regel disagreements. If it drops a lot,
      part of the gap is spelling convention, not hearing.
- [ ] Error breakdown on the 934 test sentences: substitutions vs insertions vs deletions, and
      the 50 most frequent confusions. My bet: `d'`/`de`, `an`/`a`, compounds split in two.
- [ ] Same breakdown for uni.lu and ZLS. Knowing *where* they beat us is worth more than the
      headline gap.

## 2. Cheap training wins (one Colab evening each)

- [ ] SpecAugment on (`apply_spec_augment`, `mask_time_prob` around 0.05).
- [ ] LoRA rank 16 and 64 next to 32. One table, three rows.
- [ ] 5 epochs instead of 3, keeping the best checkpoint on dev. Only if the dev curve is
      still going down at epoch 3.

## 3. Real-life audio, not only read sentences

The benchmark has clean and WhatsApp-like audio. Luxembourgish people mostly hear and speak
Luxembourgish on the radio, in meetings and in voice notes. Plan:

- [ ] `radio` condition: telephone band (300-3400 Hz) and a quiet music bed.
- [ ] `meeting` condition: room reverb at two metres, light babble.
- [ ] A small spontaneous-speech test set (30-60 min), transcribed by hand, with a clear
      licence. Recorded with volunteers rather than scraped: consent first, then data.

## 4. More data, with clean licences only

- [ ] Ask the ZLS whether the data.public.lu MIT licence or the Hugging Face CC-BY-NC-ND one
      applies to their fine-tuning set. One email, potentially 75 h of audio.
- [ ] Talk to Luxembourgish media and institutions that own spoken archives. The offer: an
      open evaluation on their audio (their data never leaves their hands), and the fine-tuned
      model back for free if they want it.
- [ ] Keep `make_manifest.py` the single entry point, so every extra hour of audio comes with
      its source and licence in the manifest.

## 5. The meeting-notes app

- [ ] Code-switching: detect the language per VAD chunk, keep `lb` unless French or German
      wins clearly, and show the switches in the transcript.
- [ ] Publish the Space. Files already purged after 50 minutes, so the one-hour promise holds.
- [ ] A "send me your worst recording" page, opt-in, for the spontaneous test set in step 3.

## Things I will not do

- Train on data with a non-commercial or no-derivatives licence, even "just to see".
- Claim we beat uni.lu or ZLS on WER until a table in this repository says so.
