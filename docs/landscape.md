# Who transcribes Luxembourgish? A survey of official documentation

Checked on 24/11/2026. For each service I reloaded the vendor's own documentation page that day
and looked for Luxembourgish in the list of supported languages. This is a survey of what vendors
write, not a test of what their tools do: I sent no audio to any of them.

How to read the second column:

- **No**: the page gives a list of supported languages and Luxembourgish is not in it. The number
  is the size of that list, counted on the page.
- **Listed**: Luxembourgish is in the list. That says nothing about quality; where the vendor
  publishes an accuracy figure or tier for the language, I quote it.
- **Partly**: listed for one product of the vendor only.
- **Not listed**: the vendor publishes no list of transcription languages as such; Luxembourgish
  is absent from the related lists I found.
- **Yes**: the tool is built for Luxembourgish.

## Meeting tools

| Service | Luxembourgish | Source | Checked on |
|---|---|---|---|
| Microsoft Teams, live transcription and live captions | No. Absent from the 57 spoken-language locales (44 languages) | [transcription](https://support.microsoft.com/en-us/teams/meetings/start-stop-and-download-live-transcripts-in-microsoft-teams-meetings), [captions](https://support.microsoft.com/en-us/teams/meetings/use-live-captions-in-microsoft-teams-meetings), [admin documentation](https://learn.microsoft.com/en-us/microsoftteams/meeting-transcription-captions); support pages dated 12 and 14/10/2026 | 24/11/2026 |
| Microsoft Teams, multilingual speech recognition | No. 9 languages | [support page](https://support.microsoft.com/en-us/teams/meetings/multilingual-speech-recognition-in-microsoft-teams), dated 12/10/2026 | 24/11/2026 |
| Google Meet, live captions | No. Absent from the 88 entries, most of them marked beta | [support page](https://support.google.com/meet/answer/15077804?hl=en) | 24/11/2026 |
| Google Meet, transcripts and "Take notes for me" | No. 8 languages each | [transcripts](https://support.google.com/meet/answer/12849897?hl=en), [Take notes for me](https://support.google.com/meet/answer/14754931?hl=en) | 24/11/2026 |
| Zoom, automated captions | No. Absent from the 49 languages and dialects | [support page](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0059762), dated 03/08/2026 | 24/11/2026 |
| Zoom, AI Companion meeting summary | No. Absent from the 33 languages | [support page](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0078144), dated 31/07/2026 | 24/11/2026 |
| Otter | No. 17 languages available, 14 more announced on 10/11/2026 | [help centre](https://help.otter.ai/hc/en-us/articles/360047247414-Supported-languages), [announcement](https://otter.ai/blog/otter-ai-announces-its-biggest-multilingual-leap-yet-27-new-languages-and-multilingual-capabilities) | 24/11/2026 |
| Fireflies | Listed (`lb`, one of 115 languages). No accuracy figure for the language | [knowledge base](https://guide.fireflies.ai/articles/2973706448-learn-about-fireflies-supported-languages), updated in November 2026 | 24/11/2026 |
| tl;dv | No. Absent from the 42 entries | [help centre](https://intercom.help/tldv/en/articles/6231678-how-to-change-transcript-language) | 24/11/2026 |
| Notion AI Meeting Notes | Not listed. The API takes 22 language hints or `auto`, and `lb` is not one of them; the help page lists 23 languages for speaker labels | [help page](https://www.notion.com/help/ai-meeting-notes), [API reference](https://developers.notion.com/reference/create-meeting-note) | 24/11/2026 |

## Dictation

| Service | Luxembourgish | Source | Checked on |
|---|---|---|---|
| Apple dictation, iOS and macOS | No. Absent from the 63 dictation locales on each system. Luxembourgish is a keyboard language only | [iOS feature availability](https://www.apple.com/ios/feature-availability/), [macOS feature availability](https://www.apple.com/macos/feature-availability/) | 24/11/2026 |

## Speech-to-text APIs

| Service | Luxembourgish | Source | Checked on |
|---|---|---|---|
| Microsoft Azure Speech | Partly. Absent from the main table of 148 speech-to-text locales (real-time and batch). `lb-LU` appears once, ticked for "Azure LLM Speech transcription" only | [language support](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt), dated 18/11/2026 | 24/11/2026 |
| Google Cloud Speech-to-Text | Listed (`lb-LU`) for the Chirp, Chirp 2 and Chirp 3 models, including EU locations. No accuracy figure for the language | [supported languages](https://docs.cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages), updated 10/11/2026 | 24/11/2026 |
| Amazon Transcribe | No. Absent from the 113 rows | [supported languages](https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html) | 24/11/2026 |
| OpenAI | Listed for the open-source Whisper model only: `lb` is one of the 100 codes in its tokenizer. The API guide says "Whisper supports 98 languages, but accuracy varies by language" without naming them, and names no languages for the current transcription model either, so the hosted models are not verified for Luxembourgish | [speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text), [Whisper tokenizer](https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py) | 24/11/2026 |
| ElevenLabs Scribe | Listed. The accuracy claims disagree: the documentation puts Luxembourgish in the 10 to 20 % WER tier for Scribe v2; the Luxembourgish product page shows 25.0 % for Scribe v1 in its own Luxembourgish FLEURS chart and files the language under 25 to 50 % WER in its FAQ. Its introduction repeats a general Scribe claim of 3.1 % WER on FLEURS, which is not specific to Luxembourgish | [documentation](https://elevenlabs.io/docs/overview/capabilities/speech-to-text), modified 23/11/2026; [Luxembourgish page](https://elevenlabs.io/speech-to-text/luxembourgish) | 24/11/2026 |
| AssemblyAI | Listed for the Universal-2 model only, in the lowest tier: "Fair accuracy (>50% WER)". Not among the 32 languages of the current model | [supported languages](https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages.md), [Luxembourgish page](https://www.assemblyai.com/languages/luxembourgish) | 24/11/2026 |
| Deepgram | No for Deepgram's own models (Nova-3 lists 100 languages). Its hosted Whisper offer: not verified | [models and languages](https://developers.deepgram.com/docs/models-languages-overview), modified 16/11/2026 | 24/11/2026 |
| Speechmatics | No. Absent from the 63 rows (language packs, including an automatic-detection row and bilingual packs) | [languages](https://docs.speechmatics.com/speech-to-text/languages) | 24/11/2026 |
| Gladia | Listed (`lb`) for the Solaria-1 model, not for Solaria-3. No accuracy figure for the language | [supported languages](https://docs.gladia.io/chapters/language/supported-languages), modified 09/09/2026 | 24/11/2026 |

## Built in Luxembourg

| Service | Luxembourgish | Source | Checked on |
|---|---|---|---|
| LuxASR, University of Luxembourg | Yes. Web interface open to the public (no price and no account shown on the page), apps and a limited API. Self-reported: trained on 150+ hours, word error rate below 10 %, punctuation and capitalisation included | [luxasr.uni.lu](https://luxasr.uni.lu/) | 24/11/2026 |
| Sproochmaschinn (Schreifmaschinn), ZLS | Yes. Web application open to the public (no price and no account shown on the page), up to 30 minutes and 100 MB per file; API for non-commercial use. No accuracy figure in the application | [sproochmaschinn.lu](https://sproochmaschinn.lu/), [government press release of 10/02/2025](https://gouvernement.lu/fr/actualites/toutes_actualites/communiques/2025/02-fevrier/10-sproochmaschinn-thill.html) | 24/11/2026 |
| braddl | Yes according to the company. Its meeting-minutes product, braddl Notes, is in pilot phase and is described as joining Microsoft Teams and Google Meet as a visible guest; no public price. Self-reported: 4.0 % word error rate on its own validation set | [braddl.lu](https://braddl.lu/en) | 24/11/2026 |

Public prices read the same day, for the APIs that list the language: ElevenLabs Scribe 0.20 USD
per hour, AssemblyAI Universal-2 0.15 USD per hour, Gladia Solaria-1 0.27 EUR per hour, Google
Cloud Speech-to-Text V2 standard recognition 0.016 USD per minute. Azure and OpenAI: not verified.

## What this means for a meeting in Luxembourg

Teams, Google Meet and Zoom have no Luxembourgish setting for captions or transcripts, and the AI
notes of Meet and Zoom do not list the language either. With the built-in captions and
transcripts, whoever runs the meeting has to pick another language or go without. Of the general
note takers I checked (Otter, Fireflies, tl;dv, Notion), Fireflies is the only one that lists the
language, and it publishes no accuracy figure for it; braddl Notes, in pilot, says it joins Teams
and Meet and transcribes Luxembourgish. On the API side a developer has a choice (Google Chirp,
ElevenLabs, Gladia, AssemblyAI Universal-2, Azure LLM Speech), but being on a list is not the same
as working well: AssemblyAI itself files Luxembourgish under more than 50 % word error rate, and
ElevenLabs puts the language in the 10 to 20 % tier in its documentation and in the 25 to 50 %
tier on its own Luxembourgish page, where its chart shows 25.0 % for Scribe v1. With these APIs
the audio goes to the vendor's cloud; among the pages I read, only Google's says where, with EU
locations for `lb-LU`.

## What already exists, and is good

Luxembourgish speech recognition is not an empty field, and the two tools anyone can use today
are public work. LuxASR is developed by Peter Gilles, Léopold Hillah and Nina Hosseini-Kivanani at
the University of Luxembourg, with the support of the Chambre des Députés. Its web interface is
open to the public, with no price and no account shown on the page. It offers diarisation,
subtitles and research formats such as Praat TextGrid, deletes the uploaded audio when processing
ends, and publishes its smaller models as open weights. Sproochmaschinn, built by the Zenter fir
d'Lëtzebuerger Sprooch, is open to the public on the same terms, processes audio on servers in the
EU, and transcribes Luxembourgish as it is spoken, with an optional normalised spelling. The ZLS
also publishes a model under MIT, and on 11/11/2026 it released about 58 hours of its training
audio with transcriptions under CC0 on
[data.public.lu](https://data.public.lu/fr/datasets/schreifmaschinn-trainingsdaten-cc0/). Anyone
who needs a Luxembourgish transcript today should start with these two. braddl, a Luxembourg
company, is piloting a meeting-minutes product hosted in the country.

## Limits of this survey

- I read documentation, I did not test. "No" means the vendor does not list the language, not that
  I measured a failure; "Listed" means the vendor accepts the language code, not that the output is
  usable.
- Every accuracy figure quoted here is self-reported by its vendor, on that vendor's own data or
  on an unspecified setup. No two are comparable. braddl's chart, for instance, states that its
  own best model and the other systems were scored on different benchmarks.
- Language lists move. Otter added 13 languages two weeks before this survey. Each row is true
  for the date in the last column only.
- The Otter help centre does not answer to a scripted client; I read it in a browser, twice, and
  kept no file of it. Otter's own announcement of 10/11/2026 names the same languages. For the two
  Zoom pages I kept the page as rendered by a browser; for every other page, the raw file. All
  kept files have a SHA-256 hash, dated 24/11/2026.
- Not checked: Copilot recap in Teams, translated captions in Zoom, Deepgram's hosted Whisper,
  on-premises or enterprise offers, and any tool not named in the tables.
