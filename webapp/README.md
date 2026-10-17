# Restaurant assistants study: web app

A browser front end for the Part 1b dialog system, built for a perception study on
text-to-speech voices. Every participant talks to the same dialog system four times,
each time with a different voice:

| Code   | Label          | Engine    | Voice       |
|--------|----------------|-----------|-------------|
| `WDTP` | robotic-male   | eSpeak NG | `en+m3`     |
| `CDUK` | robotic-female | eSpeak NG | `en+f3`     |
| `DTFL` | human-male     | Kokoro-82M| `bm_george` |
| `9TEF` | human-female   | Kokoro-82M| `bf_emma`   |

All four voices are generated **on the server**, so every participant hears exactly the
same audio, whatever their device. Both engines are free and open source and run on CPU.

## Study flow

1. Information and consent page.
2. The server assigns a participant ID and an order of the four voices. The 24 possible
   orders are dealt from a shuffled deck, so orders are random but balanced.
3. For each of the four positions: a conversation with "Assistant N" (with a task to
   complete), then a button that opens the Google Form with the participant ID and an
   opaque condition code prefilled. The participant submits it and comes back.
4. Optional final form (demographics, overall comparison).

Participants only ever see "Assistant 1-4" and the four-letter codes, never the condition
names. The code-to-voice mapping is in `study_config.json`.

## Run locally

From the repository root:

```
pip install -r webapp/requirements-server.txt
python -m uvicorn webapp.server:app --port 7860
```

Then open http://localhost:7860. To start again as a new participant, clear the site data
(the progress is kept in the browser's localStorage). The server only plays pre-rendered
audio (see below); `webapp/requirements.txt` adds the TTS engines needed to render it.

## Configuration: `study_config.json`

- `conditions`: code -> voice. eSpeak takes `rate` (words per minute), Kokoro takes `speed`.
  Other Kokoro voices: `af_heart`, `am_michael` (American), `bf_isabella`, `bm_lewis` (British).
- `form_url`: the Google Form prefill link, with `{participant_id}` and `{code}` placeholders
  (`{position}` is also available). See below.
- `final_form_url`: optional final form, with `{participant_id}`.
- `show_text`: show the system's answers as text as well as audio.
- `min_turns_before_done`: turns before the "Finish conversation" button is enabled.
- `tasks`: one task per position (shown in order, independent of the voice).

`form_url` and `final_form_url` can also be set as environment variables (`FORM_URL`,
`FINAL_FORM_URL`) in the Space settings.

### Getting the Google Form prefill link

1. In the form, add two short-answer questions, e.g. "Participant code" and "Assistant code".
2. Form menu (⋮) > **Get pre-filled link**. Type `PID` in the first and `CODE` in the second,
   click **Get link** and copy it.
3. Replace `PID` with `{participant_id}` and `CODE` with `{code}` in the link, e.g.
   `https://docs.google.com/forms/d/e/1FAIpQ.../viewform?usp=pp_url&entry.123456={participant_id}&entry.654321={code}`
4. Put it in `form_url`.

The same form is filled in four times per participant (one row per assistant), which gives
a long-format table ready for a repeated-measures analysis.

## Audio cache: `webapp/precompute.py`

Kokoro needs a few seconds per sentence on CPU, eSpeak almost nothing. Waiting time would
then differ between conditions and bias the ratings. The dialog system has a finite set of
answers (about 1200 sentences), so they are all rendered in advance, for every voice:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r webapp/requirements.txt
python -m webapp.precompute
```

Only missing files are generated. The audio (about 180 MB) is too large for git, so it is
uploaded to a public Hugging Face dataset, which Render downloads during the build:

```
set HF_TOKEN=hf_...
python webapp/audio_dataset.py upload alepese/restaurant-study-audio
```

**Re-run both after changing a voice or a response text**, then redeploy on Render. The
hosted server has no TTS engine: a sentence missing from the cache is shown without audio
(and logged as `missing_audio`).

## Logs

Every event (session start with the assigned order, condition start, each turn, condition
end) is appended as JSON to `logs/webapp/`. The hosting disk is wiped on every restart, so
when `LOG_DATASET` and `HF_TOKEN` are set the folder is committed every 2 minutes to that
private Hugging Face dataset (created automatically).

`SECRET_KEY` signs the participants' session tokens, so a restart does not lose their
condition order.

## Deploy on Render (free)

`render.yaml` describes the service. The free plan sleeps after 15 minutes without visits
and takes about a minute to wake up: open the link yourself before sending it out.

1. Push the code to GitHub.
2. On https://render.com sign in with GitHub, then **New > Blueprint** and pick this
   repository (and branch).
3. Render asks for `HF_TOKEN` (the other settings come from `render.yaml`): paste a Hugging
   Face token with write access, used only to save the logs.
4. Every push to that branch redeploys automatically.
