# Exploratory Application-Robustness Protocol (Evaluation Tier 2)

Status: protocol frozen before any exploratory recording is inspected.
Data collection: pending.

## Ground rules (non-negotiable)

- The backend recognition configuration is **frozen**
  (`development_8clip_grid_20260806`: minimum score 0.58, ambiguity margin
  0.020, minimum segment duration 0.60 s).
- The application presentation configuration is **frozen**
  (threshold-first, 1.25 s, long-X retention).
- Exploratory data **cannot tune either** — no parameter, threshold, template,
  or presentation rule may change in response to exploratory results.
- Exploratory results are **never pooled** into the formal held-out metrics.
  The formal scientific experiment remains: 8 controlled development clips +
  4 controlled unseen held-out clips.
- This study produces **descriptive/exploratory evidence only**: it
  characterizes how the frozen system behaves under intended-use domain
  shift. It is not a second model comparison, and it is not called
  "real-world validation."
- A new recording is not automatically "out of domain": the domain shifts
  studied here are the *named dimensions below*, declared per clip in the
  manifest before analysis.

## Domain-shift dimensions

```text
performer            guitar/instrument      room/acoustics
phone distance       phone-microphone capture
natural playing style
background conditions
```

## Study design (minimum practical set)

Three declared progressions (all within the controlled vocabulary):

```text
P1 — common triads:            C:maj → G:maj → A:min → E:min
P2 — mixed major/minor/7th:    G:maj → B:min → C:maj → D:7
P3 — harder qualities:         A:min7 → D:7 → G:maj7 → A:sus4 (or D:add9)
```

Planned contrast cells (≈8 clips; each clip ≈ 20–30 s, 4 chords held ≈ 3–5 s
each unless the cell says otherwise):

| Clip ID | Progression | Contrast vs baseline |
|---|---|---|
| exp_baseline_p1 | P1 | same performer, familiar guitar, phone at ≈0.5 m, quiet room, deliberate playing (the reference cell) |
| exp_baseline_p2 | P2 | as baseline |
| exp_baseline_p3 | P3 | as baseline |
| exp_distance_p1 | P1 | phone at ≈2 m |
| exp_room_p1 | P1 | different room (e.g. bathroom/kitchen acoustics) |
| exp_natural_p2 | P2 | natural playing: own tempo/strumming, no deliberate gaps |
| exp_guitarist2_p1 | P1 | different guitarist and/or different guitar (**if practically available**; omit if not) |
| exp_noise_p1 | P1 | quiet background condition changed (e.g. fan/TV at low volume) |

Justification of size: this is a descriptive characterization — one clip per
named contrast against a three-clip baseline is enough to *describe* behavior
under each shift and honest about not supporting inference. A larger sample
would suggest statistical claims this tier is explicitly not allowed to make.

## Freeze-before-inspection

Before any results are inspected, the following are frozen by committing them
to this repository:

- clip IDs and contrast cells (the table above);
- progressions P1–P3;
- annotation protocol (below);
- metrics (below).

## Annotation protocol

- Same JSON schema as `evaluation_data/chords/annotations/*.json`
  (`schema_version`, `clip_id`, `split`, `duration_sec`, `segments` with
  `start`/`end`/`label`).
- `split` = `exploratory_application_robustness`.
- Annotate by listening to the recording (declared progression + audible
  boundaries), **before** viewing any model prediction for that clip.
- Muted gaps/silence/decay annotated as `N`, as in the controlled dataset.

## Manifest schema

One JSON file declaring all clips before evaluation:

```json
{
  "schema_version": 1,
  "split": "exploratory_application_robustness",
  "include_in_formal_metrics": false,
  "clips": [
    {
      "clip_id": "exp_baseline_p1",
      "audio_path": "evaluation_data/application_robustness/audio/exp_baseline_p1.wav",
      "annotation_path": "evaluation_data/application_robustness/annotations/exp_baseline_p1.json",
      "sha256": "<hex digest of the audio file>",
      "performer_id": "performer_01",
      "guitar_id": "guitar_01",
      "capture_device": "samsung_phone_01",
      "capture_mode": "app_recording | phone_voice_recorder | wav_transfer",
      "distance_m": 0.5,
      "room_id": "room_01",
      "playing_style": "deliberate | natural",
      "noise_condition": "quiet | fan | tv_low",
      "progression_id": "P1",
      "notes": ""
    }
  ]
}
```

Additional participants are identified only by neutral IDs
(`performer_02`, `guitar_02`); no personally identifying information is
stored.

## Collection workflow (recommended)

```text
record in ChordAssist
→ stop recording
→ pull immediately
   (frontend: ./tool/pull_chordassist_recording.sh <clip_id> <dest>)
→ verify SHA (script prints it and the manifest values)
→ annotate BEFORE prediction
→ add manifest entry
→ only then evaluate
```

**Warning:** the app's recording service intentionally keeps only the latest
recording (one app-sandbox temp file, overwritten each time; its directory
varies by device, so the pull script locates it by exact filename). Pull each
clip before
starting the next one, or the previous take is lost. The per-clip checklist
lives in `TIER2_COLLECTION_WORKSHEET.md`.

**Speaker playback is not Tier 2.** Real guitar → phone microphone is the
Tier-2 intended-use condition. Recorded/YouTube audio → loudspeaker → room →
phone microphone is the optional Tier-3 acoustic re-recording stress test.
Laptop-speaker guitar may be used for device-functionality testing, but such
recordings must never receive `exp_*` Tier-2 IDs.

## Metrics

Computed by `eval/evaluate_application_robustness.py` (frozen backend API;
no recognition logic duplicated).

**RAW RECOGNITION METRICS** (per clip + descriptive aggregate; identical
definitions to the formal evaluation, reusing the same code):

```text
root / triad-family / harmonic-exact / overall-exact time-weighted accuracy
no-chord F1
X (uncertainty) rate
boundary F1 / MAE (0.25 s tolerance)
latency / real-time factor
```

**APPLICATION PRESENTATION STATISTICS** (per clip; descriptive only —
usability/presentation behavior, *never* accuracy, and never a claim of
"improved model accuracy"):

```text
raw returned segment count
prevailing-summary item count (frontend 1.25 s threshold-first mirror)
retained long-uncertainty count/duration
omitted short-uncertainty count/duration
```

## Provenance (every run)

```text
backend git SHA · configuration ID (verified == frozen) · base URL
Python/runtime + key dependency versions · manifest SHA-256
per-clip audio SHA-256 (verified against manifest) · UTC timestamp
```

Runs are written to timestamped directories under
`evaluation_results/application_robustness/`; earlier runs are never
overwritten.

## Optional Tier 3 — out-of-domain stress test (protocol note only)

If time permits, a few independently produced full-mix recordings
(guitar-dominant songs) may be analyzed as an *exploratory out-of-domain
stress test*. Two conditions must be kept distinct because they answer
different questions:

```text
A. direct digital produced-music file
B. song played through speakers → room → phone microphone
```

Condition B adds loudspeaker response, room acoustics, phone microphone
processing, and background noise on top of the full mix — a substantially
harder and different problem than A. Neither condition supports any claim of
general commercial-music performance; results, if produced, are descriptive
only and reported separately from Tiers 1 and 2. **No copyrighted commercial
audio may be committed to this repository**; such files stay local, referenced
by hash in a local-only manifest.
