# Tier-2 Collection Worksheet — Exploratory Application Robustness

Companion to `EXPLORATORY_APPLICATION_ROBUSTNESS_PROTOCOL.md`. The study
design below is the frozen ~8-cell design — do not add, remove, or redefine
clips here.

## Chord sequences to play

```text
P1
C major → G major → A minor → E minor

P2
G major → B minor → C major → D dominant seventh

P3
A minor seventh → D dominant seventh →
G major seventh → A suspended fourth
```

Hold each chord ≈ 3–5 s; clip length ≈ 20–30 s. "Deliberate" style = clean
holds with brief muted gaps (like the controlled dataset); "natural" style =
your own tempo/strumming, no deliberate gaps.

## Rules

- **Pull each recording from the phone BEFORE starting the next one** — the
  app keeps only the latest recording and overwrites it
  (`./tool/pull_chordassist_recording.sh <clip_id> <dest>` in the frontend
  repo prints the manifest values).
- **Annotate BEFORE running any prediction on that clip.**
- **Laptop/YouTube/speaker playback MUST NOT be substituted for the Tier-2
  real-guitar recordings.**
  - Real guitar → phone microphone = Tier 2 intended-use robustness.
  - Recorded/YouTube audio → loudspeaker → room → phone microphone =
    optional Tier 3 out-of-domain/acoustic re-recording stress test.
  - Laptop-speaker guitar may be used freely for device-functionality
    testing, but those recordings must never receive `exp_*` Tier-2 IDs.
- Do not download or commit copyrighted YouTube/commercial audio.

## Per-clip checklists

### exp_baseline_p1 — P1, baseline

```text
Clip ID: exp_baseline_p1        Progression: P1
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_baseline_p2 — P2, baseline

```text
Clip ID: exp_baseline_p2        Progression: P2
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_baseline_p3 — P3, baseline

```text
Clip ID: exp_baseline_p3        Progression: P3
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_distance_p1 — P1, changed phone distance

```text
Clip ID: exp_distance_p1        Progression: P1
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈2 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_room_p1 — P1, different room/acoustics

```text
Clip ID: exp_room_p1            Progression: P1
Performer ID: ____   Guitar ID: ____   Room: ____ (different from baseline)
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_natural_p2 — P2, natural playing

```text
Clip ID: exp_natural_p2         Progression: P2
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: natural   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_guitarist2_p1 — P1, different guitarist and/or guitar

**OPTIONAL / only if another guitarist or guitar is practically available**

```text
Clip ID: exp_guitarist2_p1      Progression: P1
Performer ID: performer_02 (neutral ID only — no personal information)
Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: quiet
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

### exp_noise_p1 — P1, background noise condition

```text
Clip ID: exp_noise_p1           Progression: P1
Performer ID: ____   Guitar ID: ____   Room: ____
Phone distance: ≈0.5 m   Capture mode: app_recording
Playing style: deliberate   Noise condition: fan / tv_low (state which)
Recorded? [ ]   Pulled from phone? [ ]   SHA recorded? [ ]
Annotation completed before prediction? [ ]
Manifest entry completed? [ ]   Evaluated? [ ]
Notes: ____________________________________________
```

## After all clips

```text
All manifest entries complete?           [ ]
All annotations done before prediction?  [ ]
Harness run completed (timestamped dir): ____________________
```

```bash
cd ~/Documents/sanam/thesis/chordassist/backend && .venv/bin/python eval/evaluate_application_robustness.py --manifest evaluation_data/application_robustness/manifest.json --base-url http://127.0.0.1:8000
```
