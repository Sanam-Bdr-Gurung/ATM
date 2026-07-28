# ChordAssist Thesis Master Context

**Last updated:** 2026-07-28  
**Repository:** `https://github.com/Sanam-Bdr-Gurung/ATM`  
**Active branch:** `migrate-basic-pitch-backend`

## 1. Working Thesis Title

**AI-Powered Real-Time Guitar Transcription System with Audio Feedback and Flutter Interface for Visually Impaired Musicians**

Until continuous streaming is implemented and measured, describe the implemented system as **near-real-time** or **low-latency file-based guitar transcription**, not as proven continuous real-time transcription.

## 2. Final Scope Decision

### Core scope

- Guitar-only transcription.
- Standard E tuning assumed for tablature: `[40, 45, 50, 55, 59, 64]`.
- Spotify Basic Pitch as the pretrained neural note-transcription backend.
- Existing DSP note detector retained only as an experimental baseline.
- Chord recognition as the primary user-facing musical output.
- Notes and simple guitar tablature as secondary outputs.
- FastAPI backend.
- Flutter client with accessible controls and spoken feedback using TTS.
- Evaluation of note accuracy, chord accuracy, tab accuracy, latency, and accessibility/usability.

### Removed or deferred

- Old `jongwook/onsets-and-frames` checkpoint integration.
- Training a neural model from scratch.
- Piano and violin output.
- Instrument classification.
- Automatic tuning detection and alternate-tuning support.
- Source separation.
- On-device inference.
- True WebSocket/live streaming until the file-based prototype is stable.

## 3. Thesis Narrative

1. The DSP baseline is computationally fast but has poor note-onset precision and produces many false positives.
2. Poor note events lead to poor string/fret assignments and unreliable tabs.
3. A pretrained polyphonic transcription model, Basic Pitch, replaces the abandoned piano-oriented Onsets-and-Frames route.
4. Neural note events are used for notes, chord derivation, and Standard-tuning tab mapping.
5. Baseline and neural systems are compared using accuracy and latency measurements.
6. Results are delivered through an accessibility-first Flutter interface with spoken feedback.

## 4. Environment Locked for Current Development

- macOS on Apple Silicon (`arm64`).
- Python `3.10.20`.
- Basic Pitch `0.4.0`.
- CoreML runtime selected successfully: `MODEL_TYPES.COREML` / `COREML`.
- NumPy `1.26.4` and the pinned Python 3.10 numerical/audio stack in `requirements.txt`.
- Basic Pitch warnings about missing TensorFlow, ONNX Runtime, and TFLite Runtime are expected because CoreML is the selected runtime.
- The `resampy` `pkg_resources` deprecation warning is currently non-blocking.

## 5. Completed Work

### Milestone 1 — DSP backend baseline

Completed before the reset:

- `GET /health`.
- `POST /analyze-file`.
- Audio loading.
- DSP multipitch note detection.
- Chroma/template chord detection.
- DP-based tab mapping.
- Timing response fields.
- Evaluation scripts for notes, tabs, tuning, and latency.

### Milestone 2 — Baseline evaluation

Previously observed approximate results:

- Mean note F1: `0.187`.
- Mean pitch-only accuracy: `0.868`.
- Tab string accuracy: `0.10`.
- Tab fret accuracy: `0.39`.
- Full tab accuracy: `0.00`.
- Tuning confidence commonly around `0.44–0.46` and unreliable.

These results justify the neural migration and removal of tuning detection from the core scope.

### Checkpoint 1 — Basic Pitch adapter

Implemented and pushed on `migrate-basic-pitch-backend`:

- `models/basic_pitch_inference.py`.
- `eval/debug_basic_pitch.py`.
- Python 3.10-compatible `requirements.txt`.
- Basic Pitch model loading through CoreML.
- Conversion to the stable event schema:

```json
{
  "t_on": 1.1726,
  "t_off": 3.0779,
  "midi": 40,
  "freq_hz": 82.407,
  "conf": 0.7105
}
```

- Chronological event sorting.
- Guitar MIDI filtering.
- Preservation of simultaneous/polyphonic notes.
- Temporary WAV cleanup.
- Adapter assertions passed on `data/mini_eval/clip.wav`.

### Checkpoint 2 — FastAPI migration

Implemented and tested locally; commit/push status must be checked in Git before continuing.

Behavior:

- Supported backends: `baseline`, `basic_pitch`.
- Default backend: `basic_pitch`.
- Old O&F backend disconnected from the running API.
- One lazily created and reused `BasicPitchTranscriber` instance.
- Basic Pitch failures return an explicit API error instead of silently falling back.
- Standard E is represented as a fixed assumption, not a detected tuning.
- Harmonic collapse is removed from the neural path.
- Tab mapping uses fixed Standard E open-string MIDI values.
- Piano and violin response placeholders are removed.
- Existing chroma/template chord detector is retained temporarily and reported as `chroma_template`.
- `mode=chunked` remains a compatibility input but `mode_effective` is explicitly `full`; true streaming is not implemented.

## 6. Current API Contract

### Health

`GET /health`

Expected fields:

```json
{
  "ok": true,
  "instrument": "guitar",
  "supported_backends": ["baseline", "basic_pitch"],
  "default_backend": "basic_pitch",
  "basic_pitch_loaded": false
}
```

After the first neural request, `basic_pitch_loaded` becomes `true`.

### Analyze file

`POST /analyze-file`

Important query parameters:

- `backend=basic_pitch|baseline`.
- `mode=full|chunked`; both currently execute full-file processing.
- `chords=true|false`.

Important response fields:

- `instrument_hint`.
- `tuning` with `source=fixed_assumption`.
- `notes`.
- `chords`.
- `render.guitar_tabs`.
- `tts`.
- `latency_ms`.
- `real_time_factor`.
- `mode_requested` and `mode_effective`.
- `timing_ms`.
- `audio_duration_sec`.
- `backend_requested` and `backend_effective`.
- `backend_runtime`.
- `chord_backend`.

## 7. Latest Local Smoke-Test Results

Input: `data/mini_eval/clip.wav`  
Audio duration: approximately `10.728 s`.

### Basic Pitch API request

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- First request total: `5166.59 ms`.
- First request note stage: `324.89 ms`.
- First request RTF: `0.4816`.
- Warm request total: `100.02 ms`.
- Warm request note stage: `98.73 ms`.
- Warm request RTF: `0.0093`.
- All API assertions passed.

These are smoke-test observations, not final benchmark values. Final reporting must use repeated runs, a documented warm-up policy, median/mean/p95, several clip durations, and hardware/environment details.

### DSP baseline request

- Notes: `38`.
- Tabs: `37`.
- Total latency: `47.59 ms`.
- Baseline API assertions passed.

The missing tab indicates at least one detected MIDI note had no playable position under the current combination of pitch range and `max_fret`. The guitar range and fret limit must be made consistent before final tab evaluation.

### Chord-enabled request

- Neural notes: `39`.
- Chord segments: `8`.
- Chord stage: `465.55 ms`.
- Chord backend: `chroma_template`.

Observed labels included `Cm`, `E`, `A`, `D`, `B`, `Bm`, `E`, and `Em`. These labels have not yet been validated against ground truth. A likely false chord during initial silence shows that no-chord/silence gating is required.

## 8. Important Technical Caveats

### Preprocessing consistency

The standalone Basic Pitch test returned `35` events while the API returned `39`. The API currently peak-normalizes decoded audio, which can change confidence and event counts. Before formal evaluation, lock and document one preprocessing policy and apply it consistently to both predictions and experiments.

### Timing methodology

Do not report the single cold or warm request as the final latency result. Separate and measure:

- Upload/read time.
- Decode/resample time.
- Model initialization time.
- Neural inference time.
- Chord-analysis time.
- Tab-mapping time.
- End-to-end request time.

Use at least one warm-up request and multiple measured repetitions. Report median and p95 in addition to the mean.

### Tabs and polyphony

The existing DP tab mapper processes notes sequentially and does not enforce unique strings for simultaneous chord notes. It can therefore produce physically invalid chord fingering assignments. Tabs remain secondary and must either be improved for onset groups or explicitly described as approximate single-note position suggestions.

### Pitch range versus fret limit

`midi_high=88` corresponds to high E plus 24 frets, while the current tab mapper uses `max_fret=20`. Choose one consistent configuration before evaluation, likely:

- MIDI `40–88` with `max_fret=24`, or
- MIDI `40–84` with `max_fret=20`.

### Current chord implementation

Chords are still derived from audio chroma/templates, not from Basic Pitch note groups. This is an interim baseline. The planned primary chord method will group active neural notes, match pitch-class templates, apply silence/no-chord gating, and smooth labels over time.

## 9. Next Milestones in Order

### Checkpoint 3 — Repository cleanup and durable documentation

- Commit and push the tested Checkpoint 2 API migration.
- Remove tracked `__pycache__` and `.pyc` files.
- Ensure `.gitignore` excludes generated Python files and local artifacts.
- Disconnect and then remove legacy O&F files in a separate, reviewable commit.
- Add this context file under `docs/THESIS_MASTER_CONTEXT.md`.
- Add a dated progress log under `docs/THESIS_PROGRESS_LOG.md`.
- Update the empty or outdated README with the narrowed thesis scope and startup instructions.

### Checkpoint 4 — Reproducible evaluation harness

- Run baseline and Basic Pitch over the same annotated clips.
- Calculate onset+pitch precision, recall, and F1 using a documented onset tolerance.
- Add note-with-offset metrics if feasible.
- Benchmark cold and warm latency over multiple durations and repetitions.
- Save machine-readable CSV/JSON results.
- Generate tables for the thesis.

### Checkpoint 5 — Neural-note chord recognition

- Group active Basic Pitch notes into time windows/onset groups.
- Convert MIDI notes to pitch classes.
- Match major/minor chord templates first.
- Add silence/no-chord gating.
- Smooth short unstable labels.
- Evaluate using chord ground truth and weighted chord symbol recall or a clearly documented segment metric.

### Checkpoint 6 — Tab scope and mapping

- Lock `max_fret` and MIDI range consistently.
- Decide whether tabs are approximate note positions or polyphonic chord fingerings.
- Improve simultaneous-note assignment if time permits.
- Rerun tab evaluation.

### Checkpoint 7 — Flutter accessibility prototype

- Record or select audio.
- Upload to FastAPI.
- Display chord timeline as the primary result.
- Display notes and tabs as optional detail.
- Speak concise feedback using Flutter TTS.
- Add semantic labels, large controls, screen-reader support, and predictable focus order.

### Checkpoint 8 — Final thesis material

- Methodology and architecture diagram.
- Dataset/clip description.
- Evaluation protocol.
- Accuracy and latency tables.
- Accessibility/usability evaluation.
- Discussion and limitations.
- Demo recording and reproducibility instructions.

## 10. One-Week Core Completion Priority

1. Preserve the working API checkpoint in Git.
2. Build the reproducible evaluation harness and obtain defensible results.
3. Implement and evaluate chord recognition with no-chord gating.
4. Build the minimal accessible Flutter flow.
5. Produce thesis-ready tables, figures, methodology, results, and limitations.

Streaming, alternate tunings, source separation, and advanced polyphonic fingering remain optional and must not block core completion.

## 11. Git Checkpoint Discipline

Use the active migration branch until the neural backend, evaluation, and cleanup are stable.

Recommended checkpoints:

1. `feat(transcription): add Basic Pitch CoreML adapter`
2. `feat(api): migrate transcription backend to Basic Pitch`
3. `chore(repo): remove legacy O&F path and generated files`
4. `test(evaluation): compare DSP and Basic Pitch transcription`
5. `feat(chords): derive smoothed chords from neural notes`
6. `feat(app): add accessible transcription and TTS flow`

Before every checkpoint:

```bash
git status --short
git diff --check
python -m pip check
python -m py_compile api.py models/basic_pitch_inference.py
```

Push each completed checkpoint so future work can be inspected from the exact branch state.
