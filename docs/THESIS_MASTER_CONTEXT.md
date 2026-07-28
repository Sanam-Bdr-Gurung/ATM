# ChordAssist Thesis Master Context

**Last updated:** 2026-07-28  
**Repository:** `https://github.com/Sanam-Bdr-Gurung/ATM`  
**Active branch:** `migrate-basic-pitch-backend`

## 1. Working Thesis Title

**AI-Powered Real-Time Guitar Transcription System with Audio Feedback and Flutter Interface for Visually Impaired Musicians**

Until continuous streaming is implemented and measured, describe the implemented prototype as **near-real-time** or **low-latency file-based guitar transcription**, not as proven continuous real-time transcription.

## 2. Final Scope Decision

### Core scope

- Guitar-only transcription.
- Standard E tuning assumed for tablature: `[40, 45, 50, 55, 59, 64]`.
- Spotify Basic Pitch as the pretrained neural note-transcription backend.
- Existing DSP note detector retained only as an experimental baseline.
- Chord recognition as the primary user-facing musical output.
- Notes and simple guitar tablature as secondary outputs.
- FastAPI backend.
- Flutter client with accessible controls and spoken feedback through TTS.
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
5. The DSP and neural systems are compared using accuracy and latency measurements.
6. Results are delivered through an accessibility-first Flutter interface with spoken feedback.

## 4. Environment Locked for Current Development

- macOS on Apple Silicon (`arm64`).
- Python `3.10.20`.
- Basic Pitch `0.4.0`.
- CoreML runtime selected successfully: `MODEL_TYPES.COREML` / `COREML`.
- CoreML Tools `9.0`.
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
- Initial evaluation scripts for notes, tabs, tuning, and latency.

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

Implemented and pushed on `migrate-basic-pitch-backend`.

- Commit: `b4c4018`.
- Commit message: `feat(transcription): add Basic Pitch CoreML adapter`.

Completed:

- Added `models/basic_pitch_inference.py`.
- Added `eval/debug_basic_pitch.py`.
- Locked a Python 3.10-compatible dependency stack.
- Confirmed Basic Pitch model loading through CoreML.
- Converted Basic Pitch tuples into the stable event schema:

```json
{
  "t_on": 1.1726,
  "t_off": 3.0779,
  "midi": 40,
  "freq_hz": 82.407,
  "conf": 0.7105
}
```

- Sorted events chronologically.
- Applied initial guitar MIDI filtering.
- Preserved simultaneous/polyphonic notes.
- Added temporary WAV cleanup.
- Passed adapter assertions on `data/mini_eval/clip.wav`.

### Checkpoint 2 — FastAPI Basic Pitch migration

Implemented, verified, committed, and pushed on `migrate-basic-pitch-backend`.

- Commit: `81058fc9de38968d7c98ae13ad9144c84d15820f`.
- Commit message: `feat(api): migrate transcription backend to Basic Pitch`.

Behavior:

- Supported backends: `baseline`, `basic_pitch`.
- Default backend: `basic_pitch`.
- Old O&F backend disconnected from the running API.
- One lazily initialized and reused `BasicPitchTranscriber` instance.
- Basic Pitch failures return an explicit API error rather than silently falling back.
- Standard E is represented as a fixed assumption, not a detected tuning.
- Harmonic collapse is removed from neural note events.
- Tab mapping uses fixed Standard E open-string MIDI values.
- Piano and violin response placeholders are removed.
- Existing chroma/template chord detection is retained temporarily and reported as `chroma_template`.
- `mode=chunked` remains a compatibility input while `mode_effective=full` explicitly reports the actual behavior.

### Checkpoint 3 — Repository cleanup and reproducible regression testing

Cleanup implementation committed locally.

- Commit: `69ba84e`.
- Commit message: `chore(repo): finalize Basic Pitch migration cleanup`.

Completed:

- Removed the abandoned Onsets-and-Frames implementation:
  - `models/of_inference.py`
  - `models/of_model.py`
  - `onsets_and_frames/`
- Removed O&F-specific evaluation scripts:
  - `eval/compare_baseline_vs_of.py`
  - `eval/debug_of_backend.py`
  - `eval/onset_f1.py`
  - `eval/threshold_sweep.py`
- Removed automatic tuning-detection code:
  - `tuning/detect.py`
  - `tuning/detune.py`
  - `eval/tuning_sanity.py`
- Removed the unused `util_log.py` helper.
- Confirmed that active Python code no longer references O&F or tuning-detection modules.
- Confirmed that no `__pycache__` directories or `.pyc` files remain tracked.
- Added a complete `README.md` covering scope, setup, API usage, evaluation direction, and limitations.
- Added `eval/smoke_api.py` for repeatable regression testing.
- Added `evaluation_results/` to `.gitignore`.
- Preserved the DSP baseline.
- Preserved `eval/note_f1.py` and `eval/tab_accuracy_gt.py` for formal evaluation work.
- Preserved all removed experiments through Git history.

The cleanup does not erase evidence of the previous O&F attempt. The implementation, errors, and development history remain recoverable from earlier commits and may be discussed in the thesis as an abandoned integration route.

#### Checkpoint 3 regression evidence

Test input:

- File: `data/mini_eval/clip.wav`.
- Source: self-recorded guitar audio.
- Duration: approximately `10.728 s`.
- File size: `473166` bytes.
- SHA-256: `42277f91c9ffd9f97c2c87dfa05c003963fd4427532b616c674f89f5d62e8ac5`.

A pre-commit smoke run was executed while the Checkpoint 3 changes were staged:

- Reported Git commit: `81058fc9de38968d7c98ae13ad9144c84d15820f`.
- Git dirty: `true`.
- All smoke-test assertions passed.

Observed results:

- Basic Pitch first request:
  - Notes: `39`.
  - Tabs: `39`.
  - Server total: `1433.19 ms`.
  - Audio I/O stage: `1110.16 ms`.
  - Note stage: `322.84 ms`.
  - Real-time factor: `0.1336`.
- Basic Pitch warm request:
  - Notes: `39`.
  - Tabs: `39`.
  - Server total: `98.24 ms`.
  - Audio I/O stage: `1.18 ms`.
  - Note stage: `96.88 ms`.
  - Real-time factor: `0.0092`.
- DSP baseline:
  - Notes: `38`.
  - Tabs: `37`.
  - Server total: `41.07 ms`.
  - Note stage: `40.01 ms`.
- Basic Pitch with chords:
  - Notes: `39`.
  - Tabs: `39`.
  - Chord segments: `8`.
  - Server total: `128.67 ms`.
  - Note stage: `94.42 ms`.
  - Chord stage: `32.89 ms`.

These are regression smoke-test observations, not final benchmark results.

#### Checkpoint 3 remaining verification

A clean smoke run is still required after restarting Uvicorn from a clean tree at commit `69ba84e`.

The clean run must confirm:

- `git_commit` begins with `69ba84e`.
- `git_dirty=false`.
- `basic_pitch_loaded_before=false`.
- `basic_pitch_loaded_after=true`.
- All smoke-test assertions pass.

After that run, update this document and `docs/THESIS_PROGRESS_LOG.md`, commit the documentation, and push Checkpoint 3.

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

## 7. Latest Regression Smoke-Test Results

Input: `data/mini_eval/clip.wav`  
Audio duration: approximately `10.728 s`  
Audio source: self-recorded guitar audio

### Basic Pitch first request

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Server total: `1433.19 ms`.
- Audio I/O stage: `1110.16 ms`.
- Note stage: `322.84 ms`.
- Real-time factor: `0.1336`.
- Client-observed time: `1438.29 ms`.

### Basic Pitch warm request

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Server total: `98.24 ms`.
- Audio I/O stage: `1.18 ms`.
- Note stage: `96.88 ms`.
- Real-time factor: `0.0092`.
- Client-observed time: `103.65 ms`.

### DSP baseline

- Backend: `baseline`.
- Runtime: `DSP`.
- Notes: `38`.
- Tabs: `37`.
- Server total: `41.07 ms`.
- Note stage: `40.01 ms`.
- Real-time factor: `0.0038`.
- Client-observed time: `44.18 ms`.

The missing baseline tab indicates that at least one detected note had no playable position under the current combination of pitch range and `max_fret`.

### Basic Pitch with chords

- Notes: `39`.
- Tabs: `39`.
- Chord segments: `8`.
- Server total: `128.67 ms`.
- Note stage: `94.42 ms`.
- Chord stage: `32.89 ms`.
- Real-time factor: `0.0120`.
- Client-observed time: `135.82 ms`.
- Chord backend: `chroma_template`.

These results verify regression behavior only. Final reporting must use repeated runs, a documented warm-up policy, multiple clip durations, and summary statistics such as mean, median, standard deviation, and P95.

## 8. Important Technical Caveats

### Preprocessing consistency

The standalone Basic Pitch test returned `35` events while the API returned `39`. The API currently peak-normalizes decoded audio, which may change confidence scores and event counts. Before formal evaluation, lock and document one preprocessing policy and apply it consistently across all experiments.

### Timing methodology

Do not report a single first or warm request as the final latency result. Separate and measure:

- Upload/read time.
- Decode/resample time.
- Model initialization time.
- Neural inference time.
- Chord-analysis time.
- Tab-mapping time.
- End-to-end request time.

Use at least one warm-up request and multiple measured repetitions. Report mean, median, standard deviation, and P95.

### Tabs and polyphony

The existing DP tab mapper processes notes sequentially and does not enforce unique strings for simultaneous chord notes. It may therefore produce physically invalid chord fingering assignments. Tabs remain secondary and must either be improved for onset groups or explicitly described as approximate note-position suggestions.

### Pitch range versus fret limit

`midi_high=88` corresponds to high E plus 24 frets, while the current tab mapper uses `max_fret=20`.

Choose one consistent configuration before formal evaluation:

- MIDI `40–88` with `max_fret=24`; or
- MIDI `40–84` with `max_fret=20`.

### Current chord implementation

Chords are still derived from audio chroma/templates rather than Basic Pitch note groups. This remains an interim baseline. The planned primary chord method will group active neural notes, match pitch-class templates, apply silence/no-chord gating, and smooth labels over time.

### Evaluation audio provenance

The current development and evaluation clips were recorded by the researcher using a guitar.

The local evaluation material is stored primarily under:

```text
evaluation_data/
```

The mini-evaluation regression clip is stored under:

```text
data/mini_eval/clip.wav
```

Before formal evaluation, document:

- Guitar type.
- Number of frets.
- Pickup or microphone configuration.
- Recording device.
- Recording environment.
- Original sample rate.
- Channel count.
- Audio format.
- Musical content of each clip.
- Whether each clip contains single notes, open strings, riffs, major/minor chords, silence, or muted strings.
- Ground-truth annotation method.
- Person responsible for annotation.
- Development-versus-held-out dataset split.

Threshold-development clips must remain separate from the held-out clips used for final reported results.

## 9. Next Milestones in Order

### Checkpoint 3 — Repository cleanup and reproducible regression testing

Status: **Implementation committed locally; clean commit-linked smoke rerun, documentation commit, and push pending.**

Remaining:

- Restart Uvicorn from a clean working tree at commit `69ba84e`.
- Run `eval/smoke_api.py` again.
- Confirm `git_dirty=false` and the commit hash begins with `69ba84e`.
- Add the clean-run result to both thesis context documents.
- Commit the documentation update.
- Push Checkpoint 3 to `origin/migrate-basic-pitch-backend`.

### Checkpoint 4 — Reproducible evaluation harness

Status: **Next.**

- Inspect and catalogue `evaluation_data/`.
- Separate development clips from held-out evaluation clips.
- Lock one audio preprocessing policy.
- Run the DSP baseline and Basic Pitch on the same annotated clips.
- Calculate onset-plus-pitch precision, recall, and F1 using a documented onset tolerance.
- Add note-with-offset metrics where practical.
- Benchmark first/cold and warm latency over multiple durations and repetitions.
- Save machine-readable CSV/JSON results.
- Report mean, median, standard deviation, and P95.
- Generate thesis-ready result tables and interpretations.

### Checkpoint 5 — Neural-note chord recognition

- Group active Basic Pitch notes into windows or onset groups.
- Convert MIDI notes to pitch classes.
- Match major and minor chord templates first.
- Add silence/no-chord gating.
- Smooth short unstable labels.
- Evaluate using chord ground truth and a documented segment metric or weighted chord symbol recall.

### Checkpoint 6 — Tab scope and mapping

- Lock `max_fret` and MIDI range consistently.
- Decide whether tabs represent approximate note positions or true polyphonic chord fingerings.
- Improve simultaneous-note assignment if time permits.
- Rerun tab evaluation.

### Checkpoint 7 — Flutter accessibility prototype

- Record or select audio.
- Upload to FastAPI.
- Display the chord timeline as the primary result.
- Display notes and tabs as optional detail.
- Speak concise feedback through Flutter TTS.
- Add semantic labels, large controls, screen-reader support, and predictable focus order.

### Checkpoint 8 — Final thesis material

- Methodology and architecture diagram.
- Dataset and clip description.
- Evaluation protocol.
- Accuracy and latency tables.
- Accessibility/usability evaluation.
- Discussion and limitations.
- Demo recording and reproducibility instructions.

## 10. One-Week Core Completion Priority

1. Finish and push Checkpoint 3.
2. Build the reproducible evaluation harness and obtain defensible results.
3. Implement and evaluate chord recognition with no-chord gating.
4. Build the minimal accessible Flutter flow.
5. Produce thesis-ready tables, figures, methodology, results, and limitations.

Streaming, alternate tunings, source separation, and advanced polyphonic fingering remain optional and must not block core completion.

## 11. Git Checkpoint Discipline

Use the active migration branch until the neural backend, evaluation, and cleanup are stable.

Verified checkpoints:

1. `b4c4018` — `feat(transcription): add Basic Pitch CoreML adapter`
2. `81058fc` — `feat(api): migrate transcription backend to Basic Pitch`
3. `69ba84e` — `chore(repo): finalize Basic Pitch migration cleanup`

Planned checkpoints:

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
