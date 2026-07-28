# ChordAssist Thesis Progress Log

Use one dated entry per verified checkpoint. Record only reproducible results and clearly distinguish development smoke tests from final evaluation results.

## 2026-07-28 — Checkpoint 1: Basic Pitch CoreML adapter

### Completed

- Created `models/basic_pitch_inference.py`.
- Created `eval/debug_basic_pitch.py`.
- Locked a Python 3.10-compatible Basic Pitch environment.
- Confirmed Basic Pitch 0.4.0 loads through CoreML on Apple Silicon.
- Converted Basic Pitch output into the project event format:

```text
{t_on, t_off, midi, freq_hz, conf}
```

- Sorted note events chronologically.
- Preserved simultaneous/polyphonic note events.
- Applied the initial guitar MIDI range filter.
- Added temporary WAV cleanup around Basic Pitch inference.

### Verification

- Test input: `data/mini_eval/clip.wav`.
- Audio source: self-recorded guitar audio.
- Audio duration: approximately `10.728 s`.
- Adapter event count: `35`.
- All adapter assertions passed.
- Runtime: `COREML`.

### Git

- Branch: `migrate-basic-pitch-backend`.
- Commit: `b4c4018`.
- Commit message: `feat(transcription): add Basic Pitch CoreML adapter`.

---

## 2026-07-28 — Checkpoint 2: FastAPI Basic Pitch migration

### Completed and pushed

- Replaced the active O&F API backend with `BasicPitchTranscriber`.
- Retained the DSP backend for experimental baseline comparison.
- Made `basic_pitch` the default backend.
- Added lazy shared Basic Pitch model initialization.
- Reused one CoreML model instance across requests.
- Removed tuning detection from the active API request path.
- Returned Standard E as `fixed_assumption` rather than a detected tuning.
- Removed harmonic collapse from neural note events.
- Used fixed Standard E open-string MIDI values for tab mapping.
- Removed piano and violin response placeholders.
- Exposed the actual backend runtime.
- Exposed requested and effective processing modes.
- Retained `chroma_template` as the temporary chord backend.
- Kept `mode=chunked` only as a compatibility input while explicitly reporting `mode_effective=full`.

### Verification

#### Health before first neural request

- `ok=true`.
- `basic_pitch_loaded=false`.

#### First Basic Pitch API request

- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Server total: `5166.59 ms`.
- Note stage: `324.89 ms`.
- Real-time factor: `0.4816`.
- All API assertions passed.

#### Warm Basic Pitch API request

- Server total: `100.02 ms`.
- Note stage: `98.73 ms`.
- Real-time factor: `0.0093`.

#### Health after neural request

- `basic_pitch_loaded=true`.

#### DSP baseline request

- Notes: `38`.
- Tabs: `37`.
- Server total: `47.59 ms`.
- All baseline assertions passed.

#### Chord-enabled request

- Notes: `39`.
- Chord segments: `8`.
- Chord stage: `465.55 ms`.
- Temporary chord backend: `chroma_template`.
- Request completed successfully.

### Interpretation

- The FastAPI migration is functionally successful.
- Basic Pitch warm inference is promising enough to continue the near-real-time route.
- The individual first/warm values are smoke-test observations, not final thesis results.
- Chord labels require ground-truth validation and silence/no-chord gating.
- Audio preprocessing must be locked because the standalone adapter and API produced different event counts.
- The supported MIDI range and `max_fret` must be made consistent before formal tab evaluation.

### Git

- Branch: `migrate-basic-pitch-backend`.
- Commit: `81058fc9de38968d7c98ae13ad9144c84d15820f`.
- Commit message: `feat(api): migrate transcription backend to Basic Pitch`.
- Remote status: pushed.

---

## 2026-07-28 — Checkpoint 3: Repository cleanup and reproducible API regression testing

### Completed

- Removed the abandoned O&F model implementation:
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
- Confirmed that no active Python code references:
  - `OFTranscriber`
  - `models.of_inference`
  - `onsets_and_frames`
  - tuning detection or detune estimation
- Confirmed that no `__pycache__` directories or `.pyc` files remain tracked.
- Added a complete project `README.md`.
- Added `eval/smoke_api.py` for repeatable API regression testing.
- Added `evaluation_results/` to `.gitignore`.
- Preserved the DSP baseline.
- Preserved `eval/note_f1.py` and `eval/tab_accuracy_gt.py` for the next evaluation checkpoint.
- Preserved all removed experiments through Git history.

### Repository cleanup summary

The cleanup commit contained only the intended documentation, smoke-test harness, ignore rules, and legacy-code deletions. It did not commit:

- `.venv/`
- `data/`
- `evaluation_results/`
- `__pycache__/`
- `.pyc` files

### Evaluation audio used for regression testing

- Test file: `data/mini_eval/clip.wav`.
- Source: self-recorded guitar audio.
- Duration: approximately `10.728 s`.
- File size: `473166` bytes.
- SHA-256: `42277f91c9ffd9f97c2c87dfa05c003963fd4427532b616c674f89f5d62e8ac5`.

The mini-evaluation clip is a development/regression clip. It is not yet designated as part of the final held-out thesis test set.

### Preliminary pre-commit smoke run

The regression test was run while Checkpoint 3 changes were staged but before the cleanup commit was created.

Metadata:

- Reported Git commit: `81058fc9de38968d7c98ae13ad9144c84d15820f`.
- Git dirty: `true`.
- Branch: `migrate-basic-pitch-backend`.
- Python: `3.10.20`.
- Platform: `macOS-26.5.2-arm64-arm-64bit`.
- Architecture: `arm64`.
- Basic Pitch: `0.4.0`.
- CoreML Tools: `9.0`.
- All smoke-test assertions passed.

#### Basic Pitch first request

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Server total: `1433.19 ms`.
- Audio I/O stage: `1110.16 ms`.
- Note stage: `322.84 ms`.
- Tab stage: `0.17 ms`.
- Real-time factor: `0.1336`.
- Client-observed time: `1438.29 ms`.

#### Basic Pitch warm request

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Server total: `98.24 ms`.
- Audio I/O stage: `1.18 ms`.
- Note stage: `96.88 ms`.
- Tab stage: `0.16 ms`.
- Real-time factor: `0.0092`.
- Client-observed time: `103.65 ms`.

#### DSP baseline request

- Backend: `baseline`.
- Runtime: `DSP`.
- Notes: `38`.
- Tabs: `37`.
- Server total: `41.07 ms`.
- Audio I/O stage: `0.88 ms`.
- Note stage: `40.01 ms`.
- Tab stage: `0.15 ms`.
- Real-time factor: `0.0038`.
- Client-observed time: `44.18 ms`.

#### Basic Pitch with temporary chord analysis

- Backend: `basic_pitch`.
- Runtime: `COREML`.
- Notes: `39`.
- Tabs: `39`.
- Chord segments: `8`.
- Server total: `128.67 ms`.
- Audio I/O stage: `0.85 ms`.
- Note stage: `94.42 ms`.
- Chord stage: `32.89 ms`.
- Tab stage: `0.48 ms`.
- Real-time factor: `0.0120`.
- Client-observed time: `135.82 ms`.
- Chord backend: `chroma_template`.

### Interpretation

- Removing the abandoned O&F and tuning paths did not break the active API.
- Basic Pitch continues to return chronological polyphonic note events.
- The DSP baseline remains substantially faster and is retained for comparison.
- Most of the first-request cost in this run was attributed to the audio I/O/decode stage rather than the note stage.
- The warm Basic Pitch result is promising but is not a final benchmark.
- The chord backend remains an interim chroma/template implementation and has not yet been evaluated against chord ground truth.
- The timing variation between different smoke runs confirms that formal evaluation must use repeated measurements and summary statistics.

### Clean commit-linked verification

Status: **Pending**.

A new smoke run must be performed after restarting Uvicorn from a clean working tree at commit `69ba84e`. The clean run should record:

- `git_commit` beginning with `69ba84e`.
- `git_dirty=false`.
- `basic_pitch_loaded_before=false`.
- `basic_pitch_loaded_after=true`.
- All smoke-test assertions passed.
- First, warm, baseline, and chord-enabled timing summaries.

Append the clean-run results to this section before declaring Checkpoint 3 fully documented.

### Git

- Branch: `migrate-basic-pitch-backend`.
- Cleanup commit: `69ba84e`.
- Commit message: `chore(repo): finalize Basic Pitch migration cleanup`.
- Cleanup commit status at the time of this entry: committed locally.
- Documentation commit: pending.
- Remote push of Checkpoint 3: pending.

---

## Next checkpoint — Checkpoint 4: Reproducible evaluation harness

Planned work:

- Inspect and catalog `evaluation_data/`.
- Document the researcher-recorded audio collection process.
- Separate development/threshold-tuning clips from held-out evaluation clips.
- Lock one audio preprocessing policy.
- Run the DSP baseline and Basic Pitch on the same annotated clips.
- Calculate note precision, recall, and F1 using a documented onset tolerance.
- Add offset-aware note metrics where practical.
- Benchmark cold and warm latency over multiple clip durations and repetitions.
- Save machine-readable CSV/JSON results.
- Report mean, median, standard deviation, and P95.
- Produce thesis-ready result tables and a concise interpretation.

