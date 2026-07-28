# ChordAssist Thesis Progress Log

Use one dated entry per verified checkpoint. Record only reproducible results; distinguish smoke tests from final evaluation.

## 2026-07-28 — Checkpoint 1: Basic Pitch CoreML adapter

### Completed

- Created `models/basic_pitch_inference.py`.
- Created `eval/debug_basic_pitch.py`.
- Locked a Python 3.10-compatible Basic Pitch environment.
- Confirmed Basic Pitch 0.4.0 loads with the CoreML runtime on Apple Silicon.
- Converted Basic Pitch tuples to `{t_on, t_off, midi, freq_hz, conf}`.
- Sorted events chronologically.
- Preserved simultaneous notes.
- Applied initial guitar range filtering.

### Verification

- Test input: `data/mini_eval/clip.wav`.
- Audio duration: approximately `10.728 s`.
- Adapter event count: `35`.
- All adapter assertions passed.

### Git

- Branch: `migrate-basic-pitch-backend`.
- Commit: record the commit hash here.

---

## 2026-07-28 — Checkpoint 2: FastAPI Basic Pitch migration

### Completed locally

- Replaced the active O&F API backend with `BasicPitchTranscriber`.
- Retained the DSP backend for baseline comparison.
- Made `basic_pitch` the default backend.
- Added lazy shared model initialization.
- Removed tuning detection from the request path.
- Returned Standard E as `fixed_assumption`.
- Removed harmonic collapse from neural results.
- Used Standard E values for tab mapping.
- Removed piano/violin response placeholders.
- Exposed actual backend runtime and effective processing mode.
- Retained `chroma_template` as the temporary chord backend.

### Verification

#### Health before first neural request

- `ok=true`.
- `basic_pitch_loaded=false`.

#### Basic Pitch request

- Notes: `39`.
- Tabs: `39`.
- Runtime: `COREML`.
- First total: `5166.59 ms`.
- First notes stage: `324.89 ms`.
- First RTF: `0.4816`.
- All assertions passed.

#### Warm Basic Pitch request

- Total: `100.02 ms`.
- Notes stage: `98.73 ms`.
- RTF: `0.0093`.

#### Health after neural request

- `basic_pitch_loaded=true`.

#### DSP baseline request

- Notes: `38`.
- Tabs: `37`.
- Total: `47.59 ms`.
- All assertions passed.

#### Chord-enabled request

- Notes: `39`.
- Chord segments: `8`.
- Chord stage: `465.55 ms`.
- Temporary backend: `chroma_template`.
- Request completed successfully.

### Interpretation

- API migration is functionally successful.
- Basic Pitch warm inference is fast enough to justify continuing the near-real-time route.
- The single cold/warm values are smoke tests, not final thesis results.
- Chord processing currently costs more than warm neural note inference.
- Chord labels require ground-truth validation and silence/no-chord gating.
- Preprocessing must be locked because standalone and API event counts differ.
- Tab range and `max_fret` must be made consistent.

### Git

- Branch: `migrate-basic-pitch-backend`.
- Commit: not yet confirmed on the remote branch at the time of this entry.
- Suggested commit message: `feat(api): migrate transcription backend to Basic Pitch`.

---

## Next entry — Checkpoint 3

Record:

- Checkpoint 2 commit hash.
- Generated-file cleanup.
- Legacy O&F disconnection/deletion.
- README/context documentation.
- Exact list of files removed.
- Smoke-test results after cleanup.
