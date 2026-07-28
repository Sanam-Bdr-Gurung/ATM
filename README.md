# ChordAssist

A guitar-focused, accessibility-first transcription prototype for the thesis:

**AI-Powered Real-Time Guitar Transcription System with Audio Feedback and Flutter Interface for Visually Impaired Musicians**

> Current implementation status: low-latency, file-based guitar transcription.  
> Continuous live/WebSocket streaming has not yet been implemented or evaluated.

## Current Scope

ChordAssist currently focuses on:

- guitar-only audio;
- Standard E tuning assumed for tablature;
- Spotify Basic Pitch as the pretrained neural note-transcription backend;
- the existing DSP note detector retained as an experimental baseline;
- chord recognition as the primary user-facing output;
- notes and simple guitar tablature as secondary outputs;
- a FastAPI backend;
- a future accessible Flutter client with spoken feedback.

The following features are intentionally deferred:

- automatic tuning detection;
- alternate tunings;
- piano and violin output;
- instrument classification;
- source separation;
- training a model from scratch;
- on-device inference;
- continuous live streaming.

## Repository Status

The active migration branch is:

```text
migrate-basic-pitch-backend
```

The current backend supports:

- `GET /health`
- `POST /analyze-file`
- `backend=basic_pitch`
- `backend=baseline`
- optional chroma/template chord analysis
- Standard E tab mapping
- latency and real-time-factor fields

The old Onsets-and-Frames checkpoint route has been abandoned because it was piano-oriented and incompatible with the project's current Python/PyTorch environment. Its experimental history remains available through Git history and the thesis context documents.

## Environment

The currently verified development environment is:

```text
macOS on Apple Silicon
Python 3.10.20
Basic Pitch 0.4.0
CoreML runtime
```

Other operating systems and inference runtimes have not yet been formally validated.

## Setup

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Expected Basic Pitch warnings about missing TensorFlow, ONNX Runtime, and TFLite Runtime are non-blocking on the verified macOS setup because CoreML is used.

## Verify the Basic Pitch Adapter

Run the standalone model test with a guitar recording:

```bash
python eval/debug_basic_pitch.py evaluation_data/<path-to-audio>.wav
```

A local smoke-test clip may also be used:

```bash
python eval/debug_basic_pitch.py data/mini_eval/clip.wav
```

The `data/` directory and WAV artifacts may be excluded from Git, depending on the repository ignore rules.

## Run the API

Start the FastAPI development server:

```bash
uvicorn api:app --reload
```

Check health:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Before the first neural request, `basic_pitch_loaded` should be `false`. After the first Basic Pitch request, it should become `true`.

## Analyze a Guitar Recording

Basic Pitch notes without chord analysis:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/analyze-file?backend=basic_pitch&mode=full&chords=false" \
  -F "file=@evaluation_data/<path-to-audio>.wav" \
  | python -m json.tool
```

DSP baseline:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/analyze-file?backend=baseline&mode=full&chords=false" \
  -F "file=@evaluation_data/<path-to-audio>.wav" \
  | python -m json.tool
```

Basic Pitch notes with the temporary chroma/template chord detector:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/analyze-file?backend=basic_pitch&mode=full&chords=true" \
  -F "file=@evaluation_data/<path-to-audio>.wav" \
  | python -m json.tool
```

`mode=chunked` is currently accepted only for API compatibility. The response explicitly reports `mode_effective=full`.

## Run the Reusable API Smoke Test

Keep the API running in one terminal, then run:

```bash
python eval/smoke_api.py evaluation_data/<path-to-audio>.wav
```

For the local mini-evaluation clip:

```bash
python eval/smoke_api.py data/mini_eval/clip.wav
```

The script verifies:

- health before and after model initialization;
- Basic Pitch response structure;
- chronological note events;
- guitar MIDI range;
- fixed Standard E metadata;
- warm model reuse;
- DSP baseline response;
- chord-enabled response;
- removal of piano/violin placeholders.

It also records:

- Git branch and commit;
- Python, platform, architecture, and Basic Pitch version;
- audio SHA-256 and file size;
- server timing fields;
- client-observed request timing;
- note, tab, and chord counts.

Generated smoke-test outputs are written beneath `evaluation_results/smoke/`. They are regression evidence, not final thesis benchmark results.

## Evaluation Data

The project evaluation audio is stored under `evaluation_data/`.

The current clips were recorded by the researcher using a guitar. Before final thesis evaluation, document:

- guitar type and fret count;
- recording device or microphone;
- recording environment;
- original sample rate, channel count, and file format;
- clip duration and musical content;
- how note, chord, and tab ground truth were annotated.

Do not mix development smoke clips with the final reported evaluation set without clearly labeling them.

## Current Evaluation Direction

The thesis evaluation will compare the DSP baseline and Basic Pitch using the same annotated recordings.

Planned measures include:

- note precision, recall, and F1;
- onset tolerance and, where practical, offset-aware note metrics;
- chord segment accuracy;
- tab string, fret, and full-position accuracy;
- cold-start and warm latency;
- mean, median, and P95 latency;
- real-time factor;
- accessibility and usability observations.

Single smoke-test timings must not be reported as final benchmark results.

## Project Structure

```text
api.py                              FastAPI service
audio_input.py                      Audio decoding/preprocessing
models/basic_pitch_inference.py     Basic Pitch adapter
models/notes_interface.py           Stable note-event interface
notes_baseline.py                   DSP note baseline
features.py                         Audio/chroma features
chord_match.py                      Temporary chord templates
segmentation.py                     Chord-label segmentation
tabs_guitar_dp.py                   Standard-tuning tab mapping
eval/debug_basic_pitch.py           Standalone Basic Pitch debug run
eval/smoke_api.py                   Reusable API regression smoke test
eval/note_f1.py                     Note-event evaluation
eval/tab_accuracy_gt.py             Tab evaluation
evaluation_data/                    Researcher-recorded evaluation material
docs/THESIS_MASTER_CONTEXT.md       Durable project decisions and scope
docs/THESIS_PROGRESS_LOG.md         Verified checkpoint log
```

## Documentation Discipline

At every meaningful checkpoint:

1. run the relevant verification commands;
2. save machine-readable outputs where appropriate;
3. add a concise interpretation to `docs/THESIS_PROGRESS_LOG.md`;
4. update `docs/THESIS_MASTER_CONTEXT.md` when scope, architecture, metrics, or major limitations change;
5. commit and push the checkpoint before starting the next milestone.

## Current Limitations

- Chords are still produced by the older chroma/template method, not yet from Basic Pitch note groups.
- No-chord/silence gating is not yet complete.
- Tab mapping is approximate and does not yet enforce physically valid unique-string assignments for simultaneous chord notes.
- The allowed MIDI range and supported fret count must be finalized before formal tab evaluation.
- Audio preprocessing must be locked before comparing model accuracy.
- True continuous streaming has not been implemented.
