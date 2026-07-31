# ChordAssist

ChordAssist is an AI-assisted chord-recognition prototype developed for the thesis:

**AI-Assisted Recognition of Prevailing Chords from Polyphonic Audio with a Voice-Accessible Flutter Interface**

## Project Goal

The system analyzes uploaded or recorded musical audio and estimates the prevailing chord progression over time.

Supported input may include:

* solo guitar recordings;
* mobile-microphone recordings;
* guitar with vocals;
* professionally mastered polyphonic music;
* music containing guitar, bass, piano, drums, vocals, or other supporting instruments.

The system estimates the overall harmony represented by the combined audio. It does not identify which instrument produced each note and does not isolate guitar from a full mix.

## Primary Output

The required output is a stable time-aligned chord progression containing:

* chord label;
* readable chord name;
* start time;
* end time;
* confidence score;
* spoken progression.

Example:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 2.7,
      "label": "C:maj7",
      "display": "C major seventh",
      "confidence": 0.86
    }
  ],
  "progression": ["C:maj7"]
}
```

Individual note events may be used internally but are not a required user-facing result.

Guitar tablature, string estimation, fret estimation, and exact recovery of an original guitar performance are outside the active thesis scope.

## Research Comparison

The thesis compares two chord-recognition approaches.

### Basic Pitch method

```text
Audio
  → Basic Pitch note events
  → time-windowed pitch-class evidence
  → chord classification
  → confidence gating
  → temporal smoothing
  → chord timeline
```

### Traditional method

```text
Audio
  → chroma or harmonic pitch-class features
  → chord classification
  → confidence gating
  → temporal smoothing
  → chord timeline
```

Both approaches will use the same:

* chord vocabulary;
* ground-truth annotations;
* evaluation recordings;
* output format;
* confidence rules;
* temporal smoothing;
* evaluation metrics.

## Proposed Chord Vocabulary

The planned vocabulary includes:

* major;
* minor;
* suspended second;
* suspended fourth;
* diminished;
* augmented;
* dominant seventh;
* major seventh;
* minor seventh;
* half-diminished seventh;
* diminished seventh;
* added ninth;
* dominant ninth;
* major ninth;
* minor ninth;
* `N` for no chord;
* `X` for ambiguous harmony.

The final required vocabulary may be reduced if development evaluation shows that some advanced qualities cannot be supported reliably.

## Evaluation Data

The evaluation is divided into three groups.

### Development set

Researcher-recorded guitar chord progressions captured using a mobile microphone.

These recordings may be used for:

* algorithm development;
* threshold selection;
* window-size selection;
* confidence tuning;
* smoothing adjustments.

### Held-out controlled set

Separate guitar recordings created after the algorithm and parameters are fixed.

These recordings must not be used for tuning.

### Real-world robustness case

One professionally mastered commercial-song excerpt containing a manually verified chord progression.

The commercial audio will be stored locally and excluded from the public repository. Its result will be reported separately as an exploratory real-world case.

### Legacy note recordings

Earlier single-note guitar recordings are retained only as historical development or Basic Pitch regression material. They are not part of the primary chord-accuracy evaluation.

## Evaluation Metrics

Planned measures include:

* root accuracy;
* triad-family accuracy;
* seventh-chord accuracy;
* exact chord accuracy;
* time-weighted chord accuracy;
* no-chord precision and recall;
* ambiguous-duration percentage;
* chord-boundary timing error;
* unnecessary chord-change count;
* processing latency;
* real-time factor.

## Current Implementation Status

The repository currently contains:

* a FastAPI backend;
* audio decoding and preprocessing;
* a Spotify Basic Pitch CoreML adapter;
* an older DSP note detector;
* a temporary chroma/template chord detector;
* API and adapter smoke-test utilities.

The current API remains transitional. It still contains guitar-range filtering, note output, Standard E metadata, and tablature generation from the previous scope. These paths will be removed or refactored during the chord-first migration.

The current chord implementation supports only a limited traditional chroma/template path and has not yet been replaced by the final shared chord engine.

## Verified Environment

```text
macOS on Apple Silicon
Python 3.10.20
Basic Pitch 0.4.0
CoreML runtime
```

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

## Verify Basic Pitch

```bash
python eval/debug_basic_pitch.py path/to/audio.wav
```

This command verifies model loading and note-event extraction. Its results are adapter diagnostics rather than the final chord evaluation.

## Run the API

```bash
uvicorn api:app --reload
```

The API contract will be simplified after the new Basic Pitch-derived and chroma-derived chord paths are implemented.

## Active Development Branch

```text
feature/chord-first-recognition
```

## Explicitly Excluded

The required thesis does not include:

* guitar tablature;
* fret or string estimation;
* source separation;
* instrument identification;
* exact guitar-part extraction;
* automatic guitar tuning detection;
* musical notation generation;
* melody transcription as a primary output;
* training Basic Pitch from scratch;
* recognition of every possible jazz chord;
* continuous live streaming before file-based validation.

## Planned Development Order

1. Reset documentation and remove tab-focused evaluation code.
2. Remove tablature and guitar-only assumptions from the API.
3. Define the shared chord vocabulary and label format.
4. Implement shared chord scoring and uncertainty handling.
5. Build the Basic Pitch note-event feature path.
6. Refactor the traditional chroma comparison path.
7. Create and annotate the small chord development dataset.
8. Tune only on development recordings.
9. Evaluate on held-out guitar recordings.
10. Evaluate one commercial mixed-audio robustness case.
11. Integrate the final API with the accessible Flutter interface.

## Documentation

* `docs/THESIS_MASTER_CONTEXT.md` contains the current authoritative thesis scope and architecture.
* `docs/THESIS_PROGRESS_LOG.md` preserves verified implementation checkpoints and experimental history.
