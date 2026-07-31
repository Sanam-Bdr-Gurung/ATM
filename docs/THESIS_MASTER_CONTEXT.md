# ChordAssist Thesis Master Context

**Last updated:** 2026-07-31
**Repository:** `Sanam-Bdr-Gurung/ATM`
**Active branch:** `feature/chord-first-recognition`

## 1. Working Thesis Title

**AI-Assisted Recognition of Prevailing Chords from Polyphonic Audio with a Voice-Accessible Flutter Interface**

Alternative comparison-oriented title:

**Comparative Chord Recognition from Polyphonic Audio Using Neural Note Events and Traditional Chroma Features**

## 2. Final Scope Decision

Chord recognition is the central research problem and the primary musical output.

The system accepts uploaded or recorded audio and estimates a stable, time-aligned sequence of prevailing chords.

Input may contain:

* solo guitar;
* mobile-recorded guitar;
* guitar and vocals;
* guitar, bass, piano, drums, and vocals;
* professionally mastered commercial music.

The system estimates the combined prevailing harmony. It does not identify which instrument produced individual pitches.

## 3. Definition of Prevailing Chord

The prevailing chord is the chord label that best summarizes the sustained and dominant harmonic evidence within a short analysis interval.

The detector should prioritize:

* sustained notes;
* confident note detections;
* repeated pitch-class evidence;
* chord-tone relationships;
* consistency across neighbouring windows.

Temporary melody notes should not automatically generate a new chord or extension.

## 4. Primary Research Question

Can Spotify Basic Pitch note events be transformed into reliable prevailing-chord estimates from solo-guitar and polyphonic musical audio?

## 5. Comparative Research Question

How does chord recognition derived from Basic Pitch note events compare with traditional chroma-based chord recognition?

## 6. Real-World Robustness Question

How well do both approaches transfer from controlled mobile-recorded guitar performances to a professionally mastered multi-instrument recording?

## 7. Accessibility Question

Can the recognized progression be presented through a simple Flutter interface that allows visually impaired users to hear and navigate the detected chords?

## 8. Required Output

The required API and application output is:

* chord label;
* readable chord name;
* start time;
* end time;
* confidence;
* full progression;
* text-to-speech representation.

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

Basic Pitch note events are an internal representation. They are not a required primary output.

## 9. Neural-Assisted Method

```text
Audio
  → Basic Pitch note events
  → overlapping analysis windows
  → confidence and duration weighting
  → octave folding into pitch classes
  → chord candidate scoring
  → N/X uncertainty handling
  → temporal smoothing
  → prevailing chord timeline
```

The chord path must not remain restricted to guitar MIDI pitches because bass and other instruments may provide useful root and chord-tone evidence.

## 10. Traditional Method

```text
Audio
  → harmonic or spectral preprocessing
  → chroma representation
  → chord candidate scoring
  → N/X uncertainty handling
  → temporal smoothing
  → prevailing chord timeline
```

Both methods must share:

* chord vocabulary;
* label format;
* uncertainty rules;
* temporal smoothing;
* annotations;
* evaluation metrics.

The principal experimental difference is:

```text
Basic Pitch note-event evidence
versus
traditional audio-chroma evidence
```

## 11. Planned Chord Vocabulary

### Basic qualities

* major;
* minor;
* suspended second;
* suspended fourth;
* diminished;
* augmented.

### Seventh qualities

* dominant seventh;
* major seventh;
* minor seventh;
* half-diminished seventh;
* diminished seventh.

### Added and ninth qualities

* added ninth;
* dominant ninth;
* major ninth;
* minor ninth.

### Special labels

* `N`: no chord or insufficient harmonic activity;
* `X`: harmonic activity exists but classification is uncertain.

The final required vocabulary may be reduced based on development results.

## 12. Hierarchical Recognition

The classifier should prefer a simpler reliable label over a weak complex label.

Examples:

```text
C:maj9 → C:maj7 → C:maj
A:min9 → A:min7 → A:min
G:9    → G:7    → G:maj
```

Extension labels require strong and persistent supporting evidence.

## 13. Dataset Plan

### Legacy note material

Existing single-note guitar recordings are retained only for:

* model sanity checks;
* historical evidence;
* regression diagnostics.

They are not part of the primary chord evaluation.

### Development data

Three researcher-recorded guitar clips will be used for implementation and tuning.

Initial progression plan:

```text
C → Am → F → G
Cmaj7 → Am7 → Dm7 → G7
Cadd9 → Em7 → Dsus4 → G7
```

### Held-out controlled data

Two new guitar recordings will be created after all major parameters are locked.

They must use different takes and progressions and must not influence tuning.

### Real-world case

One professionally mastered commercial-song excerpt will be used as a separate exploratory robustness case.

The commercial audio will remain local and untracked. Public repository content may include:

* metadata;
* chord annotations;
* predictions;
* metrics;
* audio hashes.

## 14. Evaluation Metrics

Primary metrics:

* root accuracy;
* triad-family accuracy;
* seventh-level accuracy;
* exact chord accuracy;
* time-weighted chord accuracy.

Additional measures:

* no-chord precision and recall;
* ambiguous-duration percentage;
* chord-boundary timing error;
* unnecessary chord changes;
* confidence;
* total latency;
* real-time factor.

Results should be separated by condition:

* controlled guitar development;
* controlled held-out guitar;
* commercial mixed-audio case.

## 15. Current Verified Environment

* macOS on Apple Silicon;
* Python 3.10.20;
* Basic Pitch 0.4.0;
* CoreML runtime;
* CoreML Tools 9.0.

## 16. Completed Historical Work

Completed work includes:

* DSP note baseline;
* FastAPI file-analysis endpoint;
* audio decoding and preprocessing;
* Basic Pitch CoreML adapter;
* normalized note-event schema;
* model reuse and runtime reporting;
* smoke-test harness;
* latency fields;
* repository cleanup;
* removal of the abandoned Onsets-and-Frames implementation;
* removal of automatic tuning detection.

Earlier note and tab evaluations remain available through Git history.

## 17. Current Transitional Code

The active implementation still contains assumptions from the previous scope:

* guitar-only MIDI filtering;
* Standard E metadata;
* notes as a primary response field;
* automatic tab mapping;
* a `chords` boolean flag;
* a DSP note backend;
* a temporary major/minor chroma chord detector;
* smoke tests that expect note and tab output.

These are transitional and must be refactored in later checkpoints.

## 18. Explicitly Removed Scope

The required thesis excludes:

* guitar tablature;
* string and fret estimation;
* exact recovery of the original guitar performance;
* source separation;
* instrument identification;
* automatic tuning detection;
* alternate guitar tunings;
* musical notation generation;
* melody transcription as a primary output;
* training Basic Pitch from scratch;
* recognition of every possible jazz chord;
* guaranteed performance across every genre;
* continuous live streaming before file-based validation.

## 19. Future Work

Potential future extensions include:

* guitar chord diagrams;
* playable guitar voicings;
* guitar tablature from detected notes;
* source separation;
* instrument-specific transcription;
* larger commercial-song evaluation;
* altered chords;
* eleventh and thirteenth chords;
* key and harmonic-function analysis;
* continuous live recognition;
* on-device inference.

## 20. Checkpoint Order

### Checkpoint 4A — Scope reset

* Replace outdated README and master context.
* Preserve the progress log.
* Remove standalone tab and note-evaluation scripts.
* Protect commercial audio from Git.

### Checkpoint 4B — API cleanup

* Remove Standard E and tablature paths.
* Remove guitar-only response metadata.
* Broaden Basic Pitch inference for harmonic analysis.
* Make chords the primary response.
* Retain note events internally.

### Checkpoint 4C — Shared chord engine

* Define labels and templates.
* Implement scoring.
* Add `N` and `X`.
* Implement hierarchical fallback.
* Add temporal smoothing.

### Checkpoint 4D — Feature paths

* Basic Pitch note-event pitch-class representation.
* Traditional chroma representation.
* Shared chord classifier.

### Checkpoint 4E — Development evaluation

* Record and annotate three guitar clips.
* Tune only on development data.
* Lock all parameters.

### Checkpoint 5 — Final evaluation

* Record two held-out guitar clips.
* Evaluate without retuning.
* Evaluate one commercial mixed-audio case separately.

### Checkpoint 6 — Accessible Flutter interface

* Record or select audio.
* Analyze audio.
* Display chord timeline.
* Speak progression.
* Navigate next, previous, and repeat chord.

## 21. Final Thesis Boundary

The thesis answers:

> What prevailing chord best represents each section of the analyzed audio?

It does not answer:

> Which instrument, string, fret, or performer produced every detected pitch?
