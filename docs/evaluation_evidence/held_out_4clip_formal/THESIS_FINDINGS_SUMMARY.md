# Controlled Chord Recognition — Thesis Findings Summary

## Experimental question

The controlled experiment compares two feature representations for prevailing
chord recognition from polyphonic guitar audio:

1. Traditional STFT-chroma features.
2. Basic Pitch note-event-derived features.

Both methods use the same chord vocabulary, shared classifier, uncertainty
handling, segmentation logic and evaluation framework.

The final classifier and segmentation parameters were selected using only the
eight-clip development split and were frozen before held-out evaluation.

## Final frozen configuration

- Configuration ID: `development_8clip_grid_20260806`
- Minimum chord score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.60` seconds

The 0.60-second duration ranked only marginally above the 0.40-second setting
under the predeclared development utility. It should therefore be reported as
a small development-set preference rather than a major optimization gain.

## Formal held-out dataset

The formal held-out evaluation contained four pre-registered recordings:

- `held_standard_01`
- `held_arpeggiated_02`
- `held_extended_03`
- `held_mixed_04`

The recordings were annotated without consulting model predictions.

No parameter was changed after inspecting held-out results.

## Primary held-out results

| Metric | Chroma | Basic Pitch |
|---|---:|---:|
| Overall exact accuracy | 22.8% | 61.1% |
| Root accuracy | 38.7% | 75.3% |
| Triad-family accuracy | 30.8% | 72.9% |
| Harmonic exact accuracy | 7.1% | 66.7% |
| No-chord F1 | 78.4% | 48.5% |
| Predicted uncertainty rate | 23.7% | 24.0% |
| Boundary F1 | 11.3% | 29.6% |
| Boundary mean absolute error | 53 ms | 65 ms |
| Median real-time factor | 0.051 | 0.022 |

## Primary finding

Basic Pitch note-event-derived features substantially outperformed traditional
STFT-chroma features for prevailing chord identity on the controlled held-out
recordings.

The largest differences were observed for:

- chord-root identification;
- triad-family identification;
- exact chord-quality recognition.

Basic Pitch achieved approximately:

- 75.3% root accuracy;
- 72.9% triad-family accuracy;
- 66.7% harmonic exact chord-only accuracy.

The corresponding chroma results were:

- 38.7% root accuracy;
- 30.8% triad-family accuracy;
- 7.1% harmonic exact chord-only accuracy.

## Generalization from development to held-out data

Basic Pitch chord recognition remained comparatively stable between
development and held-out recordings.

| Metric | Development | Held-out |
|---|---:|---:|
| Root accuracy | 74.3% | 75.3% |
| Triad-family accuracy | 73.4% | 72.9% |
| Harmonic exact accuracy | 68.2% | 66.7% |
| Overall exact accuracy | 70.4% | 61.1% |
| No-chord F1 | 87.9% | 48.5% |
| Predicted uncertainty rate | 11.9% | 24.0% |
| Boundary F1 | 36.9% | 29.6% |

The chord-only metrics changed relatively little. The larger reduction in
overall exact accuracy was associated mainly with poorer no-chord handling,
increased uncertainty and weaker segmentation rather than a collapse in chord
identity recognition.

## Performance by held-out recording condition

### Standard held strums

`held_standard_01` was the strongest Basic Pitch recording:

- Harmonic exact accuracy: 87.8%
- Root accuracy: 88.4%
- Boundary F1: 66.7%
- Predicted uncertainty rate: 4.6%

This indicates strong performance under ordinary sustained-chord conditions.

### Slow arpeggiation

`held_arpeggiated_02` remained strong:

- Harmonic exact accuracy: 84.4%
- Root accuracy: 85.5%

Therefore, slow arpeggiation alone was not a major failure condition for the
Basic Pitch note-event representation.

However, the method produced more uncertain and false transition segments than
on the standard held-strum recording.

### Extended chord qualities

`held_extended_03` produced:

- Root accuracy: 93.1%
- Harmonic exact accuracy: 63.9%

The difference shows that the method often retained the correct root while
misidentifying chord quality.

A major example was:

`A:7 -> A:maj`

for approximately 4.1 seconds.

This is a same-root chord-quality error rather than a completely unrelated
root prediction.

### Mixed repeated-strum and dynamic condition

`held_mixed_04` was the most difficult Basic Pitch recording:

- Harmonic exact accuracy: 51.0%
- Root accuracy: 59.1%
- Predicted uncertainty rate: 40.1%
- Boundary F1: 14.0%

The largest errors included long uncertain regions for:

- `E:sus4`
- `F#:min`
- `D:add9`

This suggests that the combination of repeated attacks, dynamic variation,
complex chord qualities and natural transitions creates a substantially more
difficult segmentation and classification condition.

## Basic Pitch chord-quality observations

Time-weighted held-out results by reference quality were:

| Quality | Exact | Same-root wrong quality | Wrong root | X |
|---|---:|---:|---:|---:|
| Major | 95.1% | 2.4% | 1.7% | 0.8% |
| Major seventh | 90.6% | 0.0% | 9.4% | 0.0% |
| Minor seventh | 83.5% | 0.0% | 13.5% | 2.9% |
| Minor | 64.9% | 0.2% | 3.0% | 31.9% |
| Dominant seventh | 61.0% | 21.0% | 7.1% | 10.9% |
| Added ninth | 48.0% | 29.4% | 1.2% | 14.4% |
| Suspended fourth | 34.5% | 0.0% | 3.5% | 62.0% |

These values must be interpreted cautiously because the controlled held-out
dataset is small and some qualities are represented by only one chord example.

Chord-specific inspection demonstrates this limitation clearly:

- `A:sus4` achieved 100% exact recognition.
- `E:sus4` achieved only 18.5% exact recognition.
- `A:min` achieved 85.3%.
- `F#:min` achieved 48.4%.

Therefore, the experiment supports identifying difficult examples and
conditions, but does not establish broad quality-specific population
performance.

## Chroma error pattern

The chroma method performed poorly for exact chord quality and generated many
false chord changes.

False changes included:

- 85 on `held_arpeggiated_02`;
- 31 on `held_extended_03`;
- 98 on `held_mixed_04`;
- 34 on `held_standard_01`.

Frequent errors included related chord-quality substitutions such as:

- `C:maj -> C:maj7`
- `A:maj -> A:maj7`
- `D:add9 -> D:maj7`

This suggests that chroma often retained relevant pitch-class energy while
failing to distinguish the intended chord quality reliably.

Chroma did, however, outperform Basic Pitch for no-chord F1 on the formal
held-out split.

## No-chord detection

Held-out no-chord F1:

- Chroma: 78.4%
- Basic Pitch: 48.5%

Basic Pitch frequently classified reference silence as either an uncertain
region or a lingering chord.

This is a meaningful weakness of the current note-event pipeline and should be
reported separately from chord-only classification accuracy.

## Uncertainty

Basic Pitch's predicted uncertainty rate increased from approximately 11.9%
on development data to 24.0% on held-out data.

The increase was concentrated heavily in difficult recordings rather than
being uniform across all clips.

For example:

- `held_standard_01`: 4.6% X
- `held_extended_03`: 1.3% X
- `held_arpeggiated_02`: 24.4% X
- `held_mixed_04`: 40.1% X

This indicates that uncertainty can be useful as a signal of difficult
recording conditions.

## Transition segmentation

Held-out boundary F1 remained modest:

- Chroma: 11.3%
- Basic Pitch: 29.6%

However, matched boundaries were temporally close to the reference:

- Chroma mean absolute boundary error: approximately 53 ms
- Basic Pitch mean absolute boundary error: approximately 65 ms

Therefore, the primary transition problem was not large displacement of
correctly matched boundaries. It was the creation of extra transitions and
the omission of some real transitions.

This distinction is important for the thesis discussion.

## Processing performance

Median real-time factors were:

- Chroma: 0.051
- Basic Pitch: 0.022

Both methods therefore processed the controlled recordings faster than audio
duration on the evaluation machine.

These measurements support an offline faster-than-real-time processing claim.
They do not by themselves establish a live streaming or hard real-time system.

## Research interpretation

The controlled experiment supports the following conclusion:

> For the evaluated controlled guitar recordings, converting polyphonic audio
> into Basic Pitch note events before chord classification produced
> substantially more accurate prevailing-chord identity estimates than using
> traditional STFT-chroma features with the same downstream classifier and
> segmentation pipeline.

The experiment also identifies important remaining weaknesses:

- no-chord detection for the Basic Pitch path;
- uncertain predictions under difficult mixed playing conditions;
- suspended and added-note chord ambiguity in some recordings;
- same-root seventh-versus-major quality confusions;
- extra and missed chord boundaries.

## Scope and limitations

The evidence must not be presented as broad population-level performance.

The controlled experiment is limited by:

- four held-out recordings;
- one performer;
- one guitar;
- one recording setup;
- manually selected chord progressions;
- a limited number of examples for individual chord qualities.

The findings demonstrate controlled proof-of-concept generalization and method
comparison rather than universal guitar or music chord-recognition accuracy.

## Evidence locations

Formal held-out aggregate evidence:

`docs/evaluation_evidence/held_out_4clip_formal/`

Per-clip analysis:

`PER_CLIP_ERROR_ANALYSIS.md`

Chord-level analysis:

`CHORD_LEVEL_ERROR_ANALYSIS.md`

Machine-readable derived analyses:

- `per_clip_error_analysis.json`
- `chord_level_error_analysis.json`

The original formal generated-run JSON files are preserved under:

`generated_run/`
