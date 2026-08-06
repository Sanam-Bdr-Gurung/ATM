# Shared Chord Parameter Selection

## Final selection date

August 6, 2026

## Experimental role

The shared chord-classifier and segmentation parameters were selected using
only the controlled development split.

No held-out or exploratory recording was consulted during parameter
selection.

## Chroma implementation correction

Before parameter selection, the traditional chroma path was corrected to pass
a power spectrogram to the STFT-chroma extractor.

The corrected implementation was used for all subsequent development
evaluation and parameter selection.

## Pilot selection

An initial pilot selection used three development clips:

- `dev_clean_01`
- `dev_continuous_02`
- `dev_extended_03`

That pilot selected:

- Minimum score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.40` seconds
- Configuration ID: `development_grid_20260805`

The pilot configuration was explicitly treated as provisional because the
three recordings were too limited for final parameter selection.

## Expanded development split

Five additional recordings were pre-registered and annotated before their
model outputs were inspected:

- `dev_arpeggiated_04`
- `dev_seventh_05`
- `dev_repeated_06`
- `dev_voicing_07`
- `dev_transition_08`

The final development split therefore contained eight controlled recordings.

The clips covered:

- deliberately muted chord separations;
- sustained continuous transitions;
- major and minor chords;
- seventh, suspended and added-note qualities;
- arpeggiation;
- repeated strums;
- alternate voicings;
- natural string and fret noise;
- moderate dynamic variation.

## Fixed parameter grid

The same predeclared 18-configuration grid was applied to all eight
development clips:

- Minimum score: `0.58`, `0.62`, `0.66`
- Ambiguity margin: `0.020`, `0.035`, `0.050`
- Minimum segment duration: `0.40`, `0.60` seconds

The following settings remained fixed:

- activity thresholds;
- chroma feature settings;
- Basic Pitch inference settings;
- note-event aggregation settings;
- chord templates;
- chord-scoring weights;
- development ranking weights.

## Development ranking utility

Configurations were ranked using a balanced development utility combining:

- harmonic exact accuracy;
- root accuracy;
- triad-family accuracy;
- no-chord F1;
- boundary F1;
- one minus the predicted uncertainty rate.

This utility was used only for configuration selection. It is not presented
as a final evaluation metric.

## Final selected configuration

- Configuration ID: `development_8clip_grid_20260806`
- Minimum score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.60` seconds
- Balanced development score: `0.5160`

## Comparison of the two leading configurations

| Metric | 0.60-second hold | 0.40-second hold |
|---|---:|---:|
| Balanced score | 0.5160 | 0.5150 |
| Chroma harmonic exact accuracy | 11.7% | 11.7% |
| Chroma root accuracy | 32.7% | 32.7% |
| Chroma predicted X rate | 27.8% | 27.8% |
| Chroma boundary F1 | 14.1% | 14.1% |
| Basic Pitch harmonic exact accuracy | 68.2% | 68.3% |
| Basic Pitch root accuracy | 74.3% | 74.2% |
| Basic Pitch predicted X rate | 11.9% | 12.6% |
| Basic Pitch boundary F1 | 36.9% | 36.9% |

The advantage of the selected configuration was small. The 0.60-second
configuration was selected because it ranked first under the predeclared
utility, mainly through a slightly lower Basic Pitch uncertainty rate and a
slightly higher root score.

The result must be described as a marginal development-set preference rather
than a substantial difference.

## Freeze rule

After this configuration is committed:

- no classifier threshold may be changed based on held-out results;
- no segmentation threshold may be changed based on held-out results;
- chord templates and feature parameters remain frozen;
- held-out performance is reported even when it is worse than development
  performance.

Corrections to evaluation code remain possible only when they address an
actual implementation defect and are documented transparently.

## Evidence

Curated development evidence is preserved in:

`docs/evaluation_evidence/development_8clip_selection/`

The preserved evidence includes:

- the provisional eight-clip API evaluation;
- per-clip development metrics;
- all 18 parameter configurations;
- the top-ranked configurations;
- provenance and selected-configuration metadata.

All values in this document are development-set findings and must not be
presented as held-out performance.
