# Eight-Clip Development Evaluation Evidence

## Purpose

This directory preserves the important generated evidence used to select the
final shared chord-recognition configuration.

The original generated runs remain under `evaluation_results/`, which is
excluded from version control. The files in this directory are curated,
version-controlled copies.

## Development dataset

The selection used eight controlled, manually annotated guitar recordings.
The recordings included:

- isolated chords with muted gaps;
- continuous transitions;
- suspended and seventh chords;
- arpeggiation;
- repeated strums;
- alternate voicings;
- natural fret noise;
- moderate dynamic variation.

No held-out clip was used during parameter selection.

## Provisional configuration performance

The provisional three-clip configuration was evaluated on all eight
development clips before the final parameter sweep.

### Chroma

- Root accuracy: 32.7%
- Triad-family accuracy: 29.3%
- Harmonic exact accuracy: 11.7%
- No-chord F1: 85.5%
- Predicted X rate: 27.8%
- Boundary F1: 14.1%

### Basic Pitch note events

- Root accuracy: 74.2%
- Triad-family accuracy: 73.5%
- Harmonic exact accuracy: 68.3%
- No-chord F1: 86.2%
- Predicted X rate: 12.6%
- Boundary F1: 36.9%

## Final parameter sweep

The unchanged predeclared grid contained 18 configurations:

- Minimum score: 0.58, 0.62 and 0.66
- Ambiguity margin: 0.020, 0.035 and 0.050
- Minimum segment duration: 0.40 and 0.60 seconds

The ranking utility combined chord accuracy, no-chord detection, uncertainty
rate and boundary performance.

## Selected configuration

- Configuration ID: `development_8clip_grid_20260806`
- Minimum score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.60` seconds
- Balanced development score: `0.5160`

The selected configuration ranked first. Its score was only 0.001 higher than
the equivalent 0.40-second configuration. Therefore, the selection should be
reported as a marginal development-set preference rather than a substantial
performance difference.

## Preserved files

- `provisional_configuration_comparison.json`
- `provisional_per_clip_metrics.json`
- `full_parameter_grid.json`
- `top_parameter_configurations.json`
- `provenance.json`

A separate final development verification will be preserved after the
selected configuration is applied to the production API.
