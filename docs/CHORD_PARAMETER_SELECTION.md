# Shared Chord Parameter Selection

## Selection date

August 5, 2026

## Data used

The configuration was selected using only the controlled development split:

- `dev_clean_01`
- `dev_continuous_02`
- `dev_extended_03`

No held-out or exploratory recording was consulted during selection.

## Corrected chroma baseline

Before parameter tuning, the chroma extractor was corrected to use a power spectrogram for STFT chroma. The corrected feature implementation was treated as the tuning baseline.

## Parameter grid

The predeclared grid contained 18 configurations:

- Minimum score: `0.58`, `0.62`, `0.66`
- Ambiguity margin: `0.020`, `0.035`, `0.050`
- Minimum segment duration: `0.40`, `0.60` seconds

All activity thresholds, feature windows, note-event thresholds, chord templates, and scoring weights remained fixed.

## Selection score

Configurations were ranked using a balanced development utility combining:

- harmonic exact accuracy;
- root accuracy;
- triad-family accuracy;
- no-chord F1;
- boundary F1;
- predicted uncertainty rate.

The utility was used only for development-set configuration selection and is not a final evaluation metric.

## Selected configuration

- Minimum score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.40` seconds
- Configuration ID: `development_grid_20260805`

The `0.40`- and `0.60`-second configurations achieved identical measured development results. The shorter duration was selected because it preserves greater temporal resolution and avoids unnecessary suppression of legitimate short chord changes.

## Development comparison

| Method | Metric | Corrected baseline | Selected |
|---|---|---:|---:|
| Chroma | Root accuracy | 27.8% | 34.3% |
| Chroma | Harmonic exact accuracy | 18.5% | 21.9% |
| Chroma | Predicted X rate | 37.6% | 26.7% |
| Chroma | Boundary F1 | 17.0% | 20.9% |
| Basic Pitch | Root accuracy | 82.9% | 83.6% |
| Basic Pitch | Harmonic exact accuracy | 76.9% | 77.3% |
| Basic Pitch | Predicted X rate | 9.1% | 8.2% |
| Basic Pitch | Boundary F1 | 47.6% | 50.8% |

These figures are development-set observations and must not be presented as held-out performance.
