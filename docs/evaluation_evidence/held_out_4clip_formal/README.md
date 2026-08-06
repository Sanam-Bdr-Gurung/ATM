# Formal Four-Clip Held-Out Evaluation

## Status

This directory preserves the first valid complete evaluation of the frozen
ChordAssist configuration on the four pre-registered held-out recordings.

The held-out clips were not used for parameter selection.

## Frozen configuration

- Configuration ID: `development_8clip_grid_20260806`
- Minimum score: `0.58`
- Ambiguity margin: `0.020`
- Minimum segment duration: `0.60` seconds

No parameter was changed after inspecting held-out results.

## Held-out aggregate results

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

## Main finding

Basic Pitch note-event features substantially outperformed traditional
STFT-chroma features for chord-root, chord-family and exact chord-quality
recognition on the controlled held-out split.

Basic Pitch chord-only accuracy generalized well from development to held-out
recordings. Its principal held-out weaknesses were no-chord detection,
uncertain predictions and transition segmentation.

Chroma retained reasonable no-chord detection but performed poorly on exact
chord quality.

## Scope

The formal held-out split contains four recordings from one performer and one
recording setup. The findings support a controlled proof-of-concept comparison
and should not be generalized to arbitrary music, performers or recording
conditions.
