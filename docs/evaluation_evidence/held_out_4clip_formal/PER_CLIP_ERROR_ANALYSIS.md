# Formal Held-Out Per-Clip Error Analysis

This report was generated from the archived first valid complete held-out run. No model inference or parameter tuning was performed.

## Per-clip metrics

| Clip | Method | Overall exact | Root | Triad family | Harmonic exact | N F1 | X rate | Boundary F1 | False changes | Missed changes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `held_arpeggiated_02` | Chroma | 24.3% | 46.3% | 36.3% | 5.5% | 94.2% | 25.7% | 8.5% | 85 | 1 |
| `held_arpeggiated_02` | Basic Pitch | 72.8% | 85.5% | 85.1% | 84.4% | 40.8% | 24.4% | 27.3% | 14 | 2 |
| `held_extended_03` | Chroma | 40.8% | 32.0% | 27.4% | 22.7% | 85.1% | 26.8% | 20.0% | 31 | 1 |
| `held_extended_03` | Basic Pitch | 61.3% | 93.1% | 93.1% | 63.9% | 69.2% | 1.3% | 57.1% | 5 | 1 |
| `held_mixed_04` | Chroma | 11.0% | 38.7% | 28.6% | 0.3% | 61.3% | 20.6% | 7.5% | 98 | 1 |
| `held_mixed_04` | Basic Pitch | 46.6% | 59.1% | 54.0% | 51.0% | 15.6% | 40.1% | 14.0% | 48 | 1 |
| `held_standard_01` | Chroma | 32.4% | 32.7% | 31.6% | 13.9% | 80.1% | 25.3% | 18.6% | 34 | 1 |
| `held_standard_01` | Basic Pitch | 80.5% | 88.4% | 88.4% | 87.8% | 69.9% | 4.6% | 66.7% | 5 | 0 |

## Largest chord confusions

The durations below are time-weighted reference-to-prediction errors.

### `held_arpeggiated_02`

#### Chroma

| Reference | Prediction | Duration |
|---|---|---:|
| `C:maj` | `C:maj7` | 3.252 s |
| `E:7` | `X` | 3.204 s |
| `E:7` | `E:maj7` | 3.158 s |
| `F:maj` | `X` | 2.833 s |
| `A:min` | `A:sus2` | 1.857 s |

#### Basic Pitch

| Reference | Prediction | Duration |
|---|---|---:|
| `E:7` | `X` | 2.022 s |
| `A:min` | `X` | 0.626 s |
| `F:maj` | `C:maj` | 0.285 s |
| `E:7` | `F:maj7` | 0.264 s |
| `F:maj` | `F:maj7` | 0.108 s |

Largest reference no-chord errors:

- `N → X`: 4.179 s

### `held_extended_03`

#### Chroma

| Reference | Prediction | Duration |
|---|---|---:|
| `A:sus4` | `X` | 2.031 s |
| `A:7` | `C#:min` | 1.950 s |
| `E:min7` | `X` | 1.579 s |
| `A:7` | `N` | 1.138 s |
| `A:7` | `X` | 1.068 s |

#### Basic Pitch

| Reference | Prediction | Duration |
|---|---|---:|
| `A:7` | `A:maj` | 4.119 s |
| `E:min7` | `D:sus2` | 0.418 s |
| `D:maj7` | `E:min7` | 0.156 s |
| `D:maj7` | `A:sus4` | 0.152 s |
| `A:7` | `X` | 0.139 s |

Largest reference no-chord errors:

- `N → A:maj`: 1.964 s
- `N → A:sus4`: 0.060 s

### `held_mixed_04`

#### Chroma

| Reference | Prediction | Duration |
|---|---|---:|
| `A:maj` | `A:maj7` | 7.105 s |
| `F#:min` | `X` | 4.226 s |
| `E:sus4` | `B:maj7` | 4.040 s |
| `D:add9` | `D:maj7` | 3.901 s |
| `D:add9` | `X` | 2.647 s |

#### Basic Pitch

| Reference | Prediction | Duration |
|---|---|---:|
| `E:sus4` | `X` | 8.700 s |
| `F#:min` | `X` | 4.598 s |
| `D:add9` | `D:sus2` | 1.950 s |
| `D:add9` | `X` | 1.440 s |
| `D:add9` | `D:maj` | 0.975 s |

Largest reference no-chord errors:

- `N → X`: 3.189 s
- `N → E:sus4`: 1.115 s
- `N → F#:min`: 0.029 s

### `held_standard_01`

#### Chroma

| Reference | Prediction | Duration |
|---|---|---:|
| `D:7` | `X` | 2.740 s |
| `C:maj` | `C:maj7` | 2.148 s |
| `D:7` | `N` | 1.382 s |
| `D:7` | `F#:min` | 1.068 s |
| `G:maj` | `B:min` | 0.975 s |

#### Basic Pitch

| Reference | Prediction | Duration |
|---|---|---:|
| `D:7` | `A:sus4` | 1.149 s |
| `B:min` | `G:maj7` | 0.108 s |
| `C:maj` | `D:7` | 0.105 s |
| `B:min` | `C:maj7` | 0.093 s |
| `G:maj` | `G:maj7` | 0.077 s |

Largest reference no-chord errors:

- `N → X`: 0.743 s
- `N → A:sus4`: 0.685 s
- `N → G:maj`: 0.172 s
