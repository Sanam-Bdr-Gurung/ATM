# ChordAssist Controlled Dataset Protocol

## Purpose

This protocol pre-registers the controlled development and held-out
recordings used to compare:

1. Traditional STFT-chroma chord recognition.
2. Basic Pitch note-event-derived chord recognition.

Both methods use the same chord vocabulary, classifier, uncertainty handling,
segmentation and evaluation metrics.

## General recording rules

- Standard guitar tuning.
- No capo unless explicitly documented.
- No effects, backing track or metronome.
- One continuous take per clip.
- Do not construct clips by stitching separately recorded chords.
- Approximately 1–2 seconds of silence before and after playing.
- Each chord should remain clearly identifiable for approximately 2–4 seconds.
- Natural sustain, fret noise and string squeaks are permitted.
- Transitions may contain slight natural gaps.
- Recordings should not be intentionally degraded.
- Audio is exported as WAV and preserved without additional proces.

## Annotation rules

- The timeline must start at 0.0 and end at the exact audio duration.
- There must be no unlabelled gaps or overlaps.
- Retain the previous chord label while it remains audibly identifiable.
- Begin the next chord at its first audible attack.
- Use `N` for genuine silence or clearly muted intervals.
- Use reference `X` only when neither a chord nor silence can be assigned
  defensibly.
- Annotation must be completed without consulting either model's output.

## Development split

| Clip | Progression | Recording condition |
|---|---|---|
| dev_clean_01 | C:maj, G:maj, A:min, E:min | Held strums with deliberate muted gaps |
| dev_continuous_02 | D:maj, A:maj, B:min, G:maj | Sustained chords with direct transitions |
| dev_extended_03 | D:sus2, D:sus4, G:7, C:maj7 | Suspended and seventh qualities |
| dev_arpeggiated_04 | E:min, C:maj, G:maj, D:maj | Slow arpeggiation without deliberate muting |
| dev_seventh_05 | A:min7, D:7, G:maj7, C:add9 | Extended qualities with gentle strums |
| dev_repeated_06 | A:maj, E:maj, F#:min, D:maj | Two or three steady strums per chord |
| dev_voicing_07 | F:maj, D:min, A:min, C:7 | Alternate or partial-barre voicings |
| dev_transition_08 | E:7, A:sus2, D:sus4, B:min7 | Natural transitions, squeaks and moderate dynamic variation |

## Held-out split

The held-out split must not be analyzed until the final configuration has
been selected using all eight development clips.

| Clip | Progression | Recording condition |
|---|---|---|
| held_standard_01 | G:maj, B:min, C:maj, D:7 | Continuous held strums |
| held_arpeggiated_02 | A:min, C:maj, F:maj, E:7 | Slow arpeggiation |
| held_extended_03 | A:sus4, D:maj7, E:min7, A:7 | Suspended and seventh qualities |
| held_mixed_04 | F#:min, D:add9, A:maj, E:sus4 | Repeated strums and natural dynamic variation |

## Development procedure

1. Record and annotate all eight development clips.
2. Validate annotations without running model inference.
3. Run the fixed 18-configuration parameter grid once.
4. Select the final configuration using the predeclared utility.
5. Freeze the implementation and configuration identifier.
6. Do not make further parameter changes after held-out evaluation begins.

## Held-out procedure

1. Record and annotate all four held-out clips after final development tuning.
2. Validate the complete dataset.
3. Run both methods on the held-out split.
4. Preserve the first valid complete held-out run as the formal result.
5. Do not tune thresholds, templates, feature settings or segmentation from
   held-out results.

## Scope limitation

The controlled dataset is intended for a proof-of-concept comparison and
does not establish broad population-level performance across performers,
guitars, microphones or musical genres.
