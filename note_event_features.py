from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from chord_engine import (
    ChordPrediction,
    classify_pitch_class_vector,
)


DEFAULT_FRAME_STEP_SEC = 1024 / 22050
DEFAULT_WINDOW_SEC = 0.75


@dataclass(frozen=True)
class NoteEventFeatures:
    pitch_class_frames: np.ndarray
    times: np.ndarray
    frame_activity: np.ndarray
    bass_pitch_classes: tuple[int | None, ...]
    frame_step_sec: float
    window_sec: float


@dataclass(frozen=True)
class ValidatedNoteEvent:
    onset: float
    offset: float
    midi: int
    confidence: float


def _validate_audio_duration(
    audio_duration_sec: float,
) -> float:
    duration = float(audio_duration_sec)

    if not np.isfinite(duration):
        raise ValueError(
            "audio_duration_sec must be finite."
        )

    if duration <= 0.0:
        raise ValueError(
            "audio_duration_sec must be positive."
        )

    return duration


def _validate_note_event(
    event: dict,
    *,
    audio_duration_sec: float,
) -> ValidatedNoteEvent | None:
    required_fields = {
        "t_on",
        "t_off",
        "midi",
        "conf",
    }

    missing_fields = required_fields.difference(
        event
    )

    if missing_fields:
        raise ValueError(
            "Note event is missing required fields: "
            + ", ".join(
                sorted(missing_fields)
            )
        )

    onset = float(
        event["t_on"]
    )

    offset = float(
        event["t_off"]
    )

    midi = int(
        event["midi"]
    )

    confidence = float(
        event["conf"]
    )

    numeric_values = (
        onset,
        offset,
        confidence,
    )

    if not all(
        np.isfinite(value)
        for value in numeric_values
    ):
        raise ValueError(
            "Note event contains a non-finite value."
        )

    if onset < 0.0:
        raise ValueError(
            "Note onset cannot be negative."
        )

    if offset <= onset:
        raise ValueError(
            "Note offset must be greater than onset."
        )

    if not 0 <= midi <= 127:
        raise ValueError(
            "MIDI pitch must be between 0 and 127."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Note confidence must be between 0 and 1."
        )

    if onset >= audio_duration_sec:
        return None

    clipped_offset = min(
        offset,
        audio_duration_sec,
    )

    if clipped_offset <= onset:
        return None

    return ValidatedNoteEvent(
        onset=onset,
        offset=clipped_offset,
        midi=midi,
        confidence=confidence,
    )


def _validate_note_events(
    note_events: Sequence[dict],
    *,
    audio_duration_sec: float,
    minimum_confidence: float,
) -> list[ValidatedNoteEvent]:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError(
            "minimum_confidence must be between 0 and 1."
        )

    validated: list[ValidatedNoteEvent] = []

    for event in note_events:
        converted = _validate_note_event(
            event,
            audio_duration_sec=audio_duration_sec,
        )

        if converted is None:
            continue

        if converted.confidence < minimum_confidence:
            continue

        validated.append(
            converted
        )

    validated.sort(
        key=lambda event: (
            event.onset,
            event.midi,
            event.offset,
        )
    )

    return validated


def infer_note_activity_threshold(
    frame_activity: Sequence[float] | np.ndarray,
    *,
    relative_floor: float = 0.05,
    absolute_floor: float = 1e-8,
) -> float:
    if not 0.0 <= relative_floor <= 1.0:
        raise ValueError(
            "relative_floor must be between 0 and 1."
        )

    if absolute_floor < 0.0:
        raise ValueError(
            "absolute_floor cannot be negative."
        )

    activity = np.asarray(
        frame_activity,
        dtype=np.float64,
    ).reshape(-1)

    if activity.size == 0:
        return absolute_floor

    if not np.all(
        np.isfinite(activity)
    ):
        raise ValueError(
            "frame_activity contains non-finite values."
        )

    if np.any(activity < 0.0):
        raise ValueError(
            "frame_activity cannot contain negative values."
        )

    positive = activity[
        activity > absolute_floor
    ]

    if positive.size == 0:
        return absolute_floor

    reference_activity = float(
        np.percentile(
            positive,
            95,
        )
    )

    return max(
        absolute_floor,
        relative_floor * reference_activity,
    )


def note_events_to_pitch_class_frames(
    note_events: Sequence[dict],
    *,
    audio_duration_sec: float,
    frame_step_sec: float = DEFAULT_FRAME_STEP_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
    minimum_confidence: float = 0.05,
    bass_relative_weight: float = 0.15,
) -> NoteEventFeatures:
    duration = _validate_audio_duration(
        audio_duration_sec
    )

    if frame_step_sec <= 0.0:
        raise ValueError(
            "frame_step_sec must be positive."
        )

    if window_sec <= 0.0:
        raise ValueError(
            "window_sec must be positive."
        )

    if not 0.0 <= bass_relative_weight <= 1.0:
        raise ValueError(
            "bass_relative_weight must be between 0 and 1."
        )

    events = _validate_note_events(
        note_events,
        audio_duration_sec=duration,
        minimum_confidence=minimum_confidence,
    )

    frame_times = np.arange(
        0.0,
        duration,
        frame_step_sec,
        dtype=np.float64,
    )

    if frame_times.size == 0:
        frame_times = np.array(
            [0.0],
            dtype=np.float64,
        )

    frame_count = frame_times.size

    pitch_class_frames = np.zeros(
        (
            12,
            frame_count,
        ),
        dtype=np.float64,
    )

    frame_activity = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    bass_pitch_classes: list[
        int | None
    ] = []

    half_window = window_sec / 2.0

    for frame_index, center_time in enumerate(
        frame_times
    ):
        window_start = max(
            0.0,
            float(center_time) - half_window,
        )

        window_end = min(
            duration,
            float(center_time) + half_window,
        )

        effective_window_duration = max(
            window_end - window_start,
            1e-12,
        )

        weighted_active_notes: list[
            tuple[int, float]
        ] = []

        for event in events:
            if event.offset <= window_start:
                continue

            if event.onset >= window_end:
                break

            overlap_start = max(
                event.onset,
                window_start,
            )

            overlap_end = min(
                event.offset,
                window_end,
            )

            overlap_duration = (
                overlap_end - overlap_start
            )

            if overlap_duration <= 0.0:
                continue

            overlap_fraction = (
                overlap_duration
                / effective_window_duration
            )

            weight = (
                event.confidence
                * overlap_fraction
            )

            if weight <= 0.0:
                continue

            pitch_class = (
                event.midi % 12
            )

            pitch_class_frames[
                pitch_class,
                frame_index,
            ] += weight

            frame_activity[
                frame_index
            ] += weight

            weighted_active_notes.append(
                (
                    event.midi,
                    weight,
                )
            )

        if not weighted_active_notes:
            bass_pitch_classes.append(
                None
            )
            continue

        strongest_weight = max(
            weight
            for _, weight in weighted_active_notes
        )

        minimum_bass_weight = (
            strongest_weight
            * bass_relative_weight
        )

        credible_bass_notes = [
            midi
            for midi, weight in weighted_active_notes
            if weight >= minimum_bass_weight
        ]

        bass_midi = min(
            credible_bass_notes
        )

        bass_pitch_classes.append(
            bass_midi % 12
        )

    return NoteEventFeatures(
        pitch_class_frames=pitch_class_frames,
        times=frame_times,
        frame_activity=frame_activity,
        bass_pitch_classes=tuple(
            bass_pitch_classes
        ),
        frame_step_sec=float(
            frame_step_sec
        ),
        window_sec=float(
            window_sec
        ),
    )


def classify_note_event_frames(
    features: NoteEventFeatures,
    *,
    activity_threshold: float | None = None,
    relative_activity_floor: float = 0.05,
    minimum_score: float = 0.62,
    ambiguity_margin: float = 0.035,
) -> tuple[list[ChordPrediction], float]:
    frame_count = (
        features.pitch_class_frames.shape[1]
    )

    if features.pitch_class_frames.shape[0] != 12:
        raise ValueError(
            "pitch_class_frames must have shape "
            "(12, frame_count)."
        )

    if features.frame_activity.size != frame_count:
        raise ValueError(
            "frame_activity must contain one value "
            "per pitch-class frame."
        )

    if len(
        features.bass_pitch_classes
    ) != frame_count:
        raise ValueError(
            "bass_pitch_classes must contain one "
            "value per pitch-class frame."
        )

    if activity_threshold is None:
        resolved_threshold = (
            infer_note_activity_threshold(
                features.frame_activity,
                relative_floor=relative_activity_floor,
            )
        )
    else:
        resolved_threshold = float(
            activity_threshold
        )

        if resolved_threshold < 0.0:
            raise ValueError(
                "activity_threshold cannot be negative."
            )

    predictions: list[
        ChordPrediction
    ] = []

    silence_vector = np.zeros(
        12,
        dtype=np.float64,
    )

    for frame_index in range(
        frame_count
    ):
        activity = float(
            features.frame_activity[
                frame_index
            ]
        )

        if activity <= resolved_threshold:
            prediction = classify_pitch_class_vector(
                silence_vector
            )
        else:
            prediction = classify_pitch_class_vector(
                features.pitch_class_frames[
                    :,
                    frame_index,
                ],
                bass_pitch_class=(
                    features.bass_pitch_classes[
                        frame_index
                    ]
                ),
                minimum_score=minimum_score,
                ambiguity_margin=ambiguity_margin,
            )

        predictions.append(
            prediction
        )

    return predictions, resolved_threshold
