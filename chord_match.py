from __future__ import annotations

from typing import Sequence

import numpy as np

from chord_engine import (
    ChordPrediction,
    classify_pitch_class_vector,
)


def _validate_chroma(
    chroma: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        chroma,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Chroma must be a two-dimensional matrix."
        )

    if matrix.shape[0] != 12:
        raise ValueError(
            "Chroma must have shape (12, frame_count)."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Chroma contains NaN or infinite values."
        )

    if np.any(matrix < 0.0):
        raise ValueError(
            "Chroma cannot contain negative values."
        )

    return matrix


def _validate_frame_activity(
    frame_activity: Sequence[float] | np.ndarray,
    frame_count: int,
) -> np.ndarray:
    activity = np.asarray(
        frame_activity,
        dtype=np.float64,
    ).reshape(-1)

    if activity.size != frame_count:
        raise ValueError(
            "frame_activity must contain one value per chroma frame."
        )

    if not np.all(np.isfinite(activity)):
        raise ValueError(
            "frame_activity contains NaN or infinite values."
        )

    if np.any(activity < 0.0):
        raise ValueError(
            "frame_activity cannot contain negative values."
        )

    return activity


def infer_activity_threshold(
    frame_activity: Sequence[float] | np.ndarray,
    *,
    relative_floor: float = 0.08,
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


def classify_chroma_frames(
    chroma: Sequence[Sequence[float]] | np.ndarray,
    *,
    frame_activity: Sequence[float] | np.ndarray | None = None,
    activity_threshold: float | None = None,
    relative_activity_floor: float = 0.08,
    minimum_score: float = 0.62,
    ambiguity_margin: float = 0.035,
) -> tuple[list[ChordPrediction], float]:
    """
    Classify each chroma frame through the shared chord engine.

    Activity is used only for silence/no-chord gating. Chord identity is
    determined from the 12-dimensional chroma vector.
    """
    matrix = _validate_chroma(chroma)
    frame_count = matrix.shape[1]

    if frame_count == 0:
        return [], 0.0

    if frame_activity is None:
        activity = np.sum(
            matrix,
            axis=0,
        )
    else:
        activity = _validate_frame_activity(
            frame_activity,
            frame_count,
        )

    if activity_threshold is None:
        resolved_threshold = infer_activity_threshold(
            activity,
            relative_floor=relative_activity_floor,
        )
    else:
        resolved_threshold = float(
            activity_threshold
        )

        if resolved_threshold < 0.0:
            raise ValueError(
                "activity_threshold cannot be negative."
            )

    predictions: list[ChordPrediction] = []
    silence_vector = np.zeros(
        12,
        dtype=np.float64,
    )

    for frame_index in range(frame_count):
        if activity[frame_index] <= resolved_threshold:
            prediction = classify_pitch_class_vector(
                silence_vector
            )
        else:
            prediction = classify_pitch_class_vector(
                matrix[:, frame_index],
                minimum_score=minimum_score,
                ambiguity_margin=ambiguity_margin,
            )

        predictions.append(prediction)

    return predictions, resolved_threshold
