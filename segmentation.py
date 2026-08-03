from __future__ import annotations

from typing import Sequence

import numpy as np

from chord_engine import (
    ChordPrediction,
    humanize_chord_label,
)


def group_labels(
    labels: list[str],
    times: list[float],
    min_hold_sec: float = 0.3,
) -> list[tuple[float, float, str]]:
    """
    Legacy grouping function retained temporarily for the current API.
    """
    if not labels:
        return []

    segments: list[tuple[float, float, str]] = []
    current_label = labels[0]
    start = times[0]

    for index in range(1, len(labels)):
        if labels[index] == current_label:
            continue

        end = times[index]

        if (
            segments
            and end - start < min_hold_sec
        ):
            previous_start, _, previous_label = segments[-1]
            segments[-1] = (
                previous_start,
                end,
                previous_label,
            )
        else:
            segments.append(
                (
                    start,
                    end,
                    current_label,
                )
            )

        current_label = labels[index]
        start = times[index]

    segments.append(
        (
            start,
            times[-1],
            current_label,
        )
    )

    return segments


def _validate_inputs(
    predictions: Sequence[ChordPrediction],
    times: Sequence[float] | np.ndarray,
    audio_duration_sec: float,
) -> np.ndarray:
    frame_times = np.asarray(
        times,
        dtype=np.float64,
    ).reshape(-1)

    if frame_times.size != len(predictions):
        raise ValueError(
            "times must contain one timestamp per prediction."
        )

    if not np.all(np.isfinite(frame_times)):
        raise ValueError(
            "times contains NaN or infinite values."
        )

    if np.any(frame_times < 0.0):
        raise ValueError(
            "times cannot contain negative values."
        )

    if (
        frame_times.size > 1
        and np.any(np.diff(frame_times) < 0.0)
    ):
        raise ValueError(
            "times must be sorted in ascending order."
        )

    if not np.isfinite(audio_duration_sec):
        raise ValueError(
            "audio_duration_sec must be finite."
        )

    if audio_duration_sec <= 0.0:
        raise ValueError(
            "audio_duration_sec must be positive."
        )

    return frame_times


def _run_bounds(
    labels: Sequence[str],
) -> list[tuple[int, int, str]]:
    if not labels:
        return []

    runs: list[tuple[int, int, str]] = []
    start = 0
    current = labels[0]

    for index in range(1, len(labels)):
        if labels[index] == current:
            continue

        runs.append(
            (
                start,
                index - 1,
                current,
            )
        )

        start = index
        current = labels[index]

    runs.append(
        (
            start,
            len(labels) - 1,
            current,
        )
    )

    return runs


def _run_end_time(
    end_index: int,
    frame_times: np.ndarray,
    audio_duration_sec: float,
) -> float:
    next_index = end_index + 1

    if next_index < frame_times.size:
        return min(
            float(frame_times[next_index]),
            audio_duration_sec,
        )

    return audio_duration_sec


def _mean_confidence(
    predictions: Sequence[ChordPrediction],
    start_index: int,
    end_index: int,
) -> float:
    values = [
        prediction.confidence
        for prediction in predictions[
            start_index : end_index + 1
        ]
    ]

    return float(
        np.mean(values)
    )


def _smooth_short_runs(
    predictions: Sequence[ChordPrediction],
    frame_times: np.ndarray,
    audio_duration_sec: float,
    min_hold_sec: float,
) -> list[str]:
    labels = [
        prediction.label
        for prediction in predictions
    ]

    if min_hold_sec <= 0.0:
        return labels

    # Several passes allow a short bridge to disappear and the two
    # neighbouring runs to merge.
    for _ in range(4):
        runs = _run_bounds(labels)
        changed = False

        for run_index, (
            start_index,
            end_index,
            label,
        ) in enumerate(runs):
            start_time = float(
                frame_times[start_index]
            )

            end_time = _run_end_time(
                end_index,
                frame_times,
                audio_duration_sec,
            )

            duration = end_time - start_time

            if duration >= min_hold_sec:
                continue

            previous_run = (
                runs[run_index - 1]
                if run_index > 0
                else None
            )

            next_run = (
                runs[run_index + 1]
                if run_index + 1 < len(runs)
                else None
            )

            replacement: str | None = None

            # A short label between two identical chords is treated as a
            # temporary classification interruption.
            if (
                previous_run is not None
                and next_run is not None
                and previous_run[2] == next_run[2]
            ):
                replacement = previous_run[2]

            # A brief uncertain region may be assigned to the stronger
            # neighbouring segment. Other short musical labels are
            # preserved to avoid deleting genuine rapid chord changes.
            elif label == "X":
                neighbouring_runs = [
                    run
                    for run in (
                        previous_run,
                        next_run,
                    )
                    if run is not None
                ]

                if neighbouring_runs:
                    replacement_run = max(
                        neighbouring_runs,
                        key=lambda run: _mean_confidence(
                            predictions,
                            run[0],
                            run[1],
                        ),
                    )

                    replacement = replacement_run[2]

            if (
                replacement is None
                or replacement == label
            ):
                continue

            for frame_index in range(
                start_index,
                end_index + 1,
            ):
                labels[frame_index] = replacement

            changed = True
            break

        if not changed:
            break

    return labels


def segment_chord_predictions(
    predictions: Sequence[ChordPrediction],
    times: Sequence[float] | np.ndarray,
    *,
    audio_duration_sec: float,
    min_hold_sec: float = 0.4,
) -> list[dict[str, object]]:
    if not predictions:
        return []

    frame_times = _validate_inputs(
        predictions,
        times,
        audio_duration_sec,
    )

    labels = _smooth_short_runs(
        predictions,
        frame_times,
        audio_duration_sec,
        min_hold_sec,
    )

    runs = _run_bounds(labels)
    segments: list[dict[str, object]] = []

    for start_index, end_index, label in runs:
        start_time = max(
            0.0,
            min(
                float(frame_times[start_index]),
                audio_duration_sec,
            ),
        )

        end_time = _run_end_time(
            end_index,
            frame_times,
            audio_duration_sec,
        )

        if end_time <= start_time:
            continue

        frame_predictions = predictions[
            start_index : end_index + 1
        ]

        confidence = float(
            np.mean(
                [
                    prediction.confidence
                    for prediction in frame_predictions
                ]
            )
        )

        mean_score = float(
            np.mean(
                [
                    prediction.score
                    for prediction in frame_predictions
                ]
            )
        )

        mean_margin = float(
            np.mean(
                [
                    prediction.margin
                    for prediction in frame_predictions
                ]
            )
        )

        segments.append(
            {
                "start": round(
                    start_time,
                    4,
                ),
                "end": round(
                    end_time,
                    4,
                ),
                "label": label,
                "display": humanize_chord_label(
                    label
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
                "mean_score": round(
                    mean_score,
                    4,
                ),
                "mean_margin": round(
                    mean_margin,
                    4,
                ),
                "frame_count": len(
                    frame_predictions
                ),
            }
        )

    return segments
