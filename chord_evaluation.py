from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from statistics import mean, median
from typing import Any

from chord_annotations import SUPPORTED_CHORD_LABELS


TRIAD_FAMILY = {
    "maj": "maj",
    "7": "maj",
    "maj7": "maj",
    "add9": "maj",
    "min": "min",
    "min7": "min",
    "sus2": "sus2",
    "sus4": "sus4",
}


@dataclass(frozen=True)
class TimelineSegment:
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ParsedChord:
    label: str
    root: str | None
    quality: str | None
    triad_family: str | None

    @property
    def is_chord(self) -> bool:
        return self.root is not None


def parse_chord_label(
    label: str,
) -> ParsedChord:
    if label not in SUPPORTED_CHORD_LABELS:
        raise ValueError(
            f"Unsupported chord label: {label!r}."
        )

    if label in {"N", "X"}:
        return ParsedChord(
            label=label,
            root=None,
            quality=None,
            triad_family=None,
        )

    root, quality = label.split(
        ":",
        maxsplit=1,
    )

    return ParsedChord(
        label=label,
        root=root,
        quality=quality,
        triad_family=TRIAD_FAMILY[quality],
    )


def _coerce_segment(
    raw_segment: Any,
    *,
    context: str,
) -> TimelineSegment:
    if isinstance(raw_segment, Mapping):
        start_value = raw_segment.get("start")
        end_value = raw_segment.get("end")
        label_value = raw_segment.get("label")
    else:
        start_value = getattr(
            raw_segment,
            "start",
            None,
        )

        end_value = getattr(
            raw_segment,
            "end",
            None,
        )

        label_value = getattr(
            raw_segment,
            "label",
            None,
        )

    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} start and end must be numeric."
        ) from exc

    if not isfinite(start) or not isfinite(end):
        raise ValueError(
            f"{context} contains a non-finite boundary."
        )

    if not isinstance(label_value, str):
        raise ValueError(
            f"{context}.label must be a string."
        )

    label = label_value.strip()

    parse_chord_label(label)

    if end <= start:
        raise ValueError(
            f"{context}.end must be greater than start."
        )

    return TimelineSegment(
        start=start,
        end=end,
        label=label,
    )


def _merge_adjacent_labels(
    segments: Sequence[TimelineSegment],
    *,
    tolerance_sec: float,
) -> tuple[TimelineSegment, ...]:
    merged: list[TimelineSegment] = []

    for segment in segments:
        if (
            merged
            and merged[-1].label == segment.label
            and abs(
                merged[-1].end
                - segment.start
            )
            <= tolerance_sec
        ):
            previous = merged[-1]

            merged[-1] = TimelineSegment(
                start=previous.start,
                end=segment.end,
                label=previous.label,
            )

            continue

        merged.append(segment)

    return tuple(merged)


def normalize_timeline(
    raw_segments: Sequence[Any],
    *,
    duration_sec: float,
    role: str,
    tolerance_sec: float = 0.03,
    allow_empty_prediction: bool = False,
) -> tuple[TimelineSegment, ...]:
    duration = float(duration_sec)

    if not isfinite(duration) or duration <= 0.0:
        raise ValueError(
            "duration_sec must be positive and finite."
        )

    if tolerance_sec < 0.0:
        raise ValueError(
            "tolerance_sec cannot be negative."
        )

    if not raw_segments:
        if allow_empty_prediction:
            return (
                TimelineSegment(
                    start=0.0,
                    end=duration,
                    label="X",
                ),
            )

        raise ValueError(
            f"{role} timeline cannot be empty."
        )

    normalized: list[TimelineSegment] = []

    for index, raw_segment in enumerate(
        raw_segments
    ):
        context = (
            f"{role}[{index}]"
        )

        segment = _coerce_segment(
            raw_segment,
            context=context,
        )

        start = segment.start
        end = segment.end

        if index == 0:
            if abs(start) <= tolerance_sec:
                start = 0.0
            else:
                raise ValueError(
                    f"{role} timeline must begin at "
                    f"0 seconds; received {start:.4f}."
                )
        else:
            previous = normalized[-1]
            difference = start - previous.end

            if difference > tolerance_sec:
                raise ValueError(
                    f"{role} timeline has an "
                    f"unlabelled gap of "
                    f"{difference:.4f} seconds."
                )

            if difference < -tolerance_sec:
                raise ValueError(
                    f"{role} timeline has an "
                    f"overlap of "
                    f"{abs(difference):.4f} seconds."
                )

            start = previous.end

        if end > duration + tolerance_sec:
            raise ValueError(
                f"{context} ends after the "
                "audio duration."
            )

        if (
            index == len(raw_segments) - 1
            and abs(end - duration)
            <= tolerance_sec
        ):
            end = duration

        if end <= start:
            raise ValueError(
                f"{context} became empty after "
                "boundary normalization."
            )

        normalized.append(
            TimelineSegment(
                start=start,
                end=end,
                label=segment.label,
            )
        )

    final_end = normalized[-1].end

    if abs(final_end - duration) > tolerance_sec:
        raise ValueError(
            f"{role} timeline must end at "
            f"{duration:.4f}; received "
            f"{final_end:.4f}."
        )

    normalized[-1] = TimelineSegment(
        start=normalized[-1].start,
        end=duration,
        label=normalized[-1].label,
    )

    return _merge_adjacent_labels(
        normalized,
        tolerance_sec=tolerance_sec,
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator <= 0.0:
        return None

    return numerator / denominator


def _round_optional(
    value: float | None,
    digits: int = 6,
) -> float | None:
    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


def _f1_score(
    precision: float | None,
    recall: float | None,
) -> float | None:
    if precision is None or recall is None:
        return None

    denominator = precision + recall

    if denominator <= 0.0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


def _timeline_boundaries(
    timeline: Sequence[TimelineSegment],
) -> tuple[float, ...]:
    return tuple(
        segment.start
        for segment in timeline[1:]
    )


def _match_boundaries(
    reference: Sequence[float],
    predicted: Sequence[float],
    *,
    tolerance_sec: float,
) -> tuple[
    tuple[float, float, float],
    ...,
]:
    reference_values = tuple(reference)
    predicted_values = tuple(predicted)

    @lru_cache(maxsize=None)
    def solve(
        reference_index: int,
        predicted_index: int,
    ) -> tuple[
        int,
        float,
        tuple[
            tuple[float, float, float],
            ...,
        ],
    ]:
        if (
            reference_index
            >= len(reference_values)
            and predicted_index
            >= len(predicted_values)
        ):
            return 0, 0.0, ()

        candidates: list[
            tuple[
                int,
                float,
                tuple[
                    tuple[float, float, float],
                    ...,
                ],
            ]
        ] = []

        if reference_index < len(
            reference_values
        ):
            candidates.append(
                solve(
                    reference_index + 1,
                    predicted_index,
                )
            )

        if predicted_index < len(
            predicted_values
        ):
            candidates.append(
                solve(
                    reference_index,
                    predicted_index + 1,
                )
            )

        if (
            reference_index
            < len(reference_values)
            and predicted_index
            < len(predicted_values)
        ):
            reference_time = (
                reference_values[
                    reference_index
                ]
            )

            predicted_time = (
                predicted_values[
                    predicted_index
                ]
            )

            error = abs(
                reference_time
                - predicted_time
            )

            if error <= tolerance_sec:
                (
                    matched_count,
                    total_error,
                    pairs,
                ) = solve(
                    reference_index + 1,
                    predicted_index + 1,
                )

                candidates.append(
                    (
                        matched_count + 1,
                        total_error + error,
                        (
                            (
                                reference_time,
                                predicted_time,
                                error,
                            ),
                            *pairs,
                        ),
                    )
                )

        return max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                -candidate[1],
            ),
        )

    return solve(
        0,
        0,
    )[2]


def evaluate_chord_timelines(
    reference_segments: Sequence[Any],
    predicted_segments: Sequence[Any],
    *,
    duration_sec: float,
    timeline_tolerance_sec: float = 0.03,
    boundary_tolerance_sec: float = 0.25,
) -> dict[str, Any]:
    if boundary_tolerance_sec < 0.0:
        raise ValueError(
            "boundary_tolerance_sec cannot "
            "be negative."
        )

    reference = normalize_timeline(
        reference_segments,
        duration_sec=duration_sec,
        role="reference",
        tolerance_sec=timeline_tolerance_sec,
    )

    predicted = normalize_timeline(
        predicted_segments,
        duration_sec=duration_sec,
        role="prediction",
        tolerance_sec=timeline_tolerance_sec,
        allow_empty_prediction=True,
    )

    evaluable_duration = 0.0
    chord_reference_duration = 0.0

    exact_correct_duration = 0.0
    root_correct_duration = 0.0
    triad_correct_duration = 0.0
    harmonic_exact_duration = 0.0

    no_chord_true_positive = 0.0
    no_chord_false_positive = 0.0
    no_chord_false_negative = 0.0

    predicted_x_duration = 0.0
    predicted_x_evaluable_duration = 0.0
    predicted_x_on_chord_duration = 0.0

    confusion_duration: defaultdict[
        str,
        float,
    ] = defaultdict(float)

    reference_index = 0
    predicted_index = 0

    while (
        reference_index < len(reference)
        and predicted_index < len(predicted)
    ):
        reference_segment = reference[
            reference_index
        ]

        predicted_segment = predicted[
            predicted_index
        ]

        overlap_start = max(
            reference_segment.start,
            predicted_segment.start,
        )

        overlap_end = min(
            reference_segment.end,
            predicted_segment.end,
        )

        overlap_duration = max(
            0.0,
            overlap_end - overlap_start,
        )

        reference_chord = parse_chord_label(
            reference_segment.label
        )

        predicted_chord = parse_chord_label(
            predicted_segment.label
        )

        if overlap_duration > 0.0:
            confusion_key = (
                f"{reference_segment.label}"
                " -> "
                f"{predicted_segment.label}"
            )

            confusion_duration[
                confusion_key
            ] += overlap_duration

            if predicted_segment.label == "X":
                predicted_x_duration += (
                    overlap_duration
                )

            if reference_segment.label != "X":
                evaluable_duration += (
                    overlap_duration
                )

                if predicted_segment.label == "X":
                    predicted_x_evaluable_duration += (
                        overlap_duration
                    )

                if (
                    reference_segment.label
                    == predicted_segment.label
                ):
                    exact_correct_duration += (
                        overlap_duration
                    )

                if reference_segment.label == "N":
                    if predicted_segment.label == "N":
                        no_chord_true_positive += (
                            overlap_duration
                        )
                    else:
                        no_chord_false_negative += (
                            overlap_duration
                        )

                elif predicted_segment.label == "N":
                    no_chord_false_positive += (
                        overlap_duration
                    )

                if reference_chord.is_chord:
                    chord_reference_duration += (
                        overlap_duration
                    )

                    if predicted_segment.label == "X":
                        predicted_x_on_chord_duration += (
                            overlap_duration
                        )

                    if predicted_chord.is_chord:
                        same_root = (
                            reference_chord.root
                            == predicted_chord.root
                        )

                        if same_root:
                            root_correct_duration += (
                                overlap_duration
                            )

                            if (
                                reference_chord.triad_family
                                == predicted_chord.triad_family
                            ):
                                triad_correct_duration += (
                                    overlap_duration
                                )

                            if (
                                reference_chord.quality
                                == predicted_chord.quality
                            ):
                                harmonic_exact_duration += (
                                    overlap_duration
                                )

        reference_end = (
            reference_segment.end
        )

        predicted_end = (
            predicted_segment.end
        )

        if abs(
            reference_end - predicted_end
        ) <= 1e-12:
            reference_index += 1
            predicted_index += 1
        elif reference_end < predicted_end:
            reference_index += 1
        else:
            predicted_index += 1

    no_chord_precision = _safe_ratio(
        no_chord_true_positive,
        (
            no_chord_true_positive
            + no_chord_false_positive
        ),
    )

    no_chord_recall = _safe_ratio(
        no_chord_true_positive,
        (
            no_chord_true_positive
            + no_chord_false_negative
        ),
    )

    reference_boundaries = (
        _timeline_boundaries(
            reference
        )
    )

    predicted_boundaries = (
        _timeline_boundaries(
            predicted
        )
    )

    matched_boundaries = (
        _match_boundaries(
            reference_boundaries,
            predicted_boundaries,
            tolerance_sec=(
                boundary_tolerance_sec
            ),
        )
    )

    matched_count = len(
        matched_boundaries
    )

    reference_boundary_count = len(
        reference_boundaries
    )

    predicted_boundary_count = len(
        predicted_boundaries
    )

    boundary_precision = _safe_ratio(
        matched_count,
        predicted_boundary_count,
    )

    boundary_recall = _safe_ratio(
        matched_count,
        reference_boundary_count,
    )

    boundary_errors = [
        match[2]
        for match in matched_boundaries
    ]

    return {
        "schema_version": 1,
        "duration_sec": round(
            float(duration_sec),
            6,
        ),
        "durations": {
            "evaluable_sec": round(
                evaluable_duration,
                6,
            ),
            "reference_chord_sec": round(
                chord_reference_duration,
                6,
            ),
            "predicted_x_sec": round(
                predicted_x_duration,
                6,
            ),
        },
        "accuracy": {
            "exact_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        exact_correct_duration,
                        evaluable_duration,
                    )
                )
            ),
            "root_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        root_correct_duration,
                        chord_reference_duration,
                    )
                )
            ),
            "triad_family_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        triad_correct_duration,
                        chord_reference_duration,
                    )
                )
            ),
            "harmonic_exact_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        harmonic_exact_duration,
                        chord_reference_duration,
                    )
                )
            ),
        },
        "no_chord": {
            "true_positive_sec": round(
                no_chord_true_positive,
                6,
            ),
            "false_positive_sec": round(
                no_chord_false_positive,
                6,
            ),
            "false_negative_sec": round(
                no_chord_false_negative,
                6,
            ),
            "precision": _round_optional(
                no_chord_precision
            ),
            "recall": _round_optional(
                no_chord_recall
            ),
            "f1": _round_optional(
                _f1_score(
                    no_chord_precision,
                    no_chord_recall,
                )
            ),
        },
        "ambiguity": {
            "prediction_x_rate_evaluable": (
                _round_optional(
                    _safe_ratio(
                        predicted_x_evaluable_duration,
                        evaluable_duration,
                    )
                )
            ),
            "prediction_x_on_reference_chord_rate": (
                _round_optional(
                    _safe_ratio(
                        predicted_x_on_chord_duration,
                        chord_reference_duration,
                    )
                )
            ),
        },
        "boundaries": {
            "tolerance_sec": round(
                boundary_tolerance_sec,
                6,
            ),
            "reference_count": (
                reference_boundary_count
            ),
            "predicted_count": (
                predicted_boundary_count
            ),
            "matched_count": matched_count,
            "false_change_count": (
                predicted_boundary_count
                - matched_count
            ),
            "missed_change_count": (
                reference_boundary_count
                - matched_count
            ),
            "precision": _round_optional(
                boundary_precision
            ),
            "recall": _round_optional(
                boundary_recall
            ),
            "f1": _round_optional(
                _f1_score(
                    boundary_precision,
                    boundary_recall,
                )
            ),
            "mean_absolute_error_sec": (
                round(
                    mean(boundary_errors),
                    6,
                )
                if boundary_errors
                else None
            ),
            "median_absolute_error_sec": (
                round(
                    median(boundary_errors),
                    6,
                )
                if boundary_errors
                else None
            ),
            "matches": [
                {
                    "reference_sec": round(
                        reference_time,
                        6,
                    ),
                    "predicted_sec": round(
                        predicted_time,
                        6,
                    ),
                    "absolute_error_sec": round(
                        error,
                        6,
                    ),
                }
                for (
                    reference_time,
                    predicted_time,
                    error,
                ) in matched_boundaries
            ],
        },
        "confusion_duration_sec": {
            key: round(
                value,
                6,
            )
            for key, value in sorted(
                confusion_duration.items()
            )
        },
    }
