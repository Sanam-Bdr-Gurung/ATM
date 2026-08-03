from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from statistics import mean, median
from typing import Any

from chord_evaluation import parse_chord_label


CONFUSION_SEPARATOR = " -> "


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


def _parse_confusion_key(
    key: str,
) -> tuple[str, str]:
    if CONFUSION_SEPARATOR not in key:
        raise ValueError(
            "Invalid confusion key: "
            f"{key!r}."
        )

    reference_label, predicted_label = (
        key.split(
            CONFUSION_SEPARATOR,
            maxsplit=1,
        )
    )

    parse_chord_label(
        reference_label
    )

    parse_chord_label(
        predicted_label
    )

    return (
        reference_label,
        predicted_label,
    )


def _non_negative_float(
    value: Any,
    *,
    context: str,
) -> float:
    try:
        converted = float(
            value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be numeric."
        ) from exc

    if (
        not isfinite(converted)
        or converted < 0.0
    ):
        raise ValueError(
            f"{context} must be finite "
            "and non-negative."
        )

    return converted


def _non_negative_int(
    value: Any,
    *,
    context: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{context} must be an integer."
        )

    try:
        converted = int(
            value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be an integer."
        ) from exc

    if converted < 0:
        raise ValueError(
            f"{context} cannot be negative."
        )

    return converted


def _macro_average(
    evaluations: Sequence[
        Mapping[str, Any]
    ],
    *,
    section: str,
    metric: str,
) -> float | None:
    values: list[float] = []

    for evaluation in evaluations:
        section_value = evaluation.get(
            section
        )

        if not isinstance(
            section_value,
            Mapping,
        ):
            continue

        raw_value = section_value.get(
            metric
        )

        if raw_value is None:
            continue

        values.append(
            float(raw_value)
        )

    if not values:
        return None

    return mean(values)


def aggregate_chord_evaluations(
    evaluations: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError(
            "At least one chord evaluation "
            "is required."
        )

    pooled_confusion: defaultdict[
        str,
        float,
    ] = defaultdict(float)

    reference_boundary_count = 0
    predicted_boundary_count = 0
    matched_boundary_count = 0
    boundary_errors: list[float] = []

    for evaluation_index, evaluation in enumerate(
        evaluations
    ):
        context = (
            f"evaluations[{evaluation_index}]"
        )

        if not isinstance(
            evaluation,
            Mapping,
        ):
            raise ValueError(
                f"{context} must be an object."
            )

        confusion = evaluation.get(
            "confusion_duration_sec"
        )

        if not isinstance(
            confusion,
            Mapping,
        ):
            raise ValueError(
                f"{context} is missing "
                "confusion_duration_sec."
            )

        for raw_key, raw_duration in (
            confusion.items()
        ):
            if not isinstance(
                raw_key,
                str,
            ):
                raise ValueError(
                    f"{context} contains a "
                    "non-string confusion key."
                )

            _parse_confusion_key(
                raw_key
            )

            duration = _non_negative_float(
                raw_duration,
                context=(
                    f"{context}."
                    "confusion_duration_sec"
                    f"[{raw_key!r}]"
                ),
            )

            pooled_confusion[
                raw_key
            ] += duration

        boundaries = evaluation.get(
            "boundaries"
        )

        if not isinstance(
            boundaries,
            Mapping,
        ):
            raise ValueError(
                f"{context} is missing "
                "boundary metrics."
            )

        reference_count = (
            _non_negative_int(
                boundaries.get(
                    "reference_count"
                ),
                context=(
                    f"{context}.boundaries."
                    "reference_count"
                ),
            )
        )

        predicted_count = (
            _non_negative_int(
                boundaries.get(
                    "predicted_count"
                ),
                context=(
                    f"{context}.boundaries."
                    "predicted_count"
                ),
            )
        )

        matched_count = (
            _non_negative_int(
                boundaries.get(
                    "matched_count"
                ),
                context=(
                    f"{context}.boundaries."
                    "matched_count"
                ),
            )
        )

        if matched_count > min(
            reference_count,
            predicted_count,
        ):
            raise ValueError(
                f"{context} has more matched "
                "boundaries than available "
                "boundaries."
            )

        matches = boundaries.get(
            "matches"
        )

        if not isinstance(
            matches,
            list,
        ):
            raise ValueError(
                f"{context}.boundaries.matches "
                "must be a list."
            )

        if len(matches) != matched_count:
            raise ValueError(
                f"{context} boundary match "
                "count does not match the "
                "number of match records."
            )

        for match_index, match in enumerate(
            matches
        ):
            if not isinstance(
                match,
                Mapping,
            ):
                raise ValueError(
                    f"{context}.boundaries."
                    f"matches[{match_index}] "
                    "must be an object."
                )

            error = _non_negative_float(
                match.get(
                    "absolute_error_sec"
                ),
                context=(
                    f"{context}.boundaries."
                    f"matches[{match_index}]."
                    "absolute_error_sec"
                ),
            )

            boundary_errors.append(
                error
            )

        reference_boundary_count += (
            reference_count
        )

        predicted_boundary_count += (
            predicted_count
        )

        matched_boundary_count += (
            matched_count
        )

    total_duration = 0.0
    evaluable_duration = 0.0
    reference_chord_duration = 0.0

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

    for key, duration in (
        pooled_confusion.items()
    ):
        (
            reference_label,
            predicted_label,
        ) = _parse_confusion_key(
            key
        )

        reference_chord = parse_chord_label(
            reference_label
        )

        predicted_chord = parse_chord_label(
            predicted_label
        )

        total_duration += duration

        if predicted_label == "X":
            predicted_x_duration += (
                duration
            )

        if reference_label == "X":
            continue

        evaluable_duration += duration

        if predicted_label == "X":
            predicted_x_evaluable_duration += (
                duration
            )

        if reference_label == predicted_label:
            exact_correct_duration += (
                duration
            )

        if reference_label == "N":
            if predicted_label == "N":
                no_chord_true_positive += (
                    duration
                )
            else:
                no_chord_false_negative += (
                    duration
                )

        elif predicted_label == "N":
            no_chord_false_positive += (
                duration
            )

        if not reference_chord.is_chord:
            continue

        reference_chord_duration += (
            duration
        )

        if predicted_label == "X":
            predicted_x_on_chord_duration += (
                duration
            )

        if not predicted_chord.is_chord:
            continue

        same_root = (
            reference_chord.root
            == predicted_chord.root
        )

        if not same_root:
            continue

        root_correct_duration += duration

        if (
            reference_chord.triad_family
            == predicted_chord.triad_family
        ):
            triad_correct_duration += (
                duration
            )

        if (
            reference_chord.quality
            == predicted_chord.quality
        ):
            harmonic_exact_duration += (
                duration
            )

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

    boundary_precision = _safe_ratio(
        matched_boundary_count,
        predicted_boundary_count,
    )

    boundary_recall = _safe_ratio(
        matched_boundary_count,
        reference_boundary_count,
    )

    return {
        "schema_version": 1,
        "clip_count": len(
            evaluations
        ),
        "duration_sec": round(
            total_duration,
            6,
        ),
        "durations": {
            "evaluable_sec": round(
                evaluable_duration,
                6,
            ),
            "reference_chord_sec": round(
                reference_chord_duration,
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
                        reference_chord_duration,
                    )
                )
            ),
            "triad_family_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        triad_correct_duration,
                        reference_chord_duration,
                    )
                )
            ),
            "harmonic_exact_time_weighted": (
                _round_optional(
                    _safe_ratio(
                        harmonic_exact_duration,
                        reference_chord_duration,
                    )
                )
            ),
        },
        "macro_clip_average": {
            "exact_time_weighted": (
                _round_optional(
                    _macro_average(
                        evaluations,
                        section="accuracy",
                        metric=(
                            "exact_time_weighted"
                        ),
                    )
                )
            ),
            "root_time_weighted": (
                _round_optional(
                    _macro_average(
                        evaluations,
                        section="accuracy",
                        metric=(
                            "root_time_weighted"
                        ),
                    )
                )
            ),
            "triad_family_time_weighted": (
                _round_optional(
                    _macro_average(
                        evaluations,
                        section="accuracy",
                        metric=(
                            "triad_family_"
                            "time_weighted"
                        ),
                    )
                )
            ),
            "harmonic_exact_time_weighted": (
                _round_optional(
                    _macro_average(
                        evaluations,
                        section="accuracy",
                        metric=(
                            "harmonic_exact_"
                            "time_weighted"
                        ),
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
                        reference_chord_duration,
                    )
                )
            ),
        },
        "boundaries": {
            "reference_count": (
                reference_boundary_count
            ),
            "predicted_count": (
                predicted_boundary_count
            ),
            "matched_count": (
                matched_boundary_count
            ),
            "false_change_count": (
                predicted_boundary_count
                - matched_boundary_count
            ),
            "missed_change_count": (
                reference_boundary_count
                - matched_boundary_count
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
            "error_sample_count": len(
                boundary_errors
            ),
        },
        "confusion_duration_sec": {
            key: round(
                value,
                6,
            )
            for key, value in sorted(
                pooled_confusion.items()
            )
        },
    }
