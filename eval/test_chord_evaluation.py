from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from chord_evaluation import (  # noqa: E402
    evaluate_chord_timelines,
)


def segment(
    start: float,
    end: float,
    label: str,
) -> dict:
    return {
        "start": start,
        "end": end,
        "label": label,
    }


class ChordEvaluationTests(
    unittest.TestCase
):
    def test_perfect_timeline(
        self,
    ) -> None:
        reference = [
            segment(0.0, 0.5, "N"),
            segment(0.5, 1.0, "C:maj"),
            segment(1.0, 1.5, "G:7"),
        ]

        result = evaluate_chord_timelines(
            reference,
            reference,
            duration_sec=1.5,
        )

        accuracy = result["accuracy"]

        self.assertEqual(
            accuracy["exact_time_weighted"],
            1.0,
        )

        self.assertEqual(
            accuracy["root_time_weighted"],
            1.0,
        )

        self.assertEqual(
            accuracy[
                "triad_family_time_weighted"
            ],
            1.0,
        )

        self.assertEqual(
            result["boundaries"]["f1"],
            1.0,
        )

    def test_root_and_triad_can_match_without_exact_quality(
        self,
    ) -> None:
        reference = [
            segment(
                0.0,
                1.0,
                "C:maj",
            )
        ]

        prediction = [
            segment(
                0.0,
                1.0,
                "C:7",
            )
        ]

        result = evaluate_chord_timelines(
            reference,
            prediction,
            duration_sec=1.0,
        )

        self.assertEqual(
            result["accuracy"][
                "root_time_weighted"
            ],
            1.0,
        )

        self.assertEqual(
            result["accuracy"][
                "triad_family_time_weighted"
            ],
            1.0,
        )

        self.assertEqual(
            result["accuracy"][
                "harmonic_exact_time_weighted"
            ],
            0.0,
        )

    def test_no_chord_precision_and_recall(
        self,
    ) -> None:
        reference = [
            segment(0.0, 0.5, "N"),
            segment(0.5, 1.0, "C:maj"),
        ]

        prediction = [
            segment(0.0, 0.25, "N"),
            segment(0.25, 0.75, "C:maj"),
            segment(0.75, 1.0, "N"),
        ]

        result = evaluate_chord_timelines(
            reference,
            prediction,
            duration_sec=1.0,
        )

        no_chord = result["no_chord"]

        self.assertEqual(
            no_chord["precision"],
            0.5,
        )

        self.assertEqual(
            no_chord["recall"],
            0.5,
        )

        self.assertEqual(
            no_chord["f1"],
            0.5,
        )

    def test_reference_x_is_excluded_from_accuracy(
        self,
    ) -> None:
        reference = [
            segment(0.0, 0.5, "X"),
            segment(0.5, 1.0, "C:maj"),
        ]

        prediction = [
            segment(0.0, 1.0, "G:maj"),
        ]

        result = evaluate_chord_timelines(
            reference,
            prediction,
            duration_sec=1.0,
        )

        self.assertEqual(
            result["durations"][
                "evaluable_sec"
            ],
            0.5,
        )

        self.assertEqual(
            result["accuracy"][
                "exact_time_weighted"
            ],
            0.0,
        )

    def test_empty_prediction_becomes_uncertain(
        self,
    ) -> None:
        reference = [
            segment(
                0.0,
                1.0,
                "C:maj",
            )
        ]

        result = evaluate_chord_timelines(
            reference,
            [],
            duration_sec=1.0,
        )

        self.assertEqual(
            result["ambiguity"][
                "prediction_x_rate_evaluable"
            ],
            1.0,
        )

        self.assertEqual(
            result["accuracy"][
                "exact_time_weighted"
            ],
            0.0,
        )

    def test_boundary_tolerance_and_false_changes(
        self,
    ) -> None:
        reference = [
            segment(0.0, 0.5, "C:maj"),
            segment(0.5, 1.0, "G:maj"),
        ]

        prediction = [
            segment(0.0, 0.55, "C:maj"),
            segment(0.55, 0.8, "G:maj"),
            segment(0.8, 1.0, "A:min"),
        ]

        result = evaluate_chord_timelines(
            reference,
            prediction,
            duration_sec=1.0,
            boundary_tolerance_sec=0.1,
        )

        boundaries = result["boundaries"]

        self.assertEqual(
            boundaries["matched_count"],
            1,
        )

        self.assertEqual(
            boundaries["false_change_count"],
            1,
        )

        self.assertEqual(
            boundaries[
                "mean_absolute_error_sec"
            ],
            0.05,
        )

    def test_unlabelled_gap_is_rejected(
        self,
    ) -> None:
        reference = [
            segment(0.0, 0.4, "C:maj"),
            segment(0.6, 1.0, "G:maj"),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "unlabelled gap",
        ):
            evaluate_chord_timelines(
                reference,
                reference,
                duration_sec=1.0,
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
