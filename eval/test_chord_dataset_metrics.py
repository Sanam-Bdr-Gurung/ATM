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


from chord_dataset_metrics import (  # noqa: E402
    aggregate_chord_evaluations,
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


class ChordDatasetMetricTests(
    unittest.TestCase
):
    def test_time_weighted_pooling_differs_from_macro_average(
        self,
    ) -> None:
        long_correct = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    9.0,
                    "C:maj",
                )
            ],
            [
                segment(
                    0.0,
                    9.0,
                    "C:maj",
                )
            ],
            duration_sec=9.0,
        )

        short_incorrect = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    1.0,
                    "C:maj",
                )
            ],
            [
                segment(
                    0.0,
                    1.0,
                    "G:maj",
                )
            ],
            duration_sec=1.0,
        )

        result = aggregate_chord_evaluations(
            [
                long_correct,
                short_incorrect,
            ]
        )

        self.assertEqual(
            result["accuracy"][
                "exact_time_weighted"
            ],
            0.9,
        )

        self.assertEqual(
            result[
                "macro_clip_average"
            ][
                "exact_time_weighted"
            ],
            0.5,
        )

    def test_root_and_family_metrics_are_pooled(
        self,
    ) -> None:
        quality_mismatch = (
            evaluate_chord_timelines(
                [
                    segment(
                        0.0,
                        2.0,
                        "C:maj",
                    )
                ],
                [
                    segment(
                        0.0,
                        2.0,
                        "C:7",
                    )
                ],
                duration_sec=2.0,
            )
        )

        wrong_root = (
            evaluate_chord_timelines(
                [
                    segment(
                        0.0,
                        1.0,
                        "A:min",
                    )
                ],
                [
                    segment(
                        0.0,
                        1.0,
                        "E:min",
                    )
                ],
                duration_sec=1.0,
            )
        )

        result = aggregate_chord_evaluations(
            [
                quality_mismatch,
                wrong_root,
            ]
        )

        self.assertEqual(
            result["accuracy"][
                "root_time_weighted"
            ],
            round(
                2.0 / 3.0,
                6,
            ),
        )

        self.assertEqual(
            result["accuracy"][
                "triad_family_time_weighted"
            ],
            round(
                2.0 / 3.0,
                6,
            ),
        )

        self.assertEqual(
            result["accuracy"][
                "harmonic_exact_time_weighted"
            ],
            0.0,
        )

    def test_no_chord_counts_are_pooled(
        self,
    ) -> None:
        first = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    1.0,
                    "N",
                )
            ],
            [
                segment(
                    0.0,
                    0.5,
                    "N",
                ),
                segment(
                    0.5,
                    1.0,
                    "C:maj",
                ),
            ],
            duration_sec=1.0,
        )

        second = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    1.0,
                    "C:maj",
                )
            ],
            [
                segment(
                    0.0,
                    0.5,
                    "N",
                ),
                segment(
                    0.5,
                    1.0,
                    "C:maj",
                ),
            ],
            duration_sec=1.0,
        )

        result = aggregate_chord_evaluations(
            [
                first,
                second,
            ]
        )

        self.assertEqual(
            result["no_chord"][
                "true_positive_sec"
            ],
            0.5,
        )

        self.assertEqual(
            result["no_chord"][
                "false_positive_sec"
            ],
            0.5,
        )

        self.assertEqual(
            result["no_chord"][
                "false_negative_sec"
            ],
            0.5,
        )

        self.assertEqual(
            result["no_chord"][
                "f1"
            ],
            0.5,
        )

    def test_boundary_counts_and_errors_are_pooled(
        self,
    ) -> None:
        first = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    0.5,
                    "C:maj",
                ),
                segment(
                    0.5,
                    1.0,
                    "G:maj",
                ),
            ],
            [
                segment(
                    0.0,
                    0.55,
                    "C:maj",
                ),
                segment(
                    0.55,
                    1.0,
                    "G:maj",
                ),
            ],
            duration_sec=1.0,
            boundary_tolerance_sec=0.1,
        )

        second = evaluate_chord_timelines(
            [
                segment(
                    0.0,
                    1.0,
                    "A:min",
                )
            ],
            [
                segment(
                    0.0,
                    0.5,
                    "A:min",
                ),
                segment(
                    0.5,
                    1.0,
                    "E:min",
                ),
            ],
            duration_sec=1.0,
            boundary_tolerance_sec=0.1,
        )

        result = aggregate_chord_evaluations(
            [
                first,
                second,
            ]
        )

        boundaries = result[
            "boundaries"
        ]

        self.assertEqual(
            boundaries["reference_count"],
            1,
        )

        self.assertEqual(
            boundaries["predicted_count"],
            2,
        )

        self.assertEqual(
            boundaries["matched_count"],
            1,
        )

        self.assertEqual(
            boundaries[
                "false_change_count"
            ],
            1,
        )

        self.assertEqual(
            boundaries[
                "mean_absolute_error_sec"
            ],
            0.05,
        )

    def test_empty_evaluation_collection_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "At least one",
        ):
            aggregate_chord_evaluations(
                []
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
