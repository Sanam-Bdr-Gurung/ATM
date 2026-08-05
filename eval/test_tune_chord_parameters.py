from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

EVAL_ROOT = PROJECT_ROOT / "eval"

for path in (
    PROJECT_ROOT,
    EVAL_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from tune_chord_parameters import (  # noqa: E402
    BASELINE_CONFIG,
    build_tuning_grid,
    score_configuration,
    score_method_metrics,
)


def metrics(
    *,
    harmonic_exact: float,
    root: float,
    triad: float,
    no_chord_f1: float,
    x_rate: float,
    boundary_f1: float,
) -> dict:
    return {
        "accuracy": {
            "exact_time_weighted": (
                harmonic_exact
            ),
            "root_time_weighted": root,
            "triad_family_time_weighted": (
                triad
            ),
            "harmonic_exact_time_weighted": (
                harmonic_exact
            ),
        },
        "no_chord": {
            "f1": no_chord_f1,
        },
        "ambiguity": {
            "prediction_x_rate_evaluable": (
                x_rate
            ),
        },
        "boundaries": {
            "f1": boundary_f1,
            "mean_absolute_error_sec": (
                0.05
            ),
            "false_change_count": 1,
            "missed_change_count": 1,
        },
    }


class ChordTuningTests(
    unittest.TestCase
):
    def test_grid_contains_18_configurations(
        self,
    ) -> None:
        grid = build_tuning_grid()

        self.assertEqual(
            len(grid),
            18,
        )

        self.assertIn(
            BASELINE_CONFIG,
            grid,
        )

    def test_method_score_rewards_lower_x_rate(
        self,
    ) -> None:
        low_x = metrics(
            harmonic_exact=0.7,
            root=0.8,
            triad=0.75,
            no_chord_f1=0.8,
            x_rate=0.1,
            boundary_f1=0.5,
        )

        high_x = metrics(
            harmonic_exact=0.7,
            root=0.8,
            triad=0.75,
            no_chord_f1=0.8,
            x_rate=0.5,
            boundary_f1=0.5,
        )

        self.assertGreater(
            score_method_metrics(
                low_x
            ),
            score_method_metrics(
                high_x
            ),
        )

    def test_configuration_score_averages_methods(
        self,
    ) -> None:
        strong = metrics(
            harmonic_exact=1.0,
            root=1.0,
            triad=1.0,
            no_chord_f1=1.0,
            x_rate=0.0,
            boundary_f1=1.0,
        )

        weak = metrics(
            harmonic_exact=0.0,
            root=0.0,
            triad=0.0,
            no_chord_f1=0.0,
            x_rate=1.0,
            boundary_f1=0.0,
        )

        score = score_configuration(
            {
                "first": strong,
                "second": weak,
            }
        )

        self.assertEqual(
            score,
            0.5,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
