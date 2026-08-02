from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from chord_engine import (  # noqa: E402
    classify_pitch_class_vector,
    humanize_chord_label,
)


def chord_vector(
    root_pitch_class: int,
    intervals: tuple[int, ...],
) -> np.ndarray:
    vector = np.zeros(
        12,
        dtype=np.float64,
    )

    for interval in intervals:
        vector[
            (root_pitch_class + interval) % 12
        ] = 1.0

    return vector


class ChordEngineTests(unittest.TestCase):
    def test_zero_activity_returns_no_chord(
        self,
    ) -> None:
        prediction = classify_pitch_class_vector(
            np.zeros(12)
        )

        self.assertEqual(
            prediction.label,
            "N",
        )
        self.assertEqual(
            prediction.display,
            "No chord",
        )
        self.assertEqual(
            prediction.confidence,
            1.0,
        )

    def test_major_and_minor_triads(
        self,
    ) -> None:
        c_major = classify_pitch_class_vector(
            chord_vector(
                0,
                (0, 4, 7),
            )
        )

        a_minor = classify_pitch_class_vector(
            chord_vector(
                9,
                (0, 3, 7),
            )
        )

        self.assertEqual(
            c_major.label,
            "C:maj",
        )
        self.assertEqual(
            a_minor.label,
            "A:min",
        )

    def test_seventh_chord_families(
        self,
    ) -> None:
        g_dominant = classify_pitch_class_vector(
            chord_vector(
                7,
                (0, 4, 7, 10),
            )
        )

        c_major_seventh = (
            classify_pitch_class_vector(
                chord_vector(
                    0,
                    (0, 4, 7, 11),
                )
            )
        )

        a_minor_seventh = (
            classify_pitch_class_vector(
                chord_vector(
                    9,
                    (0, 3, 7, 10),
                )
            )
        )

        self.assertEqual(
            g_dominant.label,
            "G:7",
        )
        self.assertEqual(
            c_major_seventh.label,
            "C:maj7",
        )
        self.assertEqual(
            a_minor_seventh.label,
            "A:min7",
        )

    def test_full_added_ninth_is_recognized(
        self,
    ) -> None:
        prediction = classify_pitch_class_vector(
            chord_vector(
                0,
                (0, 2, 4, 7),
            )
        )

        self.assertEqual(
            prediction.label,
            "C:add9",
        )

    def test_weak_melodic_ninth_does_not_force_add9(
        self,
    ) -> None:
        vector = chord_vector(
            0,
            (0, 4, 7),
        )

        vector[2] = 0.10

        prediction = classify_pitch_class_vector(
            vector
        )

        self.assertEqual(
            prediction.label,
            "C:maj",
        )

    def test_ambiguous_suspended_pitch_set_returns_x(
        self,
    ) -> None:
        # D-E-A can represent Dsus2 or Asus4 without
        # additional root or bass evidence.
        prediction = classify_pitch_class_vector(
            chord_vector(
                2,
                (0, 2, 7),
            )
        )

        self.assertEqual(
            prediction.label,
            "X",
        )

        self.assertAlmostEqual(
            prediction.margin,
            0.0,
            places=6,
        )

    def test_bass_evidence_resolves_suspended_root(
        self,
    ) -> None:
        prediction = classify_pitch_class_vector(
            chord_vector(
                2,
                (0, 2, 7),
            ),
            bass_pitch_class=2,
        )

        self.assertEqual(
            prediction.label,
            "D:sus2",
        )

    def test_uniform_pitch_activity_returns_x(
        self,
    ) -> None:
        prediction = classify_pitch_class_vector(
            np.ones(12)
        )

        self.assertEqual(
            prediction.label,
            "X",
        )

    def test_human_readable_labels(
        self,
    ) -> None:
        self.assertEqual(
            humanize_chord_label("C:maj7"),
            "C major seventh",
        )

        self.assertEqual(
            humanize_chord_label("G:7"),
            "G dominant seventh",
        )

        self.assertEqual(
            humanize_chord_label("N"),
            "No chord",
        )

        self.assertEqual(
            humanize_chord_label("X"),
            "Uncertain chord",
        )

    def test_invalid_vectors_are_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            classify_pitch_class_vector(
                [1.0, 0.0]
            )

        invalid = np.zeros(12)
        invalid[0] = -1.0

        with self.assertRaises(ValueError):
            classify_pitch_class_vector(
                invalid
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )
