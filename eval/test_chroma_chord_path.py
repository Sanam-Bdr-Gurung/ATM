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
)
from chord_match import (  # noqa: E402
    classify_chroma_frames,
    infer_activity_threshold,
)
from segmentation import (  # noqa: E402
    segment_chord_predictions,
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


class ChromaChordPathTests(unittest.TestCase):
    def test_activity_threshold_uses_positive_frames(
        self,
    ) -> None:
        threshold = infer_activity_threshold(
            [0.0, 0.0, 0.5, 1.0],
            relative_floor=0.10,
        )

        self.assertGreater(
            threshold,
            0.0,
        )

        self.assertLess(
            threshold,
            0.11,
        )

    def test_chroma_frames_include_no_chord_and_chords(
        self,
    ) -> None:
        chroma = np.zeros(
            (12, 10),
            dtype=np.float64,
        )

        c_major = chord_vector(
            0,
            (0, 4, 7),
        )

        g_dominant = chord_vector(
            7,
            (0, 4, 7, 10),
        )

        chroma[:, 2:6] = c_major[:, None]
        chroma[:, 6:10] = g_dominant[:, None]

        activity = np.array(
            [
                0.0,
                0.0,
                0.6,
                0.6,
                0.6,
                0.6,
                0.8,
                0.8,
                0.8,
                0.8,
            ],
            dtype=np.float64,
        )

        predictions, threshold = (
            classify_chroma_frames(
                chroma,
                frame_activity=activity,
            )
        )

        labels = [
            prediction.label
            for prediction in predictions
        ]

        self.assertGreater(
            threshold,
            0.0,
        )

        self.assertEqual(
            labels[:2],
            ["N", "N"],
        )

        self.assertEqual(
            labels[2:6],
            ["C:maj"] * 4,
        )

        self.assertEqual(
            labels[6:10],
            ["G:7"] * 4,
        )

    def test_uniform_chroma_is_uncertain(
        self,
    ) -> None:
        chroma = np.ones(
            (12, 3),
            dtype=np.float64,
        )

        predictions, _ = classify_chroma_frames(
            chroma,
            frame_activity=[1.0, 1.0, 1.0],
        )

        self.assertEqual(
            [
                prediction.label
                for prediction in predictions
            ],
            ["X", "X", "X"],
        )

    def test_predictions_form_timed_segments(
        self,
    ) -> None:
        chroma = np.zeros(
            (12, 10),
            dtype=np.float64,
        )

        chroma[:, 2:6] = chord_vector(
            0,
            (0, 4, 7),
        )[:, None]

        chroma[:, 6:10] = chord_vector(
            7,
            (0, 4, 7, 10),
        )[:, None]

        activity = np.array(
            [
                0.0,
                0.0,
                0.6,
                0.6,
                0.6,
                0.6,
                0.8,
                0.8,
                0.8,
                0.8,
            ]
        )

        predictions, _ = classify_chroma_frames(
            chroma,
            frame_activity=activity,
        )

        times = np.arange(
            10,
            dtype=np.float64,
        ) * 0.25

        segments = segment_chord_predictions(
            predictions,
            times,
            audio_duration_sec=2.5,
            min_hold_sec=0.4,
        )

        self.assertEqual(
            [
                segment["label"]
                for segment in segments
            ],
            [
                "N",
                "C:maj",
                "G:7",
            ],
        )

        self.assertEqual(
            segments[0]["start"],
            0.0,
        )

        self.assertEqual(
            segments[-1]["end"],
            2.5,
        )

    def test_short_uncertain_bridge_is_smoothed(
        self,
    ) -> None:
        c_major = classify_pitch_class_vector(
            chord_vector(
                0,
                (0, 4, 7),
            )
        )

        uncertain = classify_pitch_class_vector(
            np.ones(12)
        )

        predictions = [
            c_major,
            c_major,
            uncertain,
            c_major,
            c_major,
        ]

        times = np.arange(
            5,
            dtype=np.float64,
        ) * 0.25

        segments = segment_chord_predictions(
            predictions,
            times,
            audio_duration_sec=1.25,
            min_hold_sec=0.4,
        )

        self.assertEqual(
            len(segments),
            1,
        )

        self.assertEqual(
            segments[0]["label"],
            "C:maj",
        )

        self.assertEqual(
            segments[0]["start"],
            0.0,
        )

        self.assertEqual(
            segments[0]["end"],
            1.25,
        )

    def test_invalid_chroma_shape_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            classify_chroma_frames(
                np.zeros((10, 5))
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )
