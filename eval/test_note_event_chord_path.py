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


from note_event_features import (  # noqa: E402
    classify_note_event_frames,
    note_events_to_pitch_class_frames,
)


def note_event(
    onset: float,
    offset: float,
    midi: int,
    confidence: float = 1.0,
) -> dict:
    return {
        "t_on": onset,
        "t_off": offset,
        "midi": midi,
        "conf": confidence,
    }


class NoteEventChordPathTests(
    unittest.TestCase
):
    def test_empty_events_produce_no_chord(
        self,
    ) -> None:
        features = (
            note_events_to_pitch_class_frames(
                [],
                audio_duration_sec=2.0,
                frame_step_sec=0.25,
                window_sec=0.75,
            )
        )

        predictions, _ = (
            classify_note_event_frames(
                features
            )
        )

        self.assertTrue(
            predictions
        )

        self.assertEqual(
            {
                prediction.label
                for prediction in predictions
            },
            {"N"},
        )

    def test_c_major_events_produce_c_major(
        self,
    ) -> None:
        events = [
            note_event(
                0.0,
                2.0,
                48,
            ),
            note_event(
                0.0,
                2.0,
                52,
            ),
            note_event(
                0.0,
                2.0,
                55,
            ),
        ]

        features = (
            note_events_to_pitch_class_frames(
                events,
                audio_duration_sec=2.0,
                frame_step_sec=0.25,
                window_sec=0.75,
            )
        )

        predictions, _ = (
            classify_note_event_frames(
                features
            )
        )

        self.assertEqual(
            {
                prediction.label
                for prediction in predictions
            },
            {"C:maj"},
        )

    def test_g_dominant_seventh_events(
        self,
    ) -> None:
        events = [
            note_event(
                0.0,
                2.0,
                43,
            ),
            note_event(
                0.0,
                2.0,
                47,
            ),
            note_event(
                0.0,
                2.0,
                50,
            ),
            note_event(
                0.0,
                2.0,
                53,
            ),
        ]

        features = (
            note_events_to_pitch_class_frames(
                events,
                audio_duration_sec=2.0,
                frame_step_sec=0.25,
                window_sec=0.75,
            )
        )

        predictions, _ = (
            classify_note_event_frames(
                features
            )
        )

        self.assertEqual(
            {
                prediction.label
                for prediction in predictions
            },
            {"G:7"},
        )

    def test_low_confidence_event_is_removed(
        self,
    ) -> None:
        events = [
            note_event(
                0.0,
                2.0,
                48,
            ),
            note_event(
                0.0,
                2.0,
                52,
            ),
            note_event(
                0.0,
                2.0,
                55,
            ),
            note_event(
                0.0,
                2.0,
                50,
                confidence=0.01,
            ),
        ]

        features = (
            note_events_to_pitch_class_frames(
                events,
                audio_duration_sec=2.0,
                frame_step_sec=0.25,
                window_sec=0.75,
                minimum_confidence=0.05,
            )
        )

        predictions, _ = (
            classify_note_event_frames(
                features
            )
        )

        self.assertEqual(
            {
                prediction.label
                for prediction in predictions
            },
            {"C:maj"},
        )

    def test_lowest_note_resolves_suspended_root(
        self,
    ) -> None:
        # D, E and A can represent Dsus2 or Asus4.
        # D3 is deliberately the lowest note.
        events = [
            note_event(
                0.0,
                2.0,
                50,
            ),
            note_event(
                0.0,
                2.0,
                52,
            ),
            note_event(
                0.0,
                2.0,
                57,
            ),
        ]

        features = (
            note_events_to_pitch_class_frames(
                events,
                audio_duration_sec=2.0,
                frame_step_sec=0.25,
                window_sec=0.75,
            )
        )

        predictions, _ = (
            classify_note_event_frames(
                features
            )
        )

        self.assertEqual(
            {
                prediction.label
                for prediction in predictions
            },
            {"D:sus2"},
        )

    def test_frame_shapes_and_times(
        self,
    ) -> None:
        features = (
            note_events_to_pitch_class_frames(
                [],
                audio_duration_sec=1.0,
                frame_step_sec=0.25,
                window_sec=0.75,
            )
        )

        self.assertEqual(
            features.pitch_class_frames.shape,
            (12, 4),
        )

        np.testing.assert_allclose(
            features.times,
            [
                0.0,
                0.25,
                0.5,
                0.75,
            ],
        )

        self.assertEqual(
            len(
                features.bass_pitch_classes
            ),
            4,
        )

    def test_invalid_note_event_is_rejected(
        self,
    ) -> None:
        invalid_event = note_event(
            1.0,
            0.5,
            60,
        )

        with self.assertRaises(
            ValueError
        ):
            note_events_to_pitch_class_frames(
                [invalid_event],
                audio_duration_sec=2.0,
            )

    def test_missing_event_field_is_rejected(
        self,
    ) -> None:
        invalid_event = {
            "t_on": 0.0,
            "t_off": 1.0,
            "midi": 60,
        }

        with self.assertRaises(
            ValueError
        ):
            note_events_to_pitch_class_frames(
                [invalid_event],
                audio_duration_sec=2.0,
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )
