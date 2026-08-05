from __future__ import annotations

import inspect
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


from chord_config import (  # noqa: E402
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MINIMUM_SCORE,
    DEFAULT_MINIMUM_SEGMENT_DURATION_SEC,
    SELECTED_CONFIG_ID,
    selected_chord_configuration,
)
from chord_engine import (  # noqa: E402
    classify_pitch_class_vector,
)
from chord_match import (  # noqa: E402
    classify_chroma_frames,
)
from note_event_features import (  # noqa: E402
    classify_note_event_frames,
)
from segmentation import (  # noqa: E402
    segment_chord_predictions,
)


def default_value(
    function,
    parameter_name: str,
):
    return inspect.signature(
        function
    ).parameters[
        parameter_name
    ].default


class ChordConfigurationTests(
    unittest.TestCase
):
    def test_selected_values(
        self,
    ) -> None:
        self.assertEqual(
            SELECTED_CONFIG_ID,
            "development_grid_20260805",
        )

        self.assertEqual(
            DEFAULT_MINIMUM_SCORE,
            0.58,
        )

        self.assertEqual(
            DEFAULT_AMBIGUITY_MARGIN,
            0.020,
        )

        self.assertEqual(
            DEFAULT_MINIMUM_SEGMENT_DURATION_SEC,
            0.40,
        )

    def test_classifier_defaults_do_not_drift(
        self,
    ) -> None:
        for function in (
            classify_pitch_class_vector,
            classify_chroma_frames,
            classify_note_event_frames,
        ):
            self.assertEqual(
                default_value(
                    function,
                    "minimum_score",
                ),
                DEFAULT_MINIMUM_SCORE,
            )

            self.assertEqual(
                default_value(
                    function,
                    "ambiguity_margin",
                ),
                DEFAULT_AMBIGUITY_MARGIN,
            )

    def test_segmentation_default_does_not_drift(
        self,
    ) -> None:
        self.assertEqual(
            default_value(
                segment_chord_predictions,
                "min_hold_sec",
            ),
            DEFAULT_MINIMUM_SEGMENT_DURATION_SEC,
        )

    def test_serialized_configuration(
        self,
    ) -> None:
        configuration = (
            selected_chord_configuration()
        )

        self.assertEqual(
            configuration[
                "configuration_id"
            ],
            SELECTED_CONFIG_ID,
        )

        self.assertEqual(
            configuration[
                "selection_split"
            ],
            "development",
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
