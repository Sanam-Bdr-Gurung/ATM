from __future__ import annotations

from typing import Final


SELECTED_CONFIG_ID: Final[str] = (
    "development_8clip_grid_20260806"
)

DEFAULT_MINIMUM_SCORE: Final[float] = 0.58

DEFAULT_AMBIGUITY_MARGIN: Final[float] = 0.020

DEFAULT_MINIMUM_SEGMENT_DURATION_SEC: Final[
    float
] = 0.60


def selected_chord_configuration(
) -> dict[str, float | str]:
    return {
        "configuration_id": (
            SELECTED_CONFIG_ID
        ),
        "minimum_score": (
            DEFAULT_MINIMUM_SCORE
        ),
        "ambiguity_margin": (
            DEFAULT_AMBIGUITY_MARGIN
        ),
        "minimum_segment_duration_sec": (
            DEFAULT_MINIMUM_SEGMENT_DURATION_SEC
        ),
        "selection_split": "development",
        "selection_method": (
            "predeclared_18_configuration_grid"
        ),
    }
