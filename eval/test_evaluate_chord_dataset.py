from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

EVAL_ROOT = (
    PROJECT_ROOT / "eval"
)

for path in (
    PROJECT_ROOT,
    EVAL_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from chord_annotations import (  # noqa: E402
    AnnotatedClip,
    AnnotationSegment,
    ChordDataset,
)
from chord_evaluation import (  # noqa: E402
    evaluate_chord_timelines,
)
from evaluate_chord_dataset import (  # noqa: E402
    DatasetEvaluationError,
    build_method_summary,
    select_dataset_clips,
    validate_api_response,
)


def create_clip(
    clip_id: str,
    *,
    split: str,
    primary: bool,
    duration_sec: float = 1.0,
) -> AnnotatedClip:
    return AnnotatedClip(
        clip_id=clip_id,
        split=split,
        source="synthetic_test",
        annotation_author=(
            "Test Annotator"
        ),
        audio_path=Path(
            f"{clip_id}.wav"
        ),
        annotation_path=Path(
            f"{clip_id}.json"
        ),
        include_in_primary_metrics=(
            primary
        ),
        duration_sec=duration_sec,
        segments=(
            AnnotationSegment(
                start=0.0,
                end=duration_sec,
                label="C:maj",
            ),
        ),
    )


def response_for(
    method: str,
    *,
    duration_sec: float = 1.0,
    total_ms: float = 10.0,
) -> dict:
    feature_source = {
        "chroma": (
            "traditional_chroma"
        ),
        "basic_pitch": (
            "basic_pitch_note_events"
        ),
    }[method]

    return {
        "segments": [
            {
                "start": 0.0,
                "end": duration_sec,
                "label": "C:maj",
            }
        ],
        "progression": [
            "C:maj"
        ],
        "method": method,
        "engine_status": (
            "shared_core_v1"
        ),
        "analysis": {
            "feature_source": (
                feature_source
            ),
        },
        "audio_duration_sec": (
            duration_sec
        ),
        "timing_ms": {
            "io": 1.0,
            "chord_analysis": (
                total_ms - 1.0
            ),
            "total": total_ms,
        },
        "real_time_factor": (
            total_ms
            / (
                duration_sec
                * 1000.0
            )
        ),
    }


class DatasetRunnerTests(
    unittest.TestCase
):
    def test_primary_split_selection(
        self,
    ) -> None:
        dataset = ChordDataset(
            dataset_name="test",
            dataset_root=Path("."),
            clips=(
                create_clip(
                    "dev_primary",
                    split="development",
                    primary=True,
                ),
                create_clip(
                    "dev_secondary",
                    split="development",
                    primary=False,
                ),
                create_clip(
                    "held_primary",
                    split="held_out",
                    primary=True,
                ),
            ),
        )

        selected = (
            select_dataset_clips(
                dataset,
                split="development",
            )
        )

        self.assertEqual(
            [
                clip.clip_id
                for clip in selected
            ],
            [
                "dev_primary"
            ],
        )

    def test_exploratory_selection_includes_non_primary(
        self,
    ) -> None:
        dataset = ChordDataset(
            dataset_name="test",
            dataset_root=Path("."),
            clips=(
                create_clip(
                    "explore_01",
                    split="exploratory",
                    primary=False,
                ),
            ),
        )

        selected = (
            select_dataset_clips(
                dataset,
                split="exploratory",
            )
        )

        self.assertEqual(
            selected[0].clip_id,
            "explore_01",
        )

    def test_api_response_validation(
        self,
    ) -> None:
        response = response_for(
            "chroma"
        )

        validate_api_response(
            response,
            expected_method="chroma",
            expected_duration_sec=1.0,
        )

    def test_api_method_mismatch_is_rejected(
        self,
    ) -> None:
        response = response_for(
            "basic_pitch"
        )

        with self.assertRaisesRegex(
            DatasetEvaluationError,
            "method mismatch",
        ):
            validate_api_response(
                response,
                expected_method="chroma",
                expected_duration_sec=1.0,
            )

    def test_method_summary_uses_pooled_metrics(
        self,
    ) -> None:
        long_metrics = (
            evaluate_chord_timelines(
                [
                    {
                        "start": 0.0,
                        "end": 9.0,
                        "label": "C:maj",
                    }
                ],
                [
                    {
                        "start": 0.0,
                        "end": 9.0,
                        "label": "C:maj",
                    }
                ],
                duration_sec=9.0,
            )
        )

        short_metrics = (
            evaluate_chord_timelines(
                [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "label": "C:maj",
                    }
                ],
                [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "label": "G:maj",
                    }
                ],
                duration_sec=1.0,
            )
        )

        long_response = response_for(
            "chroma",
            duration_sec=9.0,
            total_ms=90.0,
        )

        short_response = response_for(
            "chroma",
            duration_sec=1.0,
            total_ms=10.0,
        )

        summary = build_method_summary(
            "chroma",
            [
                {
                    "clip_id": "long",
                    "metrics": (
                        long_metrics
                    ),
                    "response": (
                        long_response
                    ),
                    "client_elapsed_ms": (
                        95.0
                    ),
                },
                {
                    "clip_id": "short",
                    "metrics": (
                        short_metrics
                    ),
                    "response": (
                        short_response
                    ),
                    "client_elapsed_ms": (
                        12.0
                    ),
                },
            ],
        )

        self.assertEqual(
            summary["metrics"][
                "accuracy"
            ][
                "exact_time_weighted"
            ],
            0.9,
        )

        self.assertEqual(
            summary["latency"][
                "server_total_ms"
            ]["median"],
            50.0,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
