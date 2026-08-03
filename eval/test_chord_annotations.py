from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from chord_annotations import (  # noqa: E402
    load_chord_dataset,
    summarize_chord_dataset,
)


class ChordAnnotationTests(
    unittest.TestCase
):
    def create_dataset(
        self,
        root: Path,
        *,
        segments: list[dict] | None = None,
        clip_id: str = "dev_clean_01",
        annotation_clip_id: str | None = None,
        split: str = "development",
        include_in_primary_metrics: bool = True,
    ) -> None:
        audio_dir = (
            root / "audio"
        )

        annotation_dir = (
            root / "annotations"
        )

        audio_dir.mkdir(
            parents=True
        )

        annotation_dir.mkdir(
            parents=True
        )

        audio = np.zeros(
            22050,
            dtype=np.float32,
        )

        sf.write(
            audio_dir / f"{clip_id}.wav",
            audio,
            22050,
            subtype="PCM_16",
        )

        if segments is None:
            segments = [
                {
                    "start": 0.0,
                    "end": 0.1,
                    "label": "N",
                },
                {
                    "start": 0.1,
                    "end": 0.5,
                    "label": "C:maj",
                },
                {
                    "start": 0.5,
                    "end": 1.0,
                    "label": "G:7",
                },
            ]

        annotation = {
            "schema_version": 1,
            "clip_id": (
                annotation_clip_id
                or clip_id
            ),
            "split": split,
            "source": (
                "self_recorded_guitar"
            ),
            "annotation_author": (
                "Test Annotator"
            ),
            "duration_sec": 1.0,
            "segments": segments,
        }

        annotation_path = (
            annotation_dir
            / f"{clip_id}.json"
        )

        annotation_path.write_text(
            json.dumps(
                annotation,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "dataset_name": (
                "test_chord_dataset"
            ),
            "clips": [
                {
                    "clip_id": clip_id,
                    "split": split,
                    "audio_path": (
                        f"audio/{clip_id}.wav"
                    ),
                    "annotation_path": (
                        "annotations/"
                        f"{clip_id}.json"
                    ),
                    "include_in_primary_metrics": (
                        include_in_primary_metrics
                    ),
                }
            ],
        }

        (
            root / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_valid_dataset_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root
            )

            dataset = (
                load_chord_dataset(
                    root
                )
            )

            summary = (
                summarize_chord_dataset(
                    dataset
                )
            )

            self.assertEqual(
                summary["clip_count"],
                1,
            )

            self.assertEqual(
                summary[
                    "primary_clip_count"
                ],
                1,
            )

            self.assertEqual(
                summary["label_counts"],
                {
                    "C:maj": 1,
                    "G:7": 1,
                    "N": 1,
                },
            )

    def test_unlabelled_gap_is_rejected(
        self,
    ) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 0.4,
                "label": "C:maj",
            },
            {
                "start": 0.6,
                "end": 1.0,
                "label": "G:maj",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root,
                segments=segments,
            )

            with self.assertRaisesRegex(
                ValueError,
                "unlabelled gap",
            ):
                load_chord_dataset(
                    root
                )

    def test_overlap_is_rejected(
        self,
    ) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 0.6,
                "label": "C:maj",
            },
            {
                "start": 0.4,
                "end": 1.0,
                "label": "G:maj",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root,
                segments=segments,
            )

            with self.assertRaisesRegex(
                ValueError,
                "overlaps",
            ):
                load_chord_dataset(
                    root
                )

    def test_unsupported_label_is_rejected(
        self,
    ) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "label": "C:unknown",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root,
                segments=segments,
            )

            with self.assertRaisesRegex(
                ValueError,
                "not supported",
            ):
                load_chord_dataset(
                    root
                )

    def test_manifest_annotation_id_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root,
                annotation_clip_id=(
                    "different_clip"
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "does not match manifest",
            ):
                load_chord_dataset(
                    root
                )

    def test_exploratory_clip_cannot_be_primary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            self.create_dataset(
                root,
                split="exploratory",
                include_in_primary_metrics=True,
            )

            with self.assertRaisesRegex(
                ValueError,
                "exploratory clips",
            ):
                load_chord_dataset(
                    root
                )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
