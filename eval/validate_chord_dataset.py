#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the ChordAssist "
            "ground-truth chord dataset."
        )
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "Directory containing "
            "manifest.json."
        ),
    )

    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help=(
            "Validate JSON structure and "
            "timelines without requiring "
            "audio files."
        ),
    )

    parser.add_argument(
        "--timeline-tolerance",
        type=float,
        default=0.03,
        help=(
            "Allowed boundary difference "
            "in seconds."
        ),
    )

    args = parser.parse_args()

    dataset = load_chord_dataset(
        args.dataset_root,
        verify_audio=(
            not args.skip_audio_check
        ),
        timeline_tolerance_sec=(
            args.timeline_tolerance
        ),
    )

    summary = summarize_chord_dataset(
        dataset
    )

    print(
        "Chord dataset validation passed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    if (
        summary[
            "primary_x_segment_count"
        ]
        > 0
    ):
        print()
        print(
            "Warning: primary clips contain "
            "X reference segments. These "
            "should normally be reviewed or "
            "excluded from primary metrics."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except ValueError as exc:
        raise SystemExit(
            "Chord dataset validation failed: "
            f"{exc}"
        ) from exc
