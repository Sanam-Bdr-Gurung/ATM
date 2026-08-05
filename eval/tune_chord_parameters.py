#!/usr/bin/env python3
"""Development-only parameter sweep for shared chord recognition.

This script prepares the expensive feature sources once per clip, then
evaluates a small predeclared classifier and segmentation grid.

It must only be run on the development split.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


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


from audio_input import load_audio_bytes  # noqa: E402
from chord_annotations import (  # noqa: E402
    AnnotatedClip,
    load_chord_dataset,
)
from chord_dataset_metrics import (  # noqa: E402
    aggregate_chord_evaluations,
)
from chord_evaluation import (  # noqa: E402
    evaluate_chord_timelines,
)
from chord_match import (  # noqa: E402
    classify_chroma_frames,
)
from evaluate_chord_dataset import (  # noqa: E402
    select_dataset_clips,
)
from features import (  # noqa: E402
    ChromaFeatures,
    chroma_from_audio,
)
from models.basic_pitch_inference import (  # noqa: E402
    BasicPitchTranscriber,
)
from note_event_features import (  # noqa: E402
    NoteEventFeatures,
    classify_note_event_frames,
    note_events_to_pitch_class_frames,
)
from segmentation import (  # noqa: E402
    segment_chord_predictions,
)


MINIMUM_SCORE_GRID = (
    0.58,
    0.62,
    0.66,
)

AMBIGUITY_MARGIN_GRID = (
    0.020,
    0.035,
    0.050,
)

MINIMUM_HOLD_GRID = (
    0.40,
    0.60,
)

BASELINE_MINIMUM_SCORE = 0.62
BASELINE_AMBIGUITY_MARGIN = 0.035
BASELINE_MINIMUM_HOLD_SEC = 0.40

CHROMA_RELATIVE_ACTIVITY_FLOOR = 0.08
NOTE_RELATIVE_ACTIVITY_FLOOR = 0.05

CHROMA_HOP_LENGTH = 1024
CHROMA_FRAME_LENGTH = 4096
CHROMA_HARMONIC_MARGIN = 3.0

NOTE_WINDOW_SEC = 0.75
NOTE_MINIMUM_CONFIDENCE = 0.05
NOTE_BASS_RELATIVE_WEIGHT = 0.15


@dataclass(frozen=True)
class TuningConfig:
    minimum_score: float
    ambiguity_margin: float
    minimum_hold_sec: float

    @property
    def config_id(self) -> str:
        return (
            f"score_{self.minimum_score:.3f}"
            f"_margin_{self.ambiguity_margin:.3f}"
            f"_hold_{self.minimum_hold_sec:.2f}"
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "config_id": self.config_id,
            "minimum_score": self.minimum_score,
            "ambiguity_margin": self.ambiguity_margin,
            "minimum_hold_sec": self.minimum_hold_sec,
        }


BASELINE_CONFIG = TuningConfig(
    minimum_score=BASELINE_MINIMUM_SCORE,
    ambiguity_margin=BASELINE_AMBIGUITY_MARGIN,
    minimum_hold_sec=BASELINE_MINIMUM_HOLD_SEC,
)


@dataclass(frozen=True)
class PreparedClip:
    clip: AnnotatedClip
    audio_duration_sec: float
    chroma_features: ChromaFeatures
    note_features: NoteEventFeatures
    note_event_count: int


def git_value(
    *args: str,
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return completed.stdout.strip() or None


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_tuning_grid() -> tuple[TuningConfig, ...]:
    return tuple(
        TuningConfig(
            minimum_score=minimum_score,
            ambiguity_margin=ambiguity_margin,
            minimum_hold_sec=minimum_hold_sec,
        )
        for minimum_score in MINIMUM_SCORE_GRID
        for ambiguity_margin in AMBIGUITY_MARGIN_GRID
        for minimum_hold_sec in MINIMUM_HOLD_GRID
    )


def _metric_or_zero(
    value: Any,
) -> float:
    if value is None:
        return 0.0

    return float(value)


def metric_snapshot(
    metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    accuracy = metrics["accuracy"]
    no_chord = metrics["no_chord"]
    ambiguity = metrics["ambiguity"]
    boundaries = metrics["boundaries"]

    return {
        "exact_time_weighted": (
            accuracy["exact_time_weighted"]
        ),
        "root_time_weighted": (
            accuracy["root_time_weighted"]
        ),
        "triad_family_time_weighted": (
            accuracy[
                "triad_family_time_weighted"
            ]
        ),
        "harmonic_exact_time_weighted": (
            accuracy[
                "harmonic_exact_time_weighted"
            ]
        ),
        "no_chord_f1": (
            no_chord["f1"]
        ),
        "prediction_x_rate_evaluable": (
            ambiguity[
                "prediction_x_rate_evaluable"
            ]
        ),
        "boundary_f1": (
            boundaries["f1"]
        ),
        "boundary_mean_absolute_error_sec": (
            boundaries[
                "mean_absolute_error_sec"
            ]
        ),
        "false_change_count": (
            boundaries[
                "false_change_count"
            ]
        ),
        "missed_change_count": (
            boundaries[
                "missed_change_count"
            ]
        ),
    }


def score_method_metrics(
    metrics: Mapping[str, Any],
) -> float:
    """Balanced development utility used only to rank the grid.

    Weighting prioritizes chord-labelled time rather than overall exact
    accuracy, because overall exact accuracy can be inflated by silence.
    """

    snapshot = metric_snapshot(
        metrics
    )

    harmonic_exact = _metric_or_zero(
        snapshot[
            "harmonic_exact_time_weighted"
        ]
    )

    root = _metric_or_zero(
        snapshot[
            "root_time_weighted"
        ]
    )

    triad = _metric_or_zero(
        snapshot[
            "triad_family_time_weighted"
        ]
    )

    no_chord_f1 = _metric_or_zero(
        snapshot["no_chord_f1"]
    )

    boundary_f1 = _metric_or_zero(
        snapshot["boundary_f1"]
    )

    x_rate = _metric_or_zero(
        snapshot[
            "prediction_x_rate_evaluable"
        ]
    )

    utility = (
        0.35 * harmonic_exact
        + 0.20 * root
        + 0.15 * triad
        + 0.10 * no_chord_f1
        + 0.10 * boundary_f1
        + 0.10 * (1.0 - x_rate)
    )

    return round(
        utility,
        8,
    )


def score_configuration(
    metrics_by_method: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> float:
    if not metrics_by_method:
        raise ValueError(
            "At least one method is required."
        )

    return round(
        mean(
            score_method_metrics(
                metrics
            )
            for metrics in (
                metrics_by_method.values()
            )
        ),
        8,
    )


def prepare_development_clips(
    dataset_root: Path,
    *,
    timeline_tolerance_sec: float,
) -> tuple[
    tuple[PreparedClip, ...],
    str,
]:
    dataset = load_chord_dataset(
        dataset_root,
        verify_audio=True,
        timeline_tolerance_sec=(
            timeline_tolerance_sec
        ),
    )

    clips = select_dataset_clips(
        dataset,
        split="development",
    )

    print(
        "Loading Basic Pitch model..."
    )

    transcriber = (
        BasicPitchTranscriber()
    )

    prepared: list[
        PreparedClip
    ] = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):
        print(
            f"[prepare {index}/{len(clips)}] "
            f"{clip.clip_id}"
        )

        raw_audio = (
            clip.audio_path.read_bytes()
        )

        y, sr = load_audio_bytes(
            raw_audio,
            22050,
        )

        audio_duration_sec = (
            len(y) / float(sr)
        )

        if (
            abs(
                audio_duration_sec
                - clip.duration_sec
            )
            > timeline_tolerance_sec
        ):
            raise ValueError(
                f"{clip.clip_id}: decoded duration "
                f"{audio_duration_sec:.6f} does not "
                "match annotation duration "
                f"{clip.duration_sec:.6f}."
            )

        chroma_features = (
            chroma_from_audio(
                y,
                sr,
                hop_length=(
                    CHROMA_HOP_LENGTH
                ),
                frame_length=(
                    CHROMA_FRAME_LENGTH
                ),
                harmonic_margin=(
                    CHROMA_HARMONIC_MARGIN
                ),
            )
        )

        note_events = (
            transcriber.transcribe(
                y,
                sr,
            )
        )

        note_features = (
            note_events_to_pitch_class_frames(
                note_events,
                audio_duration_sec=(
                    audio_duration_sec
                ),
                frame_step_sec=(
                    CHROMA_HOP_LENGTH
                    / float(sr)
                ),
                window_sec=(
                    NOTE_WINDOW_SEC
                ),
                minimum_confidence=(
                    NOTE_MINIMUM_CONFIDENCE
                ),
                bass_relative_weight=(
                    NOTE_BASS_RELATIVE_WEIGHT
                ),
            )
        )

        prepared.append(
            PreparedClip(
                clip=clip,
                audio_duration_sec=(
                    audio_duration_sec
                ),
                chroma_features=(
                    chroma_features
                ),
                note_features=(
                    note_features
                ),
                note_event_count=len(
                    note_events
                ),
            )
        )

    return (
        tuple(prepared),
        transcriber.runtime_name,
    )


def evaluate_prepared_clip(
    prepared: PreparedClip,
    *,
    method: str,
    config: TuningConfig,
    timeline_tolerance_sec: float,
    boundary_tolerance_sec: float,
) -> dict[str, Any]:
    if method == "chroma":
        predictions, activity_threshold = (
            classify_chroma_frames(
                prepared.chroma_features.chroma,
                frame_activity=(
                    prepared
                    .chroma_features
                    .frame_activity
                ),
                relative_activity_floor=(
                    CHROMA_RELATIVE_ACTIVITY_FLOOR
                ),
                minimum_score=(
                    config.minimum_score
                ),
                ambiguity_margin=(
                    config.ambiguity_margin
                ),
            )
        )

        frame_times = (
            prepared.chroma_features.times
        )

    elif method == "basic_pitch":
        predictions, activity_threshold = (
            classify_note_event_frames(
                prepared.note_features,
                relative_activity_floor=(
                    NOTE_RELATIVE_ACTIVITY_FLOOR
                ),
                minimum_score=(
                    config.minimum_score
                ),
                ambiguity_margin=(
                    config.ambiguity_margin
                ),
            )
        )

        frame_times = (
            prepared.note_features.times
        )

    else:
        raise ValueError(
            f"Unsupported method: {method!r}."
        )

    segments = segment_chord_predictions(
        predictions,
        frame_times,
        audio_duration_sec=(
            prepared.audio_duration_sec
        ),
        min_hold_sec=(
            config.minimum_hold_sec
        ),
    )

    metrics = evaluate_chord_timelines(
        prepared.clip.segments,
        segments,
        duration_sec=(
            prepared.clip.duration_sec
        ),
        timeline_tolerance_sec=(
            timeline_tolerance_sec
        ),
        boundary_tolerance_sec=(
            boundary_tolerance_sec
        ),
    )

    return {
        "clip_id": (
            prepared.clip.clip_id
        ),
        "method": method,
        "activity_threshold": round(
            float(activity_threshold),
            8,
        ),
        "prediction_segment_count": len(
            segments
        ),
        "note_event_count": (
            prepared.note_event_count
            if method == "basic_pitch"
            else None
        ),
        "progression": [
            segment["label"]
            for segment in segments
            if segment["label"] != "N"
        ],
        "metrics": metrics,
    }


def evaluate_configuration(
    prepared_clips: Sequence[
        PreparedClip
    ],
    *,
    config: TuningConfig,
    timeline_tolerance_sec: float,
    boundary_tolerance_sec: float,
) -> dict[str, Any]:
    methods = (
        "chroma",
        "basic_pitch",
    )

    per_method_records: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method: []
        for method in methods
    }

    for prepared in prepared_clips:
        for method in methods:
            record = evaluate_prepared_clip(
                prepared,
                method=method,
                config=config,
                timeline_tolerance_sec=(
                    timeline_tolerance_sec
                ),
                boundary_tolerance_sec=(
                    boundary_tolerance_sec
                ),
            )

            per_method_records[
                method
            ].append(
                record
            )

    aggregate_by_method: dict[
        str,
        dict[str, Any],
    ] = {}

    method_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for method in methods:
        aggregate_metrics = (
            aggregate_chord_evaluations(
                [
                    record["metrics"]
                    for record in (
                        per_method_records[
                            method
                        ]
                    )
                ]
            )
        )

        aggregate_by_method[
            method
        ] = aggregate_metrics

        method_results[
            method
        ] = {
            "utility": (
                score_method_metrics(
                    aggregate_metrics
                )
            ),
            "summary": (
                metric_snapshot(
                    aggregate_metrics
                )
            ),
            "aggregate_metrics": (
                aggregate_metrics
            ),
            "per_clip": (
                per_method_records[
                    method
                ]
            ),
        }

    return {
        "config": config.to_dict(),
        "balanced_score": (
            score_configuration(
                aggregate_by_method
            )
        ),
        "methods": method_results,
    }


def print_ranked_results(
    ranked_results: Sequence[
        Mapping[str, Any]
    ],
    *,
    top_count: int,
) -> None:
    print()
    print(
        "Top development configurations"
    )
    print()

    header = (
        f"{'Rank':>4} "
        f"{'Score':>7} "
        f"{'Min':>5} "
        f"{'Margin':>6} "
        f"{'Hold':>5} | "
        f"{'Ch HExact':>9} "
        f"{'Ch Root':>7} "
        f"{'Ch X':>6} "
        f"{'Ch Bound':>8} | "
        f"{'BP HExact':>9} "
        f"{'BP Root':>7} "
        f"{'BP X':>6} "
        f"{'BP Bound':>8}"
    )

    print(header)
    print("-" * len(header))

    def percent(
        value: Any,
    ) -> str:
        if value is None:
            return "n/a"

        return (
            f"{100.0 * float(value):.1f}%"
        )

    for rank, result in enumerate(
        ranked_results[:top_count],
        start=1,
    ):
        config = result["config"]

        chroma = (
            result["methods"]
            ["chroma"]
            ["summary"]
        )

        basic_pitch = (
            result["methods"]
            ["basic_pitch"]
            ["summary"]
        )

        print(
            f"{rank:>4} "
            f"{float(result['balanced_score']):>7.4f} "
            f"{float(config['minimum_score']):>5.3f} "
            f"{float(config['ambiguity_margin']):>6.3f} "
            f"{float(config['minimum_hold_sec']):>5.2f} | "
            f"{percent(chroma['harmonic_exact_time_weighted']):>9} "
            f"{percent(chroma['root_time_weighted']):>7} "
            f"{percent(chroma['prediction_x_rate_evaluable']):>6} "
            f"{percent(chroma['boundary_f1']):>8} | "
            f"{percent(basic_pitch['harmonic_exact_time_weighted']):>9} "
            f"{percent(basic_pitch['root_time_weighted']):>7} "
            f"{percent(basic_pitch['prediction_x_rate_evaluable']):>6} "
            f"{percent(basic_pitch['boundary_f1']):>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared shared chord "
            "parameter grid on the development "
            "split only."
        )
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "evaluation_results/tuning"
        ),
    )

    parser.add_argument(
        "--timeline-tolerance",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if args.timeline_tolerance < 0.0:
        raise SystemExit(
            "--timeline-tolerance cannot "
            "be negative."
        )

    if args.boundary_tolerance < 0.0:
        raise SystemExit(
            "--boundary-tolerance cannot "
            "be negative."
        )

    if args.top <= 0:
        raise SystemExit(
            "--top must be positive."
        )

    grid = build_tuning_grid()

    prepared_clips, runtime_name = (
        prepare_development_clips(
            args.dataset_root,
            timeline_tolerance_sec=(
                args.timeline_tolerance
            ),
        )
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = (
        args.output_root
        / f"{timestamp}_development"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Development-only shared chord "
            "classifier and segmentation "
            "parameter sweep"
        ),
        "split": "development",
        "clip_ids": [
            prepared.clip.clip_id
            for prepared in prepared_clips
        ],
        "configuration_count": len(
            grid
        ),
        "minimum_score_grid": list(
            MINIMUM_SCORE_GRID
        ),
        "ambiguity_margin_grid": list(
            AMBIGUITY_MARGIN_GRID
        ),
        "minimum_hold_grid": list(
            MINIMUM_HOLD_GRID
        ),
        "baseline_config": (
            BASELINE_CONFIG.to_dict()
        ),
        "selection_score": {
            "harmonic_exact": 0.35,
            "root": 0.20,
            "triad_family": 0.15,
            "no_chord_f1": 0.10,
            "boundary_f1": 0.10,
            "one_minus_x_rate": 0.10,
        },
        "fixed_parameters": {
            "chroma_relative_activity_floor": (
                CHROMA_RELATIVE_ACTIVITY_FLOOR
            ),
            "note_relative_activity_floor": (
                NOTE_RELATIVE_ACTIVITY_FLOOR
            ),
            "chroma_hop_length": (
                CHROMA_HOP_LENGTH
            ),
            "chroma_frame_length": (
                CHROMA_FRAME_LENGTH
            ),
            "chroma_harmonic_margin": (
                CHROMA_HARMONIC_MARGIN
            ),
            "note_window_sec": (
                NOTE_WINDOW_SEC
            ),
            "note_minimum_confidence": (
                NOTE_MINIMUM_CONFIDENCE
            ),
            "note_bass_relative_weight": (
                NOTE_BASS_RELATIVE_WEIGHT
            ),
        },
        "basic_pitch_runtime": (
            runtime_name
        ),
        "git_branch": git_value(
            "branch",
            "--show-current",
        ),
        "git_commit": git_value(
            "rev-parse",
            "HEAD",
        ),
        "git_dirty": bool(
            git_value(
                "status",
                "--porcelain",
            )
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }

    results: list[
        dict[str, Any]
    ] = []

    for index, config in enumerate(
        grid,
        start=1,
    ):
        print(
            f"[{index}/{len(grid)}] "
            f"{config.config_id}"
        )

        result = evaluate_configuration(
            prepared_clips,
            config=config,
            timeline_tolerance_sec=(
                args.timeline_tolerance
            ),
            boundary_tolerance_sec=(
                args.boundary_tolerance
            ),
        )

        results.append(
            result
        )

    baseline_result = next(
        (
            result
            for result in results
            if result["config"]["config_id"]
            == BASELINE_CONFIG.config_id
        ),
        None,
    )

    if baseline_result is None:
        raise RuntimeError(
            "Baseline configuration was "
            "not present in the grid."
        )

    baseline_score = float(
        baseline_result[
            "balanced_score"
        ]
    )

    for result in results:
        result[
            "balanced_score_delta_vs_baseline"
        ] = round(
            float(
                result["balanced_score"]
            )
            - baseline_score,
            8,
        )

    ranked_results = sorted(
        results,
        key=lambda result: (
            float(
                result[
                    "balanced_score"
                ]
            ),
            float(
                result["methods"][
                    "basic_pitch"
                ]["summary"][
                    "harmonic_exact_time_weighted"
                ]
                or 0.0
            ),
            float(
                result["methods"][
                    "chroma"
                ]["summary"][
                    "harmonic_exact_time_weighted"
                ]
                or 0.0
            ),
        ),
        reverse=True,
    )

    output = {
        "metadata": metadata,
        "baseline_result": (
            baseline_result
        ),
        "ranked_results": (
            ranked_results
        ),
    }

    write_json(
        output_dir / "tuning_results.json",
        output,
    )

    write_json(
        output_dir / "top_configs.json",
        {
            "metadata": metadata,
            "top_configs": (
                ranked_results[
                    : args.top
                ]
            ),
        },
    )

    print_ranked_results(
        ranked_results,
        top_count=args.top,
    )

    print()
    print(
        f"Results: {output_dir.resolve()}"
    )

    print(
        "No production parameters were "
        "changed by this script."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except (
        ValueError,
        RuntimeError,
    ) as exc:
        raise SystemExit(
            "Chord tuning failed: "
            f"{exc}"
        ) from exc
