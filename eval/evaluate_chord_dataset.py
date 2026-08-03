#!/usr/bin/env python3
"""Run both ChordAssist methods against an annotated dataset split.

This script evaluates one split at a time to prevent development-set tuning
from leaking into held-out results.

It stores:

- raw API responses;
- per-clip time-aligned metrics;
- pooled metrics for each method;
- latency summaries;
- a compact method comparison.

This is an evaluation runner, not a threshold-tuning script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from chord_annotations import (  # noqa: E402
    AnnotatedClip,
    ChordDataset,
    SUPPORTED_SPLITS,
    load_chord_dataset,
)
from chord_dataset_metrics import (  # noqa: E402
    aggregate_chord_evaluations,
)
from chord_evaluation import (  # noqa: E402
    evaluate_chord_timelines,
)


SUPPORTED_METHODS = (
    "chroma",
    "basic_pitch",
)


class DatasetEvaluationError(
    RuntimeError
):
    """Raised when dataset evaluation cannot complete."""


def package_version(
    package_name: str,
) -> str | None:
    try:
        return version(
            package_name
        )
    except PackageNotFoundError:
        return None


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

    return (
        completed.stdout.strip()
        or None
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def json_request(
    request: Request,
    *,
    timeout_seconds: float,
) -> tuple[
    dict[str, Any],
    float,
]:
    started = time.perf_counter()

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read()
            status = response.status
    except HTTPError as exc:
        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise DatasetEvaluationError(
            f"HTTP {exc.code} for "
            f"{request.full_url}: {detail}"
        ) from exc
    except URLError as exc:
        raise DatasetEvaluationError(
            f"Could not reach "
            f"{request.full_url}: "
            f"{exc.reason}. Confirm that "
            "Uvicorn is running."
        ) from exc

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    if not 200 <= status < 300:
        raise DatasetEvaluationError(
            f"Unexpected HTTP status "
            f"{status} for "
            f"{request.full_url}."
        )

    try:
        payload = json.loads(
            raw.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise DatasetEvaluationError(
            f"Response from "
            f"{request.full_url} "
            "was not valid JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise DatasetEvaluationError(
            f"Expected a JSON object "
            f"from {request.full_url}."
        )

    return (
        payload,
        elapsed_ms,
    )


def get_json(
    url: str,
    *,
    timeout_seconds: float,
) -> tuple[
    dict[str, Any],
    float,
]:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
        },
    )

    return json_request(
        request,
        timeout_seconds=timeout_seconds,
    )


def encode_multipart_file(
    audio_path: Path,
    *,
    field_name: str = "file",
) -> tuple[bytes, str]:
    boundary = (
        "----ChordAssistBoundary"
        f"{uuid.uuid4().hex}"
    )

    content_type = (
        mimetypes.guess_type(
            audio_path.name
        )[0]
        or "application/octet-stream"
    )

    audio_bytes = (
        audio_path.read_bytes()
    )

    body = b"".join(
        [
            (
                f"--{boundary}\r\n"
            ).encode(),
            (
                "Content-Disposition: "
                f'form-data; name="{field_name}"; '
                f'filename="{audio_path.name}"'
                "\r\n"
            ).encode(),
            (
                f"Content-Type: "
                f"{content_type}\r\n\r\n"
            ).encode(),
            audio_bytes,
            b"\r\n",
            (
                f"--{boundary}--\r\n"
            ).encode(),
        ]
    )

    return (
        body,
        (
            "multipart/form-data; "
            f"boundary={boundary}"
        ),
    )


def analyze_audio(
    base_url: str,
    audio_path: Path,
    *,
    method: str,
    timeout_seconds: float,
) -> tuple[
    dict[str, Any],
    float,
]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: "
            f"{method!r}."
        )

    query = urlencode(
        {
            "method": method,
        }
    )

    body, content_type = (
        encode_multipart_file(
            audio_path
        )
    )

    request = Request(
        (
            f"{base_url}/analyze-file"
            f"?{query}"
        ),
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(
                len(body)
            ),
        },
    )

    return json_request(
        request,
        timeout_seconds=timeout_seconds,
    )


def select_dataset_clips(
    dataset: ChordDataset,
    *,
    split: str,
    include_non_primary: bool = False,
) -> tuple[AnnotatedClip, ...]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported dataset split: "
            f"{split!r}."
        )

    selected: list[
        AnnotatedClip
    ] = []

    for clip in dataset.clips:
        if clip.split != split:
            continue

        # Exploratory clips are intentionally marked non-primary,
        # but they should still be selectable for their own separate run.
        if (
            split != "exploratory"
            and not include_non_primary
            and not clip.include_in_primary_metrics
        ):
            continue

        selected.append(
            clip
        )

    selected.sort(
        key=lambda clip: clip.clip_id
    )

    if not selected:
        primary_description = (
            " including non-primary clips"
            if include_non_primary
            else ""
        )

        raise ValueError(
            f"No clips were selected for "
            f"split {split!r}"
            f"{primary_description}."
        )

    return tuple(
        selected
    )


def validate_api_response(
    response: Mapping[str, Any],
    *,
    expected_method: str,
    expected_duration_sec: float,
    duration_tolerance_sec: float = 0.03,
) -> None:
    if duration_tolerance_sec < 0.0:
        raise ValueError(
            "duration_tolerance_sec "
            "cannot be negative."
        )

    required_keys = {
        "segments",
        "progression",
        "method",
        "engine_status",
        "analysis",
        "audio_duration_sec",
        "timing_ms",
        "real_time_factor",
    }

    missing_keys = (
        required_keys.difference(
            response
        )
    )

    if missing_keys:
        raise DatasetEvaluationError(
            "API response is missing fields: "
            + ", ".join(
                sorted(missing_keys)
            )
        )

    if (
        response["method"]
        != expected_method
    ):
        raise DatasetEvaluationError(
            "API response method mismatch: "
            f"expected {expected_method!r}, "
            f"received "
            f"{response['method']!r}."
        )

    if (
        response["engine_status"]
        != "shared_core_v1"
    ):
        raise DatasetEvaluationError(
            "Unexpected chord-engine status: "
            f"{response['engine_status']!r}."
        )

    try:
        response_duration = float(
            response[
                "audio_duration_sec"
            ]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise DatasetEvaluationError(
            "audio_duration_sec must "
            "be numeric."
        ) from exc

    if (
        abs(
            response_duration
            - expected_duration_sec
        )
        > duration_tolerance_sec
    ):
        raise DatasetEvaluationError(
            "API duration does not match "
            "the annotation duration: "
            f"{response_duration:.4f} versus "
            f"{expected_duration_sec:.4f}."
        )

    segments = response["segments"]

    if not isinstance(
        segments,
        list,
    ):
        raise DatasetEvaluationError(
            "API segments must be a list."
        )

    for segment_index, segment in enumerate(
        segments
    ):
        if not isinstance(
            segment,
            Mapping,
        ):
            raise DatasetEvaluationError(
                "API segment "
                f"{segment_index} must be "
                "an object."
            )

        segment_keys = {
            "start",
            "end",
            "label",
        }

        if not segment_keys.issubset(
            segment
        ):
            raise DatasetEvaluationError(
                "API segment "
                f"{segment_index} is missing "
                "start, end, or label."
            )

    analysis = response["analysis"]

    if not isinstance(
        analysis,
        Mapping,
    ):
        raise DatasetEvaluationError(
            "API analysis must be an object."
        )

    expected_feature_source = {
        "chroma": (
            "traditional_chroma"
        ),
        "basic_pitch": (
            "basic_pitch_note_events"
        ),
    }[expected_method]

    if (
        analysis.get(
            "feature_source"
        )
        != expected_feature_source
    ):
        raise DatasetEvaluationError(
            "Unexpected feature source for "
            f"{expected_method}: "
            f"{analysis.get('feature_source')!r}."
        )

    timing = response["timing_ms"]

    if not isinstance(
        timing,
        Mapping,
    ):
        raise DatasetEvaluationError(
            "timing_ms must be an object."
        )

    for timing_key in (
        "io",
        "chord_analysis",
        "total",
    ):
        try:
            timing_value = float(
                timing[timing_key]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise DatasetEvaluationError(
                f"Invalid timing value: "
                f"{timing_key}."
            ) from exc

        if timing_value < 0.0:
            raise DatasetEvaluationError(
                f"{timing_key} cannot "
                "be negative."
            )

    real_time_factor = response[
        "real_time_factor"
    ]

    if (
        real_time_factor is not None
        and float(
            real_time_factor
        )
        < 0.0
    ):
        raise DatasetEvaluationError(
            "real_time_factor cannot "
            "be negative."
        )


def _summary_statistics(
    values: Sequence[float],
) -> dict[str, float] | None:
    if not values:
        return None

    converted = [
        float(value)
        for value in values
    ]

    return {
        "mean": round(
            mean(converted),
            4,
        ),
        "median": round(
            median(converted),
            4,
        ),
        "minimum": round(
            min(converted),
            4,
        ),
        "maximum": round(
            max(converted),
            4,
        ),
    }


def summarize_latency(
    clip_results: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    server_total_ms: list[
        float
    ] = []

    server_chord_ms: list[
        float
    ] = []

    client_elapsed_ms: list[
        float
    ] = []

    real_time_factors: list[
        float
    ] = []

    for result in clip_results:
        response = result[
            "response"
        ]

        timing = response[
            "timing_ms"
        ]

        server_total_ms.append(
            float(
                timing["total"]
            )
        )

        server_chord_ms.append(
            float(
                timing[
                    "chord_analysis"
                ]
            )
        )

        client_elapsed_ms.append(
            float(
                result[
                    "client_elapsed_ms"
                ]
            )
        )

        real_time_factor = response.get(
            "real_time_factor"
        )

        if real_time_factor is not None:
            real_time_factors.append(
                float(
                    real_time_factor
                )
            )

    return {
        "server_total_ms": (
            _summary_statistics(
                server_total_ms
            )
        ),
        "server_chord_analysis_ms": (
            _summary_statistics(
                server_chord_ms
            )
        ),
        "client_elapsed_ms": (
            _summary_statistics(
                client_elapsed_ms
            )
        ),
        "real_time_factor": (
            _summary_statistics(
                real_time_factors
            )
        ),
    }


def build_method_summary(
    method: str,
    clip_results: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {method!r}."
        )

    if not clip_results:
        raise ValueError(
            f"No clip results were supplied "
            f"for method {method!r}."
        )

    evaluations = [
        result["metrics"]
        for result in clip_results
    ]

    aggregate_metrics = (
        aggregate_chord_evaluations(
            evaluations
        )
    )

    summary: dict[str, Any] = {
        "method": method,
        "clip_count": len(
            clip_results
        ),
        "clip_ids": [
            result["clip_id"]
            for result in clip_results
        ],
        "metrics": aggregate_metrics,
        "latency": summarize_latency(
            clip_results
        ),
    }

    if method == "basic_pitch":
        summary["latency_note"] = (
            "The first request after server "
            "startup may include lazy CoreML "
            "model-loading overhead. Use a "
            "dedicated repeated benchmark for "
            "final latency claims."
        )

    return summary


def build_comparison(
    method_summaries: Mapping[
        str,
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    comparison: dict[
        str,
        Any,
    ] = {}

    for method, summary in (
        method_summaries.items()
    ):
        metrics = summary["metrics"]

        comparison[method] = {
            "clip_count": (
                summary["clip_count"]
            ),
            "exact_time_weighted": (
                metrics["accuracy"][
                    "exact_time_weighted"
                ]
            ),
            "root_time_weighted": (
                metrics["accuracy"][
                    "root_time_weighted"
                ]
            ),
            "triad_family_time_weighted": (
                metrics["accuracy"][
                    "triad_family_time_weighted"
                ]
            ),
            "harmonic_exact_time_weighted": (
                metrics["accuracy"][
                    "harmonic_exact_time_weighted"
                ]
            ),
            "no_chord_f1": (
                metrics["no_chord"]["f1"]
            ),
            "prediction_x_rate_evaluable": (
                metrics["ambiguity"][
                    "prediction_x_rate_evaluable"
                ]
            ),
            "boundary_f1": (
                metrics["boundaries"]["f1"]
            ),
            "boundary_mean_absolute_error_sec": (
                metrics["boundaries"][
                    "mean_absolute_error_sec"
                ]
            ),
            "median_server_total_ms": (
                summary["latency"][
                    "server_total_ms"
                ]["median"]
                if summary["latency"][
                    "server_total_ms"
                ]
                is not None
                else None
            ),
            "median_real_time_factor": (
                summary["latency"][
                    "real_time_factor"
                ]["median"]
                if summary["latency"][
                    "real_time_factor"
                ]
                is not None
                else None
            ),
        }

    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one annotated chord "
            "dataset split through the ChordAssist "
            "API."
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
        "--split",
        required=True,
        choices=sorted(
            SUPPORTED_SPLITS
        ),
        help=(
            "Evaluate exactly one dataset split."
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(
            SUPPORTED_METHODS
        ),
    )

    parser.add_argument(
        "--base-url",
        default=(
            "http://127.0.0.1:8000"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "evaluation_results/chords"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
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
        "--include-non-primary",
        action="store_true",
        help=(
            "Include development or held-out "
            "clips marked as non-primary. "
            "Exploratory clips are always "
            "evaluated only in their own split."
        ),
    )

    args = parser.parse_args()

    if args.timeout <= 0.0:
        raise SystemExit(
            "--timeout must be positive."
        )

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

    methods = tuple(
        dict.fromkeys(
            args.methods
        )
    )

    dataset = load_chord_dataset(
        args.dataset_root,
        verify_audio=True,
        timeline_tolerance_sec=(
            args.timeline_tolerance
        ),
    )

    selected_clips = (
        select_dataset_clips(
            dataset,
            split=args.split,
            include_non_primary=(
                args.include_non_primary
            ),
        )
    )

    base_url = (
        args.base_url.rstrip("/")
    )

    health, health_client_ms = (
        get_json(
            f"{base_url}/health",
            timeout_seconds=args.timeout,
        )
    )

    available_methods = set(
        health.get(
            "available_methods",
            [],
        )
    )

    unavailable_methods = (
        set(methods)
        - available_methods
    )

    if unavailable_methods:
        raise DatasetEvaluationError(
            "API does not expose requested "
            "methods: "
            + ", ".join(
                sorted(
                    unavailable_methods
                )
            )
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = (
        args.output_root
        / (
            f"{timestamp}_"
            f"{args.split}"
        )
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
            "Time-aligned chord-recognition "
            "evaluation; one dataset split "
            "per run"
        ),
        "dataset_name": (
            dataset.dataset_name
        ),
        "dataset_root": str(
            dataset.dataset_root
        ),
        "split": args.split,
        "methods": list(methods),
        "include_non_primary": (
            args.include_non_primary
        ),
        "timeline_tolerance_sec": (
            args.timeline_tolerance
        ),
        "boundary_tolerance_sec": (
            args.boundary_tolerance
        ),
        "base_url": base_url,
        "health_client_ms": round(
            health_client_ms,
            2,
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
        "basic_pitch_version": (
            package_version(
                "basic-pitch"
            )
        ),
        "coremltools_version": (
            package_version(
                "coremltools"
            )
        ),
        "librosa_version": (
            package_version(
                "librosa"
            )
        ),
        "selected_clip_count": len(
            selected_clips
        ),
        "selected_clip_ids": [
            clip.clip_id
            for clip in selected_clips
        ],
    }

    write_json(
        output_dir / "metadata.json",
        metadata,
    )

    write_json(
        output_dir / "health.json",
        health,
    )

    method_results: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method: []
        for method in methods
    }

    for clip_index, clip in enumerate(
        selected_clips,
        start=1,
    ):
        print(
            f"[{clip_index}/"
            f"{len(selected_clips)}] "
            f"Evaluating {clip.clip_id}"
        )

        clip_directory = (
            output_dir
            / "per_clip"
            / clip.clip_id
        )

        clip_metadata = {
            "clip_id": clip.clip_id,
            "split": clip.split,
            "source": clip.source,
            "annotation_author": (
                clip.annotation_author
            ),
            "include_in_primary_metrics": (
                clip.include_in_primary_metrics
            ),
            "duration_sec": (
                clip.duration_sec
            ),
            "audio_path": str(
                clip.audio_path
            ),
            "annotation_path": str(
                clip.annotation_path
            ),
            "audio_sha256": sha256_file(
                clip.audio_path
            ),
            "reference_segment_count": len(
                clip.segments
            ),
        }

        write_json(
            clip_directory
            / "clip_metadata.json",
            clip_metadata,
        )

        for method in methods:
            print(
                f"  - {method}"
            )

            (
                response,
                client_elapsed_ms,
            ) = analyze_audio(
                base_url,
                clip.audio_path,
                method=method,
                timeout_seconds=args.timeout,
            )

            validate_api_response(
                response,
                expected_method=method,
                expected_duration_sec=(
                    clip.duration_sec
                ),
                duration_tolerance_sec=(
                    args.timeline_tolerance
                ),
            )

            metrics = (
                evaluate_chord_timelines(
                    clip.segments,
                    response["segments"],
                    duration_sec=(
                        clip.duration_sec
                    ),
                    timeline_tolerance_sec=(
                        args.timeline_tolerance
                    ),
                    boundary_tolerance_sec=(
                        args.boundary_tolerance
                    ),
                )
            )

            result_record = {
                "clip_id": clip.clip_id,
                "method": method,
                "client_elapsed_ms": round(
                    client_elapsed_ms,
                    2,
                ),
                "response": response,
                "metrics": metrics,
            }

            method_results[
                method
            ].append(
                result_record
            )

            write_json(
                clip_directory
                / f"{method}_response.json",
                response,
            )

            write_json(
                clip_directory
                / f"{method}_metrics.json",
                metrics,
            )

    method_summaries = {
        method: build_method_summary(
            method,
            method_results[method],
        )
        for method in methods
    }

    for method, summary in (
        method_summaries.items()
    ):
        write_json(
            output_dir
            / f"{method}_summary.json",
            summary,
        )

    comparison = {
        "schema_version": 1,
        "dataset_name": (
            dataset.dataset_name
        ),
        "split": args.split,
        "clip_count": len(
            selected_clips
        ),
        "methods": (
            build_comparison(
                method_summaries
            )
        ),
        "interpretation_note": (
            "Development results may guide "
            "threshold selection. Held-out "
            "results must not be used for "
            "further tuning. Exploratory "
            "results are reported separately "
            "and excluded from controlled "
            "primary averages."
        ),
    }

    write_json(
        output_dir / "comparison.json",
        comparison,
    )

    print()
    print(
        "Chord dataset evaluation passed."
    )
    print(
        f"Dataset: {dataset.dataset_name}"
    )
    print(
        f"Split: {args.split}"
    )
    print(
        f"Clips: {len(selected_clips)}"
    )
    print(
        f"Results: {output_dir.resolve()}"
    )
    print()
    print(
        json.dumps(
            comparison["methods"],
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except (
        ValueError,
        DatasetEvaluationError,
    ) as exc:
        raise SystemExit(
            "Chord dataset evaluation failed: "
            f"{exc}"
        ) from exc
