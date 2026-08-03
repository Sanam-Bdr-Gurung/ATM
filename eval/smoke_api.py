#!/usr/bin/env python3
"""Regression smoke test for the dual-method ChordAssist API.

The test runs both feature sources through the same chord-classification
contract:

- traditional chroma
- Basic Pitch note events

This is a regression and reproducibility test, not an accuracy benchmark.
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
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SUPPORTED_METHODS = {
    "chroma",
    "basic_pitch",
}


class SmokeTestError(RuntimeError):
    """Raised when the API smoke test cannot complete."""


def package_version(
    package_name: str,
) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def git_value(
    *args: str,
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
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
            digest.update(chunk)

    return digest.hexdigest()


def json_request(
    request: Request,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
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

        raise SmokeTestError(
            f"HTTP {exc.code} for "
            f"{request.full_url}: {detail}"
        ) from exc
    except URLError as exc:
        raise SmokeTestError(
            f"Could not reach "
            f"{request.full_url}: "
            f"{exc.reason}. "
            "Confirm that Uvicorn is running."
        ) from exc

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    if not 200 <= status < 300:
        raise SmokeTestError(
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
        raise SmokeTestError(
            f"Response from "
            f"{request.full_url} "
            "was not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise SmokeTestError(
            f"Expected a JSON object from "
            f"{request.full_url}."
        )

    return payload, elapsed_ms


def get_json(
    url: str,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
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
            f"--{boundary}\r\n".encode(),
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
            f"--{boundary}--\r\n".encode(),
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
) -> tuple[dict[str, Any], float]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {method}"
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


def assert_health(
    response: dict[str, Any],
) -> None:
    assert response.get("ok") is True, (
        "Health response did not contain "
        "ok=true."
    )

    assert (
        response.get("service")
        == "chordassist"
    ), "Unexpected service name."

    assert (
        response.get("scope")
        == "prevailing_chord_recognition"
    ), "Unexpected API scope."

    available_methods = set(
        response.get(
            "available_methods",
            [],
        )
    )

    assert available_methods == (
        SUPPORTED_METHODS
    ), (
        "Unexpected available methods: "
        f"{available_methods}"
    )

    assert (
        response.get("default_method")
        == "chroma"
    ), "The default method must be chroma."

    assert (
        response.get(
            "chord_engine_status"
        )
        == "shared_core_chord_engine_v1"
    ), "Unexpected chord-engine status."

    assert isinstance(
        response.get(
            "basic_pitch_loaded"
        ),
        bool,
    ), (
        "basic_pitch_loaded must be "
        "a Boolean."
    )


def assert_segment(
    segment: dict[str, Any],
    *,
    duration: float,
    previous_start: float,
) -> float:
    required_keys = {
        "start",
        "end",
        "label",
        "display",
        "confidence",
        "mean_score",
        "mean_margin",
        "frame_count",
    }

    assert required_keys.issubset(
        segment
    ), (
        "Chord segment is missing fields: "
        f"{required_keys.difference(segment)}"
    )

    start = float(
        segment["start"]
    )

    end = float(
        segment["end"]
    )

    assert 0.0 <= start < end, (
        "Invalid segment interval: "
        f"{start} to {end}"
    )

    assert end <= duration + 0.01, (
        "Segment ends after audio duration: "
        f"{end} > {duration}"
    )

    assert start >= previous_start, (
        "Segments are not sorted by start time."
    )

    label = segment["label"]
    display = segment["display"]

    assert isinstance(
        label,
        str,
    ) and label, "Invalid segment label."

    assert isinstance(
        display,
        str,
    ) and display, (
        "Invalid human-readable chord label."
    )

    confidence = float(
        segment["confidence"]
    )

    mean_margin = float(
        segment["mean_margin"]
    )

    frame_count = int(
        segment["frame_count"]
    )

    assert 0.0 <= confidence <= 1.0, (
        "Segment confidence is outside "
        f"[0, 1]: {confidence}"
    )

    assert mean_margin >= 0.0, (
        "Segment mean margin cannot "
        "be negative."
    )

    assert frame_count > 0, (
        "Segment frame_count must "
        "be positive."
    )

    return start


def assert_analysis(
    analysis: dict[str, Any],
    *,
    method: str,
) -> None:
    common_keys = {
        "feature_source",
        "frame_count",
        "activity_threshold",
        "minimum_score",
        "ambiguity_margin",
        "minimum_segment_duration_sec",
        "feature_extraction_ms",
        "classification_ms",
        "segmentation_ms",
        "no_chord_frames",
        "uncertain_frames",
        "classified_chord_frames",
    }

    assert common_keys.issubset(
        analysis
    ), (
        "Analysis metadata is missing fields: "
        f"{common_keys.difference(analysis)}"
    )

    frame_count = int(
        analysis["frame_count"]
    )

    no_chord_frames = int(
        analysis["no_chord_frames"]
    )

    uncertain_frames = int(
        analysis["uncertain_frames"]
    )

    classified_frames = int(
        analysis[
            "classified_chord_frames"
        ]
    )

    assert frame_count > 0, (
        "Analysis frame_count must "
        "be positive."
    )

    assert (
        no_chord_frames
        + uncertain_frames
        + classified_frames
        == frame_count
    ), (
        "Frame-category counts do not "
        "sum to frame_count."
    )

    assert (
        float(
            analysis[
                "activity_threshold"
            ]
        )
        >= 0.0
    ), (
        "Activity threshold cannot "
        "be negative."
    )

    for timing_key in (
        "feature_extraction_ms",
        "classification_ms",
        "segmentation_ms",
    ):
        assert float(
            analysis[timing_key]
        ) >= 0.0, (
            f"{timing_key} cannot "
            "be negative."
        )

    if method == "chroma":
        assert (
            analysis["feature_source"]
            == "traditional_chroma"
        ), (
            "Chroma response has an "
            "unexpected feature source."
        )

        required_chroma_keys = {
            "hop_length",
            "frame_length",
            "harmonic_margin",
            "relative_activity_floor",
        }

        assert required_chroma_keys.issubset(
            analysis
        ), (
            "Chroma analysis is missing: "
            f"{required_chroma_keys.difference(analysis)}"
        )

        return

    assert (
        analysis["feature_source"]
        == "basic_pitch_note_events"
    ), (
        "Basic Pitch response has an "
        "unexpected feature source."
    )

    required_basic_pitch_keys = {
        "basic_pitch_runtime",
        "model_was_loaded_before_request",
        "note_event_count",
        "frame_step_sec",
        "window_sec",
        "minimum_note_confidence",
        "bass_relative_weight",
        "relative_activity_floor",
        "model_configuration",
        "model_inference_ms",
    }

    assert required_basic_pitch_keys.issubset(
        analysis
    ), (
        "Basic Pitch analysis is missing: "
        f"{required_basic_pitch_keys.difference(analysis)}"
    )

    runtime = analysis[
        "basic_pitch_runtime"
    ]

    assert isinstance(
        runtime,
        str,
    ) and runtime, (
        "Basic Pitch runtime is missing."
    )

    assert isinstance(
        analysis[
            "model_was_loaded_before_request"
        ],
        bool,
    ), (
        "model_was_loaded_before_request "
        "must be Boolean."
    )

    assert int(
        analysis["note_event_count"]
    ) >= 0, (
        "note_event_count cannot "
        "be negative."
    )

    assert float(
        analysis["model_inference_ms"]
    ) >= 0.0, (
        "model_inference_ms cannot "
        "be negative."
    )

    model_configuration = analysis[
        "model_configuration"
    ]

    assert isinstance(
        model_configuration,
        dict,
    ), (
        "model_configuration must "
        "be an object."
    )

    required_model_keys = {
        "midi_low",
        "midi_high",
        "onset_threshold",
        "frame_threshold",
        "minimum_note_length_ms",
    }

    assert required_model_keys.issubset(
        model_configuration
    ), (
        "Basic Pitch configuration "
        "is incomplete."
    )


def assert_chord_response(
    response: dict[str, Any],
    *,
    expected_method: str,
) -> None:
    required_keys = {
        "segments",
        "progression",
        "tts",
        "method",
        "engine_status",
        "analysis",
        "audio_duration_sec",
        "latency_ms",
        "real_time_factor",
        "timing_ms",
    }

    assert required_keys.issubset(
        response
    ), (
        "API response is missing fields: "
        f"{required_keys.difference(response)}"
    )

    forbidden_keys = {
        "instrument_hint",
        "tuning",
        "notes",
        "render",
        "backend_requested",
        "backend_effective",
        "backend_runtime",
    }

    assert forbidden_keys.isdisjoint(
        response
    ), (
        "Legacy response fields remain: "
        f"{forbidden_keys.intersection(response)}"
    )

    assert (
        response["method"]
        == expected_method
    ), (
        "Unexpected response method: "
        f"{response['method']}"
    )

    assert (
        response["engine_status"]
        == "shared_core_v1"
    ), "Unexpected engine status."

    duration = float(
        response["audio_duration_sec"]
    )

    assert duration > 0.0, (
        "Audio duration must be positive."
    )

    segments = response["segments"]
    progression = response["progression"]
    tts_messages = response["tts"]

    assert isinstance(
        segments,
        list,
    ), "segments must be a list."

    assert isinstance(
        progression,
        list,
    ), "progression must be a list."

    assert isinstance(
        tts_messages,
        list,
    ) and tts_messages, (
        "tts must be a non-empty list."
    )

    expected_progression = [
        segment["label"]
        for segment in segments
        if segment["label"] != "N"
    ]

    assert (
        progression
        == expected_progression
    ), (
        "Progression mismatch: "
        f"expected {expected_progression}, "
        f"received {progression}"
    )

    previous_start = -1.0

    for segment in segments:
        previous_start = assert_segment(
            segment,
            duration=duration,
            previous_start=previous_start,
        )

    analysis = response["analysis"]

    assert isinstance(
        analysis,
        dict,
    ), "analysis must be an object."

    assert_analysis(
        analysis,
        method=expected_method,
    )

    timing = response["timing_ms"]

    assert {
        "io",
        "chord_analysis",
        "total",
    }.issubset(timing), (
        "Timing metadata is incomplete."
    )

    for timing_key in (
        "io",
        "chord_analysis",
        "total",
    ):
        assert float(
            timing[timing_key]
        ) >= 0.0, (
            f"{timing_key} cannot "
            "be negative."
        )

    assert "notes" not in timing
    assert "tabs" not in timing


def summarize_response(
    response: dict[str, Any],
    *,
    client_elapsed_ms: float,
) -> dict[str, Any]:
    analysis = response.get(
        "analysis",
        {},
    )

    return {
        "method": response.get("method"),
        "feature_source": analysis.get(
            "feature_source"
        ),
        "basic_pitch_runtime": analysis.get(
            "basic_pitch_runtime"
        ),
        "model_was_loaded_before_request": (
            analysis.get(
                "model_was_loaded_before_request"
            )
        ),
        "note_event_count": analysis.get(
            "note_event_count"
        ),
        "audio_duration_sec": response.get(
            "audio_duration_sec"
        ),
        "segment_count": len(
            response.get(
                "segments",
                [],
            )
        ),
        "progression": response.get(
            "progression",
            [],
        ),
        "frame_count": analysis.get(
            "frame_count"
        ),
        "no_chord_frames": analysis.get(
            "no_chord_frames"
        ),
        "uncertain_frames": analysis.get(
            "uncertain_frames"
        ),
        "classified_chord_frames": analysis.get(
            "classified_chord_frames"
        ),
        "server_io_ms": response.get(
            "timing_ms",
            {},
        ).get("io"),
        "server_chord_analysis_ms": response.get(
            "timing_ms",
            {},
        ).get("chord_analysis"),
        "server_total_ms": response.get(
            "timing_ms",
            {},
        ).get("total"),
        "server_real_time_factor": response.get(
            "real_time_factor"
        ),
        "client_elapsed_ms": round(
            client_elapsed_ms,
            2,
        ),
    }


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify both ChordAssist feature "
            "paths through the shared API contract."
        )
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Path to an audio file.",
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "evaluation_results/smoke"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
    )

    args = parser.parse_args()

    audio_path = (
        args.audio
        .expanduser()
        .resolve()
    )

    if not audio_path.is_file():
        raise SystemExit(
            "Audio file does not exist: "
            f"{audio_path}"
        )

    base_url = (
        args.base_url.rstrip("/")
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = (
        args.output_root / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Dual-method API regression "
            "smoke test; not an accuracy "
            "benchmark"
        ),
        "base_url": base_url,
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
        "audio_path": str(
            audio_path
        ),
        "audio_filename": (
            audio_path.name
        ),
        "audio_size_bytes": (
            audio_path.stat().st_size
        ),
        "audio_sha256": sha256_file(
            audio_path
        ),
    }

    health_before, health_before_ms = (
        get_json(
            f"{base_url}/health",
            timeout_seconds=args.timeout,
        )
    )

    assert_health(
        health_before
    )

    chroma_first, chroma_first_ms = (
        analyze_audio(
            base_url,
            audio_path,
            method="chroma",
            timeout_seconds=args.timeout,
        )
    )

    assert_chord_response(
        chroma_first,
        expected_method="chroma",
    )

    chroma_repeated, chroma_repeated_ms = (
        analyze_audio(
            base_url,
            audio_path,
            method="chroma",
            timeout_seconds=args.timeout,
        )
    )

    assert_chord_response(
        chroma_repeated,
        expected_method="chroma",
    )

    basic_pitch_first, basic_pitch_first_ms = (
        analyze_audio(
            base_url,
            audio_path,
            method="basic_pitch",
            timeout_seconds=args.timeout,
        )
    )

    assert_chord_response(
        basic_pitch_first,
        expected_method="basic_pitch",
    )

    basic_pitch_warm, basic_pitch_warm_ms = (
        analyze_audio(
            base_url,
            audio_path,
            method="basic_pitch",
            timeout_seconds=args.timeout,
        )
    )

    assert_chord_response(
        basic_pitch_warm,
        expected_method="basic_pitch",
    )

    assert (
        basic_pitch_warm["analysis"][
            "model_was_loaded_before_request"
        ]
        is True
    ), (
        "The second Basic Pitch request "
        "did not reuse the loaded model."
    )

    health_after, health_after_ms = (
        get_json(
            f"{base_url}/health",
            timeout_seconds=args.timeout,
        )
    )

    assert_health(
        health_after
    )

    assert (
        health_after[
            "basic_pitch_loaded"
        ]
        is True
    ), (
        "Health endpoint did not report "
        "the Basic Pitch model as loaded."
    )

    chroma_deterministic = (
        chroma_first["progression"]
        == chroma_repeated["progression"]
    )

    basic_pitch_deterministic = (
        basic_pitch_first["progression"]
        == basic_pitch_warm["progression"]
    )

    assert chroma_deterministic, (
        "Repeated chroma requests produced "
        "different progressions."
    )

    assert basic_pitch_deterministic, (
        "Repeated Basic Pitch requests "
        "produced different progressions."
    )

    responses = {
        "health_before": health_before,
        "chroma_first": chroma_first,
        "chroma_repeated": chroma_repeated,
        "basic_pitch_first": (
            basic_pitch_first
        ),
        "basic_pitch_warm": (
            basic_pitch_warm
        ),
        "health_after": health_after,
    }

    summary = {
        "metadata": metadata,
        "health": {
            "basic_pitch_loaded_before": (
                health_before[
                    "basic_pitch_loaded"
                ]
            ),
            "basic_pitch_loaded_after": (
                health_after[
                    "basic_pitch_loaded"
                ]
            ),
            "health_before_client_ms": round(
                health_before_ms,
                2,
            ),
            "health_after_client_ms": round(
                health_after_ms,
                2,
            ),
        },
        "results": {
            "chroma_first": summarize_response(
                chroma_first,
                client_elapsed_ms=(
                    chroma_first_ms
                ),
            ),
            "chroma_repeated": summarize_response(
                chroma_repeated,
                client_elapsed_ms=(
                    chroma_repeated_ms
                ),
            ),
            "basic_pitch_first": (
                summarize_response(
                    basic_pitch_first,
                    client_elapsed_ms=(
                        basic_pitch_first_ms
                    ),
                )
            ),
            "basic_pitch_warm": (
                summarize_response(
                    basic_pitch_warm,
                    client_elapsed_ms=(
                        basic_pitch_warm_ms
                    ),
                )
            ),
        },
        "determinism": {
            "chroma": chroma_deterministic,
            "basic_pitch": (
                basic_pitch_deterministic
            ),
        },
        "assertions_passed": True,
    }

    for name, response in (
        responses.items()
    ):
        write_json(
            output_dir / f"{name}.json",
            response,
        )

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print(
        "ChordAssist dual-method "
        "API smoke test passed."
    )

    print(
        f"Audio: {audio_path}"
    )

    print(
        "Git: "
        f"{metadata['git_branch']} "
        f"@ {metadata['git_commit']}"
    )

    print(
        f"Git dirty: "
        f"{metadata['git_dirty']}"
    )

    print(
        f"Results: "
        f"{output_dir.resolve()}"
    )

    print()

    print(
        json.dumps(
            summary["results"],
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except AssertionError as exc:
        raise SystemExit(
            "Smoke-test assertion failed: "
            f"{exc}"
        ) from exc
    except SmokeTestError as exc:
        raise SystemExit(
            str(exc)
        ) from exc
