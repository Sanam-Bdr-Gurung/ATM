#!/usr/bin/env python3
"""Regression smoke test for the chord-first ChordAssist API.

This verifies the transitional chroma-only API contract. It is not a final
accuracy or latency benchmark.
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
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SmokeTestError(RuntimeError):
    """Raised when the API smoke test cannot complete."""


def git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_request(
    request: Request,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            status = response.status
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SmokeTestError(
            f"HTTP {exc.code} for {request.full_url}: {detail}"
        ) from exc
    except URLError as exc:
        raise SmokeTestError(
            f"Could not reach {request.full_url}: {exc.reason}. "
            "Confirm that Uvicorn is running."
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not 200 <= status < 300:
        raise SmokeTestError(
            f"Unexpected HTTP status {status} for {request.full_url}."
        )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeTestError(
            f"Response from {request.full_url} was not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise SmokeTestError(
            f"Expected a JSON object from {request.full_url}."
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
        headers={"Accept": "application/json"},
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
    boundary = f"----ChordAssistBoundary{uuid.uuid4().hex}"

    content_type = (
        mimetypes.guess_type(audio_path.name)[0]
        or "application/octet-stream"
    )

    audio_bytes = audio_path.read_bytes()

    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{audio_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            audio_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    return body, f"multipart/form-data; boundary={boundary}"


def analyze_audio(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    body, content_type = encode_multipart_file(audio_path)

    request = Request(
        f"{base_url}/analyze-file",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
    )

    return json_request(
        request,
        timeout_seconds=timeout_seconds,
    )


def assert_health(response: dict[str, Any]) -> None:
    assert response["ok"] is True
    assert response["service"] == "chordassist"
    assert response["scope"] == "prevailing_chord_recognition"
    assert response["available_methods"] == ["chroma"]
    assert response["planned_methods"] == ["basic_pitch"]
    assert response["basic_pitch_adapter_available"] is True


def assert_chord_response(response: dict[str, Any]) -> None:
    required_keys = {
        "segments",
        "progression",
        "tts",
        "method",
        "engine_status",
        "audio_duration_sec",
        "latency_ms",
        "real_time_factor",
        "timing_ms",
    }

    assert required_keys.issubset(response)

    # Old scope fields must no longer appear.
    forbidden_keys = {
        "instrument_hint",
        "tuning",
        "notes",
        "render",
        "backend_requested",
        "backend_effective",
        "backend_runtime",
    }

    assert forbidden_keys.isdisjoint(response)

    assert response["method"] == "chroma_template_baseline"
    assert response["engine_status"] == "temporary_baseline"

    duration = float(response["audio_duration_sec"])
    assert duration > 0.0

    segments = response["segments"]
    progression = response["progression"]
    tts_messages = response["tts"]

    assert isinstance(segments, list)
    assert isinstance(progression, list)
    assert isinstance(tts_messages, list)
    assert tts_messages

    assert progression == [
        segment["label"]
        for segment in segments
    ]

    previous_start = -1.0

    for segment in segments:
        assert {
            "start",
            "end",
            "label",
            "display",
            "confidence",
        }.issubset(segment)

        start = float(segment["start"])
        end = float(segment["end"])

        assert 0.0 <= start < end
        assert end <= duration + 0.01
        assert start >= previous_start
        assert isinstance(segment["label"], str)
        assert segment["label"]
        assert isinstance(segment["display"], str)
        assert segment["display"]

        previous_start = start

    timing = response["timing_ms"]

    assert {
        "io",
        "chord_analysis",
        "total",
    }.issubset(timing)

    assert "notes" not in timing
    assert "tabs" not in timing


def summarize_response(
    response: dict[str, Any],
    *,
    client_elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "method": response.get("method"),
        "engine_status": response.get("engine_status"),
        "audio_duration_sec": response.get("audio_duration_sec"),
        "segment_count": len(response.get("segments", [])),
        "progression": response.get("progression", []),
        "server_io_ms": response.get("timing_ms", {}).get("io"),
        "server_chord_analysis_ms": response.get(
            "timing_ms",
            {},
        ).get("chord_analysis"),
        "server_total_ms": response.get("timing_ms", {}).get("total"),
        "server_real_time_factor": response.get("real_time_factor"),
        "client_elapsed_ms": round(client_elapsed_ms, 2),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the transitional chord-first FastAPI contract and "
            "save reproducible regression evidence."
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
        default=Path("evaluation_results/smoke"),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()

    if not audio_path.is_file():
        raise SystemExit(
            f"Audio file does not exist: {audio_path}"
        )

    base_url = args.base_url.rstrip("/")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Chord-first API regression smoke test; "
            "not a final benchmark"
        ),
        "base_url": base_url,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "audio_path": str(audio_path),
        "audio_filename": audio_path.name,
        "audio_size_bytes": audio_path.stat().st_size,
        "audio_sha256": sha256_file(audio_path),
    }

    health, health_client_ms = get_json(
        f"{base_url}/health",
        timeout_seconds=args.timeout,
    )

    assert_health(health)

    first, first_client_ms = analyze_audio(
        base_url,
        audio_path,
        timeout_seconds=args.timeout,
    )

    assert_chord_response(first)

    repeated, repeated_client_ms = analyze_audio(
        base_url,
        audio_path,
        timeout_seconds=args.timeout,
    )

    assert_chord_response(repeated)

    # The temporary chroma implementation should be deterministic.
    assert first["progression"] == repeated["progression"]

    responses = {
        "health": health,
        "first_analysis": first,
        "repeated_analysis": repeated,
    }

    summary = {
        "metadata": metadata,
        "health_client_ms": round(health_client_ms, 2),
        "results": {
            "first_analysis": summarize_response(
                first,
                client_elapsed_ms=first_client_ms,
            ),
            "repeated_analysis": summarize_response(
                repeated,
                client_elapsed_ms=repeated_client_ms,
            ),
        },
        "deterministic_progression": (
            first["progression"] == repeated["progression"]
        ),
        "assertions_passed": True,
    }

    for name, response in responses.items():
        write_json(
            output_dir / f"{name}.json",
            response,
        )

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print("ChordAssist chord-first API smoke test passed.")
    print(f"Audio: {audio_path}")
    print(
        f"Git: {metadata['git_branch']} "
        f"@ {metadata['git_commit']}"
    )
    print(f"Git dirty: {metadata['git_dirty']}")
    print(f"Results: {output_dir.resolve()}")
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
        raise SystemExit(main())
    except AssertionError as exc:
        raise SystemExit(
            f"Smoke-test assertion failed: {exc}"
        ) from exc
    except SmokeTestError as exc:
        raise SystemExit(str(exc)) from exc
