#!/usr/bin/env python3
"""
Reusable ChordAssist FastAPI regression smoke test.

This script is intentionally not a final benchmark. It verifies the API contract,
records reproducibility metadata, and saves raw JSON responses for later review.

The FastAPI server must already be running.
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


STANDARD_E_OPEN_MIDI = [40, 45, 50, 55, 59, 64]
GUITAR_MIDI_LOW = 40
GUITAR_MIDI_HIGH = 88


class SmokeTestError(RuntimeError):
    """Raised when the API smoke test cannot complete."""


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


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

    value = completed.stdout.strip()
    return value or None


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
            "Confirm that uvicorn is running."
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
    return json_request(request, timeout_seconds=timeout_seconds)


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
    backend: str,
    chords: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    query = urlencode(
        {
            "backend": backend,
            "mode": "full",
            "chords": str(chords).lower(),
        }
    )
    url = f"{base_url}/analyze-file?{query}"
    body, content_type = encode_multipart_file(audio_path)

    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
    )

    return json_request(request, timeout_seconds=timeout_seconds)


def assert_common_response(response: dict[str, Any]) -> None:
    assert response["instrument_hint"] == "guitar"

    tuning = response["tuning"]
    assert tuning["name"] == "Standard E"
    assert tuning["source"] == "fixed_assumption"
    assert tuning["string_open_midi"] == STANDARD_E_OPEN_MIDI

    assert response["mode_requested"] == "full"
    assert response["mode_effective"] == "full"

    assert "tuning" not in response["timing_ms"]
    assert "guitar_tabs" in response["render"]
    assert "piano_roll" not in response["render"]
    assert "violin_fingerings" not in response["render"]

    notes = response["notes"]
    assert isinstance(notes, list)
    assert notes, "The transcription response contained no notes."

    assert all(
        float(note["t_on"]) < float(note["t_off"])
        for note in notes
    )

    assert all(
        GUITAR_MIDI_LOW <= int(note["midi"]) <= GUITAR_MIDI_HIGH
        for note in notes
    )

    assert all(
        float(notes[index]["t_on"])
        <= float(notes[index + 1]["t_on"])
        for index in range(len(notes) - 1)
    )


def summarize_response(
    response: dict[str, Any],
    *,
    client_elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "backend": response.get("backend_effective"),
        "runtime": response.get("backend_runtime"),
        "audio_duration_sec": response.get("audio_duration_sec"),
        "note_count": len(response.get("notes", [])),
        "tab_count": len(
            response.get("render", {}).get("guitar_tabs", [])
        ),
        "chord_count": len(response.get("chords", [])),
        "server_total_ms": response.get("timing_ms", {}).get("total"),
        "server_io_ms": response.get("timing_ms", {}).get("io"),
        "server_notes_ms": response.get("timing_ms", {}).get("notes"),
        "server_chords_ms": response.get("timing_ms", {}).get("chords"),
        "server_tabs_ms": response.get("timing_ms", {}).get("tabs"),
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
            "Verify the ChordAssist FastAPI contract and record a "
            "reproducible smoke-test summary."
        )
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to a guitar audio file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation_results/smoke"),
        help="Directory under which a timestamped result folder is created.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds. Default: %(default)s",
    )
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    base_url = args.base_url.rstrip("/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "API regression smoke test; not a final benchmark",
        "base_url": base_url,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "basic_pitch_version": package_version("basic-pitch"),
        "coremltools_version": package_version("coremltools"),
        "audio_path": str(audio_path),
        "audio_filename": audio_path.name,
        "audio_size_bytes": audio_path.stat().st_size,
        "audio_sha256": sha256_file(audio_path),
    }

    health_before, health_before_client_ms = get_json(
        f"{base_url}/health",
        timeout_seconds=args.timeout,
    )

    assert health_before["ok"] is True
    assert health_before["instrument"] == "guitar"
    assert set(health_before["supported_backends"]) == {
        "baseline",
        "basic_pitch",
    }
    assert health_before["default_backend"] == "basic_pitch"

    basic_first, basic_first_client_ms = analyze_audio(
        base_url,
        audio_path,
        backend="basic_pitch",
        chords=False,
        timeout_seconds=args.timeout,
    )
    assert_common_response(basic_first)
    assert basic_first["backend_requested"] == "basic_pitch"
    assert basic_first["backend_effective"] == "basic_pitch"
    assert basic_first["backend_runtime"] == "COREML"
    assert basic_first["chord_backend"] is None

    basic_warm, basic_warm_client_ms = analyze_audio(
        base_url,
        audio_path,
        backend="basic_pitch",
        chords=False,
        timeout_seconds=args.timeout,
    )
    assert_common_response(basic_warm)
    assert basic_warm["backend_effective"] == "basic_pitch"
    assert basic_warm["backend_runtime"] == "COREML"

    baseline, baseline_client_ms = analyze_audio(
        base_url,
        audio_path,
        backend="baseline",
        chords=False,
        timeout_seconds=args.timeout,
    )
    assert_common_response(baseline)
    assert baseline["backend_requested"] == "baseline"
    assert baseline["backend_effective"] == "baseline"
    assert baseline["backend_runtime"] == "DSP"

    with_chords, chords_client_ms = analyze_audio(
        base_url,
        audio_path,
        backend="basic_pitch",
        chords=True,
        timeout_seconds=args.timeout,
    )
    assert_common_response(with_chords)
    assert with_chords["backend_effective"] == "basic_pitch"
    assert with_chords["chord_backend"] == "chroma_template"
    assert isinstance(with_chords["chords"], list)

    health_after, health_after_client_ms = get_json(
        f"{base_url}/health",
        timeout_seconds=args.timeout,
    )
    assert health_after["ok"] is True
    assert health_after["basic_pitch_loaded"] is True

    responses = {
        "health_before": health_before,
        "basic_pitch_first": basic_first,
        "basic_pitch_warm": basic_warm,
        "baseline": baseline,
        "basic_pitch_with_chords": with_chords,
        "health_after": health_after,
    }

    client_timings_ms = {
        "health_before": round(health_before_client_ms, 2),
        "basic_pitch_first": round(basic_first_client_ms, 2),
        "basic_pitch_warm": round(basic_warm_client_ms, 2),
        "baseline": round(baseline_client_ms, 2),
        "basic_pitch_with_chords": round(chords_client_ms, 2),
        "health_after": round(health_after_client_ms, 2),
    }

    summary = {
        "metadata": metadata,
        "health": {
            "basic_pitch_loaded_before": health_before[
                "basic_pitch_loaded"
            ],
            "basic_pitch_loaded_after": health_after[
                "basic_pitch_loaded"
            ],
        },
        "results": {
            "basic_pitch_first": summarize_response(
                basic_first,
                client_elapsed_ms=basic_first_client_ms,
            ),
            "basic_pitch_warm": summarize_response(
                basic_warm,
                client_elapsed_ms=basic_warm_client_ms,
            ),
            "baseline": summarize_response(
                baseline,
                client_elapsed_ms=baseline_client_ms,
            ),
            "basic_pitch_with_chords": summarize_response(
                with_chords,
                client_elapsed_ms=chords_client_ms,
            ),
        },
        "client_timings_ms": client_timings_ms,
        "assertions_passed": True,
    }

    for name, response in responses.items():
        write_json(output_dir / f"{name}.json", response)

    write_json(output_dir / "summary.json", summary)

    print("ChordAssist API smoke test passed.")
    print(f"Audio: {audio_path}")
    print(f"Git: {metadata['git_branch']} @ {metadata['git_commit']}")
    print(f"Results: {output_dir.resolve()}")
    print()
    print(json.dumps(summary["results"], indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        raise SystemExit(f"Smoke-test assertion failed: {exc}") from exc
    except SmokeTestError as exc:
        raise SystemExit(str(exc)) from exc
