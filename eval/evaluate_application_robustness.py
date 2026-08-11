#!/usr/bin/env python3
"""Exploratory application-robustness evaluation runner (Tier 2).

Runs declared exploratory clips through the FROZEN backend API and produces
descriptive metrics. This runner:

- never modifies or duplicates recognition logic (metrics reuse
  chord_evaluation / chord_dataset_metrics);
- refuses manifests that are not explicitly exploratory;
- verifies audio hashes and the frozen configuration ID before evaluating;
- writes to a fresh timestamped directory (never overwrites earlier runs).

Results are descriptive/exploratory only: they are never pooled into the
formal held-out metrics and cannot be used to tune any parameter.

Usage:
  python eval/evaluate_application_robustness.py \
    --manifest evaluation_data/application_robustness/manifest.json \
    --base-url http://127.0.0.1:8000 \
    --output evaluation_results/application_robustness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = Path(__file__).resolve().parent

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from chord_dataset_metrics import aggregate_chord_evaluations  # noqa: E402
from chord_evaluation import evaluate_chord_timelines  # noqa: E402
from evaluate_chord_dataset import encode_multipart_file  # noqa: E402

REQUIRED_SPLIT = "exploratory_application_robustness"

FROZEN_CONFIGURATION_ID = "development_8clip_grid_20260806"

FROZEN_METHOD = "basic_pitch"

# Mirror of the FRONTEND presentation parameters, used ONLY to compute
# descriptive presentation statistics. This is not a recognition threshold
# and does not touch backend behaviour.
PRESENTATION_MINIMUM_DURATION_SEC = 1.25

TRACKED_PACKAGES = (
    "basic-pitch",
    "librosa",
    "numpy",
    "scipy",
    "soundfile",
    "coremltools",
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    try:
        with urlopen(Request(url), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        fail(f"request to {url} failed: {error}")

    raise AssertionError("unreachable")


def analyze_clip(base_url: str, audio_path: Path) -> dict[str, Any]:
    body, content_type = encode_multipart_file(audio_path)

    request = Request(
        f"{base_url}/analyze-file?method={FROZEN_METHOD}",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        fail(f"analysis of {audio_path.name} failed: {error}")

    raise AssertionError("unreachable")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())

    if manifest.get("split") != REQUIRED_SPLIT:
        fail(
            "manifest split must be "
            f"'{REQUIRED_SPLIT}' — this runner refuses formal or unknown "
            "splits so exploratory data can never pool into formal metrics."
        )

    if manifest.get("include_in_formal_metrics") is not False:
        fail("manifest must declare include_in_formal_metrics: false")

    clips = manifest.get("clips")

    if not isinstance(clips, list) or not clips:
        fail("manifest contains no clips")

    return manifest


def presentation_statistics(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Descriptive presentation statistics (frontend-filter mirror).

    APPLICATION PRESENTATION STATISTICS ONLY — this characterizes what the
    app would speak; it is never an accuracy metric and must not be compared
    against recognition metrics as if it were one.
    """
    raw_count = len(segments)

    kept: list[dict[str, Any]] = []

    omitted_x_count = 0
    omitted_x_duration = 0.0
    retained_x_count = 0
    retained_x_duration = 0.0

    for segment in segments:
        label = segment["label"]
        duration = float(segment["end"]) - float(segment["start"])

        if label == "N":
            continue

        if duration < PRESENTATION_MINIMUM_DURATION_SEC:
            if label == "X":
                omitted_x_count += 1
                omitted_x_duration += duration
            continue

        if label == "X":
            retained_x_count += 1
            retained_x_duration += duration

        kept.append(segment)

    summary_labels: list[str] = []

    for segment in kept:
        if summary_labels and summary_labels[-1] == segment["label"]:
            continue
        summary_labels.append(segment["label"])

    return {
        "kind": "application_presentation_statistics",
        "note": (
            "Descriptive presentation behaviour only (frontend 1.25 s "
            "threshold-first mirror). Not an accuracy metric."
        ),
        "raw_segment_count": raw_count,
        "prevailing_summary_item_count": len(summary_labels),
        "prevailing_summary_labels": summary_labels,
        "retained_uncertainty_count": retained_x_count,
        "retained_uncertainty_duration_sec": round(retained_x_duration, 4),
        "omitted_uncertainty_count": omitted_x_count,
        "omitted_uncertainty_duration_sec": round(omitted_x_duration, 4),
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}

    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"

    return versions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory application-robustness evaluation against the "
            "frozen backend (descriptive only)."
        ),
    )

    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "application_robustness",
    )

    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    manifest_sha = sha256_of(manifest_path)

    base_url = args.base_url.rstrip("/")

    health = http_get_json(f"{base_url}/health")

    reported_config = (
        health.get("selected_configuration", {}).get("configuration_id")
    )

    if reported_config != FROZEN_CONFIGURATION_ID:
        fail(
            "backend reports configuration "
            f"'{reported_config}', expected frozen "
            f"'{FROZEN_CONFIGURATION_ID}'. Refusing to evaluate."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / f"{timestamp}_application_robustness"

    if run_dir.exists():
        fail(f"run directory already exists: {run_dir}")

    (run_dir / "per_clip").mkdir(parents=True)

    evaluations: list[dict[str, Any]] = []
    per_clip_index: list[dict[str, Any]] = []

    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]

        audio_path = (manifest_path.parent / clip["audio_path"]).resolve()
        if not audio_path.exists():
            audio_path = (PROJECT_ROOT / clip["audio_path"]).resolve()
        if not audio_path.exists():
            fail(f"{clip_id}: audio file not found: {clip['audio_path']}")

        actual_sha = sha256_of(audio_path)

        if actual_sha != clip.get("sha256"):
            fail(
                f"{clip_id}: audio SHA-256 mismatch — manifest declares "
                f"{clip.get('sha256')}, file is {actual_sha}. The manifest "
                "must be regenerated deliberately, not silently."
            )

        annotation_path = (
            manifest_path.parent / clip["annotation_path"]
        ).resolve()
        if not annotation_path.exists():
            annotation_path = (
                PROJECT_ROOT / clip["annotation_path"]
            ).resolve()
        if not annotation_path.exists():
            fail(f"{clip_id}: annotation not found: {clip['annotation_path']}")

        annotation = json.loads(annotation_path.read_text())

        response = analyze_clip(base_url, audio_path)

        analysis = response.get("analysis", {})

        if analysis.get("configuration_id") != FROZEN_CONFIGURATION_ID:
            fail(
                f"{clip_id}: response configuration "
                f"'{analysis.get('configuration_id')}' is not the frozen "
                "configuration."
            )

        if response.get("method") != FROZEN_METHOD:
            fail(f"{clip_id}: response method is not {FROZEN_METHOD}")

        duration = float(
            annotation.get(
                "duration_sec",
                response.get("audio_duration_sec", 0.0),
            )
        )

        metrics = evaluate_chord_timelines(
            annotation["segments"],
            response["segments"],
            duration_sec=duration,
        )

        presentation = presentation_statistics(response["segments"])

        clip_dir = run_dir / "per_clip" / clip_id
        clip_dir.mkdir(parents=True)

        (clip_dir / "response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True)
        )
        (clip_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )
        (clip_dir / "presentation_statistics.json").write_text(
            json.dumps(presentation, indent=2, sort_keys=True)
        )
        (clip_dir / "clip_metadata.json").write_text(
            json.dumps(clip, indent=2, sort_keys=True)
        )

        evaluations.append(metrics)

        per_clip_index.append(
            {
                "clip_id": clip_id,
                "audio_sha256": actual_sha,
                "latency_ms": response.get("latency_ms"),
                "real_time_factor": response.get("real_time_factor"),
                "raw_segment_count": presentation["raw_segment_count"],
                "prevailing_summary_item_count": presentation[
                    "prevailing_summary_item_count"
                ],
            }
        )

        print(f"evaluated {clip_id}")

    aggregate = aggregate_chord_evaluations(evaluations)

    (run_dir / "aggregate_raw_recognition_metrics.json").write_text(
        json.dumps(
            {
                "kind": "raw_recognition_metrics",
                "note": (
                    "Descriptive aggregate over exploratory clips. Never "
                    "pooled with formal held-out metrics; never used for "
                    "tuning."
                ),
                "clip_count": len(evaluations),
                "metrics": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
    )

    (run_dir / "manifest_copy.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    provenance = {
        "backend_git_sha": git_head(),
        "configuration_id": FROZEN_CONFIGURATION_ID,
        "configuration_reported_by_health": health.get(
            "selected_configuration"
        ),
        "base_url": base_url,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "python": sys.version,
        "packages": package_versions(),
        "presentation_mirror_threshold_sec": (
            PRESENTATION_MINIMUM_DURATION_SEC
        ),
        "timestamp_utc": timestamp,
        "split": REQUIRED_SPLIT,
        "include_in_formal_metrics": False,
    }

    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True)
    )

    (run_dir / "per_clip_index.json").write_text(
        json.dumps(per_clip_index, indent=2, sort_keys=True)
    )

    print(f"\nrun written to {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
