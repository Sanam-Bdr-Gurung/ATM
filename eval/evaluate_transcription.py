#!/usr/bin/env python3
"""Checkpoint 4 pilot transcription evaluation for ChordAssist.

Regenerates predictions from the current DSP baseline and Basic Pitch
implementations. Historical *_pred.json and evaluation_data/of_compare files
are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from api import (  # noqa: E402
    STANDARD_E_OPEN_MIDI,
    transcribe_with_baseline,
    transcribe_with_basic_pitch,
)
from audio_input import load_audio_bytes  # noqa: E402
from tabs_guitar_dp import dp_tab_mapping  # noqa: E402

JsonObject = dict[str, Any]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_note_gt(value: Any, path: Path) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"Note ground truth must be a list: {path}")

    events: list[JsonObject] = []
    for index, event in enumerate(value):
        if not isinstance(event, dict) or not {"t_on", "midi"}.issubset(event):
            raise ValueError(f"Invalid note event {index} in {path}")
        events.append({"t_on": float(event["t_on"]), "midi": int(event["midi"])})

    return sorted(events, key=lambda event: (event["t_on"], event["midi"]))


def validate_tab_gt(value: Any, path: Path) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"Tab ground truth must be a list: {path}")

    events: list[JsonObject] = []
    for index, event in enumerate(value):
        required = {"t_on", "string", "fret"}
        if not isinstance(event, dict) or not required.issubset(event):
            raise ValueError(f"Invalid tab event {index} in {path}")
        events.append(
            {
                "t_on": float(event["t_on"]),
                "string": int(event["string"]),
                "fret": int(event["fret"]),
            }
        )

    return sorted(
        events,
        key=lambda event: (event["t_on"], event["string"], event["fret"]),
    )


def one_to_one_matches(
    truth: list[JsonObject],
    predictions: list[JsonObject],
    *,
    tolerance_seconds: float,
    require_same_midi: bool,
) -> list[tuple[int, int, float]]:
    """Globally greedy one-to-one matching ordered by onset error."""
    candidates: list[tuple[float, int, int]] = []

    for truth_index, truth_event in enumerate(truth):
        for prediction_index, prediction_event in enumerate(predictions):
            if require_same_midi and (
                int(truth_event["midi"]) != int(prediction_event["midi"])
            ):
                continue

            error = abs(
                float(prediction_event["t_on"]) - float(truth_event["t_on"])
            )
            if error <= tolerance_seconds:
                candidates.append((error, truth_index, prediction_index))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for error, truth_index, prediction_index in candidates:
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, error))

    return matches


def note_metrics(
    truth: list[JsonObject],
    predictions: list[JsonObject],
    tolerance_seconds: float,
) -> JsonObject:
    matches = one_to_one_matches(
        truth,
        predictions,
        tolerance_seconds=tolerance_seconds,
        require_same_midi=True,
    )
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth) - tp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    onset_errors_ms = [error * 1000.0 for _, _, error in matches]

    return {
        "n_ground_truth": len(truth),
        "n_predictions": len(predictions),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "matched_onset_error_mean_ms": (
            rounded(statistics.fmean(onset_errors_ms), 3)
            if onset_errors_ms
            else None
        ),
        "matched_onset_error_median_ms": (
            rounded(statistics.median(onset_errors_ms), 3)
            if onset_errors_ms
            else None
        ),
    }


def tab_metrics(
    truth: list[JsonObject],
    predictions: list[JsonObject],
    tolerance_seconds: float,
) -> JsonObject:
    matches = one_to_one_matches(
        truth,
        predictions,
        tolerance_seconds=tolerance_seconds,
        require_same_midi=False,
    )
    string_correct = 0
    fret_correct = 0
    exact_correct = 0

    for truth_index, prediction_index, _ in matches:
        gt = truth[truth_index]
        pred = predictions[prediction_index]
        string_ok = int(gt["string"]) == int(pred["string"])
        fret_ok = int(gt["fret"]) == int(pred["fret"])
        string_correct += int(string_ok)
        fret_correct += int(fret_ok)
        exact_correct += int(string_ok and fret_ok)

    matched = len(matches)
    return {
        "n_ground_truth": len(truth),
        "n_predictions": len(predictions),
        "n_matched_onset": matched,
        "onset_coverage": rounded(safe_div(matched, len(truth))),
        "string_correct": string_correct,
        "fret_correct": fret_correct,
        "exact_correct": exact_correct,
        "string_accuracy_on_matched": rounded(safe_div(string_correct, matched)),
        "fret_accuracy_on_matched": rounded(safe_div(fret_correct, matched)),
        "exact_accuracy_on_matched": rounded(safe_div(exact_correct, matched)),
        "exact_tab_recall": rounded(safe_div(exact_correct, len(truth))),
    }


def run_backend(
    backend: str,
    waveform: Any,
    sample_rate: int,
) -> tuple[list[JsonObject], str]:
    if backend == "basic_pitch":
        events, runtime = transcribe_with_basic_pitch(waveform, sample_rate)
        return events, runtime
    if backend == "baseline":
        return transcribe_with_baseline(waveform, sample_rate), "DSP"
    raise ValueError(f"Unsupported backend: {backend}")


def aggregate_notes(rows: Iterable[JsonObject]) -> JsonObject:
    rows = list(rows)
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "clips": len(rows),
        "n_ground_truth": sum(int(row["n_ground_truth"]) for row in rows),
        "n_predictions": sum(int(row["n_predictions"]) for row in rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": rounded(precision),
        "micro_recall": rounded(recall),
        "micro_f1": rounded(f1),
        "macro_f1": rounded(
            statistics.fmean(float(row["f1"]) for row in rows) if rows else 0.0
        ),
    }


def aggregate_tabs(rows: Iterable[JsonObject]) -> JsonObject:
    rows = list(rows)
    n_gt = sum(int(row["n_ground_truth"]) for row in rows)
    n_pred = sum(int(row["n_predictions"]) for row in rows)
    matched = sum(int(row["n_matched_onset"]) for row in rows)
    string_correct = sum(int(row["string_correct"]) for row in rows)
    fret_correct = sum(int(row["fret_correct"]) for row in rows)
    exact_correct = sum(int(row["exact_correct"]) for row in rows)
    return {
        "clips": len(rows),
        "n_ground_truth": n_gt,
        "n_predictions": n_pred,
        "n_matched_onset": matched,
        "onset_coverage": rounded(safe_div(matched, n_gt)),
        "string_accuracy_on_matched": rounded(safe_div(string_correct, matched)),
        "fret_accuracy_on_matched": rounded(safe_div(fret_correct, matched)),
        "exact_accuracy_on_matched": rounded(safe_div(exact_correct, matched)),
        "exact_tab_recall": rounded(safe_div(exact_correct, n_gt)),
    }


def discover_clips(data_dir: Path) -> list[JsonObject]:
    clips: list[JsonObject] = []
    for audio_path in sorted(data_dir.glob("*.wav")):
        stem = audio_path.stem
        note_gt_path = data_dir / f"{stem}.json"
        tab_gt_path = data_dir / f"{stem}_tabs_gt.json"
        if not note_gt_path.exists() or not tab_gt_path.exists():
            print(f"[WARN] Skipping incomplete clip: {audio_path.name}")
            continue
        clips.append(
            {
                "name": stem,
                "audio_path": audio_path,
                "note_gt_path": note_gt_path,
                "tab_gt_path": tab_gt_path,
            }
        )
    return clips


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("evaluation_data/accuracy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation_results/checkpoint4_pilot"),
    )
    parser.add_argument("--note-tolerance", type=float, default=0.05)
    parser.add_argument("--tab-tolerance", type=float, default=0.08)
    parser.add_argument("--max-fret", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("baseline", "basic_pitch"),
        default=("baseline", "basic_pitch"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    clips = discover_clips(data_dir)
    if not clips:
        raise SystemExit(f"No complete annotated clips found in {data_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata: JsonObject = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Checkpoint 4 pilot; not the final held-out thesis benchmark",
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "basic_pitch_version": package_version("basic-pitch"),
        "coremltools_version": package_version("coremltools"),
        "data_dir": str(data_dir),
        "clip_count": len(clips),
        "backends": list(args.backends),
        "note_tolerance_seconds": args.note_tolerance,
        "tab_tolerance_seconds": args.tab_tolerance,
        "standard_e_open_midi": list(STANDARD_E_OPEN_MIDI),
        "max_fret": args.max_fret,
        "preprocessing": (
            "load_audio_bytes: librosa mono, 22050 Hz, peak-normalized"
        ),
        "historical_prediction_files_ignored": True,
    }

    details: JsonObject = {"metadata": metadata, "clips": {}}
    csv_rows: list[JsonObject] = []

    print("Checkpoint 4 pilot transcription evaluation")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Clips: {len(clips)}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 80)

    for clip in clips:
        name = str(clip["name"])
        audio_path = Path(clip["audio_path"])
        note_gt = validate_note_gt(load_json(Path(clip["note_gt_path"])), Path(clip["note_gt_path"]))
        tab_gt = validate_tab_gt(load_json(Path(clip["tab_gt_path"])), Path(clip["tab_gt_path"]))

        decode_start = time.perf_counter()
        waveform, sample_rate = load_audio_bytes(audio_path.read_bytes(), sr=22050)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        duration = len(waveform) / float(sample_rate)

        clip_result: JsonObject = {
            "audio": {
                "path": str(audio_path),
                "sha256": sha256_file(audio_path),
                "duration_seconds": rounded(duration),
                "sample_rate": sample_rate,
                "decode_and_preprocess_ms": rounded(decode_ms, 3),
            },
            "ground_truth": {
                "note_count": len(note_gt),
                "tab_count": len(tab_gt),
            },
            "backends": {},
        }

        print(f"\n{name}: {duration:.3f}s, {len(note_gt)} GT notes")

        for backend in args.backends:
            notes_start = time.perf_counter()
            predicted_notes, runtime = run_backend(backend, waveform, sample_rate)
            notes_ms = (time.perf_counter() - notes_start) * 1000.0
            predicted_notes = sorted(
                predicted_notes,
                key=lambda event: (
                    float(event["t_on"]),
                    int(event["midi"]),
                    float(event["t_off"]),
                ),
            )

            tabs_start = time.perf_counter()
            predicted_tabs = dp_tab_mapping(
                predicted_notes,
                open_midi=list(STANDARD_E_OPEN_MIDI),
                max_fret=args.max_fret,
            )
            tabs_ms = (time.perf_counter() - tabs_start) * 1000.0

            notes_result = note_metrics(
                note_gt,
                predicted_notes,
                args.note_tolerance,
            )
            tabs_result = tab_metrics(
                tab_gt,
                predicted_tabs,
                args.tab_tolerance,
            )
            highest_playable_midi = max(STANDARD_E_OPEN_MIDI) + args.max_fret
            unplayable_count = sum(
                int(event["midi"]) > highest_playable_midi
                for event in predicted_notes
            )

            prediction_path = output_dir / "predictions" / backend / f"{name}.json"
            write_json(
                prediction_path,
                {
                    "clip": name,
                    "backend": backend,
                    "runtime": runtime,
                    "audio_duration_seconds": rounded(duration),
                    "sample_rate": sample_rate,
                    "notes": predicted_notes,
                    "render": {"guitar_tabs": predicted_tabs},
                    "timing_ms": {
                        "decode_and_preprocess": rounded(decode_ms, 3),
                        "notes": rounded(notes_ms, 3),
                        "tabs": rounded(tabs_ms, 3),
                    },
                },
            )

            backend_result = {
                "runtime": runtime,
                "prediction_path": str(prediction_path),
                "note_metrics": notes_result,
                "tab_metrics": tabs_result,
                "timing_ms": {
                    "decode_and_preprocess": rounded(decode_ms, 3),
                    "notes": rounded(notes_ms, 3),
                    "tabs": rounded(tabs_ms, 3),
                },
                "notes_real_time_factor": rounded(
                    safe_div(notes_ms, duration * 1000.0)
                ),
                "highest_playable_midi": highest_playable_midi,
                "unplayable_prediction_count": unplayable_count,
            }
            clip_result["backends"][backend] = backend_result

            csv_rows.append(
                {
                    "clip": name,
                    "backend": backend,
                    "duration_seconds": rounded(duration),
                    "n_ground_truth_notes": notes_result["n_ground_truth"],
                    "n_predicted_notes": notes_result["n_predictions"],
                    "note_true_positives": notes_result["true_positives"],
                    "note_false_positives": notes_result["false_positives"],
                    "note_false_negatives": notes_result["false_negatives"],
                    "note_precision": notes_result["precision"],
                    "note_recall": notes_result["recall"],
                    "note_f1": notes_result["f1"],
                    "n_ground_truth_tabs": tabs_result["n_ground_truth"],
                    "n_predicted_tabs": tabs_result["n_predictions"],
                    "tab_onset_coverage": tabs_result["onset_coverage"],
                    "tab_string_accuracy_on_matched": tabs_result["string_accuracy_on_matched"],
                    "tab_fret_accuracy_on_matched": tabs_result["fret_accuracy_on_matched"],
                    "tab_exact_accuracy_on_matched": tabs_result["exact_accuracy_on_matched"],
                    "tab_exact_recall": tabs_result["exact_tab_recall"],
                    "decode_and_preprocess_ms": rounded(decode_ms, 3),
                    "notes_ms": rounded(notes_ms, 3),
                    "tabs_ms": rounded(tabs_ms, 3),
                    "notes_real_time_factor": rounded(
                        safe_div(notes_ms, duration * 1000.0)
                    ),
                    "unplayable_prediction_count": unplayable_count,
                }
            )

            print(
                f"  {backend:11s} notes={len(predicted_notes):3d} "
                f"P={float(notes_result['precision']):.3f} "
                f"R={float(notes_result['recall']):.3f} "
                f"F1={float(notes_result['f1']):.3f} "
                f"tab_exact_recall={float(tabs_result['exact_tab_recall']):.3f} "
                f"notes_ms={notes_ms:.2f}"
            )

        details["clips"][name] = clip_result

    aggregate: JsonObject = {}
    for backend in args.backends:
        backend_results = [
            details["clips"][name]["backends"][backend]
            for name in details["clips"]
        ]
        aggregate[backend] = {
            "notes": aggregate_notes(
                result["note_metrics"] for result in backend_results
            ),
            "tabs": aggregate_tabs(
                result["tab_metrics"] for result in backend_results
            ),
            "timing_warning": (
                "Single pilot observations only. The first Basic Pitch clip may "
                "include lazy model initialization. Use a repeated latency "
                "benchmark for final reporting."
            ),
        }

    details["aggregate"] = aggregate
    write_json(output_dir / "summary.json", details)

    csv_path = output_dir / "per_clip_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nAggregate pilot results")
    print("=" * 80)
    for backend in args.backends:
        notes = aggregate[backend]["notes"]
        tabs = aggregate[backend]["tabs"]
        print(
            f"{backend:11s} note_micro_F1={float(notes['micro_f1']):.3f} "
            f"note_macro_F1={float(notes['macro_f1']):.3f} "
            f"tab_exact_recall={float(tabs['exact_tab_recall']):.3f}"
        )

    print("=" * 80)
    print(f"Summary JSON: {(output_dir / 'summary.json').resolve()}")
    print(f"Per-clip CSV: {csv_path.resolve()}")
    print("Pilot only; do not present this as the final held-out thesis result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
