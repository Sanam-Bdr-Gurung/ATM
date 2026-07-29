#!/usr/bin/env python3
"""Development-only Basic Pitch threshold sweep for ChordAssist."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for p in (ROOT, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audio_input import load_audio_bytes
from evaluate_transcription import (
    aggregate_notes,
    load_json,
    note_metrics,
    validate_note_gt,
)
from models.basic_pitch_inference import BasicPitchTranscriber


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(name)
    except PackageNotFoundError:
        return None


def discover_clips(data_dir: Path) -> list[dict[str, Any]]:
    clips = []
    for wav in sorted(data_dir.glob("*.wav")):
        gt_path = data_dir / f"{wav.stem}.json"
        if not gt_path.exists():
            print(f"[WARN] skipping {wav.name}: missing {gt_path.name}")
            continue
        gt = validate_note_gt(load_json(gt_path), gt_path)
        y, sr = load_audio_bytes(wav.read_bytes(), sr=22050)
        clips.append({"name": wav.stem, "gt": gt, "y": y, "sr": sr})
    return clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("evaluation_data/accuracy"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation_results/checkpoint4_threshold_sweep"),
    )
    ap.add_argument("--note-tolerance", type=float, default=0.05)
    ap.add_argument("--onset-thresholds", nargs="+", type=float,
                    default=[0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--frame-thresholds", nargs="+", type=float,
                    default=[0.20, 0.30, 0.40])
    ap.add_argument("--minimum-note-lengths-ms", nargs="+", type=float,
                    default=[60.0, 80.0, 100.0, 120.0])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Missing data directory: {data_dir}")

    clips = discover_clips(data_dir)
    if not clips:
        raise SystemExit("No annotated WAV clips found.")

    configs = list(itertools.product(
        sorted(set(args.onset_thresholds)),
        sorted(set(args.frame_thresholds)),
        sorted(set(args.minimum_note_lengths_ms)),
    ))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / stamp
    out_dir.mkdir(parents=True, exist_ok=False)

    print("Checkpoint 4 Basic Pitch threshold sweep")
    print("=" * 88)
    print(f"Development clips: {len(clips)}")
    print(f"Ground-truth notes: {sum(len(c['gt']) for c in clips)}")
    print(f"Configurations: {len(configs)}")
    print(f"Onset tolerance: {args.note_tolerance * 1000:.0f} ms")
    print(f"Output: {out_dir.resolve()}")
    print("=" * 88)

    transcriber = BasicPitchTranscriber(
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length_ms=80.0,
    )
    transcriber.transcribe(clips[0]["y"], clips[0]["sr"])

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for i, (onset, frame, min_ms) in enumerate(configs, start=1):
        transcriber.onset_threshold = onset
        transcriber.frame_threshold = frame
        transcriber.minimum_note_length_ms = min_ms

        clip_results = []
        timings = []
        for clip in clips:
            started = time.perf_counter()
            pred = transcriber.transcribe(clip["y"], clip["sr"])
            timings.append((time.perf_counter() - started) * 1000.0)
            metrics = note_metrics(
                clip["gt"], pred, tolerance_seconds=args.note_tolerance
            )
            metrics["clip"] = clip["name"]
            clip_results.append(metrics)

        agg = aggregate_notes(clip_results)
        row = {
            "onset_threshold": onset,
            "frame_threshold": frame,
            "minimum_note_length_ms": min_ms,
            "n_ground_truth": agg["n_ground_truth"],
            "n_predictions": agg["n_predictions"],
            "true_positives": agg["true_positives"],
            "false_positives": agg["false_positives"],
            "false_negatives": agg["false_negatives"],
            "micro_precision": agg["micro_precision"],
            "micro_recall": agg["micro_recall"],
            "micro_f1": agg["micro_f1"],
            "macro_f1": agg["macro_f1"],
            "mean_inference_ms": round(statistics.fmean(timings), 3),
            "median_inference_ms": round(statistics.median(timings), 3),
        }
        rows.append(row)
        details.append({"parameters": {"onset_threshold": onset,
                                        "frame_threshold": frame,
                                        "minimum_note_length_ms": min_ms},
                        "aggregate": agg,
                        "per_clip": clip_results})
        print(
            f"[{i:02d}/{len(configs):02d}] onset={onset:.2f} frame={frame:.2f} "
            f"min_ms={min_ms:.0f} P={float(agg['micro_precision']):.3f} "
            f"R={float(agg['micro_recall']):.3f} F1={float(agg['micro_f1']):.3f}"
        )

    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["micro_f1"]),
            -float(r["macro_f1"]),
            -float(r["micro_precision"]),
            int(r["n_predictions"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    default = next(
        r for r in ranked
        if r["onset_threshold"] == 0.5
        and r["frame_threshold"] == 0.3
        and r["minimum_note_length_ms"] == 80.0
    )
    best = ranked[0]

    csv_path = out_dir / "all_configurations.csv"
    fields = ["rank", "onset_threshold", "frame_threshold",
              "minimum_note_length_ms", "n_ground_truth", "n_predictions",
              "true_positives", "false_positives", "false_negatives",
              "micro_precision", "micro_recall", "micro_f1", "macro_f1",
              "mean_inference_ms", "median_inference_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)

    summary = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Development-set threshold sweep; not final held-out result",
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "basic_pitch_version": package_version("basic-pitch"),
            "coremltools_version": package_version("coremltools"),
            "runtime": transcriber.runtime_name,
            "clip_names": [c["name"] for c in clips],
            "ground_truth_note_count": sum(len(c["gt"]) for c in clips),
            "note_tolerance_seconds": args.note_tolerance,
            "configuration_count": len(configs),
            "ranking_rule": "micro F1, macro F1, precision, fewer predictions",
        },
        "default_result": default,
        "best_development_result": best,
        "top_configurations": ranked[:max(1, args.top)],
        "details": details,
        "warning": "Lock selected parameters before testing new held-out clips.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\nTop development configurations")
    print("=" * 88)
    print(f"{'Rank':>4} {'Onset':>7} {'Frame':>7} {'MinMs':>7} {'Pred':>6} "
          f"{'P':>7} {'R':>7} {'MicroF1':>8} {'MacroF1':>8}")
    print("-" * 88)
    for row in ranked[:max(1, args.top)]:
        print(f"{row['rank']:4d} {row['onset_threshold']:7.2f} "
              f"{row['frame_threshold']:7.2f} "
              f"{row['minimum_note_length_ms']:7.0f} "
              f"{row['n_predictions']:6d} "
              f"{float(row['micro_precision']):7.3f} "
              f"{float(row['micro_recall']):7.3f} "
              f"{float(row['micro_f1']):8.3f} "
              f"{float(row['macro_f1']):8.3f}")
    print("=" * 88)
    print(f"Current defaults: rank={default['rank']}, "
          f"micro_F1={float(default['micro_f1']):.3f}, "
          f"macro_F1={float(default['macro_f1']):.3f}")
    print(f"Best development configuration: onset={best['onset_threshold']:.2f}, "
          f"frame={best['frame_threshold']:.2f}, "
          f"min_ms={best['minimum_note_length_ms']:.0f}, "
          f"micro_F1={float(best['micro_f1']):.3f}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print(f"All configurations CSV: {csv_path.resolve()}")
    print("Development only; do not report as the final held-out thesis result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
