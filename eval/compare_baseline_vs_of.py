# eval/compare_baseline_vs_of.py

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np

from models.of_inference import OFTranscriber
from notes_baseline import detect_multi_pitch, frames_to_note_events


# --- Small, local postprocessing (copied from api.py, slightly simplified) ---

def postprocess_notes(
    note_events: List[Dict],
    midi_low: int = 40,     # E2
    midi_high: int = 88,    # ~E6
    min_dur: float = 0.10,  # seconds
    merge_gap: float = 0.03 # seconds
) -> List[Dict]:
    """Filter to guitar range, drop tiny blips, merge close same-pitch events."""
    if not note_events:
        return []

    filtered = []
    for ev in note_events:
        t_on = float(ev.get("t_on", 0.0))
        t_off = float(ev.get("t_off", t_on))
        dur = t_off - t_on
        midi = int(ev.get("midi", 0))

        if midi < midi_low or midi > midi_high:
            continue
        if dur < min_dur:
            continue

        filtered.append({
            **ev,
            "t_on": t_on,
            "t_off": t_off,
            "midi": midi,
        })

    if not filtered:
        return []

    filtered.sort(key=lambda e: (e["t_on"], e["midi"]))

    merged = [filtered[0]]
    for ev in filtered[1:]:
        last = merged[-1]
        same_pitch = (ev["midi"] == last["midi"])
        gap = ev["t_on"] - last["t_off"]

        if same_pitch and 0.0 <= gap <= merge_gap:
            last["t_off"] = max(last["t_off"], ev["t_off"])
            if "conf" in ev or "conf" in last:
                last_conf = float(last.get("conf", 0.0))
                ev_conf = float(ev.get("conf", 0.0))
                last["conf"] = max(last_conf, ev_conf)
        else:
            merged.append(ev)

    return merged


def collapse_harmonics(note_events: List[Dict], window: float = 0.08) -> List[Dict]:
    """In each small time window, keep only the lowest MIDI (good for monophonic-ish guitar)."""
    if not note_events:
        return []

    evs = sorted(note_events, key=lambda e: (float(e["t_on"]), int(e["midi"])))
    collapsed: List[Dict] = []
    cluster: List[Dict] = [evs[0]]

    def flush_cluster():
        if not cluster:
            return
        best = min(cluster, key=lambda e: int(e["midi"]))
        collapsed.append(best)

    for ev in evs[1:]:
        prev = cluster[-1]
        if (ev["t_on"] - prev["t_on"]) <= window:
            cluster.append(ev)
        else:
            flush_cluster()
            cluster = [ev]

    flush_cluster()
    return collapsed


# --- Core comparison logic ---

def run_baseline(y: np.ndarray, sr: int) -> List[Dict]:
    """Baseline pipeline: multi-pitch -> events -> guitar-ish cleanup."""
    mp = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
    events = frames_to_note_events(mp, min_dur=0.08)
    events = postprocess_notes(
        events,
        midi_low=40,
        midi_high=88,
        min_dur=0.10,
        merge_gap=0.03,
    )
    events = collapse_harmonics(events, window=0.08)
    return events


def run_of(y: np.ndarray, sr: int, checkpoint: str | None) -> List[Dict]:
    """OFTranscriber full-mode, using your of_inference backend."""
    tr = OFTranscriber(
        device="cpu",
        checkpoint_path=checkpoint,
        n_mels=229,
        hop_length=512,
        midi_low=21,
        midi_high=108,
    )
    events = tr.transcribe(y, sr)
    # Optional: restrict to guitar-ish range for fair comparison
    events = postprocess_notes(
        events,
        midi_low=40,
        midi_high=88,
        min_dur=0.05,
        merge_gap=0.02,
    )
    return events


def compare_on_file(
    audio_path: Path,
    out_dir: Path,
    checkpoint: str | None,
) -> Dict:
    """Run baseline + OF on a single file and return a small metrics dict."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    dur = len(y) / float(sr)

    # Baseline
    t0 = time.perf_counter()
    base_events = run_baseline(y, sr)
    t_base = (time.perf_counter() - t0) * 1000.0

    # OF
    t1 = time.perf_counter()
    of_events = run_of(y, sr, checkpoint)
    t_of = (time.perf_counter() - t1) * 1000.0

    # Save JSONs (for detailed inspection later)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_json = {
        "model": "baseline",
        "audio": str(audio_path),
        "sr": sr,
        "duration_sec": dur,
        "n_events": len(base_events),
        "events": base_events,
    }
    of_json = {
        "model": "of",
        "audio": str(audio_path),
        "sr": sr,
        "duration_sec": dur,
        "n_events": len(of_events),
        "events": of_events,
    }

    base_name = audio_path.stem
    with open(out_dir / f"{base_name}_baseline.json", "w") as f:
        json.dump(base_json, f, indent=2)
    with open(out_dir / f"{base_name}_of.json", "w") as f:
        json.dump(of_json, f, indent=2)

    return {
        "clip": base_name,
        "dur_sec": dur,
        "n_base": len(base_events),
        "n_of": len(of_events),
        "t_base_ms": t_base,
        "t_of_ms": t_of,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs OFTranscriber note predictions on one or more audio files."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to a single audio file OR a directory containing .wav files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to OFTranscriber checkpoint (.ckpt/.pth/.pt/.safetensors).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="evaluation_data/of_compare",
        help="Where to store JSON outputs.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)

    # Collect files
    if audio_path.is_dir():
        files = sorted(p for p in audio_path.iterdir() if p.suffix.lower() in [".wav", ".mp3", ".flac"])
    else:
        files = [audio_path]

    if not files:
        print(f"No audio files found at {audio_path}")
        return

    print("\nComparing baseline vs OFTranscriber")
    print(f"Checkpoint: {args.checkpoint or 'None (random weights)'}")
    print(f"Output dir: {out_dir}")
    print("-" * 72)
    print(f"{'Clip':12} {'Dur(s)':>7} {'N_base':>7} {'N_of':>7} {'t_base(ms)':>11} {'t_of(ms)':>11}")
    print("-" * 72)

    rows = []
    for f in files:
        stats = compare_on_file(f, out_dir, args.checkpoint)
        rows.append(stats)
        print(
            f"{stats['clip'][:12]:12} "
            f"{stats['dur_sec']:7.3f} "
            f"{stats['n_base']:7d} "
            f"{stats['n_of']:7d} "
            f"{stats['t_base_ms']:11.1f} "
            f"{stats['t_of_ms']:11.1f}"
        )

    # Optional mean summary
    if len(rows) > 1:
        mean_dur = sum(r["dur_sec"] for r in rows) / len(rows)
        mean_n_base = sum(r["n_base"] for r in rows) / len(rows)
        mean_n_of = sum(r["n_of"] for r in rows) / len(rows)
        mean_t_base = sum(r["t_base_ms"] for r in rows) / len(rows)
        mean_t_of = sum(r["t_of_ms"] for r in rows) / len(rows)
        print("-" * 72)
        print(
            f"{'MEAN':12} "
            f"{mean_dur:7.3f} "
            f"{mean_n_base:7.1f} "
            f"{mean_n_of:7.1f} "
            f"{mean_t_base:11.1f} "
            f"{mean_t_of:11.1f}"
        )


if __name__ == "__main__":
    main()
