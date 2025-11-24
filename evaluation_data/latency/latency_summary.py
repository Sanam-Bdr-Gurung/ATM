import json
from pathlib import Path

LAT_DIR = Path(__file__).parent

def load_metrics(name: str):
    path = LAT_DIR / f"{name}_pred.json"
    with open(path, "r") as f:
        data = json.load(f)

    dur = data.get("audio_duration_sec", None)
    total = data.get("latency_ms", None)
    timing = data.get("timing_ms", {})

    return {
        "name": name,
        "audio_sec": dur,
        "total_ms": total,
        "io_ms": timing.get("io"),
        "notes_ms": timing.get("notes"),
        "tuning_ms": timing.get("tuning"),
        "chords_ms": timing.get("chords"),
        "tabs_ms": timing.get("tabs"),
    }

def main():
    clip_names = [
        "clip_3s",
        "clip_10s",
        "clip_30s",
        "clip_60s",
    ]

    rows = [load_metrics(n) for n in clip_names]

    print("\nLatency Summary:")
    print(f"{'Clip':<10} {'Dur(s)':>8} {'Total(ms)':>10} {'IO':>8} {'Notes':>8} {'Tuning':>8} {'Chords':>8} {'Tabs':>8}")
    for r in rows:
        print(
            f"{r['name']:<10} "
            f"{r['audio_sec']:>8.3f} "
            f"{r['total_ms']:>10.1f} "
            f"{r['io_ms']:>8.1f} "
            f"{r['notes_ms']:>8.1f} "
            f"{r['tuning_ms']:>8.1f} "
            f"{r['chords_ms']:>8.1f} "
            f"{r['tabs_ms']:>8.1f}"
        )

if __name__ == "__main__":
    main()
