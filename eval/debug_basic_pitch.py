from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Final

import librosa

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict


GUITAR_LOW_MIDI: Final[int] = 40   # E2
GUITAR_HIGH_MIDI: Final[int] = 88  # E6

NOTE_NAMES: Final[list[str]] = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


def midi_to_hz(midi: int) -> float:
    """Convert a MIDI note number to frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_name(midi: int) -> str:
    """Convert a MIDI note number to a readable note name."""
    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{note_name}{octave}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Basic Pitch on one audio file without using FastAPI."
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to a WAV, MP3, FLAC, OGG, or M4A guitar recording.",
    )
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()

    if not audio_path.exists():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    if not audio_path.is_file():
        raise SystemExit(f"Audio path is not a file: {audio_path}")

    print("=== Basic Pitch standalone test ===")
    print(f"Audio: {audio_path}")
    print(f"Model path: {ICASSP_2022_MODEL_PATH}")

    try:
        audio_duration = float(
            librosa.get_duration(path=str(audio_path))
        )
    except Exception as exc:
        print(f"[WARN] Could not read audio duration: {exc}")
        audio_duration = 0.0

    print(f"Audio duration: {audio_duration:.3f} seconds")

    model_start = time.perf_counter()

    try:
        model = Model(ICASSP_2022_MODEL_PATH)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load the Basic Pitch model: {exc}"
        ) from exc

    model_load_seconds = time.perf_counter() - model_start

    print(f"Runtime: {model.model_type}")
    print(f"Model load time: {model_load_seconds:.3f} seconds")

    inference_start = time.perf_counter()

    try:
        _, _, note_events = predict(
            audio_path,
            model,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=80.0,
            minimum_frequency=midi_to_hz(GUITAR_LOW_MIDI),
            maximum_frequency=midi_to_hz(GUITAR_HIGH_MIDI),
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"Basic Pitch inference failed: {exc}"
        ) from exc

    inference_seconds = time.perf_counter() - inference_start

    print(f"Inference time: {inference_seconds:.3f} seconds")

    if audio_duration > 0:
        real_time_factor = inference_seconds / audio_duration
        print(f"Real-time factor: {real_time_factor:.3f}x")

    print(f"Detected note events: {len(note_events)}")

    print("\nFirst 30 note events:")
    print("-" * 76)

    for index, event in enumerate(note_events[:30], start=1):
        start_time, end_time, midi, amplitude, pitch_bend = event

        midi = int(midi)
        duration = float(end_time) - float(start_time)

        print(
            f"{index:02d}. "
            f"{midi_to_name(midi):4s} "
            f"MIDI={midi:3d} "
            f"start={float(start_time):7.3f}s "
            f"end={float(end_time):7.3f}s "
            f"duration={duration:6.3f}s "
            f"confidence={float(amplitude):.3f}"
        )

    print("-" * 76)

    if not note_events:
        print(
            "[WARN] Inference completed, but no notes were detected. "
            "We may need to lower the thresholds or inspect the audio."
        )
    else:
        print("Basic Pitch inference completed successfully.")


if __name__ == "__main__":
    main()