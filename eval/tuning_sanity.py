# eval/tuning_sanity.py
import argparse
import os
from pathlib import Path

import librosa
import numpy as np

from notes_baseline import detect_multi_pitch, frames_to_note_events
from tuning.detune import estimate_concert_detune_cents_from_frames
from tuning.detect import detect_tuning_class

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(m: int) -> str:
    if m < 0 or m > 127:
        return f"midi{m}"
    name = NOTE_NAMES[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"

def analyze_file(path: Path, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / float(sr)

    # baseline multi-pitch
    mp = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
    note_events = frames_to_note_events(mp, min_dur=0.08)

    # concert detune
    detune_cents = estimate_concert_detune_cents_from_frames(
        mp["pitches_hz_frames"]
    )

    # tuning class from note events
    tuning_info = detect_tuning_class(note_events, max_fret=20)

    # pretty print
    name = tuning_info.get("name", "Unknown")
    opens = tuning_info.get("string_open_midi", [])
    conf  = float(tuning_info.get("confidence", 0.0))

    opens_named = [f"{midi_to_name(m)}({m})" for m in opens]

    return {
        "duration": dur,
        "tuning_name": name,
        "string_opens": opens_named,
        "detune_cents": float(detune_cents),
        "confidence": conf,
        "n_notes": len(note_events),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Single audio file or directory of audio files."
    )
    args = ap.parse_args()

    p = Path(args.audio)
    if p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]])
    else:
        files = [p]

    print("Tuning Sanity Evaluation")
    print("-" * 72)
    print(f"{'Clip':12s} {'Dur(s)':>7s}  {'Tuning':20s} {'Detune(c)':>10s} {'Conf':>7s}  Notes")
    print("-" * 72)

    for f in files:
        try:
            res = analyze_file(f)
            print(f"{f.stem:12s} "
                  f"{res['duration']:7.3f}  "
                  f"{res['tuning_name'][:20]:20s} "
                  f"{res['detune_cents']:10.2f} "
                  f"{res['confidence']:7.3f}  "
                  f"{res['n_notes']:5d}")
            # You can uncomment this if you want to see open strings:
            # print("   Opens:", ", ".join(res["string_opens"]))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    print("-" * 72)

if __name__ == "__main__":
    main()
