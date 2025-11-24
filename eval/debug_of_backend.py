# eval/debug_of_backend.py
import time
from pathlib import Path
import numpy as np
import librosa

from models.of_inference import OFTranscriber

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(m: int) -> str:
    if m < 0 or m > 127:
        return f"midi{m}"
    name = NOTE_NAMES[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"

def load_test_clip():
    """
    Try to load a real audio clip from the repo.
    Adjust the path if your clip lives somewhere else.
    """
    root = Path(__file__).parent.parent  # backend/
    # You can change this if your file is different:
    candidates = [
        root / "data" / "mini_eval" / "clip.wav",
        root / "assets" / "clip.wav",
    ]

    for c in candidates:
        if c.exists():
            print(f"[INFO] Using test audio: {c}")
            y, sr = librosa.load(str(c), sr=22050, mono=True)
            return y, sr

    # Fallback: synthetic noise (not ideal, but avoids crash)
    print("[WARN] No real clip found, falling back to synthetic noise.")
    sr = 22050
    dur_sec = 3.0
    y = np.zeros(int(sr * dur_sec), dtype=np.float32)
    y += 1e-4 * np.random.randn(len(y)).astype(np.float32)
    return y, sr

def main():
    print("=== Debug: OFTranscriber backend (real audio) ===")

    # ---- Load test audio ----
    y, sr = load_test_clip()
    dur_sec = len(y) / float(sr)
    print(f"[INFO] Audio duration: {dur_sec:.3f} s, sr={sr}")

    # ---- Init model ----
    t0 = time.perf_counter()
    tr = OFTranscriber(
        device="cpu",           # you can try "mps" later if stable
        checkpoint_path=None,   # we'll plug a real file later
        n_mels=229,
        hop_length=512,
        midi_low=21,
        midi_high=108,
    )
    t_init = (time.perf_counter() - t0) * 1000.0
    print(f"[INFO] Init time: {t_init:.1f} ms")

    # ---- Full-mode ----
    t1 = time.perf_counter()
    events_full = tr.transcribe(y, sr)
    t_full = (time.perf_counter() - t1) * 1000.0
    print(f"[INFO] Full-mode: {len(events_full)} events, {t_full:.1f} ms")

    # ---- Chunked-mode ----
    t2 = time.perf_counter()
    events_chunked = tr.transcribe_chunked(
        y, sr,
        chunk_sec=1.0,
        hop_sec=0.5,
        onset_filt=3,
        frame_filt=5,
        th_on_hi=0.55,
        th_on_lo=0.30,
        th_fr=0.50,
    )
    t_chunk = (time.perf_counter() - t2) * 1000.0
    print(f"[INFO] Chunked-mode: {len(events_chunked)} events, {t_chunk:.1f} ms")

    # ---- Quick sanity: compare first few events ----
    def pretty(ev):
        m = int(ev["midi"])
        return f"t={ev['t_on']:.3f}s  midi={m}  {midi_to_name(m)}"

    print("\n--- First 10 events (FULL) ---")
    for ev in events_full[:10]:
        print(pretty(ev))

    print("\n--- First 10 events (CHUNKED) ---")
    for ev in events_chunked[:10]:
        print(pretty(ev))

    print("\n=== Done debug ===")

if __name__ == "__main__":
    main()
