# eval/onset_f1.py
"""
Minimal onset F1:
- Reference: a .txt file per audio, one onset (seconds) per line
  e.g., data/mini_eval/clip1.onsets.txt
- Predicted: we call your API locally or import the transcriber directly.
"""

import json, os, glob
import numpy as np

def load_onsets_txt(path):
    if not os.path.exists(path): return []
    with open(path, "r") as f:
        vals = []
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                vals.append(float(line))
            except: pass
    return sorted(vals)

def onsets_from_events(events):
    # we use note onsets; de-duplicate near hits
    ons = sorted([e["t_on"] for e in events])
    return ons

def f1_onsets(ref, hyp, tol=0.05):
    """ greedy matching within ±tol seconds """
    ref = list(ref); hyp = list(hyp)
    matched = 0
    j = 0
    used = set()
    for r in ref:
        best = None; best_d = tol+1
        for i, h in enumerate(hyp):
            if i in used: continue
            d = abs(h - r)
            if d <= tol and d < best_d:
                best_d, best = d, i
        if best is not None:
            used.add(best)
            matched += 1
    P = matched / max(1, len(hyp))
    R = matched / max(1, len(ref))
    if P+R == 0: return 0.0, 0.0, 0.0
    F1 = 2*P*R/(P+R)
    return P, R, F1

# Example usage (direct import of your transcriber):
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add backend path
    from models.of_inference import OFTranscriber
    import soundfile as sf

    audio_dir = "data/mini_eval"
    files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    tr = OFTranscriber(device="cpu", checkpoint_path=None)

    F1s = []
    for wav in files:
        y, sr = sf.read(wav)
        if y.ndim > 1: y = np.mean(y, axis=1)
        # use chunked path to mirror API default
        events = tr.transcribe_chunked(y, sr, chunk_sec=2.0, hop_sec=1.0)
        hyp = onsets_from_events(events)
        ref = load_onsets_txt(wav.replace(".wav", ".onsets.txt"))
        P, R, F1 = f1_onsets(ref, hyp, tol=0.05)
        print(os.path.basename(wav), f"P={P:.3f} R={R:.3f} F1={F1:.3f}")
        F1s.append(F1)
    if F1s:
        print("Mean F1:", sum(F1s)/len(F1s))
