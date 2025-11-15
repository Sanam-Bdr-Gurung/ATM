import os, sys, glob, numpy as np, itertools
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo)

from models.of_inference import OFTranscriber
import soundfile as sf

from onset_f1 import load_onsets_txt, onsets_from_events, f1_onsets

def run_sweep(audio_dir, ckpt=None, mode="chunked"):
    wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    tr = OFTranscriber(device="cpu", checkpoint_path=ckpt, n_mels=128, hop_length=1024)

    # small grid; you can expand later
    grid = {
        "th_on_hi": [0.5, 0.55, 0.6],
        "th_on_lo": [0.25, 0.3, 0.35],
        "th_fr":    [0.45, 0.5, 0.55],
    }

    best = (-1, None)  # (F1, params)
    for th_on_hi, th_on_lo, th_fr in itertools.product(*grid.values()):
        f1s = []
        for wav in wavs:
            y, sr = sf.read(wav);  y = y.mean(axis=1) if y.ndim>1 else y
            if mode == "chunked":
                ev = tr.transcribe_chunked(y, sr, chunk_sec=1.0, hop_sec=0.5,
                                           th_on_hi=th_on_hi, th_on_lo=th_on_lo, th_fr=th_fr)
            else:
                ev = tr.transcribe(y, sr)  # add params similarly if you modified transcribe
            hyp = onsets_from_events(ev)
            ref = load_onsets_txt(wav.replace(".wav", ".onsets.txt"))
            _, _, F1 = f1_onsets(ref, hyp, tol=0.05)
            f1s.append(F1)
        mean_f1 = np.mean(f1s) if f1s else 0.0
        params = dict(th_on_hi=th_on_hi, th_on_lo=th_on_lo, th_fr=th_fr)
        print("params", params, "meanF1", round(mean_f1,3))
        if mean_f1 > best[0]:
            best = (mean_f1, params)
    print("BEST:", best)

if __name__ == "__main__":
    run_sweep(audio_dir="backend/data/mini_eval", ckpt="backend/checkpoints/of_pretrained.pth", mode="chunked")
