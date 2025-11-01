# tuning/detect.py
from typing import Dict, List, Tuple
import numpy as np

# Base Standard: E2 A2 D3 G3 B3 E4 -> MIDI: 40,45,50,55,59,64
STANDARD = [40, 45, 50, 55, 59, 64]

CANDIDATE_TUNINGS: Dict[str, List[int]] = {
    "Standard E": STANDARD,
    "Drop D":     [38, 45, 50, 55, 59, 64],           # low E -> D
    "Eb (half-step down)": [m-1 for m in STANDARD],
    "D Standard (whole-step down)": [m-2 for m in STANDARD],
}

def score_tuning(open_midi: List[int], note_midis: List[int], max_fret: int = 20) -> float:
    """
    Score by fraction of notes that can map to SOME string within [0..max_fret].
    Simple, fast, and often enough for reliable classification.
    """
    if not note_midis:
        return 0.0
    feasible = 0
    for m in note_midis:
        ok = False
        for om in open_midi:
            fret = m - om
            if 0 <= fret <= max_fret:
                ok = True
                break
        feasible += 1 if ok else 0
    return feasible / max(1, len(note_midis))

def detect_tuning_class(note_events: List[Dict], max_fret: int = 20) -> Dict:
    """
    Pick the tuning whose open strings explain the largest fraction of detected notes.
    Returns {name, string_open_midi, confidence}.
    """
    # pull MIDI notes (unique helps stability)
    mids = [ev["midi"] for ev in note_events]
    if not mids:
        return {"name": "Standard E", "string_open_midi": STANDARD, "confidence": 0.0}

    # score all candidates
    names, scores = [], []
    for name, open_m in CANDIDATE_TUNINGS.items():
        names.append(name)
        scores.append(score_tuning(open_m, mids, max_fret=max_fret))

    scores = np.array(scores, dtype=float)
    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]
    best_open = CANDIDATE_TUNINGS[best_name]
    # confidence: normalize to [0,1] by gap to second-best
    if len(scores) > 1:
        sorted_scores = np.sort(scores)
        gap = float(sorted_scores[-1] - sorted_scores[-2])
        conf = float(min(1.0, sorted_scores[-1])) * (0.5 + 0.5 * min(1.0, 2.0*gap))
    else:
        conf = float(scores[best_idx])

    return {"name": best_name, "string_open_midi": best_open, "confidence": conf}
