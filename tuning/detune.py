# tuning/detune.py
import numpy as np
import librosa

def estimate_concert_detune_cents_from_frames(pitches_hz_frames, max_samples=4000):
    """
    Estimate global detune (cents) using raw frame-wise peak frequencies.
    Positive => sharp, negative => flat. Robust median across many peaks.
    """
    cents_residuals = []
    count = 0
    for phz in pitches_hz_frames:
        for hz in phz:
            if hz <= 0: 
                continue
            midi_f = librosa.hz_to_midi(hz)          # float midi
            nearest = round(midi_f)                   # nearest semitone
            cents = 100.0 * (midi_f - nearest)        # residual from equal temperament
            if abs(cents) <= 50:                      # ignore octave/wrong-peak outliers
                cents_residuals.append(cents)
                count += 1
                if count >= max_samples:
                    break
        if count >= max_samples:
            break
    if not cents_residuals:
        return 0.0
    # robust central tendency
    return float(np.median(cents_residuals))
