from typing import List, Dict
import numpy as np, librosa

# Converts a frequency (Hz) to its nearest MIDI note number (integer).
def hz_to_midi_safe(hz: float) -> int:
    return int(round(librosa.hz_to_midi(hz)))

# -----------------------------------------------
# MULTI-PITCH DETECTION FUNCTION
# -----------------------------------------------
def detect_multi_pitch(y, sr, hop_length=1024, top_k=3) -> Dict:
    """
    Detects multiple dominant pitches in each time frame from an audio signal.

    Args:
        y: Audio waveform (numpy array)
        sr: Sample rate (Hz)
        hop_length: Frame hop size for STFT (controls time resolution)
        top_k: Number of dominant pitches to detect per frame

    Returns:
        Dict with:
            - times: time stamps for each frame
            - pitches_hz_frames: list of lists containing top frequencies (Hz) per frame
    """
    # Compute the magnitude spectrogram (STFT)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop_length))

    # Frequency values for each FFT bin
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # Time (in seconds) for each frame
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    pitches_hz_frames: List[List[float]] = []

    # Iterate through each time frame in the spectrogram
    for t in range(S.shape[1]):
        mags = S[:, t]  # Magnitude spectrum at frame t

        # Get indices of top energy bins (oversampled to prune later)
        idx = np.argsort(mags)[::-1][:top_k * 5]

        cand = []
        for i in idx:
            hz = freqs[i]
            if hz < 55:  # Ignore sub-bass / low frequencies (noise)
                continue
            cand.append((hz, mags[i]))  # (frequency, magnitude)

        picked, used = [], []

        # Select top_k distinct pitches while avoiding near-duplicates
        for hz, m in sorted(cand, key=lambda x: x[1], reverse=True):
            # Skip if too close (within a quarter tone) to an already used pitch
            if any(abs(np.log2(hz/u)) < (1/24) for u in used):
                continue
            used.append(hz)
            picked.append(hz)
            if len(picked) >= top_k:
                break

        pitches_hz_frames.append(picked)

    return {"times": times, "pitches_hz_frames": pitches_hz_frames}

# -----------------------------------------------
# NOTE EVENT EXTRACTION FUNCTION
# -----------------------------------------------
def frames_to_note_events(multi_pitch, min_dur=0.08):
    """
    Converts per-frame multi-pitch data into discrete note events (onset/offset).

    Args:
        multi_pitch: dict from detect_multi_pitch()
        min_dur: Minimum note duration (in seconds) to keep valid notes

    Returns:
        List of dicts containing note start/end times, MIDI number, confidence, and frequency (Hz)
    """
    times = multi_pitch["times"]
    frames = multi_pitch["pitches_hz_frames"]

    active = {}   # Tracks currently active notes {midi: (start_frame, frame_count)}
    events = []   # Final list of detected notes

    # Iterate through frames
    for fi, phz in enumerate(frames):
        # Convert current frame’s detected frequencies to MIDI notes
        midi_set = set(hz_to_midi_safe(h) for h in phz)

        # Notes that ended (were active but not in current frame)
        to_close = [m for m in list(active.keys()) if m not in midi_set]
        for m in to_close:
            sfi, cnt = active.pop(m)
            t_on, t_off = times[sfi], times[fi]
            if (t_off - t_on) >= min_dur:
                # Add note event with confidence = frame count seen
                events.append({
                    "t_on": float(t_on),
                    "t_off": float(t_off),
                    "midi": int(m),
                    "conf": float(cnt)
                })

        # Update active notes (increment frame count or start new note)
        for m in midi_set:
            if m in active:
                sfi, cnt = active[m]
                active[m] = (sfi, cnt + 1)
            else:
                active[m] = (fi, 1)

    # Handle notes still active at the end of audio
    if frames:
        fi_last = len(frames) - 1
        for m, (sfi, cnt) in active.items():
            t_on, t_off = times[sfi], times[fi_last]
            if (t_off - t_on) >= min_dur:
                events.append({
                    "t_on": float(t_on),
                    "t_off": float(t_off),
                    "midi": int(m),
                    "conf": float(cnt)
                })

    # Sort by onset time
    events.sort(key=lambda e: e["t_on"])

    # Add frequency in Hz for each event
    for e in events:
        e["freq_hz"] = float(librosa.midi_to_hz(e["midi"]))

    return events
