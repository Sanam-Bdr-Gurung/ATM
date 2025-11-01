import numpy as np
import librosa

def chroma_from_audio(y, sr, hop_length=1024):
    """
    Compute a chromagram (12-note pitch energy representation) from an audio signal.

    Args:
        y: Audio waveform as a 1D NumPy array.
        sr: Sample rate of the audio (samples per second).
        hop_length: Step size (in samples) between consecutive STFT frames.
                    Controls time resolution — smaller = finer time detail.

    Returns:
        chroma: 12 x T matrix showing pitch class (C, C#, D, …, B) intensity over time.
        times:  1D array mapping each frame to its time in seconds.
    """

    # Step 1️⃣: Perform a Short-Time Fourier Transform (STFT)
    # Breaks the audio into overlapping chunks and computes frequency content for each.
    # np.abs() keeps only the magnitude (energy), discarding phase information.
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop_length))

    # Step 2️⃣: Compute the chromagram from the STFT magnitude
    # Maps the frequencies into 12 chroma bins (one for each musical note class).
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    # Step 3️⃣: Convert frame indices to actual time in seconds
    # This lets you plot or align chroma values along a real timeline.
    times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Step 4️⃣: Return both the chroma features and corresponding times
    return chroma, times
