from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class ChromaFeatures:
    chroma: np.ndarray
    times: np.ndarray
    frame_activity: np.ndarray
    hop_length: int
    frame_length: int


def _validate_audio(
    y: np.ndarray,
    sr: int,
) -> np.ndarray:
    if sr <= 0:
        raise ValueError(
            "Audio sample rate must be positive."
        )

    audio = np.asarray(
        y,
        dtype=np.float32,
    )

    audio = np.squeeze(audio)

    if audio.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional mono waveform."
        )

    if audio.size == 0:
        raise ValueError(
            "Cannot extract chroma from empty audio."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Audio contains NaN or infinite values."
        )

    return np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )


def chroma_from_audio(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 1024,
    frame_length: int = 4096,
    harmonic_margin: float = 3.0,
) -> ChromaFeatures:
    """
    Extract traditional chroma evidence and frame activity.

    Harmonic-percussive separation is used only to emphasize pitched
    harmonic content. This does not perform instrument source separation.
    """
    if hop_length <= 0:
        raise ValueError(
            "hop_length must be positive."
        )

    if frame_length <= 0:
        raise ValueError(
            "frame_length must be positive."
        )

    if harmonic_margin <= 0.0:
        raise ValueError(
            "harmonic_margin must be positive."
        )

    audio = _validate_audio(
        y,
        sr,
    )

    harmonic_audio = librosa.effects.harmonic(
        audio,
        margin=harmonic_margin,
    )

    spectrum = np.abs(
        librosa.stft(
            harmonic_audio,
            n_fft=frame_length,
            hop_length=hop_length,
            center=True,
        )
    )

    chroma = librosa.feature.chroma_stft(
        S=spectrum,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        norm=2,
    )

    frame_activity = librosa.feature.rms(
        y=harmonic_audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    frame_count = min(
        chroma.shape[1],
        frame_activity.size,
    )

    chroma = np.asarray(
        chroma[:, :frame_count],
        dtype=np.float64,
    )

    frame_activity = np.asarray(
        frame_activity[:frame_count],
        dtype=np.float64,
    )

    times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=sr,
        hop_length=hop_length,
    )

    return ChromaFeatures(
        chroma=chroma,
        times=np.asarray(
            times,
            dtype=np.float64,
        ),
        frame_activity=frame_activity,
        hop_length=hop_length,
        frame_length=frame_length,
    )
