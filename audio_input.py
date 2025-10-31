from typing import Tuple
import io, numpy as np, librosa

def load_audio_bytes(data: bytes, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load mono audio from bytes, resampled to sr, normalized to [-1,1].
    """
    bio = io.BytesIO(data)
    y, sr = librosa.load(bio, sr=sr, mono=True)
    m = float(np.max(np.abs(y)) + 1e-9)
    return (y / m), sr
