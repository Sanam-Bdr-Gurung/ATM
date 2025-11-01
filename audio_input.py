from typing import Tuple
import io
import numpy as np
import librosa

def load_audio_bytes(data: bytes, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load mono audio from bytes, resampled to 'sr', normalized to [-1, 1].
    Returns:
        y (np.ndarray): waveform samples as float32 array
        sr (int): actual sample rate used
    """
    # Wrap the raw byte data in a file-like buffer so librosa can read it
    bio = io.BytesIO(data)
    
    # Load the audio:
    #   - 'sr': target sample rate (resamples if different)
    #   - 'mono=True': convert stereo -> mono
    #   - returns waveform 'y' and used sample rate 'sr'
    y, sr = librosa.load(bio, sr=sr, mono=True)
    
    # Compute the max absolute amplitude (to normalize)
    # Add a tiny epsilon (1e-9) to avoid division by zero
    m = float(np.max(np.abs(y)) + 1e-9)
    
    # Normalize waveform so its values lie in [-1, 1]
    # This keeps loudness consistent between different files
    y = y / m
    
    # Return normalized audio and sample rate
    return y, sr
