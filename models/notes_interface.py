# models/notes_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np

class NotesTranscriber(ABC):
    """
    Stable interface for any notes model: given (y, sr) -> list of note events.
    Each event: {t_on, t_off, midi, freq_hz, conf}
    """

    @abstractmethod
    def transcribe(self, y: np.ndarray, sr: int) -> List[Dict]:
        ...

def midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))
