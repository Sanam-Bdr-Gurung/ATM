import numpy as np
from chord_templates import CHORD_TEMPLATES, CHORD_NAMES

def match_chords(chroma):
    """
    Match chroma frames (12-note energy vectors) to the closest chord template.

    Args:
        chroma: np.ndarray of shape (12, T)
                Each column is a chroma vector for a time frame (12 pitch classes × T frames).

    Returns:
        A list of predicted chord names (length = T), one for each time frame.
    """

    # Step 1️⃣: Normalize each chroma column (frame) to unit length.
    # This removes differences in overall loudness so only the *relative* pitch
    # distribution matters when matching to chords.
    denom = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9  # avoid division by zero
    norm = chroma / denom  # normalized chroma (12 × T)

    # Step 2️⃣: Compute similarity between each frame and every chord template.
    # CHORD_TEMPLATES → shape (24, 12)  → 24 chord profiles (12 major + 12 minor)
    # norm → shape (12, T)
    # Matrix multiplication gives → (24, T): similarity scores for each chord vs each frame
    scores = CHORD_TEMPLATES @ norm

    # Step 3️⃣: Find the index of the best-matching chord for each frame
    idx = scores.argmax(axis=0)

    # Step 4️⃣: Convert chord indices to their names (like "C", "Am", "F#")
    return [CHORD_NAMES[i] for i in idx]
