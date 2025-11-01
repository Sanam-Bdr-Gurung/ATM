import numpy as np

# Define note names for the 12 pitch classes in Western music
NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()

# Tonal profiles (templates) for major and minor chords from music theory.
# Each value roughly represents how "strongly" each of the 12 pitch classes
# contributes to that chord type (based on perceptual experiments).
MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], float)
MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], float)

# These lists will hold the final chord labels and their corresponding templates.
CHORD_NAMES, CHORD_TEMPLATES = [], []

# Generate templates for all 12 possible root notes (C → B)
# For each root, create both a major and a minor chord profile.
for r in range(12):
    # Add chord names like "C", "Cm", "C#", "C#m", ...
    CHORD_NAMES += [f"{NOTE_NAMES[r]}", f"{NOTE_NAMES[r]}m"]
    
    # np.roll() circularly shifts the major/minor profile so that each root
    # corresponds to the right pitch class (e.g., shifting for D major vs C major).
    CHORD_TEMPLATES += [np.roll(MAJOR, r), np.roll(MINOR, r)]

# Normalize each chord vector so that its length (L2 norm) = 1.
# This makes chord similarity comparisons (e.g., via cosine similarity) scale-invariant.
CHORD_TEMPLATES = [v / (np.linalg.norm(v) + 1e-9) for v in CHORD_TEMPLATES]

# Stack all 24 templates (12 major + 12 minor) into a single 2D NumPy array: shape (24, 12)
CHORD_TEMPLATES = np.stack(CHORD_TEMPLATES, axis=0)  # (24 chords, 12 pitch classes)
