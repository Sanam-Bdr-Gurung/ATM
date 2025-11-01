GUITAR_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
def map_note_to_string_fret(midi: int, open_midi, max_fret: int = 20):
    candidates = []
    for s, om in enumerate(open_midi):  # s=0..5
        fret = midi - om
        if 0 <= fret <= max_fret:
            candidates.append((s, fret))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[0]))  # prefer lower fret, thicker string
    s, f = candidates[0]
    return {"string": s+1, "fret": f}            # 1-indexed string