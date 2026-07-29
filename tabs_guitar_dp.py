# backend/tabs_guitar_dp.py

from typing import List, Dict


def dp_tab_mapping(
    note_events: List[Dict],
    open_midi: List[int],
    max_fret: int = 20,
):
    """
    Dynamic-programming based string/fret assignment.

    note_events: list of { "t_on": float, "t_off": float, "midi": int, ... }
    open_midi:   list of 6 MIDI values for open strings, low → high
                 e.g. [40,45,50,55,59,64] for Standard E

    Returns: list of { "t_on": float, "string": int, "fret": int }
             where string is 1..6 (to match your current JSON).
    """

    if not note_events or not open_midi:
        return []

    # ---- Hyperparameters (tweak later if needed) ----
    FRET_JUMP_PEN = 0.4       # cost per fret difference between notes
    STRING_CHANGE_PEN = 1.0   # penalty when changing strings
    HIGH_FRET_PEN = 0.2       # extra cost if fret >= 7
    OPEN_BONUS = 0.3          # reward (negative cost) for fret == 0

    # Ensure events are sorted by time
    evs = sorted(note_events, key=lambda e: float(e["t_on"]))

    # ---- For each note, find possible positions ----
    all_positions = []  # list of list of candidates per note
    for ev in evs:
        midi = int(ev["midi"])
        cands = []
        for s_idx, open_note in enumerate(open_midi):  # 0..5, low to high
            fret = midi - open_note
            if 0 <= fret <= max_fret:
                base_cost = 0.0
                if fret >= 7:
                    base_cost += HIGH_FRET_PEN
                if fret == 0:
                    base_cost -= OPEN_BONUS
                cands.append((s_idx, fret, base_cost))
        all_positions.append(cands)

    # If some notes have no playable positions (e.g., out-of-range), we skip them
    # and keep indexes aligned via a map
    index_map = []
    filtered_positions = []
    for i, cands in enumerate(all_positions):
        if cands:
            index_map.append(i)
            filtered_positions.append(cands)

    if not filtered_positions:
        return []

    # ---- Dynamic Programming over candidates ----
    dp = []    # dp[t][j] = min cost up to time t ending at candidate j
    back = []  # back pointers

    # Initialize with first note's candidates
    first_cands = filtered_positions[0]
    dp.append([c[2] for c in first_cands])  # base_cost only
    back.append([None] * len(first_cands))

    # Forward pass
    for t in range(1, len(filtered_positions)):
        cands = filtered_positions[t]
        prev_cands = filtered_positions[t-1]

        dp.append([float("inf")] * len(cands))
        back.append([None] * len(cands))

        for j, (s_j, f_j, base_cost_j) in enumerate(cands):
            best_cost = float("inf")
            best_k = None
            for k, (s_k, f_k, base_cost_k) in enumerate(prev_cands):
                # transition cost
                t_cost = (
                    abs(f_j - f_k) * FRET_JUMP_PEN +
                    (STRING_CHANGE_PEN if s_j != s_k else 0.0)
                )
                total = dp[t-1][k] + base_cost_j + t_cost
                if total < best_cost:
                    best_cost = total
                    best_k = k
            dp[t][j] = best_cost
            back[t][j] = best_k

    # ---- Backtrack best path ----
    last_t = len(filtered_positions) - 1
    end_j = min(range(len(dp[last_t])), key=lambda j: dp[last_t][j])

    path = []
    t_idx = last_t
    j_idx = end_j
    while t_idx >= 0:
        s_idx, fret, _ = filtered_positions[t_idx][j_idx]
        path.append((t_idx, s_idx, fret))
        j_idx = back[t_idx][j_idx]
        if j_idx is None:
            break
        t_idx -= 1

    path.reverse()

    # ---- Build final tabs aligned to original note_events ----
    tabs = []
    for (t_pos, s_idx, fret) in path:
        ev_idx = index_map[t_pos]
        ev = evs[ev_idx]
        tabs.append({
            "t_on": float(ev["t_on"]),
            "string": int(len(open_midi) - s_idx),
            "fret": int(fret),
        })

    return tabs
