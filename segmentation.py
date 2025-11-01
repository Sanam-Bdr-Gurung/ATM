from typing import List, Tuple

def group_labels(labels: List[str], times: List[float], min_hold_sec: float = 0.3) -> List[Tuple[float, float, str]]:
    if not labels: return []
    segs, cur, start = [], labels[0], times[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            end = times[i]
            if segs and (end - start) < min_hold_sec:
                ps, pe, pl = segs[-1]
                segs[-1] = (ps, end, pl)
            else:
                segs.append((start, end, cur))
            cur, start = labels[i], times[i]
    segs.append((start, times[-1], cur))
    return segs
