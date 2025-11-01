def seconds_to_mmss(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60); s = int(round(t - 60*m))
    return f"{m:02d}:{s:02d}"