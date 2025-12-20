from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from audio_input import load_audio_bytes
from features import chroma_from_audio
from chord_match import match_chords
from segmentation import group_labels
from notes_baseline import detect_multi_pitch, frames_to_note_events
from tabs_guitar import map_note_to_string_fret
from tuning.detune import (
    estimate_concert_detune_cents_from_frames,
    estimate_detune_cents_from_events,
)
from tuning.detect import detect_tuning_class
import time
import os, traceback
from pathlib import Path
# ----- PyTorch notes model singleton (safe) -----
from typing import Optional, Literal
from models.of_inference import OFTranscriber
from util_log import safe_call
from tabs_guitar_dp import dp_tab_mapping

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(m: int) -> str:
    if m < 0 or m > 127:
        return f"midi{m}"
    name = NOTE_NAMES[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"     

def postprocess_notes(
    note_events,
    midi_low: int = 40,     # E2
    midi_high: int = 88,    # up to ~E6
    min_dur: float = 0.10,  # seconds
    merge_gap: float = 0.03 # seconds, merge if gap smaller than this
):
    """
    Simple, cheap cleanup pass for baseline note_events:
      - keep only notes in guitar pitch range
      - drop very short notes (< min_dur)
      - merge nearly-contiguous notes of same pitch
    """
    if not note_events:
        return []

    # 1) filter by pitch + duration
    filtered = []
    for ev in note_events:
        t_on = float(ev.get("t_on", 0.0))
        t_off = float(ev.get("t_off", t_on))
        dur = t_off - t_on
        midi = int(ev.get("midi", 0))

        if midi < midi_low or midi > midi_high:
            continue
        if dur < min_dur:
            continue

        filtered.append({
            **ev,
            "t_on": t_on,
            "t_off": t_off,
            "midi": midi,
        })

    if not filtered:
        return []

    # 2) sort by onset then pitch
    filtered.sort(key=lambda e: (e["t_on"], e["midi"]))

    # 3) merge close same-pitch notes
    merged = [filtered[0]]
    for ev in filtered[1:]:
        last = merged[-1]
        same_pitch = (ev["midi"] == last["midi"])
        gap = ev["t_on"] - last["t_off"]

        if same_pitch and 0.0 <= gap <= merge_gap:
            # merge into last
            last["t_off"] = max(last["t_off"], ev["t_off"])
            # keep max confidence if present
            if "conf" in ev or "conf" in last:
                last_conf = float(last.get("conf", 0.0))
                ev_conf = float(ev.get("conf", 0.0))
                last["conf"] = max(last_conf, ev_conf)
        else:
            merged.append(ev)

    return merged

def collapse_harmonics(note_events, window: float = 0.08):
    """
    For monophonic/debug use:
    In each small time window, keep only the lowest MIDI note.
    Assumes most higher notes in the cluster are harmonics.
    """
    if not note_events:
        return []

    # Sort by onset, then midi
    evs = sorted(note_events, key=lambda e: (float(e["t_on"]), int(e["midi"])))

    collapsed = []
    cluster = [evs[0]]

    def flush_cluster():
        if not cluster:
            return
        # pick lowest midi in cluster
        best = min(cluster, key=lambda e: int(e["midi"]))
        collapsed.append(best)

    for ev in evs[1:]:
        prev = cluster[-1]
        if (ev["t_on"] - prev["t_on"]) <= window:
            # still in same cluster
            cluster.append(ev)
        else:
            # new cluster
            flush_cluster()
            cluster = [ev]

    flush_cluster()
    return collapsed

def tmark(): return time.perf_counter()
def telapsed(t0): return (time.perf_counter() - t0) * 1000.0
OF_MODEL: Optional[OFTranscriber] = None
def get_of_model() -> Optional[OFTranscriber]:
    global OF_MODEL
    if OF_MODEL is not None:
        return OF_MODEL

    try:
        base = Path(__file__).parent
        # Prefer these filenames in order
        candidates = [
            base / "checkpoints" / "of_pretrained.ckpt",
            base / "checkpoints" / "of_pretrained.pth",
            base / "checkpoints" / "of_pretrained.pt",
            base / "checkpoints" / "of_pretrained.safetensors",
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists() and c.is_file() and c.stat().st_size > 1024:
                ckpt_path = c
                break

        # If it's a safetensors file, pass path anyway; of_inference will handle it
        ckpt_arg = str(ckpt_path) if ckpt_path else None
        
        dev = "cpu"
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
        except:
            pass

        OF_MODEL = OFTranscriber(
            device=dev,
            checkpoint_path=ckpt_arg,   # your resilient pick
            sr_model=16000,      # typical O&F training SR; safe choice
            n_mels=229,
            n_fft=2048,
            hop_length=512,
            midi_low=21,                # 88-key piano range
            midi_high=108
        )
        if ckpt_arg:
            print(f"[INFO] OFTranscriber initialized with checkpoint: {ckpt_arg}")
        else:
            print("[INFO] OFTranscriber initialized with NO checkpoint path.")
    except Exception as e:
        print("[WARN] Could not init OFTranscriber:", repr(e))
        traceback.print_exc()
        OF_MODEL = None

    return OF_MODEL
    
app = FastAPI(title="ChordAssist API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}
 
@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    mode: Literal["chunked", "full"] = Query("chunked"),  # default chunked
    backend: Literal["baseline", "of"] = Query("baseline"),
    chords: bool = Query(False)):
     # ---- backend sanity / model selection ----
    backend_effective = "baseline"
    notes_model: Optional[OFTranscriber] = None

    if backend == "of":
        notes_model = get_of_model()
        if notes_model is not None:
            print("[INFO] backend=of requested, using OFTranscriber.")
            backend_effective = "of"
        else:
            print("[WARN] backend=of requested but OFTranscriber unavailable; using baseline.")
    else:
        backend_effective = "baseline"

    # ---- overall timer ----
    t_all = tmark()

    # -------- Load audio --------
    t0 = tmark()
    raw = await file.read()
    y, sr = load_audio_bytes(raw, sr=22050)

    # Duration in seconds (for evaluation / thesis reporting)
    audio_duration_sec = len(y) / float(sr)

    t_io = telapsed(t0)

    # -------- Notes (OF vs baseline) --------
    t0 = tmark()
    used_baseline = False
    note_events = None
    mp = None  # only set for baseline path

    if backend_effective == "of" and notes_model is not None:
        # Use PyTorch Onsets & Frames backend
        def _run_of():
            if mode == "chunked":
                return notes_model.transcribe_chunked(
                    y, sr,
                    chunk_sec=1.0,    # you can tweak later
                    hop_sec=0.5,
                    onset_filt=3,
                    frame_filt=5,
                    th_on_hi=0.55,
                    th_on_lo=0.30,
                    th_fr=0.50,
                )
            else:
                return notes_model.transcribe(y, sr)

        note_events = safe_call("OFTranscriber inference", _run_of, fallback=None)

    # Fallback or baseline path
    if note_events is None:
        mp = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
        note_events = frames_to_note_events(mp, min_dur=0.08)
        used_baseline = True

    t_notes = telapsed(t0)

    # -------- Tuning detection (events or frames) --------
    t0 = tmark()
    if used_baseline and mp is not None:
        detune_cents = estimate_concert_detune_cents_from_frames(mp["pitches_hz_frames"])
    else:
        detune_cents = estimate_detune_cents_from_events(note_events)

    tuning_info = detect_tuning_class(note_events, max_fret=20)
    # tuning_block = {
    #     "name": tuning_info["name"],
    #     "string_open_midi": tuning_info["string_open_midi"],
    #     "concert_detune_cents": round(detune_cents, 2),
    #     "confidence": round(float(tuning_info["confidence"]), 3),
    # }
    # TEMP: force standard E for debugging
    tuning_block = {
        "name": "Standard E (forced)",
        "string_open_midi": [40, 45, 50, 55, 59, 64],
        "concert_detune_cents": 0.0,
        "confidence": 1.0,
    }
    # --- NEW: postprocess for guitar range + cleaner notes ---
    note_events = postprocess_notes(
        note_events,
        midi_low=40,   # E2
        midi_high=88,  # ~E6
        min_dur=0.10,  # a bit stricter than 0.08
        merge_gap=0.03
    )
    # TEMP: assume mostly monophonic, collapse harmonics
    note_events = collapse_harmonics(note_events, window=0.08)
    # ---- Debug: print first 25 notes + tuning ----
    print("=== DEBUG NOTES ===")
    print("TUNING PRED:", tuning_block)
    for ev in note_events[:25]:
        m = int(ev["midi"])
        print(f"t={ev['t_on']:.3f}s  midi={m}  {midi_to_name(m)}")
    print("=== END DEBUG ===")
    t_tuning = telapsed(t0)

    # -------- Chords (baseline templates) --------
    t0 = tmark()
    chords_out = []
    if chords:
        chroma, times = chroma_from_audio(y, sr, hop_length=1024)
        chord_labels = match_chords(chroma)
        chord_segs = group_labels(chord_labels, list(times), min_hold_sec=0.4)
        for s, e, lab in chord_segs:
            human = (lab[:-1] + " Minor") if lab.endswith("m") else (lab + " Major")
            chords_out.append({
                "t_start": float(s),
                "t_end": float(e),
                "label": lab,
                "conf": 0.6,
                "tts": f"{human} from {round(s,2)} to {round(e,2)} seconds",
            })
    t_chords = telapsed(t0) if chords else 0.0

    # -------- Tabs (tuning-aware, DP smoothing) --------
    t0_tabs = tmark()
    open_midi = tuning_info["string_open_midi"]
    tabs = dp_tab_mapping(note_events, open_midi=open_midi, max_fret=20)
    t_tabs = telapsed(t0_tabs)

    # -------- TTS & latency --------
    t0 = tmark()
    tts_msgs = []
    if tuning_block["confidence"] >= 0.6:
        tts_msgs.append(f"Detected {tuning_block['name']} tuning.")
    else:
        tts_msgs.append(f"Likely {tuning_block['name']} tuning.")
    if chords and chords_out:
        tts_msgs.append(f"Detected {len(chords_out)} chord segments.")
    if note_events:
        tts_msgs.append(f"Detected {len(note_events)} notes.")
    # (no separate timer for TTS strings—they're trivial)

    latency_ms = telapsed(t_all)

    # Response
    return {
        "instrument_hint": "guitar",
        "tuning": tuning_block,
        "notes": note_events,
        "chords": chords_out,   # <-- use guarded list
        "render": {
            "guitar_tabs": tabs,
            "piano_roll": [],
            "violin_fingerings": []
        },
        "tts": tts_msgs,
        "latency_ms": round(latency_ms, 2),
        "mode": mode,
        "timing_ms": {
            "io": round(t_io, 2),
            "notes": round(t_notes, 2),
            "tuning": round(t_tuning, 2),
            "chords": round(t_chords, 2),  # 0.0 when skipped
            "tabs": round(t_tabs, 2),
            "total": round(latency_ms, 2)
        },
        "audio_duration_sec": round(audio_duration_sec, 3), 
        "backend_requested": backend,
        "backend_effective": backend_effective
    }