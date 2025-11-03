from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from audio_input import load_audio_bytes
from features import chroma_from_audio
from chord_match import match_chords
from segmentation import group_labels
from notes_baseline import detect_multi_pitch, frames_to_note_events
from tabs_guitar import map_note_to_string_fret
from tuning.detune import estimate_concert_detune_cents_from_frames, estimate_detune_cents_from_events
from tuning.detect import detect_tuning_class
import time
import os, traceback
from pathlib import Path
# ----- PyTorch notes model singleton (safe) -----
from typing import Optional
from models.of_inference import OFTranscriber


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

        OF_MODEL = OFTranscriber(device="cpu", checkpoint_path=ckpt_arg)
        if ckpt_arg:
            print(f"[INFO] OFTranscriber initialized with checkpoint: {ckpt_arg}")
        else:
            print("[INFO] OFTranscriber initialized WITHOUT checkpoint (random weights).")
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
async def analyze_file(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    raw = await file.read()
    y, sr = load_audio_bytes(raw, sr=22050)

    # ---- Notes (prefer PyTorch model; fallback to baseline) ----
    notes_model = get_of_model()
    used_baseline = False

    if notes_model is not None:
        try:
            note_events = notes_model.transcribe(y, sr)
        except Exception as e:
            print("[WARN] OFTranscriber failed; falling back to baseline:", repr(e))
            from notes_baseline import detect_multi_pitch, frames_to_note_events
            mp = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
            note_events = frames_to_note_events(mp, min_dur=0.08)
            used_baseline = True
    else:
        from notes_baseline import detect_multi_pitch, frames_to_note_events
        mp = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
        note_events = frames_to_note_events(mp, min_dur=0.08)
        used_baseline = True

    # ---- Tuning detection ----
    if used_baseline:
        detune_cents = estimate_concert_detune_cents_from_frames(mp["pitches_hz_frames"])
    else:
        detune_cents = estimate_detune_cents_from_events(note_events)

    tuning_info = detect_tuning_class(note_events, max_fret=20)
    tuning_block = {
        "name": tuning_info["name"],
        "string_open_midi": tuning_info["string_open_midi"],
        "concert_detune_cents": round(detune_cents, 2),
        "confidence": round(float(tuning_info["confidence"]), 3),
    }

    # ---- Chords (baseline templates) ----
    chroma, times = chroma_from_audio(y, sr, hop_length=1024)
    chord_labels = match_chords(chroma)
    chord_segs = group_labels(chord_labels, list(times), min_hold_sec=0.4)
    chords = []
    for s, e, lab in chord_segs:
        human = (lab[:-1] + " Minor") if lab.endswith("m") else (lab + " Major")
        chords.append({
            "t_start": float(s), "t_end": float(e), "label": lab,
            "conf": 0.6, "tts": f"{human} from {round(s,2)} to {round(e,2)} seconds"
        })

    # ---- Tabs (tuning-aware) ----
    tabs = []
    open_midi = tuning_info["string_open_midi"]
    for ev in note_events:
        m = map_note_to_string_fret(ev["midi"], open_midi=open_midi, max_fret=20)
        if m:
            tabs.append({"t_on": ev["t_on"], "string": m["string"], "fret": m["fret"]})

    # ---- TTS & latency ----
    tts_msgs = []
    if tuning_block["confidence"] >= 0.6:
        tts_msgs.append(f"Detected {tuning_block['name']} tuning.")
    else:
        tts_msgs.append(f"Likely {tuning_block['name']} tuning.")
    if chords: tts_msgs.append(f"Detected {len(chords)} chord segments.")
    if note_events: tts_msgs.append(f"Detected {len(note_events)} notes.")

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "instrument_hint": "guitar",
        "tuning": tuning_block,
        "notes": note_events,
        "chords": chords,
        "render": {
            "guitar_tabs": tabs,
            "piano_roll": [],
            "violin_fingerings": []
        },
        "tts": tts_msgs,
        "latency_ms": round(latency_ms, 2)
    }
    