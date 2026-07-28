from __future__ import annotations

import threading
import time
from typing import Final, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from audio_input import load_audio_bytes
from chord_match import match_chords
from features import chroma_from_audio
from models.basic_pitch_inference import BasicPitchTranscriber
from notes_baseline import detect_multi_pitch, frames_to_note_events
from segmentation import group_labels
from tabs_guitar_dp import dp_tab_mapping


BackendName = Literal["baseline", "basic_pitch"]
ProcessingMode = Literal["full", "chunked"]

GUITAR_MIDI_LOW: Final[int] = 40
GUITAR_MIDI_HIGH: Final[int] = 88
STANDARD_E_OPEN_MIDI: Final[tuple[int, ...]] = (40, 45, 50, 55, 59, 64)

_BASIC_PITCH_MODEL: BasicPitchTranscriber | None = None
_BASIC_PITCH_LOCK = threading.Lock()


def tmark() -> float:
    return time.perf_counter()


def telapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def postprocess_baseline_notes(
    note_events: list[dict],
    midi_low: int = GUITAR_MIDI_LOW,
    midi_high: int = GUITAR_MIDI_HIGH,
    min_duration: float = 0.10,
    merge_gap: float = 0.03,
) -> list[dict]:
    """Clean only the legacy DSP baseline output."""
    filtered: list[dict] = []

    for event in note_events or []:
        onset = float(event.get("t_on", 0.0))
        offset = float(event.get("t_off", onset))
        midi = int(event.get("midi", 0))

        if not midi_low <= midi <= midi_high:
            continue

        if offset - onset < min_duration:
            continue

        filtered.append(
            {
                **event,
                "t_on": onset,
                "t_off": offset,
                "midi": midi,
            }
        )

    filtered.sort(key=lambda event: (event["t_on"], event["midi"], event["t_off"]))

    if not filtered:
        return []

    merged: list[dict] = [filtered[0]]

    for event in filtered[1:]:
        previous = merged[-1]
        same_pitch = event["midi"] == previous["midi"]
        gap = event["t_on"] - previous["t_off"]

        if same_pitch and 0.0 <= gap <= merge_gap:
            previous["t_off"] = max(previous["t_off"], event["t_off"])
            previous["conf"] = max(
                float(previous.get("conf", 0.0)),
                float(event.get("conf", 0.0)),
            )
        else:
            merged.append(event)

    return merged


def transcribe_with_basic_pitch(
    y,
    sr: int,
) -> tuple[list[dict], str]:
    """Reuse one CoreML model and serialize access to that model."""
    global _BASIC_PITCH_MODEL

    with _BASIC_PITCH_LOCK:
        if _BASIC_PITCH_MODEL is None:
            _BASIC_PITCH_MODEL = BasicPitchTranscriber(
                midi_low=GUITAR_MIDI_LOW,
                midi_high=GUITAR_MIDI_HIGH,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length_ms=80.0,
            )

        note_events = _BASIC_PITCH_MODEL.transcribe(y, sr)
        return note_events, _BASIC_PITCH_MODEL.runtime_name


def transcribe_with_baseline(y, sr: int) -> list[dict]:
    frames = detect_multi_pitch(y, sr, hop_length=1024, top_k=3)
    events = frames_to_note_events(frames, min_dur=0.08)
    return postprocess_baseline_notes(events)


def humanize_chord(label: str) -> str:
    if label.upper() == "N":
        return "No chord"

    if label.endswith("m"):
        return f"{label[:-1]} Minor"

    return f"{label} Major"


def detect_chords_from_audio(y, sr: int) -> list[dict]:
    """Keep the existing chroma/template chord detector for this checkpoint."""
    chroma, times = chroma_from_audio(y, sr, hop_length=1024)
    chord_labels = match_chords(chroma)
    chord_segments = group_labels(chord_labels, list(times), min_hold_sec=0.4)

    return [
        {
            "t_start": float(start),
            "t_end": float(end),
            "label": label,
            "conf": 0.6,
            "tts": (
                f"{humanize_chord(label)} from "
                f"{round(float(start), 2)} to {round(float(end), 2)} seconds"
            ),
        }
        for start, end, label in chord_segments
    ]


app = FastAPI(title="ChordAssist API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "instrument": "guitar",
        "supported_backends": ["baseline", "basic_pitch"],
        "default_backend": "basic_pitch",
        "basic_pitch_loaded": _BASIC_PITCH_MODEL is not None,
    }


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    backend: BackendName = Query("basic_pitch"),
    mode: ProcessingMode = Query(
        "full",
        description=(
            "Compatibility parameter. Both values currently use full-file "
            "processing; true streaming will be implemented later."
        ),
    ),
    chords: bool = Query(False),
) -> dict:
    overall_start = tmark()

    raw_audio = await file.read()
    if not raw_audio:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

    io_start = tmark()
    try:
        y, sr = await run_in_threadpool(load_audio_bytes, raw_audio, 22050)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded audio: {exc}",
        ) from exc

    audio_duration_sec = len(y) / float(sr)
    io_ms = telapsed_ms(io_start)

    notes_start = tmark()
    runtime_name: str

    try:
        if backend == "basic_pitch":
            note_events, runtime_name = await run_in_threadpool(
                transcribe_with_basic_pitch,
                y,
                sr,
            )
        else:
            note_events = await run_in_threadpool(
                transcribe_with_baseline,
                y,
                sr,
            )
            runtime_name = "DSP"
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{backend} transcription failed: {exc}",
        ) from exc

    notes_ms = telapsed_ms(notes_start)

    tuning = {
        "name": "Standard E",
        "source": "fixed_assumption",
        "string_open_midi": list(STANDARD_E_OPEN_MIDI),
    }

    chords_start = tmark()
    chords_out: list[dict] = []

    if chords:
        try:
            chords_out = await run_in_threadpool(detect_chords_from_audio, y, sr)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Chord analysis failed: {exc}",
            ) from exc

    chords_ms = telapsed_ms(chords_start) if chords else 0.0

    tabs_start = tmark()
    tabs = dp_tab_mapping(
        note_events,
        open_midi=list(STANDARD_E_OPEN_MIDI),
        max_fret=20,
    )
    tabs_ms = telapsed_ms(tabs_start)

    tts_messages = ["Standard E tuning is assumed for tablature."]

    if chords and chords_out:
        tts_messages.append(f"Detected {len(chords_out)} chord segments.")

    if note_events:
        tts_messages.append(f"Detected {len(note_events)} notes.")

    total_ms = telapsed_ms(overall_start)
    real_time_factor = (
        total_ms / (audio_duration_sec * 1000.0)
        if audio_duration_sec > 0.0
        else None
    )

    return {
        "instrument_hint": "guitar",
        "tuning": tuning,
        "notes": note_events,
        "chords": chords_out,
        "render": {
            "guitar_tabs": tabs,
        },
        "tts": tts_messages,
        "latency_ms": round(total_ms, 2),
        "real_time_factor": (
            round(real_time_factor, 4)
            if real_time_factor is not None
            else None
        ),
        "mode_requested": mode,
        "mode_effective": "full",
        "timing_ms": {
            "io": round(io_ms, 2),
            "notes": round(notes_ms, 2),
            "chords": round(chords_ms, 2),
            "tabs": round(tabs_ms, 2),
            "total": round(total_ms, 2),
        },
        "audio_duration_sec": round(audio_duration_sec, 3),
        "backend_requested": backend,
        "backend_effective": backend,
        "backend_runtime": runtime_name,
        "chord_backend": "chroma_template" if chords else None,
    }