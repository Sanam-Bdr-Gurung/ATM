from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from audio_input import load_audio_bytes
from chord_match import match_chords
from features import chroma_from_audio
from segmentation import group_labels


def tmark() -> float:
    return time.perf_counter()


def telapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def humanize_chord(label: str) -> str:
    normalized = label.strip()

    if normalized.upper() == "N":
        return "No chord"

    if normalized.upper() == "X":
        return "Uncertain chord"

    if normalized.endswith("m"):
        return f"{normalized[:-1]} minor"

    return f"{normalized} major"


def detect_chords_from_audio(
    y,
    sr: int,
    audio_duration_sec: float,
) -> list[dict]:
    """
    Run the existing chroma/template detector.

    This is a temporary major/minor baseline. It will be replaced by the
    shared chord engine used by both chroma and Basic Pitch note events.
    """
    chroma, times = chroma_from_audio(
        y,
        sr,
        hop_length=1024,
    )

    if chroma.size == 0 or len(times) == 0:
        return []

    chord_labels = match_chords(chroma)

    if not chord_labels:
        return []

    grouped = group_labels(
        chord_labels,
        list(times),
        min_hold_sec=0.4,
    )

    segments: list[dict] = []

    for index, (start, end, label) in enumerate(grouped):
        start_sec = max(
            0.0,
            min(float(start), audio_duration_sec),
        )

        # group_labels currently ends the final segment at the timestamp of
        # the last frame. Extend it to the actual end of the audio.
        if index == len(grouped) - 1:
            end_sec = audio_duration_sec
        else:
            end_sec = min(float(end), audio_duration_sec)

        if end_sec <= start_sec:
            continue

        display = humanize_chord(label)

        segments.append(
            {
                "start": round(start_sec, 4),
                "end": round(end_sec, 4),
                "label": label,
                "display": display,
                # The temporary matcher does not yet expose calibrated scores.
                "confidence": None,
            }
        )

    return segments


app = FastAPI(
    title="ChordAssist API",
    version="0.2.0",
)

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
        "service": "chordassist",
        "scope": "prevailing_chord_recognition",
        "available_methods": ["chroma"],
        "planned_methods": ["basic_pitch"],
        "chord_engine_status": "temporary_major_minor_chroma_baseline",
        "basic_pitch_adapter_available": True,
    }


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
) -> dict:
    overall_start = tmark()

    raw_audio = await file.read()

    if not raw_audio:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty.",
        )

    io_start = tmark()

    try:
        y, sr = await run_in_threadpool(
            load_audio_bytes,
            raw_audio,
            22050,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded audio: {exc}",
        ) from exc

    audio_duration_sec = len(y) / float(sr)
    io_ms = telapsed_ms(io_start)

    chord_start = tmark()

    try:
        segments = await run_in_threadpool(
            detect_chords_from_audio,
            y,
            sr,
            audio_duration_sec,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chord analysis failed: {exc}",
        ) from exc

    chord_ms = telapsed_ms(chord_start)

    progression = [
        segment["label"]
        for segment in segments
    ]

    readable_progression = [
        segment["display"]
        for segment in segments
    ]

    if readable_progression:
        tts_messages = [
            f"Detected {len(segments)} chord segments.",
            "The detected progression is "
            + ", ".join(readable_progression)
            + ".",
        ]
    else:
        tts_messages = [
            "No chord segments were detected.",
        ]

    total_ms = telapsed_ms(overall_start)

    real_time_factor = (
        total_ms / (audio_duration_sec * 1000.0)
        if audio_duration_sec > 0.0
        else None
    )

    return {
        "segments": segments,
        "progression": progression,
        "tts": tts_messages,
        "method": "chroma_template_baseline",
        "engine_status": "temporary_baseline",
        "audio_duration_sec": round(audio_duration_sec, 3),
        "latency_ms": round(total_ms, 2),
        "real_time_factor": (
            round(real_time_factor, 4)
            if real_time_factor is not None
            else None
        ),
        "timing_ms": {
            "io": round(io_ms, 2),
            "chord_analysis": round(chord_ms, 2),
            "total": round(total_ms, 2),
        },
    }