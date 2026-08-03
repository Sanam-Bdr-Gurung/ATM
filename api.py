from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from audio_input import load_audio_bytes
from chord_match import classify_chroma_frames
from features import chroma_from_audio
from segmentation import segment_chord_predictions


def tmark() -> float:
    return time.perf_counter()


def telapsed_ms(start: float) -> float:
    return (
        time.perf_counter() - start
    ) * 1000.0


def detect_chords_from_audio(
    y,
    sr: int,
    audio_duration_sec: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    features = chroma_from_audio(
        y,
        sr,
        hop_length=1024,
        frame_length=4096,
        harmonic_margin=3.0,
    )

    predictions, activity_threshold = (
        classify_chroma_frames(
            features.chroma,
            frame_activity=features.frame_activity,
            relative_activity_floor=0.08,
            minimum_score=0.62,
            ambiguity_margin=0.035,
        )
    )

    segments = segment_chord_predictions(
        predictions,
        features.times,
        audio_duration_sec=audio_duration_sec,
        min_hold_sec=0.4,
    )

    label_counts = {
        "no_chord_frames": sum(
            prediction.label == "N"
            for prediction in predictions
        ),
        "uncertain_frames": sum(
            prediction.label == "X"
            for prediction in predictions
        ),
        "classified_chord_frames": sum(
            prediction.label not in {"N", "X"}
            for prediction in predictions
        ),
    }

    analysis = {
        "frame_count": len(predictions),
        "activity_threshold": round(
            float(activity_threshold),
            8,
        ),
        "hop_length": features.hop_length,
        "frame_length": features.frame_length,
        "minimum_score": 0.62,
        "ambiguity_margin": 0.035,
        "minimum_segment_duration_sec": 0.4,
        **label_counts,
    }

    return segments, analysis


app = FastAPI(
    title="ChordAssist API",
    version="0.3.0",
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
        "chord_engine_status": "shared_core_chord_engine_v1",
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
            detail=(
                "Could not decode the uploaded "
                f"audio: {exc}"
            ),
        ) from exc

    audio_duration_sec = (
        len(y) / float(sr)
    )

    io_ms = telapsed_ms(
        io_start
    )

    chord_start = tmark()

    try:
        segments, analysis = await run_in_threadpool(
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

    chord_ms = telapsed_ms(
        chord_start
    )

    progression = [
        segment["label"]
        for segment in segments
        if segment["label"] != "N"
    ]

    readable_progression = [
        segment["display"]
        for segment in segments
        if segment["label"] != "N"
    ]

    if readable_progression:
        tts_messages = [
            (
                f"Detected {len(progression)} "
                "harmonic segments."
            ),
            (
                "The detected progression is "
                + ", ".join(
                    str(label)
                    for label in readable_progression
                )
                + "."
            ),
        ]
    else:
        tts_messages = [
            "No reliable chord progression was detected.",
        ]

    total_ms = telapsed_ms(
        overall_start
    )

    real_time_factor = (
        total_ms
        / (
            audio_duration_sec
            * 1000.0
        )
        if audio_duration_sec > 0.0
        else None
    )

    return {
        "segments": segments,
        "progression": progression,
        "tts": tts_messages,
        "method": "chroma_shared_engine",
        "engine_status": "shared_core_v1",
        "analysis": analysis,
        "audio_duration_sec": round(
            audio_duration_sec,
            3,
        ),
        "latency_ms": round(
            total_ms,
            2,
        ),
        "real_time_factor": (
            round(
                real_time_factor,
                4,
            )
            if real_time_factor is not None
            else None
        ),
        "timing_ms": {
            "io": round(
                io_ms,
                2,
            ),
            "chord_analysis": round(
                chord_ms,
                2,
            ),
            "total": round(
                total_ms,
                2,
            ),
        },
    }
