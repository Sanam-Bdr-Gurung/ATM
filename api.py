from __future__ import annotations

import threading
import time
from typing import Final, Literal, Sequence

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from audio_input import load_audio_bytes
from chord_engine import ChordPrediction
from chord_match import classify_chroma_frames
from features import chroma_from_audio
from models.basic_pitch_inference import BasicPitchTranscriber
from note_event_features import (
    classify_note_event_frames,
    note_events_to_pitch_class_frames,
)
from segmentation import segment_chord_predictions


MethodName = Literal[
    "chroma",
    "basic_pitch",
]

MINIMUM_SCORE: Final[float] = 0.62
AMBIGUITY_MARGIN: Final[float] = 0.035
MINIMUM_SEGMENT_DURATION_SEC: Final[float] = 0.4

CHROMA_HOP_LENGTH: Final[int] = 1024
CHROMA_FRAME_LENGTH: Final[int] = 4096
CHROMA_HARMONIC_MARGIN: Final[float] = 3.0
CHROMA_RELATIVE_ACTIVITY_FLOOR: Final[float] = 0.08

NOTE_WINDOW_SEC: Final[float] = 0.75
NOTE_MINIMUM_CONFIDENCE: Final[float] = 0.05
NOTE_BASS_RELATIVE_WEIGHT: Final[float] = 0.15
NOTE_RELATIVE_ACTIVITY_FLOOR: Final[float] = 0.05

_BASIC_PITCH_MODEL: BasicPitchTranscriber | None = None
_BASIC_PITCH_LOCK = threading.Lock()


def tmark() -> float:
    return time.perf_counter()


def telapsed_ms(start: float) -> float:
    return (
        time.perf_counter() - start
    ) * 1000.0


def summarize_predictions(
    predictions: Sequence[ChordPrediction],
) -> dict[str, int]:
    return {
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


def analyze_with_chroma(
    y,
    sr: int,
    audio_duration_sec: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    feature_start = tmark()

    features = chroma_from_audio(
        y,
        sr,
        hop_length=CHROMA_HOP_LENGTH,
        frame_length=CHROMA_FRAME_LENGTH,
        harmonic_margin=CHROMA_HARMONIC_MARGIN,
    )

    feature_extraction_ms = telapsed_ms(
        feature_start
    )

    classification_start = tmark()

    predictions, activity_threshold = (
        classify_chroma_frames(
            features.chroma,
            frame_activity=features.frame_activity,
            relative_activity_floor=(
                CHROMA_RELATIVE_ACTIVITY_FLOOR
            ),
            minimum_score=MINIMUM_SCORE,
            ambiguity_margin=AMBIGUITY_MARGIN,
        )
    )

    classification_ms = telapsed_ms(
        classification_start
    )

    segmentation_start = tmark()

    segments = segment_chord_predictions(
        predictions,
        features.times,
        audio_duration_sec=audio_duration_sec,
        min_hold_sec=MINIMUM_SEGMENT_DURATION_SEC,
    )

    segmentation_ms = telapsed_ms(
        segmentation_start
    )

    analysis = {
        "feature_source": "traditional_chroma",
        "frame_count": len(predictions),
        "activity_threshold": round(
            float(activity_threshold),
            8,
        ),
        "hop_length": features.hop_length,
        "frame_length": features.frame_length,
        "harmonic_margin": CHROMA_HARMONIC_MARGIN,
        "relative_activity_floor": (
            CHROMA_RELATIVE_ACTIVITY_FLOOR
        ),
        "minimum_score": MINIMUM_SCORE,
        "ambiguity_margin": AMBIGUITY_MARGIN,
        "minimum_segment_duration_sec": (
            MINIMUM_SEGMENT_DURATION_SEC
        ),
        "feature_extraction_ms": round(
            feature_extraction_ms,
            2,
        ),
        "classification_ms": round(
            classification_ms,
            2,
        ),
        "segmentation_ms": round(
            segmentation_ms,
            2,
        ),
        **summarize_predictions(
            predictions
        ),
    }

    return segments, analysis


def transcribe_with_basic_pitch(
    y,
    sr: int,
) -> tuple[
    list[dict],
    str,
    bool,
    dict[str, object],
]:
    global _BASIC_PITCH_MODEL

    with _BASIC_PITCH_LOCK:
        model_was_loaded = (
            _BASIC_PITCH_MODEL is not None
        )

        if _BASIC_PITCH_MODEL is None:
            _BASIC_PITCH_MODEL = (
                BasicPitchTranscriber()
            )

        model = _BASIC_PITCH_MODEL

        note_events = model.transcribe(
            y,
            sr,
        )

        configuration = {
            "midi_low": model.midi_low,
            "midi_high": model.midi_high,
            "onset_threshold": (
                model.onset_threshold
            ),
            "frame_threshold": (
                model.frame_threshold
            ),
            "minimum_note_length_ms": (
                model.minimum_note_length_ms
            ),
        }

        return (
            note_events,
            model.runtime_name,
            model_was_loaded,
            configuration,
        )


def analyze_with_basic_pitch(
    y,
    sr: int,
    audio_duration_sec: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    inference_start = tmark()

    (
        note_events,
        runtime_name,
        model_was_loaded,
        model_configuration,
    ) = transcribe_with_basic_pitch(
        y,
        sr,
    )

    model_inference_ms = telapsed_ms(
        inference_start
    )

    feature_start = tmark()

    features = note_events_to_pitch_class_frames(
        note_events,
        audio_duration_sec=audio_duration_sec,
        frame_step_sec=(
            CHROMA_HOP_LENGTH / float(sr)
        ),
        window_sec=NOTE_WINDOW_SEC,
        minimum_confidence=(
            NOTE_MINIMUM_CONFIDENCE
        ),
        bass_relative_weight=(
            NOTE_BASS_RELATIVE_WEIGHT
        ),
    )

    feature_extraction_ms = telapsed_ms(
        feature_start
    )

    classification_start = tmark()

    predictions, activity_threshold = (
        classify_note_event_frames(
            features,
            relative_activity_floor=(
                NOTE_RELATIVE_ACTIVITY_FLOOR
            ),
            minimum_score=MINIMUM_SCORE,
            ambiguity_margin=AMBIGUITY_MARGIN,
        )
    )

    classification_ms = telapsed_ms(
        classification_start
    )

    segmentation_start = tmark()

    segments = segment_chord_predictions(
        predictions,
        features.times,
        audio_duration_sec=audio_duration_sec,
        min_hold_sec=MINIMUM_SEGMENT_DURATION_SEC,
    )

    segmentation_ms = telapsed_ms(
        segmentation_start
    )

    analysis = {
        "feature_source": (
            "basic_pitch_note_events"
        ),
        "basic_pitch_runtime": runtime_name,
        "model_was_loaded_before_request": (
            model_was_loaded
        ),
        "note_event_count": len(
            note_events
        ),
        "frame_count": len(
            predictions
        ),
        "activity_threshold": round(
            float(activity_threshold),
            8,
        ),
        "frame_step_sec": round(
            features.frame_step_sec,
            8,
        ),
        "window_sec": features.window_sec,
        "minimum_note_confidence": (
            NOTE_MINIMUM_CONFIDENCE
        ),
        "bass_relative_weight": (
            NOTE_BASS_RELATIVE_WEIGHT
        ),
        "relative_activity_floor": (
            NOTE_RELATIVE_ACTIVITY_FLOOR
        ),
        "minimum_score": MINIMUM_SCORE,
        "ambiguity_margin": AMBIGUITY_MARGIN,
        "minimum_segment_duration_sec": (
            MINIMUM_SEGMENT_DURATION_SEC
        ),
        "model_configuration": (
            model_configuration
        ),
        "model_inference_ms": round(
            model_inference_ms,
            2,
        ),
        "feature_extraction_ms": round(
            feature_extraction_ms,
            2,
        ),
        "classification_ms": round(
            classification_ms,
            2,
        ),
        "segmentation_ms": round(
            segmentation_ms,
            2,
        ),
        **summarize_predictions(
            predictions
        ),
    }

    return segments, analysis


def analyze_with_method(
    method: MethodName,
    y,
    sr: int,
    audio_duration_sec: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    if method == "chroma":
        return analyze_with_chroma(
            y,
            sr,
            audio_duration_sec,
        )

    return analyze_with_basic_pitch(
        y,
        sr,
        audio_duration_sec,
    )


app = FastAPI(
    title="ChordAssist API",
    version="0.4.0",
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
        "scope": (
            "prevailing_chord_recognition"
        ),
        "available_methods": [
            "chroma",
            "basic_pitch",
        ],
        "default_method": "chroma",
        "chord_engine_status": (
            "shared_core_chord_engine_v1"
        ),
        "basic_pitch_loaded": (
            _BASIC_PITCH_MODEL is not None
        ),
    }


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    method: MethodName = Query(
        "chroma",
        description=(
            "Feature source used before the shared "
            "chord classifier."
        ),
    ),
) -> dict:
    overall_start = tmark()

    raw_audio = await file.read()

    if not raw_audio:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded audio file is empty."
            ),
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
        segments, analysis = (
            await run_in_threadpool(
                analyze_with_method,
                method,
                y,
                sr,
                audio_duration_sec,
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{method} chord analysis "
                f"failed: {exc}"
            ),
        ) from exc

    chord_ms = telapsed_ms(
        chord_start
    )

    harmonic_segments = [
        segment
        for segment in segments
        if segment["label"] != "N"
    ]

    progression = [
        segment["label"]
        for segment in harmonic_segments
    ]

    readable_progression = [
        segment["display"]
        for segment in harmonic_segments
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
            (
                "No reliable chord progression "
                "was detected."
            ),
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
        "method": method,
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
