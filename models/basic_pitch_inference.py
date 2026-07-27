from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import numpy as np
import soundfile as sf

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

from models.notes_interface import NotesTranscriber, midi_to_hz


class BasicPitchTranscriber(NotesTranscriber):
    """
    Adapter between Spotify Basic Pitch and the project's stable note-event
    interface.

    Project output format:

        {
            "t_on": float,
            "t_off": float,
            "midi": int,
            "freq_hz": float,
            "conf": float
        }

    Basic Pitch currently requires an audio file path. The existing project
    interface provides an in-memory waveform, so this adapter temporarily
    writes a WAV file and removes it after inference.

    This temporary disk round trip will be optimized later, before final
    end-to-end latency measurements.
    """

    def __init__(
        self,
        midi_low: int = 40,
        midi_high: int = 88,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 80.0,
    ) -> None:
        if not 0 <= midi_low <= 127:
            raise ValueError("midi_low must be between 0 and 127.")

        if not 0 <= midi_high <= 127:
            raise ValueError("midi_high must be between 0 and 127.")

        if midi_low > midi_high:
            raise ValueError("midi_low cannot be greater than midi_high.")

        if minimum_note_length_ms <= 0:
            raise ValueError("minimum_note_length_ms must be positive.")

        self.midi_low = midi_low
        self.midi_high = midi_high
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length_ms = minimum_note_length_ms

        # Load the model once for this transcriber instance.
        self.model = Model(ICASSP_2022_MODEL_PATH)
        self.runtime_name = self.model.model_type.name

    def transcribe(
        self,
        y: np.ndarray,
        sr: int,
    ) -> List[Dict]:
        audio = self._validate_audio(y, sr)
        temporary_path = self._write_temporary_wav(audio, sr)

        try:
            _, _, basic_pitch_events = predict(
                temporary_path,
                self.model,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.minimum_note_length_ms,
                minimum_frequency=midi_to_hz(self.midi_low),
                maximum_frequency=midi_to_hz(self.midi_high),
                multiple_pitch_bends=False,
                melodia_trick=True,
            )
        finally:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(
                    f"[WARN] Could not remove temporary audio file "
                    f"{temporary_path}: {exc}"
                )

        converted_events: List[Dict] = []

        # Basic Pitch event structure:
        # (
        #   start_time_seconds,
        #   end_time_seconds,
        #   pitch_midi,
        #   amplitude,
        #   pitch_bend_values
        # )
        for event in basic_pitch_events:
            start_time = float(event[0])
            end_time = float(event[1])
            midi = int(event[2])
            amplitude = float(event[3])

            if midi < self.midi_low or midi > self.midi_high:
                continue

            if end_time <= start_time:
                continue

            converted_events.append(
                {
                    "t_on": round(start_time, 4),
                    "t_off": round(end_time, 4),
                    "midi": midi,
                    "freq_hz": round(float(midi_to_hz(midi)), 3),
                    "conf": round(amplitude, 4),
                }
            )

        # Basic Pitch does not guarantee chronological event ordering.
        converted_events.sort(
            key=lambda item: (
                item["t_on"],
                item["midi"],
                item["t_off"],
            )
        )

        return converted_events

    def transcribe_chunked(
        self,
        y: np.ndarray,
        sr: int,
        **_: object,
    ) -> List[Dict]:
        """
        Compatibility method only.

        This is not true streaming yet. It currently performs full-file
        Basic Pitch prediction so callers do not mistake an unfinished
        chunking implementation for real streaming inference.
        """
        return self.transcribe(y, sr)

    @staticmethod
    def _validate_audio(
        y: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        if sr <= 0:
            raise ValueError("Audio sample rate must be positive.")

        audio = np.asarray(y, dtype=np.float32)
        audio = np.squeeze(audio)

        if audio.ndim != 1:
            raise ValueError(
                "BasicPitchTranscriber expects a one-dimensional mono waveform."
            )

        if audio.size == 0:
            raise ValueError("Cannot transcribe empty audio.")

        if not np.all(np.isfinite(audio)):
            raise ValueError(
                "Audio contains NaN or infinite sample values."
            )

        peak = float(np.max(np.abs(audio)))

        if peak > 1.0:
            audio = audio / peak

        return np.ascontiguousarray(audio, dtype=np.float32)

    @staticmethod
    def _write_temporary_wav(
        audio: np.ndarray,
        sr: int,
    ) -> str:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="chordassist_basic_pitch_",
            suffix=".wav",
            delete=False,
        )

        temporary_path = temporary_file.name
        temporary_file.close()

        try:
            # FLOAT avoids introducing PCM-16 quantization before inference.
            sf.write(
                temporary_path,
                audio,
                int(sr),
                format="WAV",
                subtype="FLOAT",
            )
        except Exception:
            try:
                os.remove(temporary_path)
            except OSError:
                pass

            raise

        return temporary_path