from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from chord_config import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MINIMUM_SCORE,
)


PITCH_CLASS_NAMES: Final[tuple[str, ...]] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


@dataclass(frozen=True)
class ChordQuality:
    key: str
    display: str
    intervals: tuple[int, ...]
    weights: tuple[float, ...]
    complexity_penalty: float = 0.0


CORE_QUALITIES: Final[tuple[ChordQuality, ...]] = (
    ChordQuality(
        key="maj",
        display="major",
        intervals=(0, 4, 7),
        weights=(1.0, 0.95, 0.75),
    ),
    ChordQuality(
        key="min",
        display="minor",
        intervals=(0, 3, 7),
        weights=(1.0, 0.95, 0.75),
    ),
    ChordQuality(
        key="7",
        display="dominant seventh",
        intervals=(0, 4, 7, 10),
        weights=(1.0, 0.95, 0.70, 0.85),
        complexity_penalty=0.02,
    ),
    ChordQuality(
        key="maj7",
        display="major seventh",
        intervals=(0, 4, 7, 11),
        weights=(1.0, 0.95, 0.70, 0.85),
        complexity_penalty=0.02,
    ),
    ChordQuality(
        key="min7",
        display="minor seventh",
        intervals=(0, 3, 7, 10),
        weights=(1.0, 0.95, 0.70, 0.85),
        complexity_penalty=0.02,
    ),
    ChordQuality(
        key="sus2",
        display="suspended second",
        intervals=(0, 2, 7),
        weights=(1.0, 0.90, 0.75),
        complexity_penalty=0.015,
    ),
    ChordQuality(
        key="sus4",
        display="suspended fourth",
        intervals=(0, 5, 7),
        weights=(1.0, 0.90, 0.75),
        complexity_penalty=0.015,
    ),
    ChordQuality(
        key="add9",
        display="added ninth",
        intervals=(0, 2, 4, 7),
        weights=(1.0, 0.65, 0.95, 0.70),
        complexity_penalty=0.03,
    ),
)


@dataclass(frozen=True)
class CandidateScore:
    label: str
    display: str
    root_pitch_class: int
    quality: str
    score: float
    cosine_similarity: float
    tone_coverage: float
    root_support: float


@dataclass(frozen=True)
class ChordPrediction:
    label: str
    display: str
    confidence: float
    score: float
    margin: float
    root_pitch_class: int | None
    quality: str | None
    candidate_label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "display": self.display,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "root_pitch_class": self.root_pitch_class,
            "quality": self.quality,
            "candidate_label": self.candidate_label,
        }


def humanize_chord_label(label: str) -> str:
    normalized = label.strip()

    if normalized == "N":
        return "No chord"

    if normalized == "X":
        return "Uncertain chord"

    try:
        root, quality_key = normalized.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported chord label format: {label!r}."
        ) from exc

    quality = next(
        (
            definition
            for definition in CORE_QUALITIES
            if definition.key == quality_key
        ),
        None,
    )

    if quality is None:
        raise ValueError(
            f"Unsupported chord quality: {quality_key!r}."
        )

    if root not in PITCH_CLASS_NAMES:
        raise ValueError(
            f"Unsupported chord root: {root!r}."
        )

    return f"{root} {quality.display}"


def _validate_pitch_class_vector(
    vector: Sequence[float] | np.ndarray,
) -> np.ndarray:
    evidence = np.asarray(
        vector,
        dtype=np.float64,
    )

    if evidence.size != 12:
        raise ValueError(
            "A pitch-class vector must contain exactly 12 values."
        )

    evidence = evidence.reshape(12)

    if not np.all(np.isfinite(evidence)):
        raise ValueError(
            "Pitch-class evidence contains NaN or infinite values."
        )

    if np.any(evidence < 0.0):
        raise ValueError(
            "Pitch-class evidence cannot contain negative values."
        )

    return evidence


def _validate_bass_pitch_class(
    bass_pitch_class: int | None,
) -> int | None:
    if bass_pitch_class is None:
        return None

    bass = int(bass_pitch_class)

    if not 0 <= bass <= 11:
        raise ValueError(
            "bass_pitch_class must be between 0 and 11."
        )

    return bass


def rank_chord_candidates(
    vector: Sequence[float] | np.ndarray,
    *,
    bass_pitch_class: int | None = None,
) -> list[CandidateScore]:
    evidence = _validate_pitch_class_vector(vector)
    bass = _validate_bass_pitch_class(bass_pitch_class)

    total_activity = float(
        np.sum(evidence)
    )

    if total_activity <= 0.0:
        return []

    normalized = evidence / total_activity
    normalized_l2 = normalized / (
        float(np.linalg.norm(normalized)) + 1e-12
    )

    candidates: list[CandidateScore] = []

    for root_pitch_class, root_name in enumerate(
        PITCH_CLASS_NAMES
    ):
        for quality in CORE_QUALITIES:
            tone_indices = tuple(
                (root_pitch_class + interval) % 12
                for interval in quality.intervals
            )

            template = np.zeros(
                12,
                dtype=np.float64,
            )

            for pitch_class, weight in zip(
                tone_indices,
                quality.weights,
                strict=True,
            ):
                template[pitch_class] = weight

            template_l2 = template / (
                float(np.linalg.norm(template)) + 1e-12
            )

            cosine_similarity = float(
                np.dot(
                    normalized_l2,
                    template_l2,
                )
            )

            tone_coverage = float(
                np.sum(
                    normalized[list(tone_indices)]
                )
            )

            root_support = float(
                normalized[root_pitch_class]
            )

            outside_energy = max(
                0.0,
                1.0 - tone_coverage,
            )

            score = (
                0.65 * cosine_similarity
                + 0.25 * tone_coverage
                + 0.10 * root_support
                - 0.25 * outside_energy
                - quality.complexity_penalty
            )

            # Pitch-class sets such as Dsus2 and Asus4 are identical.
            # Optional low-frequency/bass evidence can help distinguish
            # the intended root.
            if bass is not None:
                if root_pitch_class == bass:
                    score += 0.08
                elif bass in tone_indices:
                    score += 0.015
                else:
                    score -= 0.04

            label = f"{root_name}:{quality.key}"

            candidates.append(
                CandidateScore(
                    label=label,
                    display=(
                        f"{root_name} "
                        f"{quality.display}"
                    ),
                    root_pitch_class=root_pitch_class,
                    quality=quality.key,
                    score=float(score),
                    cosine_similarity=cosine_similarity,
                    tone_coverage=tone_coverage,
                    root_support=root_support,
                )
            )

    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return candidates


def classify_pitch_class_vector(
    vector: Sequence[float] | np.ndarray,
    *,
    bass_pitch_class: int | None = None,
    no_chord_activity: float = 1e-8,
    minimum_score: float = (
        DEFAULT_MINIMUM_SCORE
    ),
    ambiguity_margin: float = (
        DEFAULT_AMBIGUITY_MARGIN
    ),
) -> ChordPrediction:
    if no_chord_activity < 0.0:
        raise ValueError(
            "no_chord_activity cannot be negative."
        )

    if not 0.0 <= minimum_score <= 1.5:
        raise ValueError(
            "minimum_score is outside the supported range."
        )

    if ambiguity_margin < 0.0:
        raise ValueError(
            "ambiguity_margin cannot be negative."
        )

    evidence = _validate_pitch_class_vector(vector)

    total_activity = float(
        np.sum(evidence)
    )

    if total_activity <= no_chord_activity:
        return ChordPrediction(
            label="N",
            display="No chord",
            confidence=1.0,
            score=0.0,
            margin=0.0,
            root_pitch_class=None,
            quality=None,
            candidate_label=None,
        )

    candidates = rank_chord_candidates(
        evidence,
        bass_pitch_class=bass_pitch_class,
    )

    best = candidates[0]
    second = candidates[1]

    margin = max(
        0.0,
        best.score - second.score,
    )

    if (
        best.score < minimum_score
        or margin < ambiguity_margin
    ):
        return ChordPrediction(
            label="X",
            display="Uncertain chord",
            confidence=0.0,
            score=best.score,
            margin=margin,
            root_pitch_class=None,
            quality=None,
            candidate_label=best.label,
        )

    confidence = (
        0.75
        * float(
            np.clip(
                best.score,
                0.0,
                1.0,
            )
        )
        + 0.25
        * float(
            np.clip(
                margin / 0.15,
                0.0,
                1.0,
            )
        )
    )

    return ChordPrediction(
        label=best.label,
        display=best.display,
        confidence=float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        ),
        score=best.score,
        margin=margin,
        root_pitch_class=best.root_pitch_class,
        quality=best.quality,
        candidate_label=best.label,
    )
