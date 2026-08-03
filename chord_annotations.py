from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import soundfile as sf

from chord_engine import (
    CORE_QUALITIES,
    PITCH_CLASS_NAMES,
)


SCHEMA_VERSION: Final[int] = 1

SUPPORTED_SPLITS: Final[frozenset[str]] = frozenset(
    {
        "development",
        "held_out",
        "exploratory",
    }
)

SUPPORTED_CHORD_LABELS: Final[frozenset[str]] = frozenset(
    {
        "N",
        "X",
        *{
            f"{root}:{quality.key}"
            for root in PITCH_CLASS_NAMES
            for quality in CORE_QUALITIES
        },
    }
)


@dataclass(frozen=True)
class AnnotationSegment:
    start: float
    end: float
    label: str
    notes: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class AnnotatedClip:
    clip_id: str
    split: str
    source: str
    annotation_author: str
    audio_path: Path
    annotation_path: Path
    include_in_primary_metrics: bool
    duration_sec: float
    segments: tuple[AnnotationSegment, ...]


@dataclass(frozen=True)
class ChordDataset:
    dataset_name: str
    dataset_root: Path
    clips: tuple[AnnotatedClip, ...]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    try:
        raw = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ValueError(
            f"Could not read JSON file {path}: {exc}"
        ) from exc

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return value


def _require_string(
    value: Any,
    *,
    field: str,
    context: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{context}.{field} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{context}.{field} cannot be empty."
        )

    return normalized


def _require_boolean(
    value: Any,
    *,
    field: str,
    context: str,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{context}.{field} must be Boolean."
        )

    return value


def _require_positive_float(
    value: Any,
    *,
    field: str,
    context: str,
) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}.{field} must be numeric."
        ) from exc

    if converted <= 0.0:
        raise ValueError(
            f"{context}.{field} must be positive."
        )

    return converted


def _resolve_dataset_path(
    dataset_root: Path,
    relative_path: Any,
    *,
    field: str,
    context: str,
) -> Path:
    raw_path = _require_string(
        relative_path,
        field=field,
        context=context,
    )

    path = Path(raw_path)

    if path.is_absolute():
        raise ValueError(
            f"{context}.{field} must be relative "
            "to the dataset directory."
        )

    resolved_root = dataset_root.resolve()
    resolved_path = (
        resolved_root / path
    ).resolve()

    if (
        resolved_path != resolved_root
        and resolved_root
        not in resolved_path.parents
    ):
        raise ValueError(
            f"{context}.{field} escapes the "
            "dataset directory."
        )

    return resolved_path


def _validate_schema_version(
    value: Any,
    *,
    context: str,
) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(
            f"{context}.schema_version must be "
            f"{SCHEMA_VERSION}; received {value!r}."
        )


def _load_segments(
    annotation: dict[str, Any],
    *,
    annotation_path: Path,
    duration_sec: float,
    timeline_tolerance_sec: float,
) -> tuple[AnnotationSegment, ...]:
    raw_segments = annotation.get(
        "segments"
    )

    if not isinstance(raw_segments, list):
        raise ValueError(
            f"{annotation_path}: segments must "
            "be a list."
        )

    if not raw_segments:
        raise ValueError(
            f"{annotation_path}: at least one "
            "segment is required."
        )

    segments: list[AnnotationSegment] = []

    for index, raw_segment in enumerate(
        raw_segments
    ):
        context = (
            f"{annotation_path}:"
            f"segments[{index}]"
        )

        if not isinstance(raw_segment, dict):
            raise ValueError(
                f"{context} must be an object."
            )

        try:
            start = float(
                raw_segment["start"]
            )
            end = float(
                raw_segment["end"]
            )
        except KeyError as exc:
            raise ValueError(
                f"{context} is missing "
                f"{exc.args[0]!r}."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} start and end "
                "must be numeric."
            ) from exc

        label = _require_string(
            raw_segment.get("label"),
            field="label",
            context=context,
        )

        if label not in SUPPORTED_CHORD_LABELS:
            raise ValueError(
                f"{context}.label {label!r} is "
                "not supported."
            )

        if start < 0.0:
            raise ValueError(
                f"{context}.start cannot "
                "be negative."
            )

        if end <= start:
            raise ValueError(
                f"{context}.end must be "
                "greater than start."
            )

        if end > (
            duration_sec
            + timeline_tolerance_sec
        ):
            raise ValueError(
                f"{context} ends after the "
                "declared clip duration."
            )

        raw_notes = raw_segment.get(
            "notes"
        )

        if (
            raw_notes is not None
            and not isinstance(raw_notes, str)
        ):
            raise ValueError(
                f"{context}.notes must be a "
                "string or null."
            )

        notes = (
            raw_notes.strip()
            if isinstance(raw_notes, str)
            and raw_notes.strip()
            else None
        )

        segment = AnnotationSegment(
            start=start,
            end=end,
            label=label,
            notes=notes,
        )

        if segments:
            previous = segments[-1]
            difference = (
                segment.start
                - previous.end
            )

            if (
                difference
                > timeline_tolerance_sec
            ):
                raise ValueError(
                    f"{context} leaves an "
                    f"unlabelled gap of "
                    f"{difference:.4f} seconds."
                )

            if (
                difference
                < -timeline_tolerance_sec
            ):
                raise ValueError(
                    f"{context} overlaps the "
                    f"previous segment by "
                    f"{abs(difference):.4f} "
                    "seconds."
                )

        segments.append(
            segment
        )

    first_start = segments[0].start

    if first_start > timeline_tolerance_sec:
        raise ValueError(
            f"{annotation_path}: the first "
            "segment must begin at 0 seconds. "
            f"Received {first_start:.4f}."
        )

    final_end = segments[-1].end

    if (
        abs(final_end - duration_sec)
        > timeline_tolerance_sec
    ):
        raise ValueError(
            f"{annotation_path}: final segment "
            "must end at the declared duration "
            f"{duration_sec:.4f}; received "
            f"{final_end:.4f}."
        )

    return tuple(
        segments
    )


def _load_clip(
    dataset_root: Path,
    raw_clip: Any,
    *,
    clip_index: int,
    verify_audio: bool,
    timeline_tolerance_sec: float,
) -> AnnotatedClip:
    context = (
        f"manifest.clips[{clip_index}]"
    )

    if not isinstance(raw_clip, dict):
        raise ValueError(
            f"{context} must be an object."
        )

    clip_id = _require_string(
        raw_clip.get("clip_id"),
        field="clip_id",
        context=context,
    )

    split = _require_string(
        raw_clip.get("split"),
        field="split",
        context=context,
    )

    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"{context}.split must be one of "
            f"{sorted(SUPPORTED_SPLITS)}."
        )

    include_in_primary_metrics = (
        _require_boolean(
            raw_clip.get(
                "include_in_primary_metrics"
            ),
            field=(
                "include_in_primary_metrics"
            ),
            context=context,
        )
    )

    if (
        split == "exploratory"
        and include_in_primary_metrics
    ):
        raise ValueError(
            f"{context}: exploratory clips "
            "cannot be included in primary "
            "metrics."
        )

    audio_path = _resolve_dataset_path(
        dataset_root,
        raw_clip.get("audio_path"),
        field="audio_path",
        context=context,
    )

    annotation_path = (
        _resolve_dataset_path(
            dataset_root,
            raw_clip.get(
                "annotation_path"
            ),
            field="annotation_path",
            context=context,
        )
    )

    if not annotation_path.is_file():
        raise ValueError(
            f"Annotation file does not exist: "
            f"{annotation_path}"
        )

    annotation = _load_json(
        annotation_path
    )

    annotation_context = str(
        annotation_path
    )

    _validate_schema_version(
        annotation.get(
            "schema_version"
        ),
        context=annotation_context,
    )

    annotation_clip_id = (
        _require_string(
            annotation.get("clip_id"),
            field="clip_id",
            context=annotation_context,
        )
    )

    if annotation_clip_id != clip_id:
        raise ValueError(
            f"{annotation_path}: clip_id "
            f"{annotation_clip_id!r} does not "
            f"match manifest clip_id "
            f"{clip_id!r}."
        )

    annotation_split = _require_string(
        annotation.get("split"),
        field="split",
        context=annotation_context,
    )

    if annotation_split != split:
        raise ValueError(
            f"{annotation_path}: split "
            f"{annotation_split!r} does not "
            f"match manifest split "
            f"{split!r}."
        )

    source = _require_string(
        annotation.get("source"),
        field="source",
        context=annotation_context,
    )

    annotation_author = (
        _require_string(
            annotation.get(
                "annotation_author"
            ),
            field="annotation_author",
            context=annotation_context,
        )
    )

    duration_sec = (
        _require_positive_float(
            annotation.get(
                "duration_sec"
            ),
            field="duration_sec",
            context=annotation_context,
        )
    )

    segments = _load_segments(
        annotation,
        annotation_path=annotation_path,
        duration_sec=duration_sec,
        timeline_tolerance_sec=(
            timeline_tolerance_sec
        ),
    )

    if verify_audio:
        if not audio_path.is_file():
            raise ValueError(
                f"Audio file does not exist: "
                f"{audio_path}"
            )

        try:
            audio_info = sf.info(
                str(audio_path)
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Could not inspect audio file "
                f"{audio_path}: {exc}"
            ) from exc

        actual_duration = float(
            audio_info.duration
        )

        duration_tolerance = max(
            timeline_tolerance_sec,
            (
                2.0
                / float(
                    audio_info.samplerate
                )
            ),
        )

        if (
            abs(
                actual_duration
                - duration_sec
            )
            > duration_tolerance
        ):
            raise ValueError(
                f"{annotation_path}: declared "
                f"duration {duration_sec:.4f} "
                "does not match audio duration "
                f"{actual_duration:.4f}."
            )

    return AnnotatedClip(
        clip_id=clip_id,
        split=split,
        source=source,
        annotation_author=(
            annotation_author
        ),
        audio_path=audio_path,
        annotation_path=annotation_path,
        include_in_primary_metrics=(
            include_in_primary_metrics
        ),
        duration_sec=duration_sec,
        segments=segments,
    )


def load_chord_dataset(
    dataset_root: str | Path,
    *,
    verify_audio: bool = True,
    timeline_tolerance_sec: float = 0.03,
) -> ChordDataset:
    if timeline_tolerance_sec < 0.0:
        raise ValueError(
            "timeline_tolerance_sec cannot "
            "be negative."
        )

    root = Path(
        dataset_root
    ).expanduser().resolve()

    manifest_path = (
        root / "manifest.json"
    )

    if not manifest_path.is_file():
        raise ValueError(
            "Dataset manifest does not exist: "
            f"{manifest_path}"
        )

    manifest = _load_json(
        manifest_path
    )

    _validate_schema_version(
        manifest.get(
            "schema_version"
        ),
        context=str(
            manifest_path
        ),
    )

    dataset_name = _require_string(
        manifest.get("dataset_name"),
        field="dataset_name",
        context=str(manifest_path),
    )

    raw_clips = manifest.get(
        "clips"
    )

    if not isinstance(raw_clips, list):
        raise ValueError(
            f"{manifest_path}: clips must "
            "be a list."
        )

    if not raw_clips:
        raise ValueError(
            f"{manifest_path}: at least one "
            "clip is required."
        )

    clips: list[AnnotatedClip] = []
    seen_clip_ids: set[str] = set()

    for index, raw_clip in enumerate(
        raw_clips
    ):
        clip = _load_clip(
            root,
            raw_clip,
            clip_index=index,
            verify_audio=verify_audio,
            timeline_tolerance_sec=(
                timeline_tolerance_sec
            ),
        )

        if clip.clip_id in seen_clip_ids:
            raise ValueError(
                "Duplicate clip_id in manifest: "
                f"{clip.clip_id!r}."
            )

        seen_clip_ids.add(
            clip.clip_id
        )

        clips.append(
            clip
        )

    return ChordDataset(
        dataset_name=dataset_name,
        dataset_root=root,
        clips=tuple(clips),
    )


def summarize_chord_dataset(
    dataset: ChordDataset,
) -> dict[str, Any]:
    split_counts = Counter(
        clip.split
        for clip in dataset.clips
    )

    label_counts = Counter(
        segment.label
        for clip in dataset.clips
        for segment in clip.segments
    )

    primary_clips = [
        clip
        for clip in dataset.clips
        if clip.include_in_primary_metrics
    ]

    primary_x_segments = sum(
        segment.label == "X"
        for clip in primary_clips
        for segment in clip.segments
    )

    return {
        "dataset_name": (
            dataset.dataset_name
        ),
        "clip_count": len(
            dataset.clips
        ),
        "split_counts": dict(
            sorted(
                split_counts.items()
            )
        ),
        "total_duration_sec": round(
            sum(
                clip.duration_sec
                for clip in dataset.clips
            ),
            4,
        ),
        "primary_clip_count": len(
            primary_clips
        ),
        "primary_duration_sec": round(
            sum(
                clip.duration_sec
                for clip in primary_clips
            ),
            4,
        ),
        "primary_x_segment_count": (
            primary_x_segments
        ),
        "label_counts": dict(
            sorted(
                label_counts.items()
            )
        ),
    }
