from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from features import (  # noqa: E402
    chroma_from_audio,
)


class ChromaFeatureTests(
    unittest.TestCase
):
    def test_chroma_receives_power_spectrogram(
        self,
    ) -> None:
        audio = np.ones(
            8,
            dtype=np.float32,
        )

        complex_stft = np.array(
            [
                [
                    1.0 + 1.0j,
                    2.0 + 0.0j,
                ],
                [
                    0.0 + 3.0j,
                    4.0 + 0.0j,
                ],
            ],
            dtype=np.complex128,
        )

        fake_chroma = np.ones(
            (12, 2),
            dtype=np.float64,
        )

        fake_rms = np.ones(
            (1, 2),
            dtype=np.float64,
        )

        with (
            patch(
                "features.librosa.effects.harmonic",
                return_value=audio,
            ),
            patch(
                "features.librosa.stft",
                return_value=complex_stft,
            ),
            patch(
                "features.librosa.feature.chroma_stft",
                return_value=fake_chroma,
            ) as chroma_mock,
            patch(
                "features.librosa.feature.rms",
                return_value=fake_rms,
            ),
            patch(
                "features.librosa.frames_to_time",
                return_value=np.array(
                    [
                        0.0,
                        0.1,
                    ],
                    dtype=np.float64,
                ),
            ),
        ):
            result = chroma_from_audio(
                audio,
                22050,
                hop_length=2,
                frame_length=4,
            )

        supplied_spectrogram = (
            chroma_mock.call_args.kwargs[
                "S"
            ]
        )

        expected_power = (
            np.abs(complex_stft) ** 2
        )

        np.testing.assert_allclose(
            supplied_spectrogram,
            expected_power,
        )

        self.assertEqual(
            result.chroma.shape,
            (12, 2),
        )

    def test_empty_audio_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "empty audio",
        ):
            chroma_from_audio(
                np.array(
                    [],
                    dtype=np.float32,
                ),
                22050,
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
