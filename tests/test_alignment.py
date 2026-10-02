"""Unit tests for the face alignment helper."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.constants import RECOGNIZER_INPUT_SIZE
from app.utils.alignment import align_face


def test_align_face_returns_expected_shape() -> None:
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    landmarks = np.array(
        [
            [200.0, 200.0],
            [280.0, 200.0],
            [240.0, 240.0],
            [210.0, 290.0],
            [270.0, 290.0],
        ],
        dtype=np.float32,
    )

    aligned = align_face(image, landmarks)

    assert aligned.shape == (RECOGNIZER_INPUT_SIZE[1], RECOGNIZER_INPUT_SIZE[0], 3)
    assert aligned.dtype == np.uint8


def test_align_face_rejects_wrong_landmark_shape() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bad = np.zeros((4, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        align_face(image, bad)
