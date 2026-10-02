"""
Abstract contract for face embedders (face recognition models).

The pipeline only depends on this interface, so a custom embedder can be
plugged in by subclassing `FaceEmbedder` and wiring it in
`app/dependencies.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ml.recognizer.types import FaceEmbedding


class FaceEmbedder(ABC):
    """Extracts an identity vector from an aligned face crop."""

    @abstractmethod
    def embed(self, aligned_face_bgr: np.ndarray) -> FaceEmbedding:
        """Run recognition on a single aligned face.

        Args:
            aligned_face_bgr: 112x112x3 uint8 BGR ndarray, already aligned
                via `app.utils.alignment.align_face`.

        Returns:
            FaceEmbedding with an L2-normalized 512-d vector.
        """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True once the underlying model is fully loaded and warmed up."""
