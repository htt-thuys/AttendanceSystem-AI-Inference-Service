"""Value objects returned by `FaceDetector` implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A single detection result.

    Coordinates are in the original image's pixel space (not the resized
    network input). This keeps downstream code (alignment, cropping)
    independent of detector internals.
    """

    bbox: np.ndarray         # shape (4,) float32: [x1, y1, x2, y2]
    landmarks: np.ndarray    # shape (5, 2) float32: 5 facial keypoints
    score: float             # detector confidence in [0, 1]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
