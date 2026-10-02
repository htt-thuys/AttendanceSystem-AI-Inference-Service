"""
Abstract contract for face detectors.

Any concrete detector (SCRFD, YOLO-Face, RetinaFace, ...) plugged into the
service MUST implement `FaceDetector`. The pipeline only knows this
interface, so swapping implementations is a one-line dependency change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ml.detector.types import DetectedFace


class FaceDetector(ABC):
    """Detects faces and 5-point landmarks in a BGR image."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Run detection on a single image.

        Args:
            image_bgr: HxWx3 uint8 BGR ndarray (OpenCV native).

        Returns:
            List of `DetectedFace` in original image coordinates. May be empty.
            Implementations are responsible for any NMS / score filtering
            before returning.
        """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True once the underlying model is fully loaded and warmed up."""
