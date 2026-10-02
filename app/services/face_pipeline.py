"""
Face recognition pipeline — orchestrates detect → align → embed.

Single public class `FacePipeline` so callers (API routes) interact with
one cohesive object instead of juggling detector + recognizer separately.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.ml.detector.interface import FaceDetector
from app.ml.detector.types import DetectedFace
from app.ml.recognizer.interface import FaceEmbedder
from app.ml.recognizer.types import FaceEmbedding
from app.utils.alignment import align_face

logger = get_logger(__name__)


class PipelineResult:
    """Lightweight result holder returned to the API layer.

    Using an explicit class (instead of a tuple) keeps call sites
    self-documenting and avoids accidental ordering bugs.
    """

    __slots__ = ("face", "embedding")

    def __init__(self, face: DetectedFace, embedding: FaceEmbedding) -> None:
        self.face = face
        self.embedding = embedding


class FacePipeline:
    """Stateless orchestrator. Holds references to detector + embedder."""

    def __init__(self, detector: FaceDetector | None = None, embedder: FaceEmbedder | None = None) -> None:
        self._detector = detector
        self._embedder = embedder

    @property
    def detector(self) -> FaceDetector | None:
        return self._detector

    @property
    def embedder(self) -> FaceEmbedder | None:
        return self._embedder

    @property
    def is_ready(self) -> bool:
        det_ready = self._detector.is_ready if self._detector else True
        emb_ready = self._embedder.is_ready if self._embedder else True
        return det_ready and emb_ready

    def run(self, image_bgr: np.ndarray, provided_landmarks: np.ndarray | None = None) -> PipelineResult | None:
        """Detect the dominant face, align it, return its embedding.

        Strategy: when multiple faces are present, prefer the largest
        bounding box (the user standing in front of the camera). This
        mirrors the backend's existing convention in `ml_server/embedder.py`.

        If `provided_landmarks` is given (e.g., in recognition-only mode), detection is skipped.
        Returns None when no face is detected.
        """
        if provided_landmarks is not None:
            # Bypass detection and use provided landmarks (recognition mode)
            if not self._embedder:
                raise RuntimeError("Embedder is not initialized in this mode.")
            aligned = align_face(image_bgr, provided_landmarks)
            embedding = self._embedder.embed(aligned)
            # Create a dummy face to return
            dummy_face = DetectedFace(bbox=np.array([0, 0, 0, 0]), landmarks=provided_landmarks, score=1.0)
            return PipelineResult(face=dummy_face, embedding=embedding)

        if not self._detector:
            raise RuntimeError("Detector is not initialized in this mode. Must provide landmarks.")
            
        faces = self._detector.detect(image_bgr)
        if not faces:
            return None

        face = self._pick_dominant_face(faces)
        if self._embedder:
            aligned = align_face(image_bgr, face.landmarks)
            embedding = self._embedder.embed(aligned)
        else:
            # We don't have an embedder, but we found a face
            embedding = None

        return PipelineResult(face=face, embedding=embedding)

    @staticmethod
    def _pick_dominant_face(faces: list[DetectedFace]) -> DetectedFace:
        """Pick the face with the largest area."""
        return max(faces, key=lambda f: f.area)
