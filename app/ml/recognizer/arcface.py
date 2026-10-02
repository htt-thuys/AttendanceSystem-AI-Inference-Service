
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.constants import RECOGNIZER_EMBEDDING_DIM, RECOGNIZER_INPUT_SIZE
from app.core.logging import get_logger
from app.ml.base import OnnxModel
from app.ml.recognizer.interface import FaceEmbedder
from app.ml.recognizer.types import FaceEmbedding

logger = get_logger(__name__)

# Standard ArcFace normalization constants.
_PIXEL_MEAN: float = 127.5
_PIXEL_STD: float = 127.5
_L2_EPSILON: float = 1e-12


class ArcFaceEmbedder(FaceEmbedder):
    """ArcFace R50 face embedder."""

    def __init__(self, model_path: Path) -> None:
        self._model = OnnxModel(model_path)
        self._input_name: str = self._model.input_names[0]
        self._input_size: tuple[int, int] = RECOGNIZER_INPUT_SIZE
        self._is_ready: bool = False
        self._warm_up()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def providers(self) -> list[str]:
        return self._model.providers

    def embed(self, aligned_face_bgr: np.ndarray) -> FaceEmbedding:
        if aligned_face_bgr.shape[:2] != (self._input_size[1], self._input_size[0]):
            raise ValueError(
                f"aligned_face_bgr must be {self._input_size[1]}x{self._input_size[0]}, "
                f"got {aligned_face_bgr.shape[:2]}"
            )

        blob = self._preprocess(aligned_face_bgr)
        raw = self._model.forward({self._input_name: blob})
        vector = self._postprocess(raw)
        return FaceEmbedding(vector=vector)

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #

    def _preprocess(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """Convert a 112x112 BGR uint8 face into an NCHW float32 blob."""
        return cv2.dnn.blobFromImage(
            aligned_face_bgr,
            scalefactor=1.0 / _PIXEL_STD,
            size=self._input_size,
            mean=(_PIXEL_MEAN, _PIXEL_MEAN, _PIXEL_MEAN),
            swapRB=True,
            crop=False,
        )

    def _postprocess(self, raw_outputs: list[np.ndarray]) -> np.ndarray:
        """Squeeze the batch dim and L2-normalize the 512-d embedding."""
        feat = raw_outputs[0][0]
        if feat.shape != (RECOGNIZER_EMBEDDING_DIM,):
            raise RuntimeError(
                f"Unexpected embedding shape from recognizer: {feat.shape}; "
                f"expected ({RECOGNIZER_EMBEDDING_DIM},)."
            )
        norm = float(np.linalg.norm(feat))
        return (feat / (norm + _L2_EPSILON)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _warm_up(self) -> None:
        """Run a single dummy inference to surface wiring errors at startup."""
        dummy = np.zeros(
            (self._input_size[1], self._input_size[0], 3),
            dtype=np.uint8,
        )
        try:
            self.embed(dummy)
        except Exception as exc:  # pragma: no cover — defensive
            logger.error("Recognizer warm-up failed: %s", exc)
            return

        self._is_ready = True
