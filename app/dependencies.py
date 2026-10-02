"""
FastAPI dependency providers — singleton model lifecycle.

Why a module-level singleton rather than `Depends(get_pipeline)` for every
request:
  - ONNX sessions are expensive to load (hundreds of MB).
  - They are thread-safe (the runtime serializes internally).
  - We want them initialized at app startup (lifespan), not on first
    request — otherwise the first user pays a ~5s latency tax.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.constants import DETECTOR_MODEL_FILENAME, RECOGNIZER_MODEL_FILENAME
from app.core.logging import get_logger
from app.ml.detector.scrfd import SCRFDFaceDetector
from app.ml.recognizer.arcface import ArcFaceEmbedder
from app.services.face_pipeline import FacePipeline

logger = get_logger(__name__)

_pipeline: FacePipeline | None = None


def build_pipeline() -> FacePipeline:
    """Construct the singleton pipeline. Called once from the lifespan."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    detector_path = settings.MODELS_DIR / DETECTOR_MODEL_FILENAME
    recognizer_path = settings.MODELS_DIR / RECOGNIZER_MODEL_FILENAME

    detector = None
    if settings.SERVICE_MODE in ("all", "detection"):
        logger.info("Initializing detector from %s", detector_path)
        detector = SCRFDFaceDetector(detector_path)

    recognizer = None
    if settings.SERVICE_MODE in ("all", "recognition"):
        logger.info("Initializing recognizer from %s", recognizer_path)
        recognizer = ArcFaceEmbedder(recognizer_path)

    _pipeline = FacePipeline(detector=detector, embedder=recognizer)
    return _pipeline


def get_pipeline() -> FacePipeline:
    """FastAPI dependency — returns the loaded pipeline.

    Raises a RuntimeError if called before `build_pipeline()` (would only
    happen if the lifespan was bypassed, e.g. in some tests).
    """
    if _pipeline is None:
        raise RuntimeError(
            "FacePipeline is not initialized. "
            "Did the FastAPI lifespan run? Call build_pipeline() first."
        )
    return _pipeline


def reset_pipeline_for_tests() -> None:
    """Test-only hook to discard the singleton between test cases."""
    global _pipeline
    _pipeline = None
