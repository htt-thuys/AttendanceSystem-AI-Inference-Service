"""
Single source of truth for magic numbers and string constants.

Keep this file free of any logic — values only. Anything that depends on
runtime configuration belongs in `app/core/config.py`.
"""

from __future__ import annotations

from typing import Final

# --- Model file names (under MODELS_DIR) ----------------------------------
DETECTOR_MODEL_FILENAME: Final[str] = "det_10g.onnx"
RECOGNIZER_MODEL_FILENAME: Final[str] = "w600k_r50.onnx"

# --- Detector (SCRFD det_10g) ---------------------------------------------
# Input is a square BGR image; det_10g was trained at 640x640.
DETECTOR_INPUT_SIZE: Final[tuple[int, int]] = (640, 640)
DETECTOR_SCORE_THRESHOLD: Final[float] = 0.5
DETECTOR_NMS_THRESHOLD: Final[float] = 0.4
DETECTOR_MAX_FACES: Final[int] = 10

# --- Recognizer (ArcFace w600k_r50) ---------------------------------------
# ArcFace standard: 112x112 BGR, aligned by 5-point similarity transform.
RECOGNIZER_INPUT_SIZE: Final[tuple[int, int]] = (112, 112)
RECOGNIZER_EMBEDDING_DIM: Final[int] = 512

# --- Canonical 5-point landmarks for ArcFace 112x112 alignment ------------
# Order: left_eye, right_eye, nose, left_mouth, right_mouth.
# Reference: ArcFace / InsightFace official alignment template.
ARCFACE_REFERENCE_LANDMARKS: Final[tuple[tuple[float, float], ...]] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

# --- HTTP response status codes (domain-specific, not HTTP) ---------------
STATUS_OK: Final[str] = "ok"
STATUS_NO_FACE: Final[str] = "no_face"
STATUS_INVALID_IMAGE: Final[str] = "invalid_image"

# --- Misc ------------------------------------------------------------------
MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB upload cap
SUPPORTED_IMAGE_MIMES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
)
