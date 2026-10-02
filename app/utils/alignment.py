"""
Face alignment via 5-point similarity transform.

Standard ArcFace / InsightFace preprocessing: the detector returns 5 keypoints
(eyes, nose, mouth corners). We compute a similarity (rotation + uniform scale
+ translation) that maps those points onto canonical positions inside a
112x112 image, then warp the original image with that transform.

This module is pure math; it does not depend on the detector or recognizer
classes.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.constants import ARCFACE_REFERENCE_LANDMARKS, RECOGNIZER_INPUT_SIZE

_REFERENCE = np.asarray(ARCFACE_REFERENCE_LANDMARKS, dtype=np.float32)


class AlignmentError(RuntimeError):
    """Raised when a similarity transform could not be estimated."""


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a face crop to the canonical 112x112 ArcFace template.

    Args:
        image_bgr: source image, HxWx3 BGR uint8.
        landmarks: detected 5 keypoints, shape (5, 2), float pixel coordinates.
            Order: left_eye, right_eye, nose, left_mouth, right_mouth.

    Returns:
        112x112x3 uint8 BGR ndarray ready to feed into the recognizer.

    Raises:
        ValueError: if `landmarks` does not have shape (5, 2).
        AlignmentError: if OpenCV cannot estimate a transform (degenerate points).
    """
    if landmarks.shape != (5, 2):
        raise ValueError(f"landmarks must have shape (5, 2), got {landmarks.shape}")

    src = landmarks.astype(np.float32)
    # `estimateAffinePartial2D` solves for the same 4-DoF transform (rotation +
    # uniform scale + translation) as scikit-image's SimilarityTransform.
    # `LMEDS` makes the estimate robust against a single bad keypoint.
    affine, _ = cv2.estimateAffinePartial2D(src, _REFERENCE, method=cv2.LMEDS)
    if affine is None:
        raise AlignmentError("Could not estimate similarity transform from landmarks.")

    aligned = cv2.warpAffine(
        image_bgr,
        affine,
        RECOGNIZER_INPUT_SIZE,
        borderValue=0.0,
    )
    return aligned
