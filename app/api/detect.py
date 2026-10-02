"""
POST /detect — face detection endpoint.

Returns ALL detected faces (up to DETECTOR_MAX_FACES) in the image with
their bounding boxes, 5-point landmarks, and confidence scores.

Unlike /embed, this endpoint:
  - Does NOT run alignment or embedding (faster).
  - Returns every detected face, not just the dominant one.
  - Is useful for debugging detection quality and validating thresholds.

Contract:
  - Request: multipart/form-data with a single `image` file field.
  - Response: one of DetectSuccess | DetectNoFace | DetectInvalidImage.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.constants import SUPPORTED_IMAGE_MIMES
from app.core.logging import get_logger
from app.dependencies import get_pipeline
from app.schemas.detect import (
    DetectInvalidImage,
    DetectNoFace,
    DetectResponse,
    DetectSuccess,
    DetectedFaceDTO,
)
from app.schemas.embed import BoundingBox, Landmark
from app.services.face_pipeline import FacePipeline
from app.utils.image import InvalidImageError, decode_image_bgr

router = APIRouter(tags=["Detection"])
logger = get_logger(__name__)


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect all faces in an image and return bounding boxes + landmarks.",
)
async def detect(
    image: UploadFile = File(..., description="Image file (JPEG / PNG / WEBP / BMP)."),
    pipeline: FacePipeline = Depends(get_pipeline),
) -> DetectResponse:
    """Detect every face in the uploaded image.

    Returns all detected faces sorted by confidence (highest first), each
    with its bounding box, 5-point landmarks, and detector score.
    Does **not** compute embeddings — use `/embed` for that.
    """
    _validate_upload(image)
    image_bytes = await image.read()
    _validate_size(image_bytes)

    try:
        image_bgr = decode_image_bgr(image_bytes)
    except InvalidImageError as exc:
        logger.info("Invalid image rejected: %s", exc)
        return DetectInvalidImage(message=str(exc))

    faces = pipeline.detector.detect(image_bgr)

    if not faces:
        return DetectNoFace()

    # Sort by score descending so the most confident face is first.
    faces_sorted = sorted(faces, key=lambda f: f.score, reverse=True)

    face_dtos = [
        DetectedFaceDTO(
            det_score=float(f.score),
            bbox=BoundingBox(
                x1=float(f.bbox[0]),
                y1=float(f.bbox[1]),
                x2=float(f.bbox[2]),
                y2=float(f.bbox[3]),
            ),
            landmarks=[Landmark(x=float(p[0]), y=float(p[1])) for p in f.landmarks],
        )
        for f in faces_sorted
    ]

    logger.info("Detected %d face(s) in uploaded image.", len(face_dtos))
    return DetectSuccess(faces=face_dtos, count=len(face_dtos))


# --------------------------------------------------------------------------
# Validation helpers (shared logic with /embed)
# --------------------------------------------------------------------------


def _validate_upload(image: UploadFile) -> None:
    if image.content_type and image.content_type not in SUPPORTED_IMAGE_MIMES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image MIME: {image.content_type}",
        )


def _validate_size(image_bytes: bytes) -> None:
    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.MAX_UPLOAD_BYTES} bytes.",
        )
