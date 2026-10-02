"""
POST /embed — main inference endpoint.

Contract (consumed by the BE `attendance-system` service):
  - Request: multipart/form-data with a single `image` file field.
  - Response: one of EmbedSuccess | EmbedNoFace | EmbedInvalidImage.

The endpoint is intentionally thin — all business logic lives in
`FacePipeline`; this layer only validates input and shapes the response.
"""

from __future__ import annotations

import json
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.constants import SUPPORTED_IMAGE_MIMES
from app.core.logging import get_logger
from app.dependencies import get_pipeline
from app.schemas.embed import (
    BoundingBox,
    EmbedInvalidImage,
    EmbedNoFace,
    EmbedResponse,
    EmbedSuccess,
    Landmark,
)
from app.services.face_pipeline import FacePipeline
from app.utils.image import InvalidImageError, decode_image_bgr

router = APIRouter(tags=["Embedding"])
logger = get_logger(__name__)


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Detect the dominant face and return its 512-d embedding.",
)
async def embed(
    image: UploadFile = File(..., description="Face image (JPEG / PNG / WEBP / BMP)."),
    landmarks: str | None = Form(default=None, description="Optional JSON string of 5 landmarks [[x,y]...] for recognition-only mode."),
    pipeline: FacePipeline = Depends(get_pipeline),
) -> EmbedResponse:
    _validate_upload(image)
    image_bytes = await image.read()
    _validate_size(image_bytes)

    try:
        image_bgr = decode_image_bgr(image_bytes)
    except InvalidImageError as exc:
        logger.info("Invalid image rejected: %s", exc)
        return EmbedInvalidImage(message=str(exc))

    if settings.SERVICE_MODE == "detection":
        if not settings.RECOGNITION_SERVICE_URL:
            raise HTTPException(status_code=500, detail="RECOGNITION_SERVICE_URL not configured")
            
        faces = pipeline.detector.detect(image_bgr)
        if not faces:
            return EmbedNoFace()
            
        best_face = max(faces, key=lambda f: f.score)
        landmarks_json = json.dumps(best_face.landmarks.tolist())
        
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{settings.RECOGNITION_SERVICE_URL}/api/v1/embed",
                    files={"image": (image.filename or "image.jpg", image_bytes, image.content_type)},
                    data={"landmarks": landmarks_json},
                    timeout=10.0
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Error communicating with recognition service: {e}")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    parsed_landmarks = None
    if landmarks:
        try:
            parsed_list = json.loads(landmarks)
            parsed_landmarks = np.array(parsed_list, dtype=np.float32)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid landmarks format: {e}")

    try:
        result = pipeline.run(image_bgr, provided_landmarks=parsed_landmarks)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result is None:
        return EmbedNoFace()

    face = result.face
    return EmbedSuccess(
        embedding=result.embedding.to_list(),
        det_score=float(face.score),
        bbox=BoundingBox(
            x1=float(face.bbox[0]),
            y1=float(face.bbox[1]),
            x2=float(face.bbox[2]),
            y2=float(face.bbox[3]),
        ),
        landmarks=[Landmark(x=float(p[0]), y=float(p[1])) for p in face.landmarks],
    )


# --------------------------------------------------------------------------
# Validation helpers
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
