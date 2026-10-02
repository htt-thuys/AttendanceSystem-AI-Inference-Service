"""
Pydantic DTOs for the /detect endpoint.

The detect endpoint returns ALL faces found in the image (up to
DETECTOR_MAX_FACES), unlike /embed which returns only the dominant one.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import STATUS_INVALID_IMAGE, STATUS_NO_FACE, STATUS_OK
from app.schemas.embed import BoundingBox, Landmark  # reuse shared value objects

DetectStatus = Literal["ok", "no_face", "invalid_image"]


class DetectedFaceDTO(BaseModel):
    """A single detected face with its location and confidence."""

    model_config = ConfigDict(frozen=True)

    det_score: float = Field(..., ge=0.0, le=1.0, description="Detector confidence.")
    bbox: BoundingBox
    landmarks: list[Landmark] = Field(
        ...,
        description="5 facial keypoints: left_eye, right_eye, nose, left_mouth, right_mouth.",
        min_length=5,
        max_length=5,
    )


class DetectSuccess(BaseModel):
    """One or more faces detected."""

    status: Literal["ok"] = STATUS_OK
    faces: list[DetectedFaceDTO] = Field(
        ...,
        description="All detected faces, sorted by descending confidence score.",
    )
    count: int = Field(..., ge=0, description="Number of faces detected.")


class DetectNoFace(BaseModel):
    """No face found in the image."""

    status: Literal["no_face"] = STATUS_NO_FACE


class DetectInvalidImage(BaseModel):
    """Image could not be decoded or is unsupported."""

    status: Literal["invalid_image"] = STATUS_INVALID_IMAGE
    message: str


# Discriminated union — pydantic picks the correct variant via `status` literal.
DetectResponse = Annotated[
    DetectSuccess | DetectNoFace | DetectInvalidImage,
    Field(discriminator="status"),
]
