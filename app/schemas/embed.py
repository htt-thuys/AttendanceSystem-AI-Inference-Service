"""
Pydantic DTOs for the /embed endpoint.

These are the wire contract between the AI service and the Backend
(`attendance-system`). Any breaking change here must be coordinated.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import (
    RECOGNIZER_EMBEDDING_DIM,
    STATUS_INVALID_IMAGE,
    STATUS_NO_FACE,
    STATUS_OK,
)

EmbedStatus = Literal["ok", "no_face", "invalid_image"]


class BoundingBox(BaseModel):
    """Axis-aligned face bounding box in original image coordinates (pixels)."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(..., description="Top-left X")
    y1: float = Field(..., description="Top-left Y")
    x2: float = Field(..., description="Bottom-right X")
    y2: float = Field(..., description="Bottom-right Y")


class Landmark(BaseModel):
    """A single 2D facial keypoint (pixels)."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class EmbedSuccess(BaseModel):
    """Successful embedding response."""

    status: Literal["ok"] = STATUS_OK
    embedding: list[float] = Field(
        ...,
        description=f"L2-normalized {RECOGNIZER_EMBEDDING_DIM}-d face embedding.",
        min_length=RECOGNIZER_EMBEDDING_DIM,
        max_length=RECOGNIZER_EMBEDDING_DIM,
    )
    det_score: float = Field(..., ge=0.0, le=1.0, description="Detector confidence.")
    bbox: BoundingBox
    landmarks: list[Landmark] = Field(
        ...,
        description="5 facial keypoints: left_eye, right_eye, nose, left_mouth, right_mouth.",
        min_length=5,
        max_length=5,
    )


class EmbedNoFace(BaseModel):
    """No face detected (or all faces filtered out)."""

    status: Literal["no_face"] = STATUS_NO_FACE


class EmbedInvalidImage(BaseModel):
    """Image could not be decoded / is too large / unsupported MIME."""

    status: Literal["invalid_image"] = STATUS_INVALID_IMAGE
    message: str


# Discriminated union — pydantic uses the `status` literal to pick the variant,
# which keeps OpenAPI schemas clean and serialization deterministic.
EmbedResponse = Annotated[
    EmbedSuccess | EmbedNoFace | EmbedInvalidImage,
    Field(discriminator="status"),
]
