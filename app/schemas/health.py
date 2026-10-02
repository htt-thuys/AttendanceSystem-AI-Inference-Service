"""Health probe DTOs."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    detector_ready: bool
    recognizer_ready: bool
    providers: list[str]
