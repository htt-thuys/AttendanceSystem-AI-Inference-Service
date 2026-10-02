"""Health & readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import settings
from app.dependencies import get_pipeline
from app.schemas.health import HealthResponse, ReadinessResponse
from app.services.face_pipeline import FacePipeline

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Cheap liveness check — does NOT touch the ML models."""
    return HealthResponse(
        status="ok",
        service=settings.SERVICE_NAME,
        version=__version__,
        environment=settings.ENVIRONMENT,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(pipeline: FacePipeline = Depends(get_pipeline)) -> ReadinessResponse:
    """Returns the loaded state of each model and the active ORT providers."""
    detector_ready = pipeline.detector.is_ready
    recognizer_ready = pipeline.embedder.is_ready
    providers = getattr(pipeline.detector, "providers", [])
    status_label = "ok" if (detector_ready and recognizer_ready) else "degraded"
    return ReadinessResponse(
        status=status_label,
        detector_ready=detector_ready,
        recognizer_ready=recognizer_ready,
        providers=list(providers),
    )
