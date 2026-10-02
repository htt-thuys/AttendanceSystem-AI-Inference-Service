"""Top-level API router — aggregates all feature routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import detect, embed, health

from app.core.config import settings

api_router = APIRouter()
api_router.include_router(health.router)

if settings.SERVICE_MODE in ("all", "recognition", "detection"):
    api_router.include_router(embed.router)

if settings.SERVICE_MODE in ("all", "detection"):
    api_router.include_router(detect.router)
