"""
FastAPI entry point.

Layout:
  - Lifespan: bootstraps the ML pipeline (singleton) and tears it down.
  - CORS: opt-in via `CORS_ORIGINS` env (default `*` in dev).
  - Routes: mounted under `settings.API_PREFIX`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

# Windows: register NVIDIA wheel DLL paths before any onnxruntime import.
from app.core import runtime as _runtime  # noqa: F401

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.dependencies import build_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Boot heavy resources once, before traffic is accepted."""
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Starting %s v%s (env=%s)", settings.SERVICE_NAME, __version__, settings.ENVIRONMENT)

    build_pipeline()
    logger.info("AI service ready on %s:%s%s", settings.HOST, settings.PORT, settings.API_PREFIX)
    yield
    logger.info("Shutting down %s", settings.SERVICE_NAME)


def create_app() -> FastAPI:
    """Application factory (kept separate so tests can build alternate apps)."""
    app = FastAPI(
        title="AI Service System",
        description="Face detection + embedding microservice for the attendance system.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()
