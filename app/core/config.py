"""
Application configuration loaded from environment variables / .env.

Single instance: `settings` (imported throughout the codebase).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. All values overridable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service identity ---
    SERVICE_NAME: str = "ai-service-system"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = Field(default="INFO")
    SERVICE_MODE: str = Field(
        default="all", 
        description="Mode to run the service in: all | detection | recognition"
    )
    RECOGNITION_SERVICE_URL: str | None = Field(
        default=None,
        description="URL of the recognition service to proxy requests to when running in detection mode."
    )

    # --- HTTP server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    API_PREFIX: str = "/api/v1"

    # --- CORS ---
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    # --- Model weights ---
    MODELS_DIR: Path = Field(
        default=Path("./buffalo_l"),
        description="Directory containing ONNX weights (det_10g.onnx, w600k_r50.onnx).",
    )

    # --- Inference runtime ---
    USE_GPU: bool = Field(
        default=True,
        description="Try CUDAExecutionProvider first; fall back to CPU if unavailable.",
    )
    ORT_INTRA_OP_THREADS: int = Field(
        default=0,
        description="0 = let ONNX Runtime pick. Set explicitly on CPU-only nodes.",
    )

    # --- Detector tuning (overridable per-deployment) ---
    DETECTOR_SCORE_THRESHOLD: float = 0.5
    DETECTOR_NMS_THRESHOLD: float = 0.4
    DETECTOR_MAX_FACES: int = 10

    # --- Upload guard ---
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

    @field_validator("LOG_LEVEL")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("MODELS_DIR")
    @classmethod
    def _resolve_models_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (instantiated once per process)."""
    return Settings()


settings: Settings = get_settings()
