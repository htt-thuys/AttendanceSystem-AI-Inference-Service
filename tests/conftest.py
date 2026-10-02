"""
Pytest fixtures shared across the test suite.

We do NOT load real ONNX models in tests:
  - Weights are hundreds of MB and machine-dependent.
  - CI runs CPU-only and would not exercise the GPU path anyway.

Instead, we install lightweight in-memory fakes that satisfy the
`FaceDetector` and `FaceEmbedder` interfaces. This way the API layer,
schemas, pipeline orchestration, and error paths are fully covered
without the real models being present.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.ml.detector.interface import FaceDetector
from app.ml.detector.types import DetectedFace
from app.ml.recognizer.interface import FaceEmbedder
from app.ml.recognizer.types import FaceEmbedding
from app.services.face_pipeline import FacePipeline


class FakeDetector(FaceDetector):
    """Always returns one face roughly in the centre of the image."""

    def __init__(self, return_face: bool = True) -> None:
        self._return_face = return_face

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if not self._return_face:
            return []
        h, w = image_bgr.shape[:2]
        cx, cy = w / 2, h / 2
        bbox = np.array([cx - 50, cy - 50, cx + 50, cy + 50], dtype=np.float32)
        landmarks = np.array(
            [
                [cx - 20, cy - 10],
                [cx + 20, cy - 10],
                [cx, cy + 5],
                [cx - 15, cy + 25],
                [cx + 15, cy + 25],
            ],
            dtype=np.float32,
        )
        return [DetectedFace(bbox=bbox, landmarks=landmarks, score=0.99)]


class FakeEmbedder(FaceEmbedder):
    """Returns a deterministic L2-normalized vector."""

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def embed(self, aligned_face_bgr: np.ndarray) -> FaceEmbedding:
        rng = np.random.default_rng(seed=42)
        vec = rng.standard_normal(512).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return FaceEmbedding(vector=vec)


@pytest.fixture
def fake_pipeline_with_face() -> FacePipeline:
    return FacePipeline(detector=FakeDetector(return_face=True), embedder=FakeEmbedder())


@pytest.fixture
def fake_pipeline_no_face() -> FacePipeline:
    return FacePipeline(detector=FakeDetector(return_face=False), embedder=FakeEmbedder())


@pytest.fixture
def client_with_face(fake_pipeline_with_face: FacePipeline) -> Iterator[TestClient]:
    """TestClient with the pipeline replaced by a face-returning fake."""
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[dependencies.get_pipeline] = lambda: fake_pipeline_with_face
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_face(fake_pipeline_no_face: FacePipeline) -> Iterator[TestClient]:
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[dependencies.get_pipeline] = lambda: fake_pipeline_no_face
    with TestClient(app) as client:
        yield client


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A tiny valid JPEG image (320x240 black canvas)."""
    import cv2

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok, "Failed to encode test JPEG"
    return buf.tobytes()
