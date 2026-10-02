"""Integration tests for POST /embed."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.constants import RECOGNIZER_EMBEDDING_DIM


def test_embed_returns_vector_when_face_present(
    client_with_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    resp = client_with_face.post(
        "/api/v1/embed",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert len(body["embedding"]) == RECOGNIZER_EMBEDDING_DIM
    assert 0.0 <= body["det_score"] <= 1.0
    assert {"x1", "y1", "x2", "y2"} <= body["bbox"].keys()
    assert len(body["landmarks"]) == 5


def test_embed_returns_no_face_when_detector_empty(
    client_no_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    resp = client_no_face.post(
        "/api/v1/embed",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_face"}


def test_embed_rejects_invalid_image(client_with_face: TestClient) -> None:
    resp = client_with_face.post(
        "/api/v1/embed",
        files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "invalid_image"
    assert "message" in body


def test_embed_rejects_unsupported_mime(client_with_face: TestClient) -> None:
    resp = client_with_face.post(
        "/api/v1/embed",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 415
