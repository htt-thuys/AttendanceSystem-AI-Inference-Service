"""Tests for POST /api/v1/detect.

Uses the same FakeDetector fixtures from conftest.py — no real weights needed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_detect_returns_faces_when_face_present(
    client_with_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    assert len(body["faces"]) == 1

    face = body["faces"][0]
    assert 0.0 <= face["det_score"] <= 1.0
    assert {"x1", "y1", "x2", "y2"} <= face["bbox"].keys()
    assert len(face["landmarks"]) == 5
    for lm in face["landmarks"]:
        assert "x" in lm and "y" in lm


def test_detect_bbox_is_valid(
    client_with_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    """x1 < x2 and y1 < y2 (valid non-degenerate box)."""
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    face = resp.json()["faces"][0]
    bbox = face["bbox"]
    assert bbox["x1"] < bbox["x2"]
    assert bbox["y1"] < bbox["y2"]


def test_detect_returns_no_face_when_none_detected(
    client_no_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    resp = client_no_face.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_face"}


def test_detect_rejects_invalid_image(client_with_face: TestClient) -> None:
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "invalid_image"
    assert "message" in body


def test_detect_rejects_unsupported_mime(client_with_face: TestClient) -> None:
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 415


def test_detect_count_matches_faces_length(
    client_with_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    """count field must always equal len(faces)."""
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    body = resp.json()
    assert body["count"] == len(body["faces"])


def test_detect_does_not_return_embedding(
    client_with_face: TestClient,
    jpeg_bytes: bytes,
) -> None:
    """/detect must NOT include an embedding field — that's /embed's job."""
    resp = client_with_face.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    body = resp.json()
    assert "embedding" not in body
    for face in body.get("faces", []):
        assert "embedding" not in face
