"""Smoke tests for /health and /ready."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client_with_face: TestClient) -> None:
    resp = client_with_face.get("/api/v1/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "ai-service-system"
    assert "version" in body


def test_ready_reports_pipeline_state(client_with_face: TestClient) -> None:
    resp = client_with_face.get("/api/v1/ready")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["detector_ready"] is True
    assert body["recognizer_ready"] is True
    assert "CPUExecutionProvider" in body["providers"]
