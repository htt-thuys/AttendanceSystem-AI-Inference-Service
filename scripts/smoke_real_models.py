"""
Smoke test against the real ONNX weights.

Runs OUTSIDE pytest so the full ~340 MB model load happens only when this
script is called explicitly (not on every `pytest` run on CI).

Validates:
  1. FacePipeline can be built with the real det_10g.onnx + w600k_r50.onnx.
  2. Both models become `is_ready` after warm-up.
  3. /api/v1/embed returns the expected envelope shape end-to-end.

Usage (from repo root):
    python scripts/smoke_real_models.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Make `import app` work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # noqa: E402


def _synthetic_face_image() -> bytes:
    """Build a tiny synthetic 'face-like' image so SCRFD has something to find.

    SCRFD is robust enough to fire on rough oval shapes with eye/nose
    contrast. We don't care about identity correctness — only the
    end-to-end envelope.
    """
    canvas = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Face oval
    cv2.ellipse(canvas, (320, 240), (110, 140), 0, 0, 360, (180, 160, 150), -1)
    # Eyes
    cv2.circle(canvas, (280, 220), 12, (40, 40, 40), -1)
    cv2.circle(canvas, (360, 220), 12, (40, 40, 40), -1)
    # Nose
    cv2.line(canvas, (320, 235), (320, 270), (90, 80, 80), 3)
    # Mouth
    cv2.ellipse(canvas, (320, 305), (30, 10), 0, 0, 180, (60, 40, 50), 3)

    ok, buf = cv2.imencode(".jpg", canvas)
    if not ok:
        raise RuntimeError("Failed to encode synthetic test image.")
    return buf.tobytes()


def main() -> int:
    print("[smoke] building FastAPI app (this loads both ONNX models)...")
    app = create_app()

    with TestClient(app) as client:
        print("[smoke] GET /api/v1/health")
        r = client.get("/api/v1/health")
        print("   ", r.status_code, r.json())
        assert r.status_code == 200

        print("[smoke] GET /api/v1/ready")
        r = client.get("/api/v1/ready")
        print("   ", r.status_code, r.json())
        assert r.status_code == 200
        body = r.json()
        assert body["detector_ready"] is True, "Detector failed to warm up"
        assert body["recognizer_ready"] is True, "Recognizer failed to warm up"

        print("[smoke] POST /api/v1/embed with synthetic face image")
        image_bytes = _synthetic_face_image()
        r = client.post(
            "/api/v1/embed",
            files={"image": ("face.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        print("   ", r.status_code)
        assert r.status_code == 200
        body = r.json()

        if body["status"] == "ok":
            emb = body["embedding"]
            vec = np.array(emb, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            print(f"    status=ok | det_score={body['det_score']:.3f} | "
                  f"|embedding|={len(emb)} | ||v||_2={norm:.6f}")
            assert len(emb) == 512, f"Expected 512-d, got {len(emb)}"
            assert abs(norm - 1.0) < 1e-3, f"Embedding not L2-normalized (||v||={norm})"
            assert len(body["landmarks"]) == 5
            print("[smoke] PASS — pipeline works end-to-end with real weights.")
        elif body["status"] == "no_face":
            print("    status=no_face — synthetic image was too simple for SCRFD.")
            print("    Pipeline is wired correctly; rerun with a real photo to validate inference.")
        else:
            print("    unexpected response:", body)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
