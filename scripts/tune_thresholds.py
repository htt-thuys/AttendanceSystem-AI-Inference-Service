"""
Threshold tuning script for SCRFD detector.

Tests multiple threshold combinations on a test image and reports
how many faces are detected and with what confidence scores.

Usage (from project root, venv active):
    python scripts/tune_thresholds.py --image path/to/face.jpg

If no --image is given, a synthetic test is run (no real faces).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

# Bootstrap CUDA before ORT import
from app.core.runtime import bootstrap_windows_cuda_runtime

bootstrap_windows_cuda_runtime()

from app.core.constants import DETECTOR_MODEL_FILENAME
from app.core.config import settings
from app.ml.detector.scrfd import SCRFDFaceDetector


@dataclass
class ThresholdConfig:
    score: float
    nms: float
    max_faces: int
    label: str


# Threshold combinations to compare
CONFIGS: list[ThresholdConfig] = [
    ThresholdConfig(score=0.3, nms=0.4, max_faces=10, label="Low score (catches more, risk of FP)"),
    ThresholdConfig(score=0.5, nms=0.4, max_faces=10, label="Default (balanced)"),
    ThresholdConfig(score=0.6, nms=0.4, max_faces=1,  label="Recommended for attendance (1 person, strict)"),
    ThresholdConfig(score=0.7, nms=0.4, max_faces=1,  label="Strict (very confident faces only)"),
    ThresholdConfig(score=0.5, nms=0.3, max_faces=10, label="Tight NMS (fewer overlapping boxes)"),
    ThresholdConfig(score=0.5, nms=0.6, max_faces=10, label="Loose NMS (more overlapping boxes kept)"),
]

SEP = "-" * 70


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot read image: {path}")
        sys.exit(1)
    return img


def run_with_config(
    model_path: Path,
    image: np.ndarray,
    cfg: ThresholdConfig,
) -> None:
    """Temporarily override config and run detection."""
    # Patch settings in-place for this call
    original_score = settings.DETECTOR_SCORE_THRESHOLD
    original_nms   = settings.DETECTOR_NMS_THRESHOLD
    original_max   = settings.DETECTOR_MAX_FACES

    settings.DETECTOR_SCORE_THRESHOLD = cfg.score
    settings.DETECTOR_NMS_THRESHOLD   = cfg.nms
    settings.DETECTOR_MAX_FACES       = cfg.max_faces

    try:
        detector = SCRFDFaceDetector(model_path)
        faces = detector.detect(image)
    finally:
        settings.DETECTOR_SCORE_THRESHOLD = original_score
        settings.DETECTOR_NMS_THRESHOLD   = original_nms
        settings.DETECTOR_MAX_FACES       = original_max

    print(f"\n  [{cfg.label}]")
    print(f"  score_threshold={cfg.score}  nms_threshold={cfg.nms}  max_faces={cfg.max_faces}")
    print(f"  --> Faces detected: {len(faces)}")
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        print(f"      Face {i}: score={f.score:.4f}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]  size={w:.0f}x{h:.0f}px")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCRFD threshold tuning tool")
    parser.add_argument("--image", type=str, default=None, help="Path to a face image (JPEG/PNG)")
    args = parser.parse_args()

    model_path = settings.MODELS_DIR / DETECTOR_MODEL_FILENAME

    if args.image:
        image = load_image(args.image)
        h, w = image.shape[:2]
        print(f"\nImage: {args.image}  ({w}x{h})")
    else:
        # Synthetic image — white oval on dark background to simulate a face shape
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(image, (320, 240), (90, 120), 0, 0, 360, (220, 200, 180), -1)
        print("\nNo --image provided. Using synthetic oval (may not trigger detection).")
        print("For real results run:  python scripts/tune_thresholds.py --image your_face.jpg")

    print(f"\n{'='*70}")
    print("SCRFD Threshold Tuning Report")
    print(f"{'='*70}")
    print(f"Model: {model_path}")

    for cfg in CONFIGS:
        print(f"\n{SEP}")
        run_with_config(model_path, image, cfg)

    print(f"\n{SEP}")
    print("\nRECOMMENDATION for attendance system:")
    print("  DETECTOR_SCORE_THRESHOLD=0.6   # confident faces only")
    print("  DETECTOR_NMS_THRESHOLD=0.4     # default works well")
    print("  DETECTOR_MAX_FACES=1           # only 1 person checks in at a time")
    print("\nEdit these in your .env file, then restart uvicorn.")
    print(SEP)


if __name__ == "__main__":
    main()
