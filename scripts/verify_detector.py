"""
Diagnostic script — Verify SCRFD postprocess assumptions.

Run from project root (with venv active):
    python scripts/verify_detector.py

What this checks:
  1. ONNX output tensor names and shapes
  2. Anchor grid sizes per stride
  3. Coordinate remapping (bbox in original image space, not padded space)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# Bootstrap Windows CUDA path before importing ORT
from app.core.runtime import bootstrap_windows_cuda_runtime

bootstrap_windows_cuda_runtime()

from app.core.config import settings
from app.core.constants import DETECTOR_INPUT_SIZE, DETECTOR_MODEL_FILENAME
from app.ml.base import OnnxModel
from app.ml.detector.scrfd import SCRFDFaceDetector, _STRIDES, _NUM_ANCHORS

SEPARATOR = "-" * 60


# ── 1. Raw ONNX output names & shapes ────────────────────────────────────────

def check_output_tensors() -> None:
    print(f"\n{SEPARATOR}")
    print("CHECK 1: ONNX output tensor names and shapes")
    print(SEPARATOR)

    model_path = settings.MODELS_DIR / DETECTOR_MODEL_FILENAME
    model = OnnxModel(model_path)

    # Run one dummy forward pass
    in_w, in_h = DETECTOR_INPUT_SIZE
    dummy_blob = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
    raw_outputs = model.forward({model.input_names[0]: dummy_blob})

    expected_names = [
        "score_8", "score_16", "score_32",
        "bbox_8",  "bbox_16",  "bbox_32",
        "kps_8",   "kps_16",   "kps_32",
    ]

    print(f"\nModel output names: {model.output_names}")
    print(f"\n{'Index':<6} {'Name from model':<30} {'Shape':<25} {'Expected name':<20} {'Match?'}")
    print("-" * 90)

    all_match = True
    for i, (tensor, raw) in enumerate(zip(model.output_names, raw_outputs)):
        exp = expected_names[i] if i < len(expected_names) else "?"
        # The actual model may use different name strings; we check shapes instead
        print(f"{i:<6} {tensor:<30} {str(raw.shape):<25} {exp:<20}")

    # Expected shape breakdown for 640×640 input, 2 anchors/cell
    print("\nExpected shape breakdown for 640×640 input, 2 anchors/cell:")
    print(f"  stride=8  -> cells=80x80={80*80}, anchors={_NUM_ANCHORS} -> {80*80*_NUM_ANCHORS} rows")
    print(f"  stride=16 -> cells=40x40={40*40}, anchors={_NUM_ANCHORS} -> {40*40*_NUM_ANCHORS} rows")
    print(f"  stride=32 -> cells=20x20={20*20}, anchors={_NUM_ANCHORS} -> {20*20*_NUM_ANCHORS} rows")

    # Validate shape sizes
    expected_rows = {8: 80*80*_NUM_ANCHORS, 16: 40*40*_NUM_ANCHORS, 32: 20*20*_NUM_ANCHORS}
    for i, stride in enumerate(_STRIDES):
        score_tensor = raw_outputs[i]          # score at stride i
        bbox_tensor  = raw_outputs[i + 3]      # bbox at stride i
        kps_tensor   = raw_outputs[i + 6]      # kps at stride i
        rows = score_tensor.shape[1] if score_tensor.ndim == 2 else score_tensor.reshape(-1).shape[0]

        ok_score = score_tensor.reshape(-1).shape[0] == expected_rows[stride]
        ok_bbox  = bbox_tensor.shape[-1] == 4
        ok_kps   = kps_tensor.shape[-1] == 10

        print(f"\n  stride={stride}:")
        print(f"    score shape={score_tensor.shape}  rows={score_tensor.reshape(-1).shape[0]} expected={expected_rows[stride]}  {'OK' if ok_score else 'MISMATCH'}")
        print(f"    bbox  shape={bbox_tensor.shape}  last_dim=4  {'OK' if ok_bbox else 'MISMATCH'}")
        print(f"    kps   shape={kps_tensor.shape}  last_dim=10  {'OK' if ok_kps else 'MISMATCH'}")

    return raw_outputs, model


# ── 2. Anchor grid cache ──────────────────────────────────────────────────────

def check_anchor_grids(detector: SCRFDFaceDetector) -> None:
    print(f"\n{SEPARATOR}")
    print("CHECK 2: Anchor center grids per stride")
    print(SEPARATOR)

    expected_rows = {8: 80*80*_NUM_ANCHORS, 16: 40*40*_NUM_ANCHORS, 32: 20*20*_NUM_ANCHORS}

    for stride in _STRIDES:
        centers = detector._get_anchor_centers(stride)
        exp = expected_rows[stride]
        ok = centers.shape == (exp, 2)
        print(f"  stride={stride}: shape={centers.shape}  expected=({exp}, 2)  {'OK' if ok else 'MISMATCH'}")
        print(f"           x range=[{centers[:, 0].min():.1f}, {centers[:, 0].max():.1f}]  "
              f"y range=[{centers[:, 1].min():.1f}, {centers[:, 1].max():.1f}]")


# ── 3. Coordinate remapping ───────────────────────────────────────────────────

def check_coordinate_remapping(detector: SCRFDFaceDetector) -> None:
    print(f"\n{SEPARATOR}")
    print("CHECK 3: Coordinate remapping to original image space")
    print(SEPARATOR)

    # Use an image that is NOT square so scale < 1 and remapping is visible
    orig_h, orig_w = 480, 640
    test_image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    # Draw a white rectangle in the centre to vaguely simulate a face region
    cx, cy = orig_w // 2, orig_h // 2
    cv2.rectangle(test_image, (cx - 80, cy - 80), (cx + 80, cy + 80), (255, 255, 255), -1)

    faces = detector.detect(test_image)

    in_w, in_h = DETECTOR_INPUT_SIZE
    scale = min(in_w / orig_w, in_h / orig_h)
    print(f"\n  Original image size : {orig_w}×{orig_h}")
    print(f"  Detector input size : {in_w}×{in_h}")
    print(f"  Scale factor applied: {scale:.4f}")
    print(f"  Faces detected      : {len(faces)}")

    if faces:
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = f.bbox
            valid_coords = (0 <= x1 < x2 <= orig_w) and (0 <= y1 < y2 <= orig_h)
            print(f"\n  Face {i}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]  score={f.score:.3f}")
            print(f"           bbox within original dims ({orig_w}x{orig_h}): {'OK' if valid_coords else 'OUT OF BOUNDS - coordinate remapping broken'}")
            print(f"           landmarks[0]={f.landmarks[0]}  (should be in original pixel space)")
    else:
        print("  No faces detected on test image (expected -- blank canvas).")
        print("  Coordinate path untested here; use a real face image in Swagger.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SCRFD Postprocess Verification Script")
    print("=" * 60)

    model_path = settings.MODELS_DIR / DETECTOR_MODEL_FILENAME
    print(f"\nLoading model: {model_path}")

    detector = SCRFDFaceDetector(model_path)

    raw_outputs, model = check_output_tensors()
    check_anchor_grids(detector)
    check_coordinate_remapping(detector)

    print(f"\n{SEPARATOR}")
    print("Done. Fix any ❌ items before moving to Phase 4.")
    print(SEPARATOR)
