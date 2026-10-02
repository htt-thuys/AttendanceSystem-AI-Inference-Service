"""
SCRFD face detector — concrete `FaceDetector` implementation.

Wraps `det_10g.onnx` from the InsightFace `buffalo_l` model pack.
Architecture: SCRFD-10GF, anchor-free with FCOS-style distance regression,
multi-scale outputs at strides 8 / 16 / 32.

This file is a SCAFFOLD owned by the team member responsible for the
detect module. The plumbing (session, preprocess, NMS, coordinate
remapping) is implemented end-to-end; the only remaining work is to
verify / refine the postprocess loop against your validation set.

References:
  - SCRFD paper: https://arxiv.org/abs/2105.04714
  - InsightFace impl: https://github.com/deepinsight/insightface
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import DETECTOR_INPUT_SIZE
from app.core.logging import get_logger
from app.ml.base import OnnxModel
from app.ml.detector.interface import FaceDetector
from app.ml.detector.types import DetectedFace

logger = get_logger(__name__)

# SCRFD det_10g uses 3 output scales with stride 8/16/32 and 2 anchors per cell.
_STRIDES: tuple[int, ...] = (8, 16, 32)
_NUM_ANCHORS: int = 2


class SCRFDFaceDetector(FaceDetector):
    """SCRFD-10GF face + 5-landmark detector."""

    def __init__(self, model_path: Path) -> None:
        self._model = OnnxModel(model_path)
        self._input_name: str = self._model.input_names[0]
        self._input_size: tuple[int, int] = DETECTOR_INPUT_SIZE
        self._score_threshold: float = settings.DETECTOR_SCORE_THRESHOLD
        self._nms_threshold: float = settings.DETECTOR_NMS_THRESHOLD
        self._max_faces: int = settings.DETECTOR_MAX_FACES

        # Anchor centers are deterministic per input size — cache them.
        self._anchor_centers: dict[int, np.ndarray] = {}
        self._warm_up()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def providers(self) -> list[str]:
        return self._model.providers

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        blob, scale = self._preprocess(image_bgr)
        raw_outputs = self._model.forward({self._input_name: blob})
        return self._postprocess(raw_outputs, scale)

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox-resize to the network input and produce an NCHW blob.

        Returns the blob and the uniform scale factor applied, so the
        postprocess step can map predictions back to original pixels.
        """
        target_w, target_h = self._input_size
        src_h, src_w = image_bgr.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)

        new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        # SCRFD expects (input - 127.5) / 128.0, RGB order, NCHW.
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=self._input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        return blob, scale

    def _postprocess(
        self,
        raw_outputs: list[np.ndarray],
        scale: float,
    ) -> list[DetectedFace]:
        """Decode multi-scale FCOS-style outputs and apply NMS."""
        scores_all: list[np.ndarray] = []
        bboxes_all: list[np.ndarray] = []
        kps_all: list[np.ndarray] = []

        # ONNX output order: [score_8, score_16, score_32,
        #                    bbox_8, bbox_16, bbox_32,
        #                    kps_8,  kps_16,  kps_32]
        num_strides = len(_STRIDES)
        for idx, stride in enumerate(_STRIDES):
            scores = raw_outputs[idx]
            bbox_preds = raw_outputs[idx + num_strides] * stride
            kps_preds = raw_outputs[idx + num_strides * 2] * stride

            anchor_centers = self._get_anchor_centers(stride)

            keep = (scores >= self._score_threshold).reshape(-1)
            if not keep.any():
                continue

            decoded_bboxes = _distance_to_bbox(anchor_centers, bbox_preds)
            decoded_kps = _distance_to_keypoints(anchor_centers, kps_preds)

            scores_all.append(scores.reshape(-1)[keep])
            bboxes_all.append(decoded_bboxes[keep])
            kps_all.append(decoded_kps[keep])

        if not scores_all:
            return []

        scores = np.concatenate(scores_all, axis=0)
        bboxes = np.concatenate(bboxes_all, axis=0) / scale
        kps = np.concatenate(kps_all, axis=0) / scale

        keep_idx = _nms(bboxes, scores, self._nms_threshold)
        keep_idx = keep_idx[: self._max_faces]

        return [
            DetectedFace(
                bbox=bboxes[i].astype(np.float32),
                landmarks=kps[i].astype(np.float32),
                score=float(scores[i]),
            )
            for i in keep_idx
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_anchor_centers(self, stride: int) -> np.ndarray:
        """Build and cache (H*W*A, 2) anchor centers for a given stride."""
        if stride in self._anchor_centers:
            return self._anchor_centers[stride]

        in_w, in_h = self._input_size
        height = in_h // stride
        width = in_w // stride

        ys, xs = np.mgrid[:height, :width]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        # Repeat per anchor on the same cell.
        centers = np.stack([centers] * _NUM_ANCHORS, axis=2).reshape(-1, 2)

        self._anchor_centers[stride] = centers
        return centers

    def _warm_up(self) -> None:
        """Run one dummy inference so the first real call is not slow."""
        dummy = np.zeros((self._input_size[1], self._input_size[0], 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Detector warm-up failed: %s", exc)


# --------------------------------------------------------------------------
# Module-level helpers (pure functions, easy to unit-test in isolation).
# --------------------------------------------------------------------------


def _distance_to_bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode (l, t, r, b) distances at each anchor center into (x1,y1,x2,y2)."""
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_keypoints(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode 10 distances into 5 (x, y) keypoints relative to each anchor."""
    pts = []
    for i in range(0, distances.shape[1], 2):
        px = centers[:, 0] + distances[:, i]
        py = centers[:, 1] + distances[:, i + 1]
        pts.append(np.stack([px, py], axis=-1))
    return np.stack(pts, axis=1)


def _nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy Non-Maximum Suppression. Returns indices sorted by score desc."""
    if bboxes.size == 0:
        return []

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_threshold]

    return keep
