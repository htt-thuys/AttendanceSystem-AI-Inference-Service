"""Unit tests for the SCRFD face detector module.

Covers the pure-math helper functions (no ONNX model required):
  - _distance_to_bbox
  - _distance_to_keypoints
  - _nms

And the value-object DetectedFace:
  - area property
  - frozen/immutability guarantee

All tests run without any model weights and are safe for CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.detector.scrfd import (
    _distance_to_bbox,
    _distance_to_keypoints,
    _nms,
)
from app.ml.detector.types import DetectedFace


# ---------------------------------------------------------------------------
# _distance_to_bbox
# ---------------------------------------------------------------------------


class TestDistanceToBbox:
    """FCOS-style (l, t, r, b) distance → (x1, y1, x2, y2) conversion."""

    def test_single_anchor_centre(self) -> None:
        """A single anchor at (100, 200) with known distances."""
        centers = np.array([[100.0, 200.0]], dtype=np.float32)
        # l=10, t=20, r=30, b=40
        distances = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        result = _distance_to_bbox(centers, distances)

        assert result.shape == (1, 4)
        x1, y1, x2, y2 = result[0]
        assert x1 == pytest.approx(90.0)   # 100 - 10
        assert y1 == pytest.approx(180.0)  # 200 - 20
        assert x2 == pytest.approx(130.0)  # 100 + 30
        assert y2 == pytest.approx(240.0)  # 200 + 40

    def test_multiple_anchors(self) -> None:
        """Vectorised: two anchors produce two bboxes."""
        centers = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float32)
        distances = np.array([[5.0, 5.0, 5.0, 5.0],
                              [10.0, 10.0, 10.0, 10.0]], dtype=np.float32)

        result = _distance_to_bbox(centers, distances)

        assert result.shape == (2, 4)
        # First bbox: centred at (0,0) with distance 5 → [-5,-5,5,5]
        np.testing.assert_allclose(result[0], [-5.0, -5.0, 5.0, 5.0])
        # Second bbox: centred at (50,50) with distance 10 → [40,40,60,60]
        np.testing.assert_allclose(result[1], [40.0, 40.0, 60.0, 60.0])

    def test_zero_distances_produce_degenerate_bbox(self) -> None:
        """Zero distances → point bbox (x1==x2, y1==y2)."""
        centers = np.array([[30.0, 40.0]], dtype=np.float32)
        distances = np.zeros((1, 4), dtype=np.float32)

        result = _distance_to_bbox(centers, distances)

        assert result[0, 0] == pytest.approx(result[0, 2])  # x1 == x2
        assert result[0, 1] == pytest.approx(result[0, 3])  # y1 == y2

    def test_output_order_is_x1_y1_x2_y2(self) -> None:
        """Ensure column order: x1 < x2 and y1 < y2 for positive distances."""
        centers = np.array([[50.0, 50.0]], dtype=np.float32)
        distances = np.array([[5.0, 5.0, 5.0, 5.0]], dtype=np.float32)

        result = _distance_to_bbox(centers, distances)
        x1, y1, x2, y2 = result[0]

        assert x1 < x2
        assert y1 < y2


# ---------------------------------------------------------------------------
# _distance_to_keypoints
# ---------------------------------------------------------------------------


class TestDistanceToKeypoints:
    """Decode 10 distances into 5 (x, y) landmark offsets from anchor centres."""

    def test_single_anchor_produces_five_points(self) -> None:
        centers = np.array([[100.0, 200.0]], dtype=np.float32)
        # 5 pairs of (dx, dy): all offsets = +10
        distances = np.full((1, 10), 10.0, dtype=np.float32)

        result = _distance_to_keypoints(centers, distances)

        assert result.shape == (1, 5, 2)
        # Each keypoint should be at (100+10, 200+10) = (110, 210)
        np.testing.assert_allclose(result[0], [[110.0, 210.0]] * 5)

    def test_zero_offsets_equal_anchor_centres(self) -> None:
        """Zero distances → keypoints coincide with anchor centres."""
        centers = np.array([[64.0, 32.0]], dtype=np.float32)
        distances = np.zeros((1, 10), dtype=np.float32)

        result = _distance_to_keypoints(centers, distances)

        np.testing.assert_allclose(result[0], [[64.0, 32.0]] * 5)

    def test_multiple_anchors(self) -> None:
        """Two anchors → shape (2, 5, 2)."""
        centers = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        distances = np.zeros((2, 10), dtype=np.float32)

        result = _distance_to_keypoints(centers, distances)

        assert result.shape == (2, 5, 2)

    def test_distinct_offsets_per_landmark(self) -> None:
        """Each of the 5 landmarks can have a unique offset."""
        centers = np.array([[0.0, 0.0]], dtype=np.float32)
        # dx, dy pairs: (1,2), (3,4), (5,6), (7,8), (9,10)
        distances = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32)

        result = _distance_to_keypoints(centers, distances)

        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        np.testing.assert_allclose(result[0], expected)


# ---------------------------------------------------------------------------
# _nms
# ---------------------------------------------------------------------------


class TestNms:
    """Greedy NMS — returns indices sorted by score, suppresses overlapping boxes."""

    def test_empty_input_returns_empty(self) -> None:
        bboxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert result == []

    def test_single_box_always_kept(self) -> None:
        bboxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert result == [0]

    def test_non_overlapping_boxes_all_kept(self) -> None:
        """Two boxes far apart → both survive NMS."""
        bboxes = np.array([
            [0.0,   0.0,  10.0,  10.0],
            [100.0, 100.0, 110.0, 110.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert set(result) == {0, 1}

    def test_identical_boxes_suppresses_lower_score(self) -> None:
        """Two perfectly overlapping boxes → only the highest-score survives."""
        box = [0.0, 0.0, 10.0, 10.0]
        bboxes = np.array([box, box], dtype=np.float32)
        scores = np.array([0.6, 0.9], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert len(result) == 1
        assert result[0] == 1  # index of the higher score

    def test_output_sorted_by_score_descending(self) -> None:
        """Returned indices must be in descending confidence order."""
        bboxes = np.array([
            [0.0,  0.0,  5.0,  5.0],
            [20.0, 20.0, 25.0, 25.0],
            [40.0, 40.0, 45.0, 45.0],
        ], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        # All non-overlapping → all kept, but ordered: 0.9 > 0.7 > 0.5
        assert result == [1, 2, 0]

    def test_partial_overlap_below_threshold_kept(self) -> None:
        """IoU slightly below threshold → both boxes kept."""
        # Box A: [0,0,10,10] area=100
        # Box B: [8,8,18,18] area=100  overlap=[8,8,10,10]=4  union=196  IoU=4/196≈0.02
        bboxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [8.0, 8.0, 18.0, 18.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert set(result) == {0, 1}

    def test_high_overlap_above_threshold_suppressed(self) -> None:
        """IoU well above threshold → lower-score box suppressed."""
        # Box A: [0,0,10,10]  Box B: [1,1,11,11]
        # Intersection: [1,1,10,10]=81  Union: 100+100-81=119  IoU≈0.68 > 0.4
        bboxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ], dtype=np.float32)
        scores = np.array([0.95, 0.80], dtype=np.float32)

        result = _nms(bboxes, scores, iou_threshold=0.4)

        assert result == [0]  # only the best box


# ---------------------------------------------------------------------------
# DetectedFace value object
# ---------------------------------------------------------------------------


class TestDetectedFace:
    """Verify the DetectedFace dataclass behaves as a correct value object."""

    def _make_face(
        self,
        x1: float = 10.0, y1: float = 20.0,
        x2: float = 110.0, y2: float = 120.0,
        score: float = 0.9,
    ) -> DetectedFace:
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        return DetectedFace(bbox=bbox, landmarks=landmarks, score=score)

    def test_area_correct(self) -> None:
        face = self._make_face(x1=0, y1=0, x2=50, y2=80)
        assert face.area == pytest.approx(50.0 * 80.0)

    def test_area_zero_for_degenerate_bbox(self) -> None:
        face = self._make_face(x1=5, y1=5, x2=5, y2=5)  # zero-size box
        assert face.area == pytest.approx(0.0)

    def test_area_zero_when_inverted_bbox(self) -> None:
        """If x2 < x1, area should be 0 (clamped), not negative."""
        face = self._make_face(x1=50, y1=50, x2=10, y2=10)
        assert face.area == pytest.approx(0.0)

    def test_is_frozen(self) -> None:
        """DetectedFace is a frozen dataclass — mutation must raise."""
        face = self._make_face()
        with pytest.raises((AttributeError, TypeError)):
            face.score = 0.0  # type: ignore[misc]

    def test_bbox_shape(self) -> None:
        face = self._make_face()
        assert face.bbox.shape == (4,)

    def test_landmarks_shape(self) -> None:
        face = self._make_face()
        assert face.landmarks.shape == (5, 2)
