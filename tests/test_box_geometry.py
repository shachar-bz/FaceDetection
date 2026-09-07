"""Tests for box overlap and the greedy pairing that binds labelled faces to detected ones.

These decide which detection a ground-truth label is scored against, so a bug here silently
shifts every identification metric without any error being raised.
"""
import numpy as np
import pytest

from face_identity.detection.box_geometry import (
    compute_intersection_over_union,
    expand_box_to_square,
    match_boxes_by_overlap,
)


def box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Builds one x1/y1/x2/y2 box."""
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class TestComputeIntersectionOverUnion:
    def test_identical_boxes_overlap_completely(self):
        assert compute_intersection_over_union(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes_do_not_overlap(self):
        assert compute_intersection_over_union(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0

    def test_touching_edges_do_not_count_as_overlap(self):
        assert compute_intersection_over_union(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
        assert compute_intersection_over_union(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_contained_box_scores_the_area_ratio(self):
        # A 5x5 box entirely inside a 10x10 one: intersection 25, union 100.
        assert compute_intersection_over_union(box(0, 0, 10, 10), box(0, 0, 5, 5)) == pytest.approx(0.25)

    def test_overlap_is_symmetric(self):
        first, second = box(0, 0, 10, 10), box(3, 4, 12, 16)
        assert compute_intersection_over_union(first, second) == pytest.approx(
            compute_intersection_over_union(second, first))


class TestMatchBoxesByOverlap:
    def test_each_box_pairs_with_its_own_match(self):
        labelled = np.array([box(0, 0, 10, 10), box(100, 100, 110, 110)])
        detected = np.array([box(100, 100, 110, 110), box(0, 0, 10, 10)])
        assert match_boxes_by_overlap(labelled, detected, 0.5) == {0: 1, 1: 0}

    def test_pairing_is_one_to_one_and_the_strongest_overlap_wins(self):
        # Both labelled boxes overlap detection 0, but only one may claim it.
        labelled = np.array([box(0, 0, 10, 10), box(1, 1, 11, 11)])
        detected = np.array([box(0, 0, 10, 10)])
        matches = match_boxes_by_overlap(labelled, detected, 0.5)
        assert matches == {0: 0}

    def test_overlap_below_the_minimum_is_not_a_match(self):
        labelled = np.array([box(0, 0, 10, 10)])
        detected = np.array([box(8, 8, 18, 18)])
        assert match_boxes_by_overlap(labelled, detected, 0.5) == {}

    def test_no_detections_yields_no_matches(self):
        labelled = np.array([box(0, 0, 10, 10)])
        assert match_boxes_by_overlap(labelled, np.zeros((0, 4), dtype=np.float32), 0.5) == {}

    def test_no_labels_yields_no_matches(self):
        detected = np.array([box(0, 0, 10, 10)])
        assert match_boxes_by_overlap(np.zeros((0, 4), dtype=np.float32), detected, 0.5) == {}


class TestExpandBoxToSquare:
    def test_the_result_is_square_when_it_fits_in_the_image(self):
        x1, y1, x2, y2 = expand_box_to_square(100, 100, 20, 40, 0.5, 1000, 1000)
        assert (x2 - x1) == (y2 - y1)

    def test_the_longer_side_drives_the_square(self):
        # Longer side 40, grown by 0.5 per side -> 40 * 2 = 80.
        x1, y1, x2, y2 = expand_box_to_square(100, 100, 20, 40, 0.5, 1000, 1000)
        assert (x2 - x1) == 80

    def test_a_zero_margin_still_squares_the_box(self):
        x1, y1, x2, y2 = expand_box_to_square(100, 100, 20, 40, 0.0, 1000, 1000)
        assert (x2 - x1, y2 - y1) == (40, 40)

    def test_the_square_is_clamped_to_the_image_bounds(self):
        x1, y1, x2, y2 = expand_box_to_square(0, 0, 20, 20, 1.0, 30, 30)
        assert (x1, y1) == (0, 0)
        assert x2 <= 30 and y2 <= 30

    def test_the_square_stays_centred_on_the_original_box(self):
        x1, y1, x2, y2 = expand_box_to_square(100, 200, 20, 40, 0.5, 1000, 1000)
        assert (x1 + x2) / 2 == pytest.approx(110)
        assert (y1 + y2) / 2 == pytest.approx(220)
