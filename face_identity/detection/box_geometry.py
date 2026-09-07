"""Bounding-box arithmetic shared by annotation and by evaluation-set box matching."""
import numpy as np


def expand_box_to_square(
    x: float, y: float, width: float, height: float,
    margin_ratio: float, image_width: int, image_height: int,
) -> tuple[int, int, int, int]:
    """Grows a detection box into a square crop centred on it, clamped to the image bounds.

    The square's side is the longer box side grown by margin_ratio on each side.
    Returns (x1, y1, x2, y2).
    """
    centre_x, centre_y = x + width / 2, y + height / 2
    side = max(width, height) * (1 + 2 * margin_ratio)
    half_side = side / 2
    x1, y1 = max(0, centre_x - half_side), max(0, centre_y - half_side)
    x2, y2 = min(image_width, centre_x + half_side), min(image_height, centre_y + half_side)
    return int(x1), int(y1), int(x2), int(y2)


def compute_intersection_over_union(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Measures how much two x1/y1/x2/y2 boxes overlap, from 0.0 (disjoint) to 1.0 (identical)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def match_boxes_by_overlap(
    reference_boxes: np.ndarray,
    candidate_boxes: np.ndarray,
    min_intersection_over_union: float,
) -> dict[int, int]:
    """Greedily pairs each reference box with the candidate box it overlaps most, one to one.

    Pairs are taken in descending overlap order so the clearest matches win before weaker
    ones. Returns {reference index: candidate index} for the pairs that met the minimum.
    """
    candidate_pairs = []
    for reference_index, reference_box in enumerate(reference_boxes):
        for candidate_index, candidate_box in enumerate(candidate_boxes):
            overlap = compute_intersection_over_union(reference_box, candidate_box)
            if overlap >= min_intersection_over_union:
                candidate_pairs.append((overlap, reference_index, candidate_index))

    matches: dict[int, int] = {}
    used_candidates: set[int] = set()
    for _, reference_index, candidate_index in sorted(candidate_pairs, reverse=True):
        if reference_index in matches or candidate_index in used_candidates:
            continue
        matches[reference_index] = candidate_index
        used_candidates.add(candidate_index)
    return matches
