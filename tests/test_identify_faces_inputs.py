"""Tests folder discovery and CSV rows for user-supplied evaluation images."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from identify_faces import RESULT_COLUMNS, build_result_rows, discover_input_images


def test_directory_input_finds_supported_images_recursively_in_stable_order(tmp_path):
    """Directory mode finds nested images while ignoring unrelated files."""
    for relative_path in ["nested/b.png", "a.jpg", "notes.txt"]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    images, is_directory = discover_input_images(tmp_path)

    assert is_directory
    assert [relative_path.as_posix() for _, relative_path in images] == ["a.jpg", "nested/b.png"]


def test_one_image_input_remains_supported(tmp_path):
    """The original single-image CLI behavior remains available."""
    image_path = tmp_path / "photo.webp"
    image_path.write_bytes(b"test")

    images, is_directory = discover_input_images(image_path)

    assert not is_directory
    assert images == [(image_path, Path("photo.webp"))]


def test_no_face_image_still_has_a_csv_row():
    """A batch result never silently loses an image with no detected faces."""
    [row] = build_result_rows(Path("empty.jpg"), [])

    assert list(row) == RESULT_COLUMNS
    assert row["relative_path"] == "empty.jpg"
    assert row["face_index"] == ""
    assert row["error"] == ""


def test_identified_face_fields_are_written_to_the_csv_row():
    """A detected face produces a complete reviewable result row."""
    face = SimpleNamespace(
        predicted_person="Person One",
        is_known=True,
        similarity_score=0.87654,
        bounding_box_xyxy=np.array([1.2, 2.3, 30.4, 40.5]),
        detection_confidence=0.91234,
    )

    [row] = build_result_rows(Path("group/photo.jpg"), [face])

    assert row["relative_path"] == "group/photo.jpg"
    assert row["predicted_person"] == "Person One"
    assert row["similarity_score"] == 0.8765
    assert row["detection_confidence"] == 0.9123
    assert [row[name] for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")] == [
        1.2,
        2.3,
        30.4,
        40.5,
    ]
