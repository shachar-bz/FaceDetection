"""Deletes images under a people-root directory where SCRFD detects zero or more than one face.

A per-person face database needs exactly one face per image. Walks
--people-root/<group>/<person>/*, runs SCRFD-10G-KPS on each image, and
deletes any image with fewer than MIN_FACES_ALLOWED or more than
MAX_FACES_ALLOWED detections at or above --min-confidence. Writes a CSV log
of every deleted file. Defaults to a dry run (--apply actually deletes).
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_face_database import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SCRFD_INPUT_SIZE,
    DEFAULT_SCRFD_NMS_THRESH,
    build_scrfd_detector,
    detect_faces,
    discover_person_images,
)

MIN_FACES_ALLOWED = 1
MAX_FACES_ALLOWED = 1

LOG_COLUMNS = ["group", "person", "relative_path", "face_count", "status"]


def find_bad_face_count_images(people_root: Path, scrfd_model: Path, min_confidence: float,
                                input_size: int, nms_thresh: float) -> list[dict]:
    """Runs SCRFD over every image under people_root and returns a log row for each image outside the allowed face-count range."""
    detector = build_scrfd_detector(scrfd_model, min_confidence, input_size, nms_thresh)
    person_images = discover_person_images(people_root)
    print(f"Found {len(person_images)} images under {people_root}")

    rows = []
    for i, person_image in enumerate(person_images, start=1):
        image_bgr = cv2.imread(str(person_image.path))
        if image_bgr is None:
            rows.append({
                "group": person_image.group,
                "person": person_image.person,
                "relative_path": str(person_image.relative_path),
                "face_count": 0,
                "status": "unreadable",
            })
            continue

        face_count = len(detect_faces(detector, image_bgr, min_confidence))
        if face_count < MIN_FACES_ALLOWED:
            status = "no_face"
        elif face_count > MAX_FACES_ALLOWED:
            status = "multiple_faces"
        else:
            status = None

        if status is not None:
            rows.append({
                "group": person_image.group,
                "person": person_image.person,
                "relative_path": str(person_image.relative_path),
                "face_count": face_count,
                "status": status,
            })

        if i % 200 == 0:
            print(f"  processed {i}/{len(person_images)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-root", type=Path, required=True,
                         help="Directory laid out as <group>/<person>/<image files>")
    parser.add_argument("--scrfd-model", type=Path, default=Path("models/scrfd_10g_kps.onnx"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--scrfd-input-size", type=int, default=DEFAULT_SCRFD_INPUT_SIZE)
    parser.add_argument("--scrfd-nms-thresh", type=float, default=DEFAULT_SCRFD_NMS_THRESH)
    parser.add_argument("--log-path", type=Path, default=Path("results_scrfd/removed_bad_face_count_images.csv"))
    parser.add_argument("--apply", action="store_true",
                         help="Actually delete the flagged images. Without this flag, only logs them.")
    args = parser.parse_args()

    rows = find_bad_face_count_images(
        args.people_root, args.scrfd_model, args.min_confidence, args.scrfd_input_size, args.scrfd_nms_thresh
    )

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(args.log_path, index=False)
    print(f"{len(rows)} images flagged (no face, multiple faces, or unreadable) -> {args.log_path}")

    if args.apply:
        deleted = 0
        for row in rows:
            image_path = args.people_root / row["relative_path"]
            image_path.unlink(missing_ok=True)
            deleted += 1
        print(f"Deleted {deleted} images.")
    else:
        print("Dry run only (no files deleted). Re-run with --apply to delete the flagged images.")


if __name__ == "__main__":
    main()
