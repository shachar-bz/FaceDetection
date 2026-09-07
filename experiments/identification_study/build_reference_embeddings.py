"""Builds the known-people reference embedding database the identification study scores against.

Walks every image under --people-root, detects the faces in it once, and stores each face's
embedding under both study models plus a manifest row naming the person it belongs to. The
manifest is the authoritative index: stale .npz files left behind by an earlier build are
ignored by everything downstream.
"""
import argparse
from pathlib import Path

import pandas as pd

from experiments.identification_study.study_face_embedding import (
    build_study_detector,
    build_study_embedders,
    embed_image_faces,
    save_face_embedding_npz,
    unreadable_image_row,
)
from face_identity.configuration import (
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
)
from face_identity.image_io import discover_person_images, read_image_bgr

DEFAULT_OUTPUT_DIR = Path("results/embeddings")
MANIFEST_FILENAME = "embeddings_manifest.csv"

MANIFEST_COLUMNS = [
    "group",
    "person",
    "relative_path",
    "face_index",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "embedding_path",
    "error",
]


def build_manifest_row(person_image, record, embedding_path: Path) -> dict:
    """Describes one embedded reference face as a manifest row."""
    x1, y1, x2, y2 = record.bounding_box_xyxy
    return {
        "group": person_image.group,
        "person": person_image.person,
        "relative_path": str(person_image.relative_path),
        "face_index": record.face_index,
        "confidence": round(record.detection_confidence, 4),
        "bbox_x": round(float(x1), 1),
        "bbox_y": round(float(y1), 1),
        "bbox_w": round(float(x2 - x1), 1),
        "bbox_h": round(float(y2 - y1), 1),
        "embedding_path": str(embedding_path),
        "error": "",
    }


def main() -> None:
    """Extracts and stores both models' embeddings for every reference image of every person."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-root", type=Path, required=True,
                        help="Directory laid out as <group>/<person>/<image files>")
    parser.add_argument("--include-groups", nargs="+", default=None,
                        help="Only scan these top-level group folder names (default: all)")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--det-size", type=int, default=DEFAULT_DETECTION_INPUT_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()

    embeddings_dir = arguments.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    detector = build_study_detector(arguments.models_dir, arguments.min_confidence, arguments.det_size)
    embedders_by_model = build_study_embedders(arguments.models_dir)

    include_groups = set(arguments.include_groups) if arguments.include_groups else None
    person_images = discover_person_images(arguments.people_root, include_groups)
    print(f"Found {len(person_images)} images under {arguments.people_root}")

    manifest_rows = []
    for person_image in person_images:
        image_bgr = read_image_bgr(person_image.path)
        if image_bgr is None:
            manifest_rows.append(unreadable_image_row(
                MANIFEST_COLUMNS,
                group=person_image.group,
                person=person_image.person,
                relative_path=str(person_image.relative_path),
            ))
            continue

        for record in embed_image_faces(image_bgr, detector, embedders_by_model):
            embedding_path = (
                embeddings_dir / person_image.group / person_image.person
                / f"{person_image.path.stem}_face{record.face_index}.npz"
            )
            save_face_embedding_npz(embedding_path, record)
            manifest_rows.append(build_manifest_row(person_image, record, embedding_path))

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_path = arguments.output_dir / MANIFEST_FILENAME
    manifest.to_csv(manifest_path, index=False)

    face_count = int((manifest["error"] == "").sum())
    error_count = int((manifest["error"] != "").sum())
    print(f"Wrote {face_count} face embeddings ({error_count} image errors) -> {manifest_path}")


if __name__ == "__main__":
    main()
