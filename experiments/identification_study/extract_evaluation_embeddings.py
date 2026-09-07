"""Extracts both study models' embeddings for the real social-media evaluation images.

Runs the exact same detection and embedding path that built the reference database, so the
evaluation embeddings are directly comparable to it. Boxes are recorded as x1/y1/x2/y2 to
match the format the human-reviewed ground-truth table uses.
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
from face_identity.image_io import discover_images_in_groups, read_image_bgr

DEFAULT_EVAL_ROOT = Path("real_data")
DEFAULT_EVAL_GROUPS = ["one_person", "few_people"]
DEFAULT_OUTPUT_DIR = Path("results/eval_embeddings")
MANIFEST_FILENAME = "eval_embeddings_manifest.csv"

MANIFEST_COLUMNS = [
    "image_id",
    "group",
    "relative_path",
    "face_index",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "embedding_path",
    "error",
]


def build_manifest_row(image_id: str, group: str, relative_path: Path, record, embedding_path: Path) -> dict:
    """Describes one embedded evaluation face as a manifest row."""
    x1, y1, x2, y2 = record.bounding_box_xyxy
    return {
        "image_id": image_id,
        "group": group,
        "relative_path": str(relative_path),
        "face_index": record.face_index,
        "confidence": round(record.detection_confidence, 4),
        "bbox_x1": round(float(x1), 1),
        "bbox_y1": round(float(y1), 1),
        "bbox_x2": round(float(x2), 1),
        "bbox_y2": round(float(y2), 1),
        "embedding_path": str(embedding_path),
        "error": "",
    }


def main() -> None:
    """Extracts and stores both models' embeddings for every face in the evaluation images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
                        help="Directory holding the evaluation image group folders")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_EVAL_GROUPS,
                        help="Group folder names to scan under --eval-root")
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

    evaluation_images = discover_images_in_groups(arguments.eval_root, arguments.groups)
    print(f"Found {len(evaluation_images)} evaluation images under {arguments.eval_root}")

    manifest_rows = []
    for image_id, group, image_path in evaluation_images:
        relative_path = image_path.relative_to(arguments.eval_root)
        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            manifest_rows.append(unreadable_image_row(
                MANIFEST_COLUMNS, image_id=image_id, group=group, relative_path=str(relative_path)
            ))
            continue

        for record in embed_image_faces(image_bgr, detector, embedders_by_model):
            embedding_path = embeddings_dir / group / f"{image_id}_face{record.face_index}.npz"
            save_face_embedding_npz(embedding_path, record)
            manifest_rows.append(
                build_manifest_row(image_id, group, relative_path, record, embedding_path)
            )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_path = arguments.output_dir / MANIFEST_FILENAME
    manifest.to_csv(manifest_path, index=False)

    face_count = int((manifest["error"] == "").sum())
    error_count = int((manifest["error"] != "").sum())
    print(f"Wrote {face_count} face embeddings ({error_count} image errors) -> {manifest_path}")


if __name__ == "__main__":
    main()
