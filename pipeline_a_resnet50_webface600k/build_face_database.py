"""Builds the known-people face database for pipeline A from your own folder of labelled images.

Walks every image under PEOPLE_IMAGES_ROOT, detects the faces in it, and stores each face's
ResNet50@WebFace600K embedding together with the name of the person it belongs to. The result
is a single face_database.npz that identify_faces.py matches new images against, plus a CSV
manifest listing every face that went into it.

>>> FILL THIS IN BEFORE RUNNING <<<
Set PEOPLE_IMAGES_ROOT below to the folder holding your labelled people images (or pass
--people-images-root on the command line, which overrides it).

    NOTE TO A CODING AGENT EDITING THIS FILE FOR A USER:
    The only value that must be supplied is PEOPLE_IMAGES_ROOT. It is a *local filesystem
    path*, not a web URL - if the user's images live in cloud storage (Google Drive,
    Dropbox, OneDrive, a network share), point this at the locally synced folder for that
    storage, or download the images locally first. Do not add cloud-API download code here
    unless the user explicitly asks for it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from face_pipeline import (
    FACE_DATABASE_MANIFEST_FILENAME,
    IMAGE_EXTENSIONS,
    build_face_analysis_app,
    detect_and_embed_faces,
    read_image,
    save_face_database,
)

# >>> FILL THIS IN: the folder holding your labelled people images. <<<
# Expected layout - the name of the folder directly containing an image is the person's name:
#     <PEOPLE_IMAGES_ROOT>/<person name>/<image files>
# A deeper grouping layout also works, since only the immediate parent folder names the person:
#     <PEOPLE_IMAGES_ROOT>/<group>/<person name>/<image files>
PEOPLE_IMAGES_ROOT = Path("PUT/THE/PATH/TO/YOUR/PEOPLE/IMAGES/HERE")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "face_database"

MANIFEST_COLUMNS = [
    "person",
    "relative_path",
    "face_index",
    "detection_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "error",
]


def discover_person_images(people_images_root: Path) -> list[tuple[str, Path]]:
    """Finds every image under people_images_root, naming each one's person after its parent folder."""
    person_images = []
    for image_path in sorted(people_images_root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            person_images.append((image_path.parent.name, image_path))
    return person_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-images-root", type=Path, default=PEOPLE_IMAGES_ROOT,
                        help="Folder of labelled people images (overrides PEOPLE_IMAGES_ROOT)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where face_database.npz and its manifest are written")
    args = parser.parse_args()

    if not args.people_images_root.is_dir():
        raise SystemExit(
            f"People images folder not found: {args.people_images_root}\n"
            "Set PEOPLE_IMAGES_ROOT at the top of this file, or pass --people-images-root."
        )

    face_analysis_app = build_face_analysis_app()
    person_images = discover_person_images(args.people_images_root)
    print(f"Found {len(person_images)} images under {args.people_images_root}")

    person_names: list[str] = []
    embeddings: list[np.ndarray] = []
    manifest_rows: list[dict] = []

    for person, image_path in person_images:
        relative_path = image_path.relative_to(args.people_images_root)
        image_bgr = read_image(image_path)
        if image_bgr is None:
            manifest_rows.append({column: "" for column in MANIFEST_COLUMNS}
                                 | {"person": person, "relative_path": str(relative_path),
                                    "error": "unreadable"})
            continue

        for face_index, face in enumerate(detect_and_embed_faces(face_analysis_app, image_bgr)):
            person_names.append(person)
            embeddings.append(face.embedding)
            x1, y1, x2, y2 = face.bounding_box_xyxy
            manifest_rows.append({
                "person": person,
                "relative_path": str(relative_path),
                "face_index": face_index,
                "detection_confidence": round(face.detection_confidence, 4),
                "bbox_x1": round(float(x1), 1),
                "bbox_y1": round(float(y1), 1),
                "bbox_x2": round(float(x2), 1),
                "bbox_y2": round(float(y2), 1),
                "error": "",
            })

    if not embeddings:
        raise SystemExit("No faces were detected - nothing to write.")

    database_path = save_face_database(args.output_dir, person_names, np.stack(embeddings))
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_path = args.output_dir / FACE_DATABASE_MANIFEST_FILENAME
    manifest.to_csv(manifest_path, index=False)

    error_count = int((manifest["error"] != "").sum())
    print(f"Wrote {len(embeddings)} face embeddings for {len(set(person_names))} people "
          f"({error_count} unreadable images) -> {database_path}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
