"""Builds the known-people face database from your own folder of labelled images.

Walks every image under --people-images-root, detects the faces in it, and stores each face's
embedding together with the name of the person it belongs to. The result is a single
face_database.npz that identify_faces.py matches new images against, plus a CSV manifest
listing every face that went into it.

One reference image must contain exactly one detected face. An image with zero or multiple
faces is skipped with a warning and recorded in the manifest, so a bystander's face can never
be silently assigned the folder's identity.

The database is specific to the pipeline that built it, because embeddings from different
models are not comparable. Each pipeline therefore writes to its own subfolder, and the model
name is recorded inside the .npz so querying it with the wrong pipeline fails loudly.

    NOTE TO A CODING AGENT EDITING THIS FILE FOR A USER:
    The only value that must be supplied is --people-images-root (or PEOPLE_IMAGES_ROOT below).
    It is a *local filesystem path*, not a web URL -- if the user's images live in cloud storage
    (Google Drive, Dropbox, OneDrive, a network share), point this at the locally synced folder
    for that storage, or download the images locally first. Do not add cloud-API download code
    here unless the user explicitly asks for it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from face_identity.configuration import (
    DEFAULT_PIPELINE_KEY,
    FACE_DATABASE_MANIFEST_FILENAME,
    PIPELINE_CONFIGURATIONS,
    resolve_pipeline_configuration,
)
from face_identity.face_recognition_pipeline import FaceRecognitionPipeline
from face_identity.image_io import discover_person_images, read_image_bgr
from face_identity.matching.face_database import save_face_database

# Optional default for the folder holding your labelled people images, so the script can be run
# with no arguments. --people-images-root overrides it. Expected layout -- the folder directly
# containing an image names the person:
#     <PEOPLE_IMAGES_ROOT>/<person name>/<image files>
# Every image must show that person alone and yield exactly one detected face.
# A deeper grouping layout also works, since only the immediate parent folder names the person:
#     <PEOPLE_IMAGES_ROOT>/<group>/<person name>/<image files>
PEOPLE_IMAGES_ROOT = Path("PUT/THE/PATH/TO/YOUR/PEOPLE/IMAGES/HERE")

DEFAULT_FACE_DATABASE_ROOT = Path("face_database")

MANIFEST_COLUMNS = [
    "person",
    "group",
    "relative_path",
    "face_index",
    "detection_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "error",
]


def build_manifest_row(person_image, face_index: int, face) -> dict:
    """Describes one stored reference face as a manifest row."""
    x1, y1, x2, y2 = face.bounding_box_xyxy
    return {
        "person": person_image.person,
        "group": person_image.group,
        "relative_path": str(person_image.relative_path),
        "face_index": face_index,
        "detection_confidence": round(face.detection_confidence, 4),
        "bbox_x1": round(float(x1), 1),
        "bbox_y1": round(float(y1), 1),
        "bbox_x2": round(float(x2), 1),
        "bbox_y2": round(float(y2), 1),
        "error": "",
    }


def build_error_manifest_row(person_image, error: str) -> dict:
    """Describes a reference image that was skipped without storing an embedding."""
    return {column: "" for column in MANIFEST_COLUMNS} | {
        "person": person_image.person,
        "group": person_image.group,
        "relative_path": str(person_image.relative_path),
        "error": error,
    }


def reference_face_count_error(detected_face_count: int) -> str | None:
    """Returns why a reference image must be skipped, or None when it has exactly one face."""
    if detected_face_count == 0:
        return "no_face_detected"
    if detected_face_count > 1:
        return f"multiple_faces_detected:{detected_face_count}"
    return None


def main() -> None:
    """Embeds every labelled reference image and writes the face database and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_CONFIGURATIONS), default=DEFAULT_PIPELINE_KEY,
                        help="Which pipeline configuration to build the database with")
    parser.add_argument("--people-images-root", type=Path, default=PEOPLE_IMAGES_ROOT,
                        help="Folder of labelled people images")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where face_database.npz and its manifest go "
                             "(default: face_database/<pipeline name>/)")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    arguments = parser.parse_args()

    configuration = resolve_pipeline_configuration(arguments.pipeline)
    output_directory = arguments.output_dir or DEFAULT_FACE_DATABASE_ROOT / configuration.name

    if not arguments.people_images_root.is_dir():
        raise SystemExit(
            f"People images folder not found: {arguments.people_images_root}\n"
            "Pass --people-images-root, or set PEOPLE_IMAGES_ROOT at the top of this file."
        )

    print(f"Pipeline: {configuration.name} ({configuration.description})")
    person_images = discover_person_images(arguments.people_images_root)
    print(f"Found {len(person_images)} images under {arguments.people_images_root}")

    person_name_per_row: list[str] = []
    embeddings: list[np.ndarray] = []
    manifest_rows: list[dict] = []
    skipped_image_count = 0

    with FaceRecognitionPipeline(configuration, arguments.models_dir) as pipeline:
        for person_image in person_images:
            image_bgr = read_image_bgr(person_image.path)
            if image_bgr is None:
                skipped_image_count += 1
                manifest_rows.append(build_error_manifest_row(person_image, "unreadable"))
                print(f"WARNING: skipping unreadable reference image: {person_image.path}", file=sys.stderr)
                continue

            detected_faces = pipeline.detect_and_embed_faces(image_bgr)
            face_count_error = reference_face_count_error(len(detected_faces))
            if face_count_error is not None:
                skipped_image_count += 1
                manifest_rows.append(build_error_manifest_row(person_image, face_count_error))
                print(
                    f"WARNING: skipping reference image {person_image.path}: expected exactly "
                    f"one detected face, found {len(detected_faces)}",
                    file=sys.stderr,
                )
                continue

            [face] = detected_faces
            person_name_per_row.append(person_image.person)
            embeddings.append(face.embedding)
            manifest_rows.append(build_manifest_row(person_image, 0, face))

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / FACE_DATABASE_MANIFEST_FILENAME
    manifest.to_csv(manifest_path, index=False)

    if not embeddings:
        raise SystemExit(
            f"No valid single-face reference images were found - no database was written.\n"
            f"Manifest -> {manifest_path}"
        )

    database_path = save_face_database(
        output_directory, person_name_per_row, np.stack(embeddings), configuration.embedding_model_name
    )

    print(f"Wrote {len(embeddings)} face embeddings for {len(set(person_name_per_row))} people "
          f"({skipped_image_count} skipped images) -> {database_path}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
