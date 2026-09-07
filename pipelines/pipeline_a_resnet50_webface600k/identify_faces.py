"""Identifies who appears in an image, using the face database built by build_face_database.py.

Detects every face in the given image, embeds it, matches it against the known-people database,
and reports the person's name -- or `unknown` when the best similarity falls below the
pipeline's threshold. Optionally writes a copy of the image with the boxes and names drawn on it.

Which model, matching strategy and threshold are used is decided entirely by --pipeline; see
face_identity/configuration.py for the configurations and README.md for how they were chosen.
"""
import argparse
from pathlib import Path

from face_identity.annotation import draw_identified_faces, save_annotated_image
from face_identity.configuration import (
    DEFAULT_PIPELINE_KEY,
    FACE_DATABASE_FILENAME,
    PIPELINE_CONFIGURATIONS,
    resolve_pipeline_configuration,
)
from face_identity.face_recognition_pipeline import FaceRecognitionPipeline
from face_identity.image_io import read_image_bgr
from face_identity.matching.face_database import load_face_database

DEFAULT_FACE_DATABASE_ROOT = Path("face_database")


def main() -> None:
    """Names every face in one image and prints the result, optionally saving an annotated copy."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Image to identify the people in")
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_CONFIGURATIONS), default=DEFAULT_PIPELINE_KEY,
                        help="Which pipeline configuration to identify with")
    parser.add_argument("--face-database", type=Path, default=None,
                        help="face_database.npz written by build_face_database.py "
                             "(default: face_database/<pipeline name>/face_database.npz)")
    parser.add_argument("--annotated-output", type=Path, default=None,
                        help="Optional path to save a copy of the image with boxes and names drawn on it")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    arguments = parser.parse_args()

    configuration = resolve_pipeline_configuration(arguments.pipeline)
    face_database_path = arguments.face_database or (
        DEFAULT_FACE_DATABASE_ROOT / configuration.name / FACE_DATABASE_FILENAME
    )
    if not face_database_path.is_file():
        raise SystemExit(
            f"Face database not found: {face_database_path}\n"
            f"Run: python build_face_database.py --pipeline {arguments.pipeline} "
            "--people-images-root <your images>"
        )

    image_bgr = read_image_bgr(arguments.image)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {arguments.image}")

    database = load_face_database(face_database_path)
    try:
        database.require_embedding_model(configuration.embedding_model_name)
    except ValueError as mismatch:
        raise SystemExit(str(mismatch)) from None

    with FaceRecognitionPipeline(configuration, arguments.models_dir) as pipeline:
        identified_faces = pipeline.identify_faces_in_image(image_bgr, database)

    print(f"Pipeline: {configuration.name} | {configuration.description}")
    print(f"Database: {database.person_count} known people from {face_database_path}")
    print(f"Detected {len(identified_faces)} face(s) in {arguments.image}")
    for face_index, face in enumerate(identified_faces):
        x1, y1, x2, y2 = (round(float(value), 1) for value in face.bounding_box_xyxy)
        print(f"  face {face_index}: {face.predicted_person:<30} similarity={face.similarity_score:.4f}  "
              f"box=({x1}, {y1}, {x2}, {y2})  detection_confidence={face.detection_confidence:.3f}")

    recognized_people = sorted({face.predicted_person for face in identified_faces if face.is_known})
    print(f"People identified: {', '.join(recognized_people) if recognized_people else 'none'}")

    if arguments.annotated_output is not None:
        save_annotated_image(draw_identified_faces(image_bgr, identified_faces), arguments.annotated_output)
        print(f"Annotated image -> {arguments.annotated_output}")


if __name__ == "__main__":
    main()
