"""Identifies who appears in an image or folder, using a database built by build_face_database.py.

Detects every face in the given image(s), embeds it, matches it against the known-people
database, and reports the person's name -- or `unknown` when the best similarity falls below
the pipeline's threshold. Folder input is recursive and writes a CSV suitable for reviewing a
user's own evaluation images. Annotated copies are optional.

Which model, matching strategy and threshold are used is decided entirely by --pipeline; see
face_identity/configuration.py for the configurations and README.md for how they were chosen.
"""
import argparse
import csv
from pathlib import Path

from face_identity.annotation import draw_identified_faces, save_annotated_image
from face_identity.configuration import (
    DEFAULT_PIPELINE_KEY,
    FACE_DATABASE_FILENAME,
    PIPELINE_CONFIGURATIONS,
    resolve_pipeline_configuration,
)
from face_identity.face_recognition_pipeline import FaceRecognitionPipeline
from face_identity.image_io import is_image_file, read_image_bgr
from face_identity.matching.face_database import load_face_database
from face_identity.matching.face_identifier import IdentifiedFace

DEFAULT_FACE_DATABASE_ROOT = Path("face_database")
DEFAULT_BATCH_RESULTS_CSV = Path("identification_results.csv")

RESULT_COLUMNS = [
    "relative_path",
    "face_index",
    "predicted_person",
    "is_known",
    "similarity_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "detection_confidence",
    "error",
]


def discover_input_images(input_path: Path) -> tuple[list[tuple[Path, Path]], bool]:
    """Finds one input image or every supported image recursively inside an input directory."""
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Unsupported image file: {input_path}")
        return [(input_path, Path(input_path.name))], False
    if input_path.is_dir():
        images = [
            (image_path, image_path.relative_to(input_path))
            for image_path in sorted(input_path.rglob("*"))
            if is_image_file(image_path)
        ]
        if not images:
            raise ValueError(f"No supported images found under: {input_path}")
        return images, True
    raise ValueError(f"Image or directory not found: {input_path}")


def build_result_rows(
    relative_path: Path, identified_faces: list[IdentifiedFace], error: str = ""
) -> list[dict]:
    """Builds CSV rows for one input image, including unreadable and no-face images."""
    if error or not identified_faces:
        return [{column: "" for column in RESULT_COLUMNS} | {
            "relative_path": relative_path.as_posix(),
            "error": error,
        }]

    rows = []
    for face_index, face in enumerate(identified_faces):
        x1, y1, x2, y2 = (round(float(value), 1) for value in face.bounding_box_xyxy)
        rows.append({
            "relative_path": relative_path.as_posix(),
            "face_index": face_index,
            "predicted_person": face.predicted_person,
            "is_known": face.is_known,
            "similarity_score": round(face.similarity_score, 4),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "detection_confidence": round(face.detection_confidence, 4),
            "error": "",
        })
    return rows


def write_results_csv(result_rows: list[dict], output_path: Path) -> None:
    """Writes batch identification results as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)


def main() -> None:
    """Names every face in one image or every supported image under one directory."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Image or directory of images to identify")
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_CONFIGURATIONS), default=DEFAULT_PIPELINE_KEY,
                        help="Which pipeline configuration to identify with")
    parser.add_argument("--face-database", type=Path, default=None,
                        help="face_database.npz written by build_face_database.py "
                             "(default: face_database/<pipeline name>/face_database.npz)")
    parser.add_argument("--annotated-output", type=Path, default=None,
                        help="Annotated output file for one image, or output directory for folder input")
    parser.add_argument("--results-csv", type=Path, default=None,
                        help="Optional CSV output (folder input defaults to identification_results.csv)")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    arguments = parser.parse_args()

    try:
        input_images, input_is_directory = discover_input_images(arguments.image)
    except ValueError as invalid_input:
        raise SystemExit(str(invalid_input)) from None

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

    database = load_face_database(face_database_path)
    try:
        database.require_embedding_model(configuration.embedding_model_name)
    except ValueError as mismatch:
        raise SystemExit(str(mismatch)) from None

    print(f"Pipeline: {configuration.name} | {configuration.description}")
    print(f"Database: {database.person_count} known people from {face_database_path}")
    print(f"Input images: {len(input_images)} from {arguments.image}")

    result_rows = []
    recognized_people = set()
    with FaceRecognitionPipeline(configuration, arguments.models_dir) as pipeline:
        for image_path, relative_path in input_images:
            image_bgr = read_image_bgr(image_path)
            if image_bgr is None:
                print(f"WARNING: could not read image: {image_path}")
                result_rows.extend(build_result_rows(relative_path, [], "unreadable"))
                continue

            identified_faces = pipeline.identify_faces_in_image(image_bgr, database)
            result_rows.extend(build_result_rows(relative_path, identified_faces))
            print(f"Detected {len(identified_faces)} face(s) in {image_path}")
            for face_index, face in enumerate(identified_faces):
                x1, y1, x2, y2 = (round(float(value), 1) for value in face.bounding_box_xyxy)
                print(
                    f"  face {face_index}: {face.predicted_person:<30} "
                    f"similarity={face.similarity_score:.4f}  box=({x1}, {y1}, {x2}, {y2})  "
                    f"detection_confidence={face.detection_confidence:.3f}"
                )
                if face.is_known:
                    recognized_people.add(face.predicted_person)

            if arguments.annotated_output is not None:
                annotated_output_path = (
                    arguments.annotated_output / relative_path
                    if input_is_directory
                    else arguments.annotated_output
                )
                save_annotated_image(
                    draw_identified_faces(image_bgr, identified_faces), annotated_output_path
                )
                print(f"Annotated image -> {annotated_output_path}")

    print(f"People identified: {', '.join(sorted(recognized_people)) if recognized_people else 'none'}")

    results_csv_path = arguments.results_csv
    if input_is_directory and results_csv_path is None:
        results_csv_path = DEFAULT_BATCH_RESULTS_CSV
    if results_csv_path is not None:
        write_results_csv(result_rows, results_csv_path)
        print(f"Results CSV -> {results_csv_path}")


if __name__ == "__main__":
    main()
