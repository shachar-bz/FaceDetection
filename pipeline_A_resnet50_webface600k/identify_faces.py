"""Identifies who appears in a new image, using the face database built by build_face_database.py.

Detects every face in the given image, embeds it with ResNet50@WebFace600K, matches it against
the known-people database with the TOP2 strategy, and reports the person's name - or `unknown`
when the best similarity falls below the threshold (0.30, see README.md). Optionally writes a
copy of the image with the boxes and names drawn on it.
"""
import argparse
from pathlib import Path

import cv2

from face_pipeline import (
    FACE_DATABASE_FILENAME,
    IDENTIFICATION_THRESHOLD,
    MATCHING_STRATEGY_TOP_K,
    UNKNOWN_LABEL,
    IdentifiedFace,
    build_face_analysis_app,
    detect_and_embed_faces,
    identify_faces,
    load_face_database,
    read_image,
)

DEFAULT_FACE_DATABASE_PATH = Path(__file__).resolve().parent / "face_database" / FACE_DATABASE_FILENAME

KNOWN_FACE_BOX_COLOR_BGR = (0, 200, 0)
UNKNOWN_FACE_BOX_COLOR_BGR = (0, 0, 220)
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6


def identify_people_in_image(image_path: Path, face_database_path: Path) -> list[IdentifiedFace]:
    """Returns one identity decision per face found in the image at image_path."""
    image_bgr = read_image(image_path)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    face_analysis_app = build_face_analysis_app()
    database = load_face_database(face_database_path)
    detected_faces = detect_and_embed_faces(face_analysis_app, image_bgr)
    return identify_faces(detected_faces, database)


def draw_identified_faces(image_path: Path, identified_faces: list[IdentifiedFace], output_path: Path) -> None:
    """Saves a copy of the image with each face boxed and labelled with its predicted name."""
    image_bgr = read_image(image_path)
    for face in identified_faces:
        x1, y1, x2, y2 = (int(round(value)) for value in face.bounding_box_xyxy)
        is_known = face.predicted_person != UNKNOWN_LABEL
        color = KNOWN_FACE_BOX_COLOR_BGR if is_known else UNKNOWN_FACE_BOX_COLOR_BGR
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{face.predicted_person} {face.similarity_score:.2f}"
        cv2.putText(image_bgr, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, color, BOX_THICKNESS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image to identify the people in")
    parser.add_argument("--face-database", type=Path, default=DEFAULT_FACE_DATABASE_PATH,
                        help="face_database.npz written by build_face_database.py")
    parser.add_argument("--annotated-output", type=Path, default=None,
                        help="Optional path to save a copy of the image with boxes and names drawn on it")
    args = parser.parse_args()

    if not args.face_database.is_file():
        raise SystemExit(f"Face database not found: {args.face_database}\nRun build_face_database.py first.")

    identified_faces = identify_people_in_image(args.image, args.face_database)

    print(f"Model: ResNet50@WebFace600K | strategy: TOP{MATCHING_STRATEGY_TOP_K} | "
          f"threshold: {IDENTIFICATION_THRESHOLD}")
    print(f"Detected {len(identified_faces)} face(s) in {args.image}")
    for face_index, face in enumerate(identified_faces):
        x1, y1, x2, y2 = (round(float(value), 1) for value in face.bounding_box_xyxy)
        print(f"  face {face_index}: {face.predicted_person:<30} similarity={face.similarity_score:.4f}  "
              f"box=({x1}, {y1}, {x2}, {y2})  detection_confidence={face.detection_confidence:.3f}")

    recognized_people = sorted({face.predicted_person for face in identified_faces
                                if face.predicted_person != UNKNOWN_LABEL})
    print(f"People identified: {', '.join(recognized_people) if recognized_people else 'none'}")

    if args.annotated_output is not None:
        draw_identified_faces(args.image, identified_faces, args.annotated_output)
        print(f"Annotated image -> {args.annotated_output}")


if __name__ == "__main__":
    main()
