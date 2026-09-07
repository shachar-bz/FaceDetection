"""Drawing detection boxes and identity labels onto a copy of an image, for eyeballing results."""
from pathlib import Path

import cv2
import numpy as np

from face_identity.configuration import DEFAULT_CROP_MARGIN_RATIO
from face_identity.detection.box_geometry import expand_box_to_square
from face_identity.detection.face_detector import DetectedFace
from face_identity.matching.face_identifier import IdentifiedFace

DETECTION_BOX_COLOR_BGR = (0, 255, 0)
KNOWN_FACE_BOX_COLOR_BGR = (0, 200, 0)
UNKNOWN_FACE_BOX_COLOR_BGR = (0, 0, 220)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
IDENTITY_BOX_THICKNESS = 2


def draw_detection_boxes(
    image_bgr: np.ndarray,
    detected_faces: list[DetectedFace],
    margin_ratio: float = DEFAULT_CROP_MARGIN_RATIO,
) -> np.ndarray:
    """Draws each detection as the square crop box it would produce, labelled with its confidence."""
    image_height, image_width = image_bgr.shape[:2]
    thickness = max(2, image_width // 300)
    for detected_face in detected_faces:
        x1, y1, x2, y2 = detected_face.bounding_box_xyxy
        crop_x1, crop_y1, crop_x2, crop_y2 = expand_box_to_square(
            x1, y1, x2 - x1, y2 - y1, margin_ratio, image_width, image_height
        )
        cv2.rectangle(image_bgr, (crop_x1, crop_y1), (crop_x2, crop_y2), DETECTION_BOX_COLOR_BGR, thickness)
        cv2.putText(
            image_bgr,
            f"{detected_face.detection_confidence:.2f}",
            (crop_x1, max(0, crop_y1 - 6)),
            LABEL_FONT, LABEL_FONT_SCALE, DETECTION_BOX_COLOR_BGR, 2, cv2.LINE_AA,
        )
    return image_bgr


def draw_identified_faces(image_bgr: np.ndarray, identified_faces: list[IdentifiedFace]) -> np.ndarray:
    """Draws each face's raw detection box labelled with its predicted name and similarity."""
    for face in identified_faces:
        x1, y1, x2, y2 = (int(round(float(value))) for value in face.bounding_box_xyxy)
        color = KNOWN_FACE_BOX_COLOR_BGR if face.is_known else UNKNOWN_FACE_BOX_COLOR_BGR
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, IDENTITY_BOX_THICKNESS)
        cv2.putText(
            image_bgr,
            f"{face.predicted_person} {face.similarity_score:.2f}",
            (x1, max(y1 - 6, 12)),
            LABEL_FONT, LABEL_FONT_SCALE, color, IDENTITY_BOX_THICKNESS,
        )
    return image_bgr


def save_annotated_image(image_bgr: np.ndarray, output_path: Path) -> None:
    """Writes an annotated image, creating its parent directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
