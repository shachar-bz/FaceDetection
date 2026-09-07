"""OpenCV SFace recognition (128-d), aligning each face with SFace's own official alignCrop().

The detection's box, keypoints and confidence are adapted into the row alignCrop() expects and
handed to it on the full unaligned image, so the crop is never one pre-aligned for another model.
"""
from pathlib import Path

import cv2
import numpy as np

from face_identity.detection.face_detector import DetectedFace
from face_identity.embedding.face_embedder import l2_normalize

EMBEDDING_MODEL_NAME = "sface"
EMBEDDING_DIMENSIONS = 128
# Filename the weights are written under by face_identity.model_downloads.
SFACE_MODEL_FILENAME = "sface_2021dec.onnx"


def build_sface_face_row(
    bounding_box_xyxy: np.ndarray, keypoints: np.ndarray, detection_confidence: float
) -> np.ndarray:
    """Adapts one detection into the 15-value row `FaceRecognizerSF.alignCrop()` expects.

    The row is [x, y, w, h, 5x(landmark_x, landmark_y), confidence]. No landmark reordering is
    needed: OpenCV's alignCrop warps to the same ArcFace reference points InsightFace's own
    alignment uses, so SCRFD's native order (left eye, right eye, nose, left mouth, right
    mouth) is already the order it wants.
    """
    x1, y1, x2, y2 = bounding_box_xyxy
    return np.array(
        [x1, y1, x2 - x1, y2 - y1, *np.asarray(keypoints).reshape(-1), detection_confidence],
        dtype=np.float32,
    )


class SFaceEmbedder:
    """Aligns and embeds a detected face with OpenCV's SFace recognizer."""

    embedding_model_name = EMBEDDING_MODEL_NAME
    requires_pack_recognition = False

    def __init__(self, model_path: Path) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"SFace model not found: {model_path}\n"
                "Run: python -m face_identity.model_downloads"
            )
        self._recognizer = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, image_bgr: np.ndarray, detected_face: DetectedFace) -> np.ndarray:
        """Returns the face's L2-normalized 128-d SFace embedding."""
        if detected_face.keypoints is None:
            raise ValueError("SFace alignment needs the detector's 5 facial keypoints, but none were supplied.")
        face_row = build_sface_face_row(
            detected_face.bounding_box_xyxy, detected_face.keypoints, detected_face.detection_confidence
        )
        aligned_face_bgr = self._recognizer.alignCrop(image_bgr, face_row)
        return l2_normalize(self._recognizer.feature(aligned_face_bgr).flatten()).astype(np.float32)
