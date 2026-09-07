"""SCRFD-10G-KPS face detection, in the two forms this project uses it.

`SCRFDDetector` runs the standalone `scrfd_10g_kps.onnx` and reports boxes only -- the form
the detection benchmark compares against BlazeFace. `InsightFacePackDetector` runs the same
detector from inside InsightFace's `buffalo_l` pack, which also yields the 5 facial keypoints
the embedding models align from and, when recognition is enabled, the ResNet50@WebFace600K
embedding from that same pass.
"""
from pathlib import Path

import numpy as np

from face_identity.configuration import (
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_SCRFD_NMS_THRESHOLD,
    resolve_models_directory,
)
from face_identity.detection.face_detector import DetectedFace

# The InsightFace pack bundling SCRFD-10GF detection, 5-point alignment and
# ResNet50@WebFace600K recognition. FaceAnalysis downloads and caches it on first use.
INSIGHTFACE_MODEL_PACK_NAME = "buffalo_l"
# Filename the standalone detector is written under by face_identity.model_downloads.
SCRFD_MODEL_FILENAME = "scrfd_10g_kps.onnx"


class SCRFDDetector:
    """Detects faces with the standalone SCRFD-10G-KPS ONNX model, on CPU."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        input_size: int = DEFAULT_DETECTION_INPUT_SIZE,
        nms_threshold: float = DEFAULT_SCRFD_NMS_THRESHOLD,
    ) -> None:
        from insightface.model_zoo.scrfd import SCRFD

        self._detector = SCRFD(model_file=str(model_path))
        self._detector.prepare(
            ctx_id=-1,
            det_thresh=min_detection_confidence,
            input_size=(input_size, input_size),
            nms_thresh=nms_threshold,
        )

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Returns one record per detected face, with its box and confidence."""
        boxes, keypoints_per_face = self._detector.detect(image_bgr)
        detected_faces = []
        for face_index, (x1, y1, x2, y2, confidence) in enumerate(boxes):
            keypoints = None
            if keypoints_per_face is not None and len(keypoints_per_face) > face_index:
                keypoints = np.asarray(keypoints_per_face[face_index], dtype=np.float32)
            detected_faces.append(
                DetectedFace(
                    bounding_box_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    detection_confidence=float(confidence),
                    keypoints=keypoints,
                )
            )
        return detected_faces

    def close(self) -> None:
        """Nothing to release; the ONNX session is freed with the object."""


class InsightFacePackDetector:
    """Detects faces with InsightFace's `buffalo_l` pack, yielding boxes, keypoints and confidence.

    When with_recognition is True the pack's ResNet50@WebFace600K model also runs, so each face
    comes back carrying its own L2-normalized embedding from the very same detection pass.
    """

    def __init__(
        self,
        models_directory: Path | None = None,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        input_size: int = DEFAULT_DETECTION_INPUT_SIZE,
        with_recognition: bool = False,
        model_pack_name: str = INSIGHTFACE_MODEL_PACK_NAME,
    ) -> None:
        from insightface.app import FaceAnalysis

        allowed_modules = ["detection", "recognition"] if with_recognition else ["detection"]
        self.model_pack_name = model_pack_name
        self.with_recognition = with_recognition
        self.models_directory = resolve_models_directory(models_directory)
        self._face_analysis = FaceAnalysis(
            name=model_pack_name,
            root=str(self.models_directory),
            allowed_modules=allowed_modules,
            providers=["CPUExecutionProvider"],
        )
        self._face_analysis.prepare(
            ctx_id=-1,
            det_size=(input_size, input_size),
            det_thresh=min_detection_confidence,
        )

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Returns one record per detected face, with keypoints and, if enabled, its embedding."""
        return [
            DetectedFace(
                bounding_box_xyxy=face.bbox.astype(np.float32),
                detection_confidence=float(face.det_score),
                keypoints=np.asarray(face.kps, dtype=np.float32),
                pack_recognition_embedding=(
                    face.normed_embedding.astype(np.float32) if self.with_recognition else None
                ),
            )
            for face in self._face_analysis.get(image_bgr)
        ]

    def close(self) -> None:
        """Nothing to release; the ONNX sessions are freed with the object."""
