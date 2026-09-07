"""BlazeFace detection via MediaPipe, used only by the detection benchmark in experiments/.

MediaPipe ships two variants: `short_range`, tuned for faces within about 2 m of the camera,
and `full_range`, a dense-anchor variant that reaches smaller and farther faces.
"""
from pathlib import Path

import numpy as np

from face_identity.configuration import DEFAULT_MIN_DETECTION_CONFIDENCE
from face_identity.detection.face_detector import DetectedFace

# Filenames these variants are written under by face_identity.model_downloads.
BLAZEFACE_MODEL_FILENAMES = {
    "blazeface_short_range": "blaze_face_short_range.tflite",
    "blazeface_full_range": "blaze_face_full_range.tflite",
}


class BlazeFaceDetector:
    """Detects faces with one MediaPipe BlazeFace variant. Holds a native handle -- call close()."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Returns one record per detected face, with its box and confidence."""
        import cv2

        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mediapipe_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        result = self._detector.detect(mediapipe_image)

        detected_faces = []
        for detection in result.detections:
            box = detection.bounding_box
            confidence = detection.categories[0].score if detection.categories else 0.0
            detected_faces.append(
                DetectedFace(
                    bounding_box_xyxy=np.array(
                        [box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height],
                        dtype=np.float32,
                    ),
                    detection_confidence=float(confidence),
                )
            )
        return detected_faces

    def close(self) -> None:
        """Releases the MediaPipe detector's native resources."""
        self._detector.close()
