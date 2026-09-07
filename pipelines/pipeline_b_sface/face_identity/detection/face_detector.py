"""The one record every detector in this package returns, and the interface they all satisfy."""
from typing import Protocol, runtime_checkable

import numpy as np
from dataclasses import dataclass


@dataclass
class DetectedFace:
    """One face found in an image, plus whatever extras the detector was able to supply.

    `keypoints` holds the 5 facial landmarks the embedding models align from; detectors that
    do not produce them leave it None. `pack_recognition_embedding` is filled in only by the
    InsightFace pack detector when its recognition module is enabled, so the embedding comes
    back from the same pass that found the face.
    """

    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    keypoints: np.ndarray | None = None
    pack_recognition_embedding: np.ndarray | None = None


@runtime_checkable
class FaceDetector(Protocol):
    """Anything that can find the faces in a BGR image."""

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Returns one record per face found in the image."""
        ...

    def close(self) -> None:
        """Releases whatever the detector holds open."""
        ...
