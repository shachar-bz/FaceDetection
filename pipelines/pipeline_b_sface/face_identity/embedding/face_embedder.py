"""The interface every embedding model satisfies, and the vector normalization they share."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from face_identity.detection.face_detector import DetectedFace


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scales a vector to unit length, so a dot product between two of them is a cosine similarity."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@dataclass
class EmbeddedFace:
    """One detected face together with the embedding that identifies it."""

    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    embedding: np.ndarray
    keypoints: np.ndarray | None = None


@runtime_checkable
class FaceEmbedder(Protocol):
    """Anything that turns a detected face into an L2-normalized embedding vector."""

    embedding_model_name: str
    # True when the embedding is produced by the InsightFace pack's own recognition module, so
    # the detector must be built with recognition enabled rather than detection alone.
    requires_pack_recognition: bool

    def embed(self, image_bgr: np.ndarray, detected_face: DetectedFace) -> np.ndarray:
        """Returns the L2-normalized embedding of one detected face."""
        ...
