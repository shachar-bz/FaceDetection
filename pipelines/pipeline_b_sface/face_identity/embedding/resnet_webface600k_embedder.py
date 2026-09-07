"""ResNet50 trained on WebFace600K (512-d), as shipped in InsightFace's `buffalo_l` pack.

Alignment and recognition both happen inside `FaceAnalysis.get()`, using InsightFace's own
tested preprocessing, so this module never computes an alignment transform itself -- it reads
back the embedding the detection pass already produced.
"""
import numpy as np

from face_identity.detection.face_detector import DetectedFace

EMBEDDING_MODEL_NAME = "resnet_webface600k"
EMBEDDING_DIMENSIONS = 512


class ResNet50WebFace600KEmbedder:
    """Reads back the L2-normalized embedding the InsightFace pack computed during detection."""

    embedding_model_name = EMBEDDING_MODEL_NAME
    requires_pack_recognition = True

    def embed(self, image_bgr: np.ndarray, detected_face: DetectedFace) -> np.ndarray:
        """Returns the face's 512-d embedding, already L2-normalized by InsightFace."""
        if detected_face.pack_recognition_embedding is None:
            raise ValueError(
                "This face carries no InsightFace recognition embedding. Build the detector with "
                "with_recognition=True so the pack's ResNet50@WebFace600K model runs during detection."
            )
        return detected_face.pack_recognition_embedding
