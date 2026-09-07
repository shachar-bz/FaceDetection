"""Wires one PipelineConfiguration into a working detect -> embed -> identify pipeline.

This is the single entry point callers should use. Detection runs once per image and its
result feeds the embedding model, so a face's box and its embedding can never come from
different detections.
"""
from pathlib import Path

import numpy as np

from face_identity.configuration import PipelineConfiguration, resolve_models_directory
from face_identity.detection.face_detector import DetectedFace
from face_identity.detection.scrfd_detector import InsightFacePackDetector
from face_identity.embedding.embedder_registry import build_face_embedder
from face_identity.embedding.face_embedder import EmbeddedFace, FaceEmbedder
from face_identity.matching.face_database import FaceDatabase
from face_identity.matching.face_identifier import IdentifiedFace, identify_faces


def embed_detected_faces(
    image_bgr: np.ndarray, detected_faces: list[DetectedFace], embedder: FaceEmbedder
) -> list[EmbeddedFace]:
    """Runs one embedding model over faces that have already been detected in an image."""
    return [
        EmbeddedFace(
            bounding_box_xyxy=detected_face.bounding_box_xyxy,
            detection_confidence=detected_face.detection_confidence,
            embedding=embedder.embed(image_bgr, detected_face),
            keypoints=detected_face.keypoints,
        )
        for detected_face in detected_faces
    ]


class FaceRecognitionPipeline:
    """Detects, embeds and identifies faces according to one named pipeline configuration."""

    def __init__(self, configuration: PipelineConfiguration, models_directory: Path | None = None) -> None:
        self.configuration = configuration
        self.models_directory = resolve_models_directory(models_directory)
        self.embedder = build_face_embedder(configuration.embedding_model_name, self.models_directory)
        self.detector = InsightFacePackDetector(
            models_directory=self.models_directory,
            min_detection_confidence=configuration.min_detection_confidence,
            input_size=configuration.detection_input_size,
            with_recognition=self.embedder.requires_pack_recognition,
        )

    def detect_and_embed_faces(self, image_bgr: np.ndarray) -> list[EmbeddedFace]:
        """Finds every face in one BGR image and returns each one with its embedding."""
        return embed_detected_faces(image_bgr, self.detector.detect(image_bgr), self.embedder)

    def identify_faces_in_image(self, image_bgr: np.ndarray, database: FaceDatabase) -> list[IdentifiedFace]:
        """Names every face in one BGR image against the known-people database."""
        return identify_faces(
            self.detect_and_embed_faces(image_bgr),
            database,
            self.configuration.matching_strategy,
            self.configuration.identification_threshold,
        )

    def close(self) -> None:
        """Releases the detector's resources."""
        self.detector.close()

    def __enter__(self) -> "FaceRecognitionPipeline":
        return self

    def __exit__(self, *exception_details: object) -> None:
        self.close()
