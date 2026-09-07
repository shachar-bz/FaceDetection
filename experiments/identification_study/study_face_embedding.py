"""Embeds every face in an image with both study models from a single shared detection pass.

The reference database and the evaluation set must be embedded identically for their
similarities to mean anything, so both extraction scripts in this folder go through here.
Detection runs once per image via InsightFace's buffalo_l pack; that one detection then feeds
ResNet50@WebFace600K (whose embedding the pack produces during the same pass) and OpenCV SFace
(which aligns and embeds the face itself from the pack's keypoints). Detecting once means the
two embeddings for a face can never come from different detections.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_identity.configuration import (
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
)
from face_identity.detection.face_detector import DetectedFace
from face_identity.detection.scrfd_detector import InsightFacePackDetector
from face_identity.embedding.embedder_registry import build_face_embedder
from face_identity.embedding.face_embedder import FaceEmbedder

# The two models the identification study compares.
STUDY_EMBEDDING_MODEL_NAMES = ["resnet_webface600k", "sface"]
# The key each model's vector is stored under inside every per-face .npz. These names are part
# of the on-disk format the already-extracted embeddings use -- changing one invalidates them.
EMBEDDING_KEYS_IN_NPZ = {
    "resnet_webface600k": "embedding_resnet_webface600k",
    "sface": "embedding_sface",
}


@dataclass
class FaceEmbeddingRecord:
    """One detected face and its embedding under each study model."""

    face_index: int
    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    keypoints: np.ndarray
    embeddings_by_model: dict[str, np.ndarray]


def build_study_detector(
    models_directory: Path | None = None,
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    detection_input_size: int = DEFAULT_DETECTION_INPUT_SIZE,
) -> InsightFacePackDetector:
    """Creates the shared buffalo_l detector, with recognition on so ResNet embeddings come free."""
    return InsightFacePackDetector(
        models_directory=models_directory,
        min_detection_confidence=min_detection_confidence,
        input_size=detection_input_size,
        with_recognition=True,
    )


def build_study_embedders(models_directory: Path | None = None) -> dict[str, FaceEmbedder]:
    """Creates both study embedding models, keyed by name."""
    return {
        model_name: build_face_embedder(model_name, models_directory)
        for model_name in STUDY_EMBEDDING_MODEL_NAMES
    }


def embed_image_faces(
    image_bgr: np.ndarray,
    detector: InsightFacePackDetector,
    embedders_by_model: dict[str, FaceEmbedder],
) -> list[FaceEmbeddingRecord]:
    """Detects every face in one image once and embeds each of them with every study model."""
    records = []
    for face_index, detected_face in enumerate(detector.detect(image_bgr)):
        records.append(
            FaceEmbeddingRecord(
                face_index=face_index,
                bounding_box_xyxy=detected_face.bounding_box_xyxy,
                detection_confidence=float(detected_face.detection_confidence),
                keypoints=detected_face.keypoints,
                embeddings_by_model={
                    model_name: embedder.embed(image_bgr, detected_face)
                    for model_name, embedder in embedders_by_model.items()
                },
            )
        )
    return records


def save_face_embedding_npz(embedding_path: Path, record: FaceEmbeddingRecord) -> Path:
    """Writes one face's embeddings, box, keypoints and confidence to a .npz."""
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    stored_arrays = {
        EMBEDDING_KEYS_IN_NPZ[model_name]: embedding
        for model_name, embedding in record.embeddings_by_model.items()
    }
    np.savez(
        embedding_path,
        confidence=record.detection_confidence,
        bbox=record.bounding_box_xyxy,
        kps=record.keypoints,
        **stored_arrays,
    )
    return embedding_path


def load_embeddings_by_model(embedding_paths: list[str]) -> dict[str, np.ndarray]:
    """Loads a list of per-face .npz files into one stacked embedding matrix per study model."""
    embeddings_by_model: dict[str, list[np.ndarray]] = {
        model_name: [] for model_name in STUDY_EMBEDDING_MODEL_NAMES
    }
    for embedding_path in embedding_paths:
        with np.load(embedding_path) as face_data:
            for model_name, embedding_key in EMBEDDING_KEYS_IN_NPZ.items():
                embeddings_by_model[model_name].append(face_data[embedding_key])
    return {
        model_name: (
            np.stack(rows).astype(np.float32) if rows else np.zeros((0, 1), dtype=np.float32)
        )
        for model_name, rows in embeddings_by_model.items()
    }


def unreadable_image_row(manifest_columns: list[str], **identifying_fields: object) -> dict:
    """Builds the manifest row recorded for an image that could not be decoded."""
    row = {column: "" for column in manifest_columns}
    row.update(**identifying_fields, error="unreadable")
    return row
