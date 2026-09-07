"""Builds any of the supported embedding models by name, so callers never import a backend directly."""
from pathlib import Path

from face_identity.configuration import resolve_models_directory
from face_identity.embedding.face_embedder import FaceEmbedder
from face_identity.embedding.resnet_webface600k_embedder import (
    EMBEDDING_MODEL_NAME as RESNET_WEBFACE600K_MODEL_NAME,
)
from face_identity.embedding.resnet_webface600k_embedder import ResNet50WebFace600KEmbedder
from face_identity.embedding.sface_embedder import EMBEDDING_MODEL_NAME as SFACE_MODEL_NAME
from face_identity.embedding.sface_embedder import SFACE_MODEL_FILENAME, SFaceEmbedder

EMBEDDING_MODEL_NAMES = [RESNET_WEBFACE600K_MODEL_NAME, SFACE_MODEL_NAME]


def build_face_embedder(embedding_model_name: str, models_directory: Path | None = None) -> FaceEmbedder:
    """Creates one of the supported embedding models by name, loading its weights from the cache."""
    if embedding_model_name == RESNET_WEBFACE600K_MODEL_NAME:
        return ResNet50WebFace600KEmbedder()

    if embedding_model_name == SFACE_MODEL_NAME:
        models_directory = resolve_models_directory(models_directory)
        return SFaceEmbedder(models_directory / SFACE_MODEL_FILENAME)

    raise ValueError(
        f"Unknown embedding model {embedding_model_name!r}; choose one of {EMBEDDING_MODEL_NAMES}"
    )
