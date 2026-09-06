"""Shared face detection + ResNet50@WebFace600K embedding + identification logic for pipeline A.

This is the one place the model, the matching strategy and the decision threshold are
defined; `build_face_database.py` and `identify_faces.py` both import from here so the
database and the queries can never be built with mismatched settings.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Configuration - everything you may want to change lives here.
# ---------------------------------------------------------------------------

# InsightFace model pack bundling SCRFD-10GF detection + 5-point alignment +
# ResNet50@WebFace600K recognition. `download_models.py` fetches it into MODELS_ROOT.
MODEL_PACK_NAME = "buffalo_l"
MODELS_ROOT = Path(__file__).resolve().parent / "models"

# Detection settings. Faces scoring below MIN_DETECTION_CONFIDENCE are dropped.
MIN_DETECTION_CONFIDENCE = 0.5
DETECTION_INPUT_SIZE = 640

# Identification settings, chosen from the experiment in README.md.
# TOP2 = a person's score is the mean cosine similarity to their 2 closest reference images.
MATCHING_STRATEGY_TOP_K = 2
IDENTIFICATION_THRESHOLD = 0.30
UNKNOWN_LABEL = "unknown"

# Filenames written by build_face_database.py and read back by identify_faces.py.
FACE_DATABASE_FILENAME = "face_database.npz"
FACE_DATABASE_MANIFEST_FILENAME = "face_database_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetectedFace:
    """One face found in an image, with its L2-normalized 512-d ResNet50@WebFace600K embedding."""
    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    embedding: np.ndarray


@dataclass
class FaceDatabase:
    """The known-people database: all reference embeddings, grouped into contiguous per-person slices."""
    person_names: np.ndarray
    person_slices: list[tuple[int, int]]
    embeddings: np.ndarray


@dataclass
class IdentifiedFace:
    """A detected face plus the identity decision made for it."""
    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    predicted_person: str
    similarity_score: float


def build_face_analysis_app() -> FaceAnalysis:
    """Loads the buffalo_l pack; detection, alignment and ResNet50 recognition all run inside it."""
    face_analysis_app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        root=str(MODELS_ROOT),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    face_analysis_app.prepare(
        ctx_id=-1,
        det_size=(DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE),
        det_thresh=MIN_DETECTION_CONFIDENCE,
    )
    return face_analysis_app


def detect_and_embed_faces(face_analysis_app: FaceAnalysis, image_bgr: np.ndarray) -> list[DetectedFace]:
    """Detects every face in one BGR image and returns each one's ResNet50@WebFace600K embedding."""
    return [
        DetectedFace(
            bounding_box_xyxy=face.bbox.astype(np.float32),
            detection_confidence=float(face.det_score),
            embedding=face.normed_embedding.astype(np.float32),
        )
        for face in face_analysis_app.get(image_bgr)
    ]


def read_image(image_path: Path) -> np.ndarray | None:
    """Reads an image as BGR, returning None when the file is missing or undecodable."""
    return cv2.imread(str(image_path))


def save_face_database(output_dir: Path, person_names: list[str], embeddings: np.ndarray) -> Path:
    """Writes one .npz holding every reference embedding and the person each one belongs to."""
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / FACE_DATABASE_FILENAME
    np.savez(database_path, person_names=np.array(person_names), embeddings=embeddings.astype(np.float32))
    return database_path


def load_face_database(database_path: Path) -> FaceDatabase:
    """Loads the reference database, sorting it so every person owns one contiguous row range."""
    with np.load(database_path, allow_pickle=False) as stored:
        person_names_per_row = stored["person_names"]
        embeddings = stored["embeddings"].astype(np.float32)

    sort_order = np.argsort(person_names_per_row, kind="stable")
    person_names_per_row = person_names_per_row[sort_order]
    embeddings = embeddings[sort_order]

    person_names, start_indices, counts = np.unique(
        person_names_per_row, return_index=True, return_counts=True)
    person_slices = [(int(start), int(start + count)) for start, count in zip(start_indices, counts)]
    return FaceDatabase(person_names=person_names, person_slices=person_slices, embeddings=embeddings)


def score_faces_against_database(face_embeddings: np.ndarray, database: FaceDatabase) -> np.ndarray:
    """Scores every face against every known person using the TOP-K strategy.

    Embeddings are L2-normalized, so one matrix product gives all cosine similarities;
    each person's score is then the mean of their MATCHING_STRATEGY_TOP_K closest images.
    """
    similarities = face_embeddings @ database.embeddings.T
    scores = np.empty((face_embeddings.shape[0], len(database.person_names)), dtype=np.float32)
    for person_index, (start, end) in enumerate(database.person_slices):
        person_similarities = similarities[:, start:end]
        available_k = min(MATCHING_STRATEGY_TOP_K, end - start)
        if available_k == 1:
            scores[:, person_index] = person_similarities.max(axis=1)
        else:
            closest = np.partition(person_similarities, -available_k, axis=1)[:, -available_k:]
            scores[:, person_index] = closest.mean(axis=1)
    return scores


def identify_faces(detected_faces: list[DetectedFace], database: FaceDatabase) -> list[IdentifiedFace]:
    """Names each detected face, or calls it `unknown` when its best score is below the threshold."""
    if not detected_faces:
        return []

    face_embeddings = np.stack([face.embedding for face in detected_faces])
    person_scores = score_faces_against_database(face_embeddings, database)
    best_person_indices = person_scores.argmax(axis=1)
    best_scores = person_scores.max(axis=1)

    identified_faces = []
    for face, best_person_index, best_score in zip(detected_faces, best_person_indices, best_scores):
        is_known = best_score >= IDENTIFICATION_THRESHOLD
        identified_faces.append(IdentifiedFace(
            bounding_box_xyxy=face.bounding_box_xyxy,
            detection_confidence=face.detection_confidence,
            predicted_person=str(database.person_names[best_person_index]) if is_known else UNKNOWN_LABEL,
            similarity_score=float(best_score),
        ))
    return identified_faces
