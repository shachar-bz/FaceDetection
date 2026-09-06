"""Shared face detection + OpenCV SFace embedding + identification logic for pipeline B.

This is the one place the models, the matching strategy and the decision threshold are
defined; `build_face_database.py` and `identify_faces.py` both import from here so the
database and the queries can never be built with mismatched settings.

Detection uses InsightFace's SCRFD-10GF (from the `buffalo_l` pack) because it supplies the
5 facial keypoints SFace's own alignment needs; recognition is entirely OpenCV SFace, which
performs its own official `alignCrop()` from those keypoints.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Configuration - everything you may want to change lives here.
# ---------------------------------------------------------------------------

# Detection: SCRFD-10GF + 5 keypoints, taken from InsightFace's `buffalo_l` pack.
# Only the detector is loaded; the pack's recognition model is never used by this pipeline.
DETECTION_MODEL_PACK_NAME = "buffalo_l"
MODELS_ROOT = Path(__file__).resolve().parent / "models"

# Recognition: OpenCV SFace, downloaded from the OpenCV Zoo by download_models.py.
SFACE_MODEL_FILENAME = "sface_2021dec.onnx"
SFACE_MODEL_PATH = MODELS_ROOT / SFACE_MODEL_FILENAME
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Detection settings. Faces scoring below MIN_DETECTION_CONFIDENCE are dropped.
MIN_DETECTION_CONFIDENCE = 0.5
DETECTION_INPUT_SIZE = 640

# Identification settings, chosen from the experiment in README.md.
# TOP2 = a person's score is the mean cosine similarity to their 2 closest reference images.
MATCHING_STRATEGY_TOP_K = 2
IDENTIFICATION_THRESHOLD = 0.45
UNKNOWN_LABEL = "unknown"

# Filenames written by build_face_database.py and read back by identify_faces.py.
FACE_DATABASE_FILENAME = "face_database.npz"
FACE_DATABASE_MANIFEST_FILENAME = "face_database_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetectedFace:
    """One face found in an image, with its L2-normalized 128-d SFace embedding."""
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


def build_face_detector() -> FaceAnalysis:
    """Loads only the SCRFD-10GF detector from the buffalo_l pack; recognition is SFace's job."""
    face_detector = FaceAnalysis(
        name=DETECTION_MODEL_PACK_NAME,
        root=str(MODELS_ROOT),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    face_detector.prepare(
        ctx_id=-1,
        det_size=(DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE),
        det_thresh=MIN_DETECTION_CONFIDENCE,
    )
    return face_detector


def build_sface_recognizer() -> cv2.FaceRecognizerSF:
    """Loads the OpenCV SFace recognizer, which does its own alignment and embedding."""
    if not SFACE_MODEL_PATH.is_file():
        raise SystemExit(f"SFace model not found: {SFACE_MODEL_PATH}\nRun download_models.py first.")
    return cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scales a vector to unit length, so a dot product between two of them is a cosine similarity."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_sface_face_row(bounding_box_xyxy: np.ndarray, keypoints: np.ndarray, confidence: float) -> np.ndarray:
    """Adapts one detection into the 15-value row `FaceRecognizerSF.alignCrop()` expects.

    The row is [x, y, w, h, 5x(landmark_x, landmark_y), confidence]. No landmark reordering is
    needed: OpenCV's alignCrop warps to the same ArcFace reference points InsightFace's own
    alignment uses, so SCRFD's native order (left eye, right eye, nose, left mouth, right
    mouth) is already the order it wants.
    """
    x1, y1, x2, y2 = bounding_box_xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1, *keypoints.reshape(-1), confidence], dtype=np.float32)


def detect_and_embed_faces(
    face_detector: FaceAnalysis,
    sface_recognizer: cv2.FaceRecognizerSF,
    image_bgr: np.ndarray,
) -> list[DetectedFace]:
    """Detects every face in one BGR image and returns each one's SFace embedding.

    SFace aligns and crops each face itself from the full image using the detection's box and
    keypoints, so no alignment transform is ever computed here.
    """
    detected_faces = []
    for face in face_detector.get(image_bgr):
        face_row = build_sface_face_row(face.bbox, face.kps, face.det_score)
        aligned_face_bgr = sface_recognizer.alignCrop(image_bgr, face_row)
        embedding = l2_normalize(sface_recognizer.feature(aligned_face_bgr).flatten())
        detected_faces.append(DetectedFace(
            bounding_box_xyxy=face.bbox.astype(np.float32),
            detection_confidence=float(face.det_score),
            embedding=embedding.astype(np.float32),
        ))
    return detected_faces


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
