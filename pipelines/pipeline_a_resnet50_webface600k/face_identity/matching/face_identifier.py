"""Turns similarity scores into an identity decision: a person's name, or the unknown label."""
from dataclasses import dataclass

import numpy as np

from face_identity.configuration import UNKNOWN_PERSON_LABEL
from face_identity.embedding.face_embedder import EmbeddedFace
from face_identity.matching.face_database import FaceDatabase
from face_identity.matching.matching_strategies import score_faces_against_people


@dataclass
class IdentifiedFace:
    """A detected face plus the identity decision made for it."""

    bounding_box_xyxy: np.ndarray
    detection_confidence: float
    predicted_person: str
    similarity_score: float

    @property
    def is_known(self) -> bool:
        """Says whether the face was matched to a person rather than rejected as unknown."""
        return self.predicted_person != UNKNOWN_PERSON_LABEL


def identify_faces(
    embedded_faces: list[EmbeddedFace],
    database: FaceDatabase,
    matching_strategy: str,
    identification_threshold: float,
) -> list[IdentifiedFace]:
    """Names each face, or calls it unknown when its best person score is below the threshold."""
    if not embedded_faces:
        return []

    face_embeddings = np.stack([face.embedding for face in embedded_faces])
    person_scores = score_faces_against_people(face_embeddings, database, matching_strategy)
    best_person_indices = person_scores.argmax(axis=1)
    best_scores = person_scores.max(axis=1)

    identified_faces = []
    for face, best_person_index, best_score in zip(embedded_faces, best_person_indices, best_scores):
        is_known = best_score >= identification_threshold
        predicted_person = str(database.person_names[best_person_index]) if is_known else UNKNOWN_PERSON_LABEL
        identified_faces.append(
            IdentifiedFace(
                bounding_box_xyxy=face.bounding_box_xyxy,
                detection_confidence=face.detection_confidence,
                predicted_person=predicted_person,
                similarity_score=float(best_score),
            )
        )
    return identified_faces
