"""How a query face is scored against each known person -- the one implementation of every strategy.

Every strategy scores a face against each candidate *person* rather than running a global
nearest-neighbour search over all reference images, so a person with many reference images
gains no advantage over one with few.
"""
import numpy as np

from face_identity.matching.face_database import FaceDatabase

# TOP-K: a person's score is the mean similarity to their K closest reference images.
TOP_K_BY_STRATEGY = {"top1": 1, "top2": 2, "top3": 3}
# CENTROID: a person's score is the similarity to the L2-renormalized mean of their embeddings.
CENTROID_STRATEGY = "centroid"
MATCHING_STRATEGIES = [*TOP_K_BY_STRATEGY, CENTROID_STRATEGY]


def score_faces_against_people(
    face_embeddings: np.ndarray, database: FaceDatabase, matching_strategy: str
) -> np.ndarray:
    """Scores every query face against every known person, returning a faces x people matrix.

    Embeddings are L2-normalized, so one matrix product yields all cosine similarities at once.
    A person with fewer reference images than the strategy needs is scored on all the images
    they do have.
    """
    if matching_strategy == CENTROID_STRATEGY:
        return face_embeddings @ database.person_centroids.T

    if matching_strategy not in TOP_K_BY_STRATEGY:
        raise ValueError(
            f"Unknown matching strategy {matching_strategy!r}; choose one of {MATCHING_STRATEGIES}"
        )

    top_k = TOP_K_BY_STRATEGY[matching_strategy]
    similarities = face_embeddings @ database.embeddings.T
    scores = np.empty((face_embeddings.shape[0], database.person_count), dtype=np.float32)
    for person_index, (start, end) in enumerate(database.person_slices):
        person_similarities = similarities[:, start:end]
        available_k = min(top_k, end - start)
        if available_k == 1:
            scores[:, person_index] = person_similarities.max(axis=1)
        else:
            closest = np.partition(person_similarities, -available_k, axis=1)[:, -available_k:]
            scores[:, person_index] = closest.mean(axis=1)
    return scores
