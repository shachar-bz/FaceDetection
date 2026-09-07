"""Tests for the face database layout and the strategies that score a face against each person.

Every identification decision the project makes runs through this code, and a wrong per-person
slice or a wrong TOP-K average produces plausible-looking numbers rather than a crash.
"""
import numpy as np
import pytest

from face_identity.configuration import UNKNOWN_PERSON_LABEL
from face_identity.embedding.face_embedder import EmbeddedFace, l2_normalize
from face_identity.matching.face_database import (
    FaceDatabase,
    load_face_database,
    save_face_database,
)
from face_identity.matching.face_identifier import identify_faces
from face_identity.matching.matching_strategies import (
    MATCHING_STRATEGIES,
    score_faces_against_people,
)


def unit_vectors(rows: list[list[float]]) -> np.ndarray:
    """Builds an L2-normalized embedding matrix from plain numbers."""
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def two_person_database() -> FaceDatabase:
    """A database where 'amy' has three reference faces of varying similarity and 'bob' has one."""
    embeddings = unit_vectors([
        [1.0, 0.0],       # amy, identical to the query below
        [0.8, 0.6],       # amy, close
        [0.0, 1.0],       # amy, orthogonal
        [-1.0, 0.0],      # bob, opposite
    ])
    return FaceDatabase.from_labelled_embeddings(
        np.array(["amy", "amy", "amy", "bob"]), embeddings, "test_model")


class TestFaceDatabaseLayout:
    def test_each_person_owns_one_contiguous_slice(self):
        embeddings = unit_vectors([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]])
        database = FaceDatabase.from_labelled_embeddings(
            np.array(["bob", "amy", "bob", "amy", "cy"]), embeddings)
        assert list(database.person_names) == ["amy", "bob", "cy"]
        assert database.person_slices == [(0, 2), (2, 4), (4, 5)]

    def test_every_row_belongs_to_exactly_one_slice(self):
        embeddings = unit_vectors([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]])
        database = FaceDatabase.from_labelled_embeddings(
            np.array(["bob", "amy", "bob", "amy", "cy"]), embeddings)
        covered = sum(end - start for start, end in database.person_slices)
        assert covered == len(database.embeddings)

    def test_person_count_counts_people_not_images(self, two_person_database):
        assert two_person_database.person_count == 2
        assert len(two_person_database.embeddings) == 4

    def test_centroids_are_unit_length(self, two_person_database):
        norms = np.linalg.norm(two_person_database.person_centroids, axis=1)
        assert norms == pytest.approx(np.ones(2), abs=1e-6)

    def test_there_is_one_centroid_per_person(self, two_person_database):
        assert two_person_database.person_centroids.shape[0] == two_person_database.person_count


class TestMatchingStrategies:
    def test_top1_takes_the_single_closest_reference_image(self, two_person_database):
        query = unit_vectors([[1.0, 0.0]])
        scores = score_faces_against_people(query, two_person_database, "top1")
        assert scores[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_top3_averages_the_three_closest_and_so_scores_lower_than_top1(self, two_person_database):
        query = unit_vectors([[1.0, 0.0]])
        top1 = score_faces_against_people(query, two_person_database, "top1")[0, 0]
        top3 = score_faces_against_people(query, two_person_database, "top3")[0, 0]
        # amy's similarities are 1.0, 0.8 and 0.0, so the mean of all three is 0.6.
        assert top3 == pytest.approx(0.6, abs=1e-6)
        assert top3 < top1

    def test_a_person_with_fewer_images_than_k_is_scored_on_what_they_have(self, two_person_database):
        # bob has a single reference image, so TOP3 cannot average three of them.
        query = unit_vectors([[-1.0, 0.0]])
        scores = score_faces_against_people(query, two_person_database, "top3")
        assert scores[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_centroid_scores_against_the_mean_of_a_persons_embeddings(self, two_person_database):
        query = unit_vectors([[1.0, 0.0]])
        expected = float(query[0] @ two_person_database.person_centroids[0])
        scores = score_faces_against_people(query, two_person_database, "centroid")
        assert scores[0, 0] == pytest.approx(expected, abs=1e-6)

    def test_every_strategy_returns_one_score_per_face_per_person(self, two_person_database):
        queries = unit_vectors([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        for strategy in MATCHING_STRATEGIES:
            scores = score_faces_against_people(queries, two_person_database, strategy)
            assert scores.shape == (3, two_person_database.person_count), strategy

    def test_an_unknown_strategy_is_rejected(self, two_person_database):
        with pytest.raises(ValueError, match="Unknown matching strategy"):
            score_faces_against_people(unit_vectors([[1.0, 0.0]]), two_person_database, "top7")

    def test_scoring_is_per_person_not_a_global_nearest_neighbour(self):
        """A person with many mediocre images must not out-score one with a single perfect match."""
        embeddings = unit_vectors([
            [0.9, 0.44], [0.9, 0.44], [0.9, 0.44], [0.9, 0.44],  # crowded, all merely close
            [1.0, 0.0],                                            # single exact match
        ])
        database = FaceDatabase.from_labelled_embeddings(
            np.array(["crowded", "crowded", "crowded", "crowded", "exact"]), embeddings)
        scores = score_faces_against_people(unit_vectors([[1.0, 0.0]]), database, "top1")
        crowded_index = list(database.person_names).index("crowded")
        exact_index = list(database.person_names).index("exact")
        assert scores[0, exact_index] > scores[0, crowded_index]


class TestIdentifyFaces:
    def build_face(self, embedding: np.ndarray) -> EmbeddedFace:
        return EmbeddedFace(
            bounding_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            detection_confidence=0.9,
            embedding=embedding,
        )

    def test_a_close_match_above_the_threshold_is_named(self, two_person_database):
        face = self.build_face(unit_vectors([[1.0, 0.0]])[0])
        [identified] = identify_faces([face], two_person_database, "top1", 0.30)
        assert identified.predicted_person == "amy"
        assert identified.is_known

    def test_the_same_match_below_the_threshold_is_rejected(self, two_person_database):
        face = self.build_face(unit_vectors([[1.0, 0.0]])[0])
        [identified] = identify_faces([face], two_person_database, "top1", 1.01)
        assert identified.predicted_person == UNKNOWN_PERSON_LABEL
        assert not identified.is_known

    def test_a_score_exactly_on_the_threshold_is_accepted(self, two_person_database):
        face = self.build_face(unit_vectors([[1.0, 0.0]])[0])
        [identified] = identify_faces([face], two_person_database, "top1", 1.0)
        assert identified.predicted_person == "amy"

    def test_the_reported_box_and_confidence_come_from_the_detection(self, two_person_database):
        face = self.build_face(unit_vectors([[1.0, 0.0]])[0])
        [identified] = identify_faces([face], two_person_database, "top1", 0.30)
        assert list(identified.bounding_box_xyxy) == [0, 0, 10, 10]
        assert identified.detection_confidence == pytest.approx(0.9)

    def test_no_faces_yields_no_decisions(self, two_person_database):
        assert identify_faces([], two_person_database, "top3", 0.30) == []


class TestFaceDatabaseStorage:
    def test_a_saved_database_loads_back_unchanged(self, tmp_path):
        embeddings = unit_vectors([[1, 0], [0, 1], [1, 1]])
        names = ["bob", "amy", "bob"]
        save_face_database(tmp_path, names, embeddings, "resnet_webface600k")
        loaded = load_face_database(tmp_path / "face_database.npz")
        assert list(loaded.person_names) == ["amy", "bob"]
        assert loaded.embeddings.shape == (3, 2)
        assert loaded.embedding_model_name == "resnet_webface600k"

    def test_querying_with_the_model_that_built_it_is_allowed(self, tmp_path):
        save_face_database(tmp_path, ["amy"], unit_vectors([[1, 0]]), "sface")
        load_face_database(tmp_path / "face_database.npz").require_embedding_model("sface")

    def test_querying_with_a_different_model_is_refused(self, tmp_path):
        save_face_database(tmp_path, ["amy"], unit_vectors([[1, 0]]), "sface")
        loaded = load_face_database(tmp_path / "face_database.npz")
        with pytest.raises(ValueError, match="not comparable"):
            loaded.require_embedding_model("resnet_webface600k")


class TestL2Normalize:
    def test_a_normalized_vector_has_unit_length(self):
        assert np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)

    def test_direction_is_preserved(self):
        normalized = l2_normalize(np.array([3.0, 4.0]))
        assert normalized == pytest.approx(np.array([0.6, 0.8]))

    def test_a_zero_vector_is_left_alone_rather_than_dividing_by_zero(self):
        assert l2_normalize(np.zeros(3)) == pytest.approx(np.zeros(3))
