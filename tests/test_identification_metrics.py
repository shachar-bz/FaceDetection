"""Tests for the outcome counting that produces every number in the published results.

compute_metrics decides what counts as a correct identification, a false accept and a false
reject. Those definitions are the whole claim of the study, so they are pinned here.
"""
import numpy as np
import pytest

from experiments.identification_study.evaluate_identification import compute_metrics
from face_identity.configuration import UNKNOWN_PERSON_LABEL

UNKNOWN = UNKNOWN_PERSON_LABEL


def score(ground_truth: list[str], predictions: list[str], undetected: list[bool] | None = None) -> dict:
    """Runs compute_metrics over plain lists."""
    undetected = undetected if undetected is not None else [False] * len(ground_truth)
    return compute_metrics(np.array(ground_truth), np.array(predictions), np.array(undetected))


class TestOutcomeCounting:
    def test_a_known_face_named_correctly_is_a_correct_identification(self):
        metrics = score(["amy"], ["amy"])
        assert metrics["correct_identifications"] == 1
        assert metrics["incorrect_identifications"] == 0

    def test_a_distractor_called_unknown_is_a_correct_rejection(self):
        metrics = score([UNKNOWN], [UNKNOWN])
        assert metrics["correct_rejections"] == 1
        assert metrics["incorrect_identifications"] == 0

    def test_a_distractor_given_a_name_is_a_false_accept(self):
        metrics = score([UNKNOWN], ["amy"])
        assert metrics["unknown_as_known"] == 1
        assert metrics["incorrect_identifications"] == 1
        assert metrics["correct_identifications"] == 0

    def test_a_known_face_called_unknown_is_a_false_reject(self):
        metrics = score(["amy"], [UNKNOWN])
        assert metrics["known_as_unknown"] == 1
        # A rejection asserts no identity, so it costs recall but not precision.
        assert metrics["incorrect_identifications"] == 0

    def test_a_known_face_given_the_wrong_name_is_a_wrong_identity(self):
        metrics = score(["amy"], ["bob"])
        assert metrics["wrong_identity"] == 1
        assert metrics["incorrect_identifications"] == 1
        assert metrics["correct_identifications"] == 0

    def test_every_face_lands_in_exactly_one_outcome(self):
        metrics = score(
            ["amy", "bob", UNKNOWN, UNKNOWN, "cy"],
            ["amy", UNKNOWN, UNKNOWN, "amy", "bob"],
        )
        counted = (metrics["correct_identifications"] + metrics["correct_rejections"]
                   + metrics["wrong_identity"] + metrics["known_as_unknown"]
                   + metrics["unknown_as_known"])
        assert counted == metrics["total_faces"] == 5


class TestDerivedRates:
    def test_a_perfect_run_scores_one_everywhere(self):
        metrics = score(["amy", "bob", UNKNOWN], ["amy", "bob", UNKNOWN])
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_precision_is_correct_names_over_names_asserted(self):
        # Three names asserted, two of them right. Rates are rounded to 4dp for the CSV.
        metrics = score(["amy", "bob", UNKNOWN], ["amy", "bob", "cy"])
        assert metrics["precision"] == pytest.approx(2 / 3, abs=5e-5)

    def test_recall_is_correct_names_over_known_faces(self):
        # Two known faces, one of them found.
        metrics = score(["amy", "bob", UNKNOWN], ["amy", UNKNOWN, UNKNOWN])
        assert metrics["recall"] == pytest.approx(0.5)

    def test_a_wrong_identity_costs_both_precision_and_recall(self):
        metrics = score(["amy"], ["bob"])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_rejecting_everything_gives_no_precision_rather_than_dividing_by_zero(self):
        metrics = score(["amy", UNKNOWN], [UNKNOWN, UNKNOWN])
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0

    def test_accuracy_counts_correct_rejections_as_successes(self):
        metrics = score(["amy", UNKNOWN, UNKNOWN], ["amy", UNKNOWN, UNKNOWN])
        assert metrics["accuracy"] == 1.0

    def test_known_and_unknown_faces_are_counted_separately(self):
        metrics = score(["amy", "bob", UNKNOWN, UNKNOWN], [UNKNOWN] * 4)
        assert metrics["known_faces"] == 2
        assert metrics["unknown_faces"] == 2


class TestUndetectedFaces:
    def test_undetected_faces_are_reported_by_whether_the_person_was_known(self):
        metrics = score(
            ["amy", UNKNOWN, "bob"],
            [UNKNOWN, UNKNOWN, "bob"],
            undetected=[True, True, False],
        )
        assert metrics["undetected_known"] == 1
        assert metrics["undetected_unknown"] == 1

    def test_an_undetected_known_face_still_counts_as_a_missed_person(self):
        metrics = score(["amy"], [UNKNOWN], undetected=[True])
        assert metrics["known_as_unknown"] == 1
        assert metrics["recall"] == 0.0
