"""Tests for the pipeline configurations and for how labelled reference images are discovered.

The configurations are the study's conclusion written down as code, so they are pinned against
the published results. Image discovery decides which folder name becomes a person's label,
which is the difference between a correct database and a silently mislabelled one.
"""
import numpy as np
import pytest

from build_face_database import reference_face_count_error
from face_identity.configuration import (
    PIPELINE_A_RESNET50_WEBFACE600K,
    PIPELINE_B_SFACE,
    PIPELINE_CONFIGURATIONS,
    resolve_models_directory,
    resolve_pipeline_configuration,
)
from face_identity.embedding.embedder_registry import EMBEDDING_MODEL_NAMES, build_face_embedder
from face_identity.embedding.sface_embedder import build_sface_face_row
from face_identity.image_io import discover_person_images
from face_identity.matching.matching_strategies import MATCHING_STRATEGIES


class TestPipelineConfigurations:
    def test_pipeline_a_matches_the_configuration_the_study_selected(self):
        assert PIPELINE_A_RESNET50_WEBFACE600K.embedding_model_name == "resnet_webface600k"
        assert PIPELINE_A_RESNET50_WEBFACE600K.matching_strategy == "top3"
        assert PIPELINE_A_RESNET50_WEBFACE600K.identification_threshold == 0.30

    def test_pipeline_b_matches_the_runner_up_configuration(self):
        assert PIPELINE_B_SFACE.embedding_model_name == "sface"
        assert PIPELINE_B_SFACE.matching_strategy == "top2"
        assert PIPELINE_B_SFACE.identification_threshold == 0.45

    def test_both_pipelines_detect_at_the_confidence_the_benchmark_used(self):
        for configuration in PIPELINE_CONFIGURATIONS.values():
            assert configuration.min_detection_confidence == 0.5
            assert configuration.detection_input_size == 640

    def test_every_configuration_names_a_real_model_and_a_real_strategy(self):
        for configuration in PIPELINE_CONFIGURATIONS.values():
            assert configuration.embedding_model_name in EMBEDDING_MODEL_NAMES
            assert configuration.matching_strategy in MATCHING_STRATEGIES

    def test_configurations_are_frozen_so_a_caller_cannot_retune_them_by_accident(self):
        with pytest.raises(Exception):
            PIPELINE_A_RESNET50_WEBFACE600K.identification_threshold = 0.9

    def test_an_unknown_pipeline_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            resolve_pipeline_configuration("z")

    def test_an_unknown_embedding_model_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            build_face_embedder("not_a_model")


class TestReferenceImageValidation:
    def test_exactly_one_detected_face_is_accepted(self):
        assert reference_face_count_error(1) is None

    def test_an_image_without_a_detected_face_is_skipped(self):
        assert reference_face_count_error(0) == "no_face_detected"

    def test_an_image_with_multiple_detected_faces_is_skipped(self):
        assert reference_face_count_error(3) == "multiple_faces_detected:3"


class TestResolveModelsDirectory:
    def test_an_explicit_directory_is_used_and_created(self, tmp_path):
        target = tmp_path / "weights"
        assert resolve_models_directory(target) == target.resolve()
        assert target.is_dir()

    def test_the_environment_variable_overrides_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACE_IDENTITY_MODELS_DIR", str(tmp_path / "from_environment"))
        assert resolve_models_directory() == (tmp_path / "from_environment").resolve()


class TestDiscoverPersonImages:
    def write_image_files(self, root, relative_paths):
        for relative_path in relative_paths:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not really an image, only the name matters here")

    def test_the_folder_containing_an_image_names_the_person(self, tmp_path):
        self.write_image_files(tmp_path, ["Amy Adams/one.jpg", "Bob Barker/one.jpg"])
        found = discover_person_images(tmp_path)
        assert sorted(image.person for image in found) == ["Amy Adams", "Bob Barker"]

    def test_a_grouped_layout_still_names_the_person_from_the_nearest_folder(self, tmp_path):
        self.write_image_files(tmp_path, ["ministers/Amy Adams/one.jpg", "press/Bob Barker/one.jpg"])
        found = {image.person: image.group for image in discover_person_images(tmp_path)}
        assert found == {"Amy Adams": "ministers", "Bob Barker": "press"}

    def test_a_flat_layout_reports_no_group(self, tmp_path):
        self.write_image_files(tmp_path, ["Amy Adams/one.jpg"])
        assert discover_person_images(tmp_path)[0].group == ""

    def test_only_groups_that_were_asked_for_are_returned(self, tmp_path):
        self.write_image_files(tmp_path, ["ministers/Amy Adams/one.jpg", "press/Bob Barker/one.jpg"])
        found = discover_person_images(tmp_path, include_groups={"ministers"})
        assert [image.person for image in found] == ["Amy Adams"]

    def test_files_that_are_not_images_are_skipped(self, tmp_path):
        self.write_image_files(tmp_path, ["Amy Adams/one.jpg", "Amy Adams/notes.txt"])
        found = discover_person_images(tmp_path)
        assert [image.path.name for image in found] == ["one.jpg"]

    def test_every_supported_extension_is_found_whatever_its_case(self, tmp_path):
        self.write_image_files(tmp_path, [
            "Amy Adams/a.jpg", "Amy Adams/b.JPEG", "Amy Adams/c.png",
            "Amy Adams/d.BMP", "Amy Adams/e.webp",
        ])
        assert len(discover_person_images(tmp_path)) == 5

    def test_discovery_is_ordered_so_a_rebuild_is_reproducible(self, tmp_path):
        self.write_image_files(tmp_path, ["Bob/2.jpg", "Amy/1.jpg", "Bob/1.jpg"])
        first_pass = [str(image.relative_path) for image in discover_person_images(tmp_path)]
        second_pass = [str(image.relative_path) for image in discover_person_images(tmp_path)]
        assert first_pass == second_pass == sorted(first_pass)

    def test_the_relative_path_is_relative_to_the_root(self, tmp_path):
        self.write_image_files(tmp_path, ["ministers/Amy Adams/one.jpg"])
        [image] = discover_person_images(tmp_path)
        assert image.relative_path.as_posix() == "ministers/Amy Adams/one.jpg"


class TestBuildSFaceFaceRow:
    def test_the_row_is_the_fifteen_values_align_crop_expects(self):
        box = np.array([10, 20, 40, 70], dtype=np.float32)
        keypoints = np.arange(10, dtype=np.float32).reshape(5, 2)
        row = build_sface_face_row(box, keypoints, 0.87)
        assert row.shape == (15,)

    def test_the_box_is_converted_from_corners_to_width_and_height(self):
        box = np.array([10, 20, 40, 70], dtype=np.float32)
        keypoints = np.zeros((5, 2), dtype=np.float32)
        row = build_sface_face_row(box, keypoints, 0.87)
        assert list(row[:4]) == [10, 20, 30, 50]

    def test_the_keypoints_keep_the_detector_order(self):
        box = np.array([0, 0, 1, 1], dtype=np.float32)
        keypoints = np.arange(10, dtype=np.float32).reshape(5, 2)
        row = build_sface_face_row(box, keypoints, 0.5)
        assert list(row[4:14]) == list(range(10))

    def test_the_confidence_is_the_last_value(self):
        row = build_sface_face_row(
            np.array([0, 0, 1, 1], dtype=np.float32), np.zeros((5, 2), dtype=np.float32), 0.87)
        assert row[14] == pytest.approx(0.87)
