"""Tests that each generated standalone bundle is pinned to one validated pipeline."""
import pytest

from make_pipeline import bundle_pipeline


@pytest.mark.parametrize(
    ("pipeline_key", "configuration_constant", "expected_model_names"),
    [
        ("a", "PIPELINE_A_RESNET50_WEBFACE600K", "[]"),
        ("b", "PIPELINE_B_SFACE", "['sface']"),
    ],
)
def test_bundle_exposes_only_its_selected_pipeline(
    tmp_path, pipeline_key, configuration_constant, expected_model_names
):
    """A standalone folder cannot silently select or download the other pipeline."""
    bundle_directory = bundle_pipeline(pipeline_key, tmp_path)
    configuration = (bundle_directory / "face_identity" / "configuration.py").read_text(
        encoding="utf-8"
    )
    model_downloads = (bundle_directory / "face_identity" / "model_downloads.py").read_text(
        encoding="utf-8"
    )

    mapping = (
        "PIPELINE_CONFIGURATIONS = {\n"
        f'    "{pipeline_key}": {configuration_constant},\n'
        "}"
    )
    assert mapping in configuration
    assert f'DEFAULT_PIPELINE_KEY = "{pipeline_key}"' in configuration
    assert f"PIPELINE_MODEL_NAMES = {expected_model_names}" in model_downloads
