"""Builds any of the benchmarked detectors by name, so callers never import a backend directly."""
from pathlib import Path

from face_identity.configuration import (
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_SCRFD_NMS_THRESHOLD,
    resolve_models_directory,
)
from face_identity.detection.blazeface_detector import BLAZEFACE_MODEL_FILENAMES, BlazeFaceDetector
from face_identity.detection.face_detector import FaceDetector
from face_identity.detection.scrfd_detector import SCRFD_MODEL_FILENAME, SCRFDDetector

SCRFD_DETECTOR_NAME = "scrfd_10g_kps"
BENCHMARK_DETECTOR_NAMES = [*BLAZEFACE_MODEL_FILENAMES, SCRFD_DETECTOR_NAME]


def build_detector(
    detector_name: str,
    models_directory: Path | None = None,
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    input_size: int = DEFAULT_DETECTION_INPUT_SIZE,
    nms_threshold: float = DEFAULT_SCRFD_NMS_THRESHOLD,
) -> FaceDetector:
    """Creates one of the benchmarked detectors by name, loading its weights from the cache."""
    models_directory = resolve_models_directory(models_directory)

    if detector_name in BLAZEFACE_MODEL_FILENAMES:
        model_path = models_directory / BLAZEFACE_MODEL_FILENAMES[detector_name]
        return BlazeFaceDetector(model_path, min_detection_confidence)

    if detector_name == SCRFD_DETECTOR_NAME:
        model_path = models_directory / SCRFD_MODEL_FILENAME
        return SCRFDDetector(model_path, min_detection_confidence, input_size, nms_threshold)

    raise ValueError(f"Unknown detector {detector_name!r}; choose one of {BENCHMARK_DETECTOR_NAMES}")
