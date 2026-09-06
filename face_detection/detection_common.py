"""Helpers shared by the face-detector evaluation scripts in this folder.

Holds the pieces that were duplicated across detect.py, detect_scrfd.py,
annotate_dataset.py and annotate_dataset_scrfd.py: the two detector
constructors, the square crop-box expansion formula, and the BlazeFace model
filenames. Detection behaviour, thresholds and drawing are unchanged --
this module only removes the duplication.
"""
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported for type annotations only; see the note on the builders below
    from insightface.model_zoo.scrfd import SCRFD
    from mediapipe.tasks.python import vision as mp_vision

# BlazeFace model filenames, as written into models/ by
# downloading_models_scripts/download_face_detector_models.py -- these two
# names must stay in step with that script.
BLAZE_FACE_SHORT_RANGE_FILENAME = "blaze_face_short_range.tflite"
BLAZE_FACE_FULL_RANGE_FILENAME = "blaze_face_full_range.tflite"

# Detection defaults, matching the values the committed results in
# results_blazeface/ and results_scrfd/ were produced with.
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_SCRFD_INPUT_SIZE = 640
DEFAULT_SCRFD_NMS_THRESH = 0.4
# Crop margin per side, as a fraction of the longer box side, used when
# drawing annotated boxes. See the crop-margin finding in README.md.
DEFAULT_CROP_MARGIN_RATIO = 0.5


def build_blazeface_detector(model_path: Path, min_confidence: float) -> "mp_vision.FaceDetector":
    """Creates a MediaPipe Face Detector for one BlazeFace .tflite variant."""
    # Imported here, not at module scope, so the SCRFD scripts don't load
    # MediaPipe (and the BlazeFace scripts don't load InsightFace/onnxruntime)
    # just by importing this module.
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def build_scrfd_detector(model_path: Path, min_confidence: float, input_size: int, nms_thresh: float) -> "SCRFD":
    """Creates an SCRFD-10G-KPS detector running on CPU."""
    from insightface.model_zoo.scrfd import SCRFD

    detector = SCRFD(model_file=str(model_path))
    detector.prepare(ctx_id=-1, det_thresh=min_confidence, input_size=(input_size, input_size), nms_thresh=nms_thresh)
    return detector


def expand_box_square(x: int, y: int, w: int, h: int, margin_ratio: float, img_w: int, img_h: int):
    """Expands a detection box into a square crop centred on it, grown by
    margin_ratio per side (as a fraction of the longer side) and clamped to
    the image bounds. Returns (x0, y0, x1, y1)."""
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin_ratio)
    half = side / 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(img_w, cx + half), min(img_h, cy + half)
    return int(x0), int(y0), int(x1), int(y1)
