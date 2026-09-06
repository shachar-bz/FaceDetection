"""Download the MediaPipe Face Detector .tflite models used by face_detection/detect.py."""
import urllib.request
from pathlib import Path

# These filenames must stay in step with face_detection/detection_common.py's
# BLAZE_FACE_SHORT_RANGE_FILENAME / BLAZE_FACE_FULL_RANGE_FILENAME.
MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "blaze_face_full_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
    ),
}


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        dest = models_dir / filename
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main()
