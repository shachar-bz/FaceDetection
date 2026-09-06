"""Downloads the two models pipeline B needs into this folder's models/ directory.

Run this once, before building the face database:
  1. The OpenCV SFace recognition model, fetched from the OpenCV Zoo (it has no auto-download).
  2. InsightFace's `buffalo_l` pack, which supplies the SCRFD-10GF detector and its 5 facial
     keypoints; InsightFace downloads and caches it itself the first time a FaceAnalysis app
     is prepared, so this script just triggers that.
"""
import urllib.request

from face_pipeline import (
    DETECTION_MODEL_PACK_NAME,
    MODELS_ROOT,
    SFACE_MODEL_PATH,
    SFACE_MODEL_URL,
    build_face_detector,
)


def download_sface_model() -> None:
    """Fetches the SFace ONNX weights from the OpenCV Zoo, unless they are already present."""
    if SFACE_MODEL_PATH.is_file():
        print(f"SFace model already present: {SFACE_MODEL_PATH}")
        return
    print(f"Downloading {SFACE_MODEL_URL} -> {SFACE_MODEL_PATH}")
    request = urllib.request.Request(SFACE_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(SFACE_MODEL_PATH, "wb") as out_file:
        out_file.write(response.read())


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    download_sface_model()
    print(f"Downloading the '{DETECTION_MODEL_PACK_NAME}' detector pack into {MODELS_ROOT} (first run only)...")
    build_face_detector()
    print(f"Done. Models are in {MODELS_ROOT}")


if __name__ == "__main__":
    main()
