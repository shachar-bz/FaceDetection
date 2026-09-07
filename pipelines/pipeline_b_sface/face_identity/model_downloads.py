"""Fetches every model this project uses into one shared cache directory.

This is the only place a model URL is written down. Weights already present are left alone,
so the script is safe to re-run. The InsightFace `buffalo_l` pack has no direct download here
because InsightFace fetches and caches it itself the first time a detector is prepared -- this
script just triggers that.
"""
import argparse
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from face_identity.configuration import resolve_models_directory
from face_identity.detection.blazeface_detector import BLAZEFACE_MODEL_FILENAMES
from face_identity.detection.scrfd_detector import INSIGHTFACE_MODEL_PACK_NAME, SCRFD_MODEL_FILENAME
from face_identity.embedding.sface_embedder import SFACE_MODEL_FILENAME

# Some hosts reject the default urllib agent, so every request identifies as a browser.
DOWNLOAD_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
INSIGHTFACE_MODEL_PACK_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
# SCRFD-10G-KPS ships inside the pack above; only the detector is extracted from it.
SCRFD_MEMBER_IN_MODEL_PACK = "det_10g.onnx"


@dataclass(frozen=True)
class DownloadableModel:
    """One model file, where it comes from, and the filename it is cached under."""

    filename: str
    url: str
    member_in_zip: str | None = None


DOWNLOADABLE_MODELS = {
    "sface": DownloadableModel(
        filename=SFACE_MODEL_FILENAME,
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
    ),
    "scrfd_10g_kps": DownloadableModel(
        filename=SCRFD_MODEL_FILENAME,
        url=INSIGHTFACE_MODEL_PACK_URL,
        member_in_zip=SCRFD_MEMBER_IN_MODEL_PACK,
    ),
    "blazeface_short_range": DownloadableModel(
        filename=BLAZEFACE_MODEL_FILENAMES["blazeface_short_range"],
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
    ),
    "blazeface_full_range": DownloadableModel(
        filename=BLAZEFACE_MODEL_FILENAMES["blazeface_full_range"],
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
        ),
    ),
}

# What each caller needs: the pipelines only need SFace on top of the auto-downloaded pack,
# while the detection benchmark needs the standalone detectors it compares.
PIPELINE_MODEL_NAMES = ['sface']
BENCHMARK_MODEL_NAMES = ["scrfd_10g_kps", "blazeface_short_range", "blazeface_full_range"]
MODEL_GROUPS = {
    "pipelines": PIPELINE_MODEL_NAMES,
    "benchmark": BENCHMARK_MODEL_NAMES,
    "all": [*PIPELINE_MODEL_NAMES, *BENCHMARK_MODEL_NAMES],
}
DEFAULT_MODEL_GROUP = "pipelines"


def download_model(model_name: str, models_directory: Path, overwrite: bool = False) -> Path:
    """Downloads one registered model into the cache, skipping it when already present."""
    model = DOWNLOADABLE_MODELS[model_name]
    destination = models_directory / model.filename
    if destination.is_file() and not overwrite:
        print(f"  already present: {destination}")
        return destination

    print(f"  downloading {model_name} from {model.url}")
    request = urllib.request.Request(model.url, headers=DOWNLOAD_REQUEST_HEADERS)
    with urllib.request.urlopen(request) as response:
        downloaded_bytes = response.read()

    if model.member_in_zip is not None:
        with zipfile.ZipFile(io.BytesIO(downloaded_bytes)) as archive:
            downloaded_bytes = archive.read(model.member_in_zip)

    destination.write_bytes(downloaded_bytes)
    print(f"  wrote {destination}")
    return destination


def download_insightface_model_pack(models_directory: Path, with_recognition: bool = True) -> None:
    """Triggers InsightFace's own download of the buffalo_l pack into the cache directory."""
    from face_identity.detection.scrfd_detector import InsightFacePackDetector

    print(f"  preparing the {INSIGHTFACE_MODEL_PACK_NAME} pack in {models_directory} (first run downloads it)")
    InsightFacePackDetector(models_directory=models_directory, with_recognition=with_recognition).close()


def main() -> None:
    """Downloads the requested group of models into the shared cache directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Where to cache the weights (default: the FACE_IDENTITY_MODELS_DIR "
                             "environment variable, else models/ beside this repository)")
    parser.add_argument("--group", choices=sorted(MODEL_GROUPS), default=DEFAULT_MODEL_GROUP,
                        help="Which set of models to fetch")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--skip-insightface-pack", action="store_true",
                        help="Do not trigger the buffalo_l pack download")
    arguments = parser.parse_args()

    models_directory = resolve_models_directory(arguments.models_dir)
    print(f"Model cache: {models_directory}")

    for model_name in MODEL_GROUPS[arguments.group]:
        download_model(model_name, models_directory, arguments.overwrite)

    if not arguments.skip_insightface_pack:
        download_insightface_model_pack(models_directory)

    print("Done.")


if __name__ == "__main__":
    main()
