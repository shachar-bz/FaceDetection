"""Download the OpenCV SFace recognition model used by face_embedding/build_face_database.py.

ResNet50@WebFace600K is no longer downloaded here: it ships inside InsightFace's
`buffalo_l` model pack, which `insightface.app.FaceAnalysis` downloads and caches
under `--insightface-root` automatically on first use. SFace has no such
auto-download path, so it's still fetched directly from the OpenCV Zoo.
"""
import urllib.request
from pathlib import Path

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_DEST_FILENAME = "sface_2021dec.onnx"


def download_sface(models_dir: Path) -> None:
    dest = models_dir / SFACE_DEST_FILENAME
    print(f"Downloading {SFACE_URL} -> {dest}")
    req = urllib.request.Request(SFACE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        out.write(response.read())


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    download_sface(models_dir)


if __name__ == "__main__":
    main()
