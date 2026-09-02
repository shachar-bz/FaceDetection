"""Download the two face-recognition ONNX models used by face_embedding/build_face_database.py.

ResNet50@WebFace600K (w600k_r50.onnx) ships inside InsightFace's `buffalo_l`
model pack, alongside the SCRFD detector already downloaded by
download_scrfd_model.py — this extracts just the recognition model.
OpenCV SFace is fetched directly from the OpenCV Zoo.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

BUFFALO_L_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
RESNET_WEBFACE600K_NAME_IN_PACK = "w600k_r50.onnx"
RESNET_WEBFACE600K_DEST_FILENAME = "resnet50_webface600k.onnx"

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_DEST_FILENAME = "sface_2021dec.onnx"


def download_resnet_webface600k(models_dir: Path) -> None:
    dest = models_dir / RESNET_WEBFACE600K_DEST_FILENAME
    print(f"Downloading {BUFFALO_L_PACK_URL}")
    req = urllib.request.Request(BUFFALO_L_PACK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        pack_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(pack_bytes)) as zf:
        with zf.open(RESNET_WEBFACE600K_NAME_IN_PACK) as src, open(dest, "wb") as out:
            out.write(src.read())
    print(f"Extracted {RESNET_WEBFACE600K_NAME_IN_PACK} -> {dest}")


def download_sface(models_dir: Path) -> None:
    dest = models_dir / SFACE_DEST_FILENAME
    print(f"Downloading {SFACE_URL} -> {dest}")
    req = urllib.request.Request(SFACE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        out.write(response.read())


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    download_resnet_webface600k(models_dir)
    download_sface(models_dir)


if __name__ == "__main__":
    main()
