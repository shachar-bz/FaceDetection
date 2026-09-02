"""Download the SCRFD-10G-KPS ONNX model used by face_detection/detect_scrfd.py.

SCRFD-10G-KPS (det_10g.onnx) ships inside InsightFace's `buffalo_l` model
pack. This downloads that pack and extracts just the detector, since the
pack's other models (recognition, landmarks, age/gender) aren't needed here.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

MODEL_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_IN_PACK = "det_10g.onnx"
DEST_FILENAME = "scrfd_10g_kps.onnx"


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / DEST_FILENAME

    print(f"Downloading {MODEL_PACK_URL}")
    req = urllib.request.Request(MODEL_PACK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        pack_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(pack_bytes)) as zf:
        with zf.open(MODEL_IN_PACK) as src, open(dest, "wb") as out:
            out.write(src.read())

    print(f"Extracted {MODEL_IN_PACK} -> {dest}")


if __name__ == "__main__":
    main()
