"""Run the MediaPipe Face Detector over the sample image collection.

For every image listed in manifest.csv, runs both the short-range and
full-range BlazeFace variants and records the raw detections (score +
bounding box) to a CSV per variant. summarize.py turns these into
accuracy stats.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MANIFEST_COLUMNS = [
    "image_id",
    "group",
    "relative_path",
    "verified_person_count",
    "orientation",
    "brightness_bucket",
    "framing_proxy",
    "width",
    "height",
]


@dataclass
class Detection:
    score: float
    x: int
    y: int
    w: int
    h: int


def build_detector(model_path: Path, min_confidence: float) -> mp_vision.FaceDetector:
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detect_faces(detector: mp_vision.FaceDetector, image_bgr: np.ndarray) -> list[Detection]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    detections = []
    for d in result.detections:
        bbox = d.bounding_box
        score = d.categories[0].score if d.categories else 0.0
        detections.append(Detection(score=score, x=bbox.origin_x, y=bbox.origin_y, w=bbox.width, h=bbox.height))
    return detections


def run_variant(
    variant_name: str,
    model_path: Path,
    manifest: pd.DataFrame,
    images_root: Path,
    min_confidence: float,
) -> pd.DataFrame:
    detector = build_detector(model_path, min_confidence)
    rows = []
    try:
        for _, row in manifest.iterrows():
            image_path = images_root / row["relative_path"]
            image_bgr = cv2.imread(str(image_path))
            record = {col: row[col] for col in MANIFEST_COLUMNS}
            record["variant"] = variant_name
            if image_bgr is None:
                record.update(detected_count=None, scores="[]", boxes="[]", error="unreadable")
                rows.append(record)
                continue
            detections = detect_faces(detector, image_bgr)
            record.update(
                detected_count=len(detections),
                scores=json.dumps([round(d.score, 4) for d in detections]),
                boxes=json.dumps([[d.x, d.y, d.w, d.h] for d in detections]),
                error="",
            )
            rows.append(record)
    finally:
        detector.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Only process the first N rows per group (useful for a quick smoke test)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    if args.limit_per_group:
        manifest = manifest.groupby("group", group_keys=False).head(args.limit_per_group)

    model_variants = {
        "short_range": args.models_dir / "blaze_face_short_range.tflite",
        "full_range": args.models_dir / "blaze_face_full_range.tflite",
    }

    for variant_name, model_path in model_variants.items():
        df = run_variant(variant_name, model_path, manifest, args.images_root, args.min_confidence)
        out_csv = args.output_dir / f"detections_{variant_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[{variant_name}] wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
