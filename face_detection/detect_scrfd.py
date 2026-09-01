"""Run the SCRFD-10G-KPS face detector over the sample image collection.

Same manifest/CSV contract as detect.py, so summarize.py, threshold_sweep.py,
and accuracy_report.py can be reused against its output by passing
`--variants scrfd_10g_kps`.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.model_zoo.scrfd import SCRFD

VARIANT_NAME = "scrfd_10g_kps"

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


def build_detector(model_path: Path, min_confidence: float, input_size: int, nms_thresh: float) -> SCRFD:
    detector = SCRFD(model_file=str(model_path))
    detector.prepare(ctx_id=-1, det_thresh=min_confidence, input_size=(input_size, input_size), nms_thresh=nms_thresh)
    return detector


def detect_faces(detector: SCRFD, image_bgr: np.ndarray) -> list[Detection]:
    boxes, _ = detector.detect(image_bgr)
    detections = []
    for x1, y1, x2, y2, score in boxes:
        detections.append(Detection(score=float(score), x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1)))
    return detections


def run_variant(
    model_path: Path,
    manifest: pd.DataFrame,
    images_root: Path,
    min_confidence: float,
    input_size: int,
    nms_thresh: float,
) -> pd.DataFrame:
    detector = build_detector(model_path, min_confidence, input_size, nms_thresh)
    rows = []
    for _, row in manifest.iterrows():
        image_path = images_root / row["relative_path"]
        image_bgr = cv2.imread(str(image_path))
        record = {col: row[col] for col in MANIFEST_COLUMNS}
        record["variant"] = VARIANT_NAME
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
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=640, help="Square SCRFD input resolution")
    parser.add_argument("--nms-thresh", type=float, default=0.4)
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

    df = run_variant(args.model, manifest, args.images_root, args.min_confidence, args.input_size, args.nms_thresh)
    out_csv = args.output_dir / f"detections_{VARIANT_NAME}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{VARIANT_NAME}] wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
