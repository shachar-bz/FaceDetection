"""Run the face detector over every image in the manifest and save annotated copies.

For each image, draws every detection's bounding box — expanded by
--margin per side (square crop centered on the raw box, clamped to image
bounds) — plus its confidence
score, and writes the result to <output-dir>/<group>/<original filename>
— one folder per group (no_person, one_person, multiple_people), so
results can be eyeballed category by category.
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from detection_common import (
    DEFAULT_CROP_MARGIN_RATIO,
    DEFAULT_MIN_CONFIDENCE,
    build_blazeface_detector,
    expand_box_square,
)

MANIFEST_COLUMNS_NEEDED = ["group", "relative_path"]


def annotate(image_bgr, detections, margin_ratio: float) -> None:
    img_h, img_w = image_bgr.shape[:2]
    thickness = max(2, img_w // 300)
    for d in detections:
        bbox = d.bounding_box
        score = d.categories[0].score if d.categories else 0.0
        x0, y0, x1, y1 = expand_box_square(
            bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, margin_ratio, img_w, img_h
        )
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 255, 0), thickness)
        label = f"{score:.2f}"
        cv2.putText(
            image_bgr, label, (x0, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--margin", type=float, default=DEFAULT_CROP_MARGIN_RATIO,
                         help="Fraction of max(box_w, box_h) to expand the box by, on each side")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    detector = build_blazeface_detector(args.model, args.min_confidence)

    written = 0
    skipped = 0
    try:
        for _, row in manifest.iterrows():
            group = row["group"]
            image_path = args.images_root / row["relative_path"]
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"skip (unreadable): {row['relative_path']}")
                skipped += 1
                continue

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            annotate(image_bgr, result.detections, args.margin)

            group_dir = args.output_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)
            out_path = group_dir / Path(row["relative_path"]).name
            cv2.imwrite(str(out_path), image_bgr)
            written += 1
    finally:
        detector.close()

    print(f"\nwrote {written} annotated images, skipped {skipped} unreadable -> {args.output_dir}")


if __name__ == "__main__":
    main()
