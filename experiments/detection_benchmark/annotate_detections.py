"""Saves a copy of every dataset image with one detector's boxes drawn on it, for eyeballing.

Each detection is drawn as the square crop box it would produce, labelled with its confidence,
and written to <output-dir>/<group>/<original filename> so results can be reviewed group by group.
"""
import argparse
from pathlib import Path

import pandas as pd

from face_identity.annotation import draw_detection_boxes, save_annotated_image
from face_identity.configuration import (
    DEFAULT_CROP_MARGIN_RATIO,
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_SCRFD_NMS_THRESHOLD,
)
from face_identity.detection.detector_registry import BENCHMARK_DETECTOR_NAMES, build_detector
from face_identity.image_io import read_image_bgr


def main() -> None:
    """Annotates every manifest image with one detector's boxes and writes them out by group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector", required=True, choices=BENCHMARK_DETECTOR_NAMES)
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--input-size", type=int, default=DEFAULT_DETECTION_INPUT_SIZE,
                        help="Square SCRFD input resolution; ignored by BlazeFace")
    parser.add_argument("--nms-thresh", type=float, default=DEFAULT_SCRFD_NMS_THRESHOLD,
                        help="SCRFD non-maximum-suppression threshold; ignored by BlazeFace")
    parser.add_argument("--margin", type=float, default=DEFAULT_CROP_MARGIN_RATIO,
                        help="Fraction of the longer box side to expand the box by, on each side")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = pd.read_csv(arguments.manifest, encoding="utf-8-sig")
    detector = build_detector(
        arguments.detector, arguments.models_dir, arguments.min_confidence,
        arguments.input_size, arguments.nms_thresh)

    written_count = 0
    skipped_count = 0
    try:
        for _, manifest_row in manifest.iterrows():
            relative_path = Path(manifest_row["relative_path"])
            image_bgr = read_image_bgr(arguments.images_root / relative_path)
            if image_bgr is None:
                print(f"skip (unreadable): {relative_path}")
                skipped_count += 1
                continue

            draw_detection_boxes(image_bgr, detector.detect(image_bgr), arguments.margin)
            save_annotated_image(image_bgr, arguments.output_dir / manifest_row["group"] / relative_path.name)
            written_count += 1
    finally:
        detector.close()

    print(f"\nwrote {written_count} annotated images, skipped {skipped_count} unreadable "
          f"-> {arguments.output_dir}")


if __name__ == "__main__":
    main()
