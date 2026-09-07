"""Runs each benchmarked face detector over the sample image collection and records what it found.

For every image listed in the manifest, writes one row per image per detector holding the
detected face count, the confidence scores and the boxes. accuracy_report.py, summarize.py and
threshold_sweep.py all read these CSVs, so the models never have to be re-run to re-score them.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from face_identity.configuration import (
    DEFAULT_DETECTION_INPUT_SIZE,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_SCRFD_NMS_THRESHOLD,
)
from face_identity.detection.detector_registry import BENCHMARK_DETECTOR_NAMES, build_detector
from face_identity.image_io import read_image_bgr

# Columns copied straight through from the dataset manifest onto every detection row.
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


def run_detector_over_manifest(
    detector_name: str,
    manifest: pd.DataFrame,
    images_root: Path,
    models_directory: Path | None,
    min_confidence: float,
    input_size: int,
    nms_threshold: float,
) -> pd.DataFrame:
    """Detects faces in every manifest image with one detector, returning a row per image."""
    detector = build_detector(detector_name, models_directory, min_confidence, input_size, nms_threshold)
    rows = []
    try:
        for _, manifest_row in manifest.iterrows():
            image_path = images_root / manifest_row["relative_path"]
            image_bgr = read_image_bgr(image_path)
            record = {column: manifest_row[column] for column in MANIFEST_COLUMNS}
            record["variant"] = detector_name

            if image_bgr is None:
                record.update(detected_count=None, scores="[]", boxes="[]", error="unreadable")
                rows.append(record)
                continue

            detected_faces = detector.detect(image_bgr)
            record.update(
                detected_count=len(detected_faces),
                scores=json.dumps(
                    [round(face.detection_confidence, 4) for face in detected_faces]
                ),
                boxes=json.dumps([
                    [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    for x1, y1, x2, y2 in (face.bounding_box_xyxy for face in detected_faces)
                ]),
                error="",
            )
            rows.append(record)
    finally:
        detector.close()
    return pd.DataFrame(rows)


def main() -> None:
    """Runs the requested detectors over the manifest and writes one detections CSV per detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=BENCHMARK_DETECTOR_NAMES,
                        choices=BENCHMARK_DETECTOR_NAMES)
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Model weights cache (default: FACE_IDENTITY_MODELS_DIR, else models/)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--input-size", type=int, default=DEFAULT_DETECTION_INPUT_SIZE,
                        help="Square SCRFD input resolution; ignored by BlazeFace")
    parser.add_argument("--nms-thresh", type=float, default=DEFAULT_SCRFD_NMS_THRESHOLD,
                        help="SCRFD non-maximum-suppression threshold; ignored by BlazeFace")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=None,
                        help="Only process the first N rows per group (useful for a quick smoke test)")
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(arguments.manifest, encoding="utf-8-sig")
    if arguments.limit_per_group:
        manifest = manifest.groupby("group", group_keys=False).head(arguments.limit_per_group)

    for detector_name in arguments.detectors:
        detections = run_detector_over_manifest(
            detector_name, manifest, arguments.images_root, arguments.models_dir,
            arguments.min_confidence, arguments.input_size, arguments.nms_thresh)
        output_csv = arguments.output_dir / f"detections_{detector_name}.csv"
        detections.to_csv(output_csv, index=False)
        print(f"[{detector_name}] wrote {len(detections)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
