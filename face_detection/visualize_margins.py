"""Build contact sheets showing a detected face crop at several margin ratios.

For each requested image, runs the face detector once, then for every
detection draws: the source image with the raw bounding box, followed by
square crops expanded by each margin ratio (a fraction of the box size
added on every side, clamped to the image bounds). Used to eyeball which
margin keeps the whole face (forehead/chin/ears) without pulling in too
much background, ahead of feeding crops to an embedding model.
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

THUMB_SIZE = 220
LABEL_HEIGHT = 24


def build_detector(model_path: Path, min_confidence: float) -> mp_vision.FaceDetector:
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
    return mp_vision.FaceDetector.create_from_options(options)


def expand_box_square(x: int, y: int, w: int, h: int, margin_ratio: float, img_w: int, img_h: int):
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin_ratio)
    half = side / 2
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    return int(x0), int(y0), int(x1), int(y1)


def make_thumb(crop: np.ndarray, label: str) -> np.ndarray:
    if crop.size == 0:
        crop = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
    canvas = np.full((THUMB_SIZE + LABEL_HEIGHT, THUMB_SIZE, 3), 255, dtype=np.uint8)
    canvas[:THUMB_SIZE] = crop
    cv2.putText(canvas, label, (4, THUMB_SIZE + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--relative-paths", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args.model, args.min_confidence)

    for rel_path in args.relative_paths:
        image_path = args.images_root / rel_path
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"skip (unreadable): {rel_path}")
            continue
        img_h, img_w = image_bgr.shape[:2]

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        annotated = image_bgr.copy()
        for d in result.detections:
            bbox = d.bounding_box
            cv2.rectangle(
                annotated,
                (bbox.origin_x, bbox.origin_y),
                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                (0, 255, 0),
                max(2, img_w // 300),
            )
        scale = THUMB_SIZE / max(img_h, img_w)
        annotated_thumb = cv2.resize(annotated, (int(img_w * scale), int(img_h * scale)))
        header_canvas = np.full((THUMB_SIZE + LABEL_HEIGHT, THUMB_SIZE, 3), 255, dtype=np.uint8)
        y_off = (THUMB_SIZE - annotated_thumb.shape[0]) // 2
        x_off = (THUMB_SIZE - annotated_thumb.shape[1]) // 2
        header_canvas[y_off:y_off + annotated_thumb.shape[0], x_off:x_off + annotated_thumb.shape[1]] = annotated_thumb
        cv2.putText(header_canvas, "source+bbox", (4, THUMB_SIZE + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if not result.detections:
            print(f"no detections: {rel_path}")
            continue

        for face_idx, d in enumerate(result.detections):
            bbox = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            row_thumbs = [header_canvas]
            for margin in args.margins:
                x0, y0, x1, y1 = expand_box_square(
                    bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, margin, img_w, img_h
                )
                crop = image_bgr[y0:y1, x0:x1]
                row_thumbs.append(make_thumb(crop, f"margin {margin:.0%}"))
            row = np.hstack(row_thumbs)
            stem = Path(rel_path).stem
            out_path = args.output_dir / f"{stem}_face{face_idx}_score{score:.2f}.jpg"
            cv2.imwrite(str(out_path), row)
            print(f"wrote {out_path}")

    detector.close()


if __name__ == "__main__":
    main()
