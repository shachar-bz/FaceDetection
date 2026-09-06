"""Extracts face embeddings for the evaluation images under real_data/.

Runs the exact same detection + embedding pipeline that built the known-people
database (face_embedding/build_face_database.py): one buffalo_l detection pass
per image feeding both ResNet50@WebFace600K and OpenCV SFace, so the evaluation
embeddings are directly comparable to the reference database. Writes one .npz
per detected face plus a manifest CSV indexing them.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_embedding.build_face_database import (  # noqa: E402
    DEFAULT_DET_SIZE,
    DEFAULT_INSIGHTFACE_ROOT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MODEL_PACK_NAME,
    DEFAULT_SFACE_MODEL_PATH,
    IMAGE_EXTENSIONS,
    build_face_analysis_app,
    build_sface_face_box,
    build_sface_recognizer,
    embed_sface,
)

DEFAULT_EVAL_ROOT = Path("real_data")
DEFAULT_EVAL_GROUPS = ["one_person", "few_people"]
DEFAULT_OUTPUT_DIR = Path("results_eval_embeddings")

MANIFEST_COLUMNS = [
    "image_id",
    "group",
    "relative_path",
    "face_index",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "embedding_path",
    "error",
]


def discover_eval_images(eval_root: Path, groups: list[str]) -> list[tuple[str, str, Path]]:
    """Finds every evaluation image under eval_root/<group>/, returning (image_id, group, path).

    The image_id is the file stem, which is the key the ground-truth table joins on.
    """
    images = []
    for group in groups:
        group_dir = eval_root / group
        for image_path in sorted(group_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path.stem, group, image_path))
    return images


def extract_image_faces(
    image_id: str,
    group: str,
    image_path: Path,
    eval_root: Path,
    face_app,
    sface_recognizer: cv2.FaceRecognizerSF,
    embeddings_dir: Path,
) -> list[dict]:
    """Detects every face in one evaluation image and writes both models' embeddings per face.

    Boxes are recorded as x1/y1/x2/y2 to match the ground-truth table's bbox_xyxy format.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        record = {column: "" for column in MANIFEST_COLUMNS}
        record.update(image_id=image_id, group=group,
                      relative_path=str(image_path.relative_to(eval_root)), error="unreadable")
        return [record]

    faces = face_app.get(image_bgr)

    rows = []
    for face_index, face in enumerate(faces):
        resnet_embedding = face.normed_embedding
        face_box = build_sface_face_box(face.bbox, face.kps, face.det_score)
        sface_embedding = embed_sface(sface_recognizer, image_bgr, face_box)

        confidence = float(face.det_score)
        bbox = face.bbox.astype(np.float32)

        out_dir = embeddings_dir / group
        out_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = out_dir / f"{image_id}_face{face_index}.npz"
        np.savez(
            embedding_path,
            embedding_resnet_webface600k=resnet_embedding,
            embedding_sface=sface_embedding,
            confidence=confidence,
            bbox=bbox,
            kps=face.kps,
        )

        x1, y1, x2, y2 = bbox
        rows.append({
            "image_id": image_id,
            "group": group,
            "relative_path": str(image_path.relative_to(eval_root)),
            "face_index": face_index,
            "confidence": round(confidence, 4),
            "bbox_x1": round(float(x1), 1),
            "bbox_y1": round(float(y1), 1),
            "bbox_x2": round(float(x2), 1),
            "bbox_y2": round(float(y2), 1),
            "embedding_path": str(embedding_path),
            "error": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
                        help="Directory holding the evaluation image group folders")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_EVAL_GROUPS,
                        help="Group folder names to scan under --eval-root")
    parser.add_argument("--model-pack", default=DEFAULT_MODEL_PACK_NAME)
    parser.add_argument("--insightface-root", type=Path, default=DEFAULT_INSIGHTFACE_ROOT)
    parser.add_argument("--sface-model", type=Path, default=DEFAULT_SFACE_MODEL_PATH)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    face_app = build_face_analysis_app(args.model_pack, args.insightface_root,
                                       args.min_confidence, args.det_size)
    sface_recognizer = build_sface_recognizer(args.sface_model)

    eval_images = discover_eval_images(args.eval_root, args.groups)
    print(f"Found {len(eval_images)} evaluation images under {args.eval_root}")

    all_rows = []
    for image_id, group, image_path in eval_images:
        all_rows.extend(extract_image_faces(image_id, group, image_path, args.eval_root,
                                            face_app, sface_recognizer, embeddings_dir))

    manifest = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    manifest_path = args.output_dir / "eval_embeddings_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    face_count = (manifest["error"] == "").sum()
    error_count = (manifest["error"] != "").sum()
    print(f"Wrote {face_count} face embeddings ({error_count} image errors) -> {manifest_path}")


if __name__ == "__main__":
    main()
