"""Build a face-embedding database from a directory of known people.

For every image under --people-root/<group>/<person>/*, runs face detection
exactly once — via InsightFace's official `buffalo_l` pipeline, `FaceAnalysis.
get()` — then feeds each detection's box/keypoints to two independent
recognizers, each performing its own official alignment from that shared
detection (never a crop pre-aligned for the other model):
  - **ResNet50@WebFace600K**: alignment + recognition happen inside
    `FaceAnalysis.get()` itself (InsightFace's own tested preprocessing).
  - **OpenCV SFace**: the same detection's box/keypoints/confidence are
    adapted into the 15-value row `cv2.FaceRecognizerSF.alignCrop()` expects,
    then that recognizer's own `alignCrop()` -> `feature()`.
Detecting once means there's no risk of pairing embeddings from different
faces — both embeddings always come from the same detected face. This module
never computes either alignment transform itself. Each face's two
L2-normalized embeddings are written to a .npz file; one row per face is
written to a summary CSV.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_DET_SIZE = 640
# buffalo_l bundles SCRFD-10GF detection + 5-point alignment + ResNet50@WebFace600K
# recognition; FaceAnalysis downloads/caches it under --insightface-root on first use.
DEFAULT_MODEL_PACK_NAME = "buffalo_l"
DEFAULT_INSIGHTFACE_ROOT = Path("~/.insightface")

MANIFEST_COLUMNS = [
    "group",
    "person",
    "relative_path",
    "face_index",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "embedding_path",
    "error",
]


@dataclass
class PersonImage:
    group: str
    person: str
    path: Path
    relative_path: Path


def discover_person_images(people_root: Path, include_groups: set[str] | None = None) -> list[PersonImage]:
    """Finds every image under people_root/<group>/<person>/*, skipping empty dirs.

    If include_groups is given, only those top-level group folder names are scanned.
    """
    images = []
    for group_dir in sorted(p for p in people_root.iterdir() if p.is_dir()):
        if include_groups is not None and group_dir.name not in include_groups:
            continue
        for person_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(
                        PersonImage(
                            group=group_dir.name,
                            person=person_dir.name,
                            path=image_path,
                            relative_path=image_path.relative_to(people_root),
                        )
                    )
    return images


def build_face_analysis_app(model_pack_name: str, insightface_root: Path, min_confidence: float, det_size: int) -> FaceAnalysis:
    """Loads InsightFace's official model pack: SCRFD detection + 5-point alignment + ResNet50@WebFace600K
    recognition all run inside `FaceAnalysis.get()`, so no alignment transform or ResNet preprocessing
    is computed by this module. Downloads/caches the pack under insightface_root on first use."""
    app = FaceAnalysis(
        name=model_pack_name,
        root=str(insightface_root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=min_confidence)
    return app


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_sface_recognizer(model_path: Path) -> cv2.FaceRecognizerSF:
    return cv2.FaceRecognizerSF_create(str(model_path), "")


def build_sface_face_box(bbox: np.ndarray, kps: np.ndarray, confidence: float) -> np.ndarray:
    """Adapts a detection into the 15-value [x, y, w, h, 5x(landmark_x, landmark_y), confidence]
    row `FaceRecognizerSF.alignCrop()` expects. No landmark reordering needed: OpenCV's alignCrop
    (face_recognize.cpp's getSimilarityTransformMatrix) warps to the exact same ArcFace reference
    points InsightFace's own alignment uses, so SCRFD's native 5-point order (left_eye, right_eye,
    nose, left_mouth, right_mouth) is already in the order it wants."""
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2 - x1, y2 - y1, *kps.reshape(-1), confidence], dtype=np.float32)


def embed_sface(recognizer: cv2.FaceRecognizerSF, image_bgr: np.ndarray, face_box: np.ndarray) -> np.ndarray:
    """Runs SFace's own official alignCrop() -> feature() pipeline on the full (unaligned)
    image, rather than reusing a crop aligned for another model."""
    aligned_bgr = recognizer.alignCrop(image_bgr, face_box)
    embedding = recognizer.feature(aligned_bgr).flatten()
    return l2_normalize(embedding)


def process_image(
    person_image: PersonImage,
    face_app: FaceAnalysis,
    sface_recognizer: cv2.FaceRecognizerSF,
    embeddings_dir: Path,
) -> list[dict]:
    image_bgr = cv2.imread(str(person_image.path))
    if image_bgr is None:
        record = {col: "" for col in MANIFEST_COLUMNS}
        record.update(group=person_image.group, person=person_image.person,
                       relative_path=str(person_image.relative_path), error="unreadable")
        return [record]

    # Single detection pass (buffalo_l's SCRFD, run inside FaceAnalysis.get()) feeds both
    # recognizers, so both embeddings for a face always come from that same detected face.
    faces = face_app.get(image_bgr)

    rows = []
    for face_index, face in enumerate(faces):
        resnet_embedding = face.normed_embedding
        face_box = build_sface_face_box(face.bbox, face.kps, face.det_score)
        sface_embedding = embed_sface(sface_recognizer, image_bgr, face_box)

        confidence = float(face.det_score)
        bbox = face.bbox.astype(np.float32)

        out_dir = embeddings_dir / person_image.group / person_image.person
        out_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = out_dir / f"{person_image.path.stem}_face{face_index}.npz"
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
            "group": person_image.group,
            "person": person_image.person,
            "relative_path": str(person_image.relative_path),
            "face_index": face_index,
            "confidence": round(confidence, 4),
            "bbox_x": round(float(x1), 1),
            "bbox_y": round(float(y1), 1),
            "bbox_w": round(float(x2 - x1), 1),
            "bbox_h": round(float(y2 - y1), 1),
            "embedding_path": str(embedding_path),
            "error": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-root", type=Path, required=True,
                         help="Directory laid out as <group>/<person>/<image files>")
    parser.add_argument("--include-groups", nargs="+", default=None,
                         help="Only scan these top-level group folder names (default: all)")
    parser.add_argument("--model-pack", default=DEFAULT_MODEL_PACK_NAME,
                         help="InsightFace model pack name (bundles SCRFD detection + ResNet50@WebFace600K recognition)")
    parser.add_argument("--insightface-root", type=Path, default=DEFAULT_INSIGHTFACE_ROOT,
                         help="Where FaceAnalysis caches/downloads --model-pack")
    parser.add_argument("--sface-model", type=Path, default=Path("models/sface_2021dec.onnx"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    face_app = build_face_analysis_app(args.model_pack, args.insightface_root, args.min_confidence, args.det_size)
    sface_recognizer = build_sface_recognizer(args.sface_model)

    include_groups = set(args.include_groups) if args.include_groups else None
    person_images = discover_person_images(args.people_root, include_groups)
    print(f"Found {len(person_images)} images under {args.people_root}")

    all_rows = []
    for person_image in person_images:
        all_rows.extend(process_image(person_image, face_app, sface_recognizer, embeddings_dir))

    manifest = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    manifest_path = args.output_dir / "embeddings_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    face_count = (manifest["error"] == "").sum()
    error_count = (manifest["error"] != "").sum()
    print(f"Wrote {face_count} face embeddings ({error_count} image errors) -> {manifest_path}")


if __name__ == "__main__":
    main()
