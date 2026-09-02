"""Build a face-embedding database from a directory of known people.

For every image under --people-root/<group>/<person>/*, runs SCRFD-10G-KPS
to get face boxes + confidence + 5 keypoints, keeps detections at or above
--min-confidence, aligns each kept face to a canonical 112x112 RGB crop from
its keypoints (similarity transform: rotation + scale + translation, no box
needed), then computes two L2-normalized embeddings per face — one from
ResNet50@WebFace600K using InsightFace's official preprocessing, one from
OpenCV SFace using its own model-specific preprocessing. Each face's vectors
are written to a .npz file; one row per face is written to a summary CSV.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import pandas as pd
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_SCRFD_INPUT_SIZE = 640
DEFAULT_SCRFD_NMS_THRESH = 0.4

ALIGNED_FACE_SIZE = 112

# InsightFace ArcFace-style preprocessing: RGB, scaled to [-1, 1].
RESNET_INPUT_MEAN = 127.5
RESNET_INPUT_STD = 127.5

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
class FaceDetection:
    score: float
    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    kps: np.ndarray  # (5, 2) left_eye, right_eye, nose, left_mouth, right_mouth


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


def build_scrfd_detector(model_path: Path, min_confidence: float, input_size: int, nms_thresh: float) -> SCRFD:
    detector = SCRFD(model_file=str(model_path))
    detector.prepare(ctx_id=-1, det_thresh=min_confidence, input_size=(input_size, input_size), nms_thresh=nms_thresh)
    return detector


def detect_faces(detector: SCRFD, image_bgr: np.ndarray, min_confidence: float) -> list[FaceDetection]:
    """Runs SCRFD and keeps only detections at or above min_confidence."""
    boxes, kpss = detector.detect(image_bgr)
    detections = []
    for box, kps in zip(boxes, kpss if kpss is not None else []):
        score = float(box[4])
        if score < min_confidence:
            continue
        detections.append(FaceDetection(score=score, bbox=box[:4].astype(np.float32), kps=kps.astype(np.float32)))
    return detections


def align_face(image_rgb: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Warps the face to a canonical ALIGNED_FACE_SIZE x ALIGNED_FACE_SIZE RGB crop via a similarity transform fit to the 5 keypoints."""
    return norm_crop(image_rgb, landmark=kps, image_size=ALIGNED_FACE_SIZE)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_resnet_webface600k_session(model_path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def embed_resnet_webface600k(session: onnxruntime.InferenceSession, aligned_face_rgb: np.ndarray) -> np.ndarray:
    """Embeds an aligned RGB face with InsightFace's official ArcFace-style preprocessing (RGB, (x-127.5)/127.5, NCHW)."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    blob = cv2.dnn.blobFromImage(
        aligned_face_rgb,
        scalefactor=1.0 / RESNET_INPUT_STD,
        size=(ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE),
        mean=(RESNET_INPUT_MEAN, RESNET_INPUT_MEAN, RESNET_INPUT_MEAN),
        swapRB=False,  # already RGB
    )
    embedding = session.run([output_name], {input_name: blob})[0].flatten()
    return l2_normalize(embedding)


def build_sface_recognizer(model_path: Path) -> cv2.FaceRecognizerSF:
    return cv2.FaceRecognizerSF_create(str(model_path), "")


def embed_sface(recognizer: cv2.FaceRecognizerSF, aligned_face_rgb: np.ndarray) -> np.ndarray:
    """Embeds an aligned RGB face with OpenCV SFace's own preprocessing (BGR, raw pixel values, baked into the model graph)."""
    aligned_face_bgr = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)
    embedding = recognizer.feature(aligned_face_bgr).flatten()
    return l2_normalize(embedding)


def process_image(
    person_image: PersonImage,
    scrfd_detector: SCRFD,
    resnet_session: onnxruntime.InferenceSession,
    sface_recognizer: cv2.FaceRecognizerSF,
    min_confidence: float,
    embeddings_dir: Path,
) -> list[dict]:
    image_bgr = cv2.imread(str(person_image.path))
    if image_bgr is None:
        record = {col: "" for col in MANIFEST_COLUMNS}
        record.update(group=person_image.group, person=person_image.person,
                       relative_path=str(person_image.relative_path), error="unreadable")
        return [record]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = detect_faces(scrfd_detector, image_bgr, min_confidence)

    rows = []
    for face_index, detection in enumerate(detections):
        aligned_face_rgb = align_face(image_rgb, detection.kps)
        resnet_embedding = embed_resnet_webface600k(resnet_session, aligned_face_rgb)
        sface_embedding = embed_sface(sface_recognizer, aligned_face_rgb)

        out_dir = embeddings_dir / person_image.group / person_image.person
        out_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = out_dir / f"{person_image.path.stem}_face{face_index}.npz"
        np.savez(
            embedding_path,
            embedding_resnet_webface600k=resnet_embedding,
            embedding_sface=sface_embedding,
            confidence=detection.score,
            bbox=detection.bbox,
            kps=detection.kps,
        )

        x1, y1, x2, y2 = detection.bbox
        rows.append({
            "group": person_image.group,
            "person": person_image.person,
            "relative_path": str(person_image.relative_path),
            "face_index": face_index,
            "confidence": round(detection.score, 4),
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
    parser.add_argument("--scrfd-model", type=Path, default=Path("models/scrfd_10g_kps.onnx"))
    parser.add_argument("--resnet-webface600k-model", type=Path, default=Path("models/resnet50_webface600k.onnx"))
    parser.add_argument("--sface-model", type=Path, default=Path("models/sface_2021dec.onnx"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--scrfd-input-size", type=int, default=DEFAULT_SCRFD_INPUT_SIZE)
    parser.add_argument("--scrfd-nms-thresh", type=float, default=DEFAULT_SCRFD_NMS_THRESH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    scrfd_detector = build_scrfd_detector(
        args.scrfd_model, args.min_confidence, args.scrfd_input_size, args.scrfd_nms_thresh
    )
    resnet_session = build_resnet_webface600k_session(args.resnet_webface600k_model)
    sface_recognizer = build_sface_recognizer(args.sface_model)

    include_groups = set(args.include_groups) if args.include_groups else None
    person_images = discover_person_images(args.people_root, include_groups)
    print(f"Found {len(person_images)} images under {args.people_root}")

    all_rows = []
    for person_image in person_images:
        all_rows.extend(process_image(
            person_image, scrfd_detector, resnet_session, sface_recognizer, args.min_confidence, embeddings_dir
        ))

    manifest = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    manifest_path = args.output_dir / "embeddings_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    face_count = (manifest["error"] == "").sum()
    error_count = (manifest["error"] != "").sum()
    print(f"Wrote {face_count} face embeddings ({error_count} image errors) -> {manifest_path}")


if __name__ == "__main__":
    main()
