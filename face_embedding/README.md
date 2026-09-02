# Face embedding database

Builds a face-embedding database from a directory of known people, for
downstream face matching/tracking.

## Pipeline (`face_embedding/build_face_database.py`)

1. Run SCRFD-10G-KPS on each image -> per face: bounding box, confidence,
   5 keypoints (left eye, right eye, nose, left mouth corner, right mouth
   corner).
2. Keep only detections at or above `--min-confidence`.
3. Align: fit a similarity transform (rotation + scale + translation) from
   the 5 keypoints to InsightFace's standard reference points, and warp to
   a 112x112 RGB crop. The bounding box itself is discarded here — only the
   keypoints matter for alignment.
4. From the same aligned crop, compute two independent, L2-normalized
   embeddings:
   - **ResNet50@WebFace600K**, with InsightFace's official ArcFace-style
     preprocessing (RGB, `(x - 127.5) / 127.5`).
   - **OpenCV SFace**, with its own model-specific preprocessing (BGR, raw
     pixel values — SFace's normalization is baked into the ONNX graph).

## Setup

```
pip install -r requirements.txt
python downloading_models_scripts/download_scrfd_model.py         # -> models/scrfd_10g_kps.onnx
python downloading_models_scripts/download_recognition_models.py  # -> models/resnet50_webface600k.onnx, models/sface_2021dec.onnx
```

## Usage

Expects `--people-root` laid out as `<group>/<person>/<image files>`:

```
python face_embedding/build_face_database.py --people-root path/to/people --output-dir results_embeddings
```

Writes:
- `<output-dir>/embeddings/<group>/<person>/<image>_face<N>.npz` — one file
  per kept face, holding `embedding_resnet_webface600k` (512-d),
  `embedding_sface` (128-d), `confidence`, `bbox`, and `kps`.
- `<output-dir>/embeddings_manifest.csv` — one row per kept face, with
  identity, source image, confidence, box, and the path to its `.npz`.
