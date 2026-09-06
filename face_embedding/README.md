# Face embedding database

Builds a face-embedding database from a directory of known people, for
downstream face matching/tracking.

## Pipeline (`face_embedding/build_face_database.py`)

Face detection runs **exactly once per image** — via InsightFace's official
`buffalo_l` pipeline, `FaceAnalysis.get()` (SCRFD-10GF detection + 5-point
keypoints). That single detection's box/keypoints/confidence then feed two
independent recognizers, each performing its own official alignment (never a
crop pre-aligned for the other model):

- **ResNet50@WebFace600K.** Alignment (rotation + scale + translation to
  InsightFace's standard reference points, 112x112 crop) and recognition
  happen inside `FaceAnalysis.get()` itself, using its own tested
  preprocessing. This code never computes that transform or the ResNet blob.
  Detections below `--min-confidence` are dropped internally (`det_thresh`).
- **OpenCV SFace.** The same detection's box/keypoints/confidence are
  adapted into the 15-value `[x, y, w, h, 5x(landmark_x, landmark_y),
  confidence]` row `cv2.FaceRecognizerSF.alignCrop()` expects, then that
  recognizer's own `alignCrop()` -> `feature()`. No landmark reordering is
  needed: OpenCV's `alignCrop` warps to the exact same ArcFace reference
  points InsightFace uses (verified against OpenCV's own
  `face_recognize.cpp` source), so SCRFD's native 5-point order (left eye,
  right eye, nose, left mouth corner, right mouth corner) is already what it
  wants.

Detecting once (rather than running SCRFD twice and pairing results by
index or position) means there's no risk of ever pairing embeddings from two
different faces — both embeddings for a row always come from the same
detected face.

## Setup

```
pip install -r requirements.txt
python downloading_models_scripts/download_recognition_models.py  # -> models/sface_2021dec.onnx
```

The `buffalo_l` pack (SCRFD-10GF detection + ResNet50@WebFace600K
recognition) is downloaded automatically by `FaceAnalysis` on first run and
cached under `--insightface-root` (default `~/.insightface`) — no separate
download step needed for it.

## Usage

Expects `--people-root` laid out as `<group>/<person>/<image files>`:

```
python face_embedding/build_face_database.py --people-root path/to/people --output-dir results/embeddings
```

Writes:
- `<output-dir>/embeddings/<group>/<person>/<image>_face<N>.npz` — one file
  per kept face, holding `embedding_resnet_webface600k` (512-d),
  `embedding_sface` (128-d), `confidence`, `bbox`, and `kps`.
- `<output-dir>/embeddings_manifest.csv` — one row per kept face, with
  identity, source image, confidence, box, and the path to its `.npz`.
