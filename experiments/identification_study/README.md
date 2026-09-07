# Identification study

Compares two face embedding models, four matching strategies, and a sweep of decision
thresholds, to choose the configuration the packaged pipelines ship with. The study runs in
three steps: embed the known people, embed the evaluation images, then score every
combination.

All three scripts import the shared library in [face_identity/](../../face_identity/) and run
as modules from the repository root.

## Setup

```bash
pip install -e .
python -m face_identity.model_downloads --group pipelines
```

The `buffalo_l` pack (SCRFD-10GF detection + ResNet50@WebFace600K recognition) is downloaded
by InsightFace itself on first use and cached in the same model directory, so only SFace needs
fetching explicitly.

## How faces are embedded

Face detection runs **exactly once per image** — via InsightFace's official `buffalo_l`
pipeline, `FaceAnalysis.get()` (SCRFD-10GF detection + 5-point keypoints). That single
detection's box/keypoints/confidence then feed two independent recognizers, each performing
its own official alignment (never a crop pre-aligned for the other model):

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
detected face. This is implemented once, in
[study_face_embedding.py](study_face_embedding.py), and used by both extraction scripts.

## 1. Reference embeddings

Expects `--people-root` laid out as `<group>/<person>/<image files>`:

```bash
python -m experiments.identification_study.build_reference_embeddings \
  --people-root reference_people --output-dir results/embeddings
```

Writes:
- `<output-dir>/embeddings/<group>/<person>/<image>_face<N>.npz` — one file
  per kept face, holding `embedding_resnet_webface600k` (512-d),
  `embedding_sface` (128-d), `confidence`, `bbox`, and `kps`.
- `<output-dir>/embeddings_manifest.csv` — one row per kept face, with
  identity, source image, confidence, box, and the path to its `.npz`.

The manifest is authoritative: `.npz` files left on disk by an earlier build are ignored by
everything downstream, so a rebuild never has to start from an empty folder.

## 2. Evaluation embeddings

```bash
python -m experiments.identification_study.extract_evaluation_embeddings \
  --eval-root evaluation_images --output-dir results/eval_embeddings
```

Same models and same detection pass as step 1, so the two sets of embeddings are directly
comparable. Boxes are recorded as x1/y1/x2/y2, matching the human-reviewed ground-truth table.

## 3. Threshold sweep

```bash
python -m experiments.identification_study.evaluate_identification \
  --output-dir results/identification
```

Pairs every labelled face with the detected face it overlaps (IoU ≥ 0.5), then scores every
model × strategy × threshold × scope combination into
`results/identification/identification_metrics.csv` and prints the best-F1 threshold per
model and strategy. The strategies and the scoring maths come from
[face_identity/matching/](../../face_identity/matching/) — the same code the packaged pipelines
run, so the operating point the study picks is the operating point they use.

Results and their interpretation are in the [top-level README](../../README.md) and
[results/identification/RESULTS.md](../../results/identification/RESULTS.md).
