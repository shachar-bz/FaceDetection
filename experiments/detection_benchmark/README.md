# Face detector benchmark

Evaluates face detectors — MediaPipe Face Detector (BlazeFace `blazeface_short_range`
and `blazeface_full_range`) and `scrfd_10g_kps` — against the 411-image sample
in `detecting_faces_data/`, to pick a model, confidence
threshold, and crop margin for the face-embedding pipeline.

## Setup

```bash
pip install -e ".[benchmark]"
python -m face_identity.model_downloads --group benchmark
```

## Scripts

Run every script as a module from the repository root, so `face_identity` resolves:

- `run_detection_benchmark.py` — runs any set of detectors over every image in
  the manifest, writing per-image detection counts/scores/boxes to
  `<output-dir>/detections_<detector>.csv`. One script for every detector;
  pick with `--detectors`.
- `accuracy_report.py` — scores a detections CSV against per-group ground truth
  (TP/TN/FP/FN, precision/recall/accuracy). Accepts `--detectors` to target any
  set of detector names.
- `summarize_detections.py` — turns a detections CSV into recall /
  false-positive stats, broken down by brightness, framing, and orientation.
- `threshold_sweep.py` — re-thresholds a low-confidence
  (`--min-confidence 0.1`) detection run to show the recall vs.
  false-positive trade-off across confidence cutoffs without re-running
  the model.
- `annotate_detections.py` — saves a copy of every image with one detector's
  boxes drawn on it, for eyeballing.

The detectors themselves live in [face_identity/detection/](../../face_identity/detection/);
this folder only runs and scores them.

## Reproducing the results

BlazeFace results go to `results/blazeface/`, SCRFD to `results/scrfd/`:

```bash
python -m experiments.detection_benchmark.run_detection_benchmark \
  --images-root detecting_faces_data --manifest detecting_faces_data/manifest.csv \
  --detectors blazeface_short_range blazeface_full_range --output-dir results/blazeface
python -m experiments.detection_benchmark.accuracy_report --results-dir results/blazeface \
  --detectors blazeface_short_range blazeface_full_range

python -m experiments.detection_benchmark.run_detection_benchmark \
  --images-root detecting_faces_data --manifest detecting_faces_data/manifest.csv \
  --detectors scrfd_10g_kps --output-dir results/scrfd
python -m experiments.detection_benchmark.accuracy_report --results-dir results/scrfd \
  --detectors scrfd_10g_kps
```

At the default 0.5 confidence threshold, SCRFD-10G-KPS clearly outperforms
both BlazeFace variants on this sample: 96.8% accuracy / 100% precision /
95.0% recall (n=411), vs. `blazeface_full_range`'s 81.0% accuracy / 99.5% precision /
70.5% recall. It also had zero false positives on `no_person` images
(`blazeface_full_range` had 1).

## Findings (411-image sample: 150 no_person / 145 one_person / 116 multiple_people)

- **`full_range` beats `short_range` at every confidence threshold** tested
  (0.1-0.7) on both recall and false-positive rate — expected, since this
  dataset has group photos and non-selfie framing that `short_range`
  (tuned for faces within ~2m) isn't built for. Use `full_range`.
- **Confidence threshold is a precision/recall dial, not a bug fix:**
  at the default 0.5, `full_range` gets ~32% mean per-image recall with a
  5.4% false-positive rate on no-person images; dropping to 0.3 roughly
  doubles recall (~47%) but false positives jump to ~25%. Since a wrong
  detection becomes a garbage embedding in the face DB (worse than a
  missed face), **keep the 0.5 default** unless the product can tolerate
  reviewing/filtering more junk crops downstream.
- **A meaningful share of "misses" aren't real misses.** Manual review of
  false-positive-heavy low-confidence samples turned up MIAP "person"
  boxes drawn on garden statues, a light fixture, and a helmet — the
  ground-truth label counts a "person" box, not a visible face, so raw
  recall against `verified_person_count` understates true accuracy on
  photos where the face is actually visible. Likewise, several
  one_person "misses" were legitimately hard: crowd shots where everyone's
  back is to the camera, or a face mostly covered by a costume mask.
- **Crop margin:** across every real (non-false-positive) detection
  inspected, the raw `full_range` bounding box is tight enough that it
  regularly clips the chin or forehead. Expanding the box by **~20-30%**
  on each side (square crop centered on the box, margin as a fraction of
  `max(box_w, box_h)`) reliably captures the full head with a little hair/
  neck margin. Beyond ~40% the crop increasingly wastes area on
  background, which would dilute a downstream embedding model's input.
  **Superseded for the embedding pipeline:** once alignment moved into the
  packaged pipelines (InsightFace `FaceAnalysis` and OpenCV SFace's
  `alignCrop()`), the crop margin stopped being ours to pick — each
  recognizer warps the face to its own reference points straight from the
  detected keypoints, with no intermediate square crop. The finding still
  describes how tight the raw boxes are, and `--margin` still controls the
  annotated preview images.
