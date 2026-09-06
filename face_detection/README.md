# Face detector evaluation

Evaluates face detectors — MediaPipe Face Detector (BlazeFace `short_range`
and `full_range` variants) and SCRFD-10G-KPS — against the 500-image sample
in `detecting_faces_data/`, to pick a model, confidence
threshold, and crop margin for the face-embedding pipeline.

## Setup

```
pip install -r requirements.txt
python downloading_models_scripts/download_face_detector_models.py  # -> models/*.tflite
python downloading_models_scripts/download_scrfd_model.py           # -> models/scrfd_10g_kps.onnx
```

## Scripts

- `face_detection/detect.py` — runs both BlazeFace variants over every image
  in the manifest, writes per-image detection counts/scores/boxes to
  `<output-dir>/detections_<variant>.csv`.
- `face_detection/detect_scrfd.py` — same contract as `detect.py`, for the
  SCRFD-10G-KPS ONNX model; writes `detections_scrfd_10g_kps.csv`.
- `face_detection/accuracy_report.py` — scores a detections CSV against
  per-group ground truth (TP/TN/FP/FN, precision/recall/accuracy). Accepts
  `--variants` to target any set of variant names.
- `face_detection/summarize.py` — turns a detections CSV into recall /
  false-positive stats, broken down by brightness, framing, and orientation.
- `face_detection/detection_common.py` — helpers shared by the scripts
  above: the two detector constructors, the square crop-box expansion
  formula, the BlazeFace model filenames, and the detection defaults.
- `face_detection/threshold_sweep.py` — re-thresholds a low-confidence
  (`--min-confidence 0.1`) detection run to show the recall vs.
  false-positive trade-off across confidence cutoffs without re-running
  the model.

## SCRFD-10G-KPS results

Run into a separate `results_scrfd/` folder (vs. `results/` for the
MediaPipe variants) since it's a different model with a different accuracy
report:

```
python face_detection/detect_scrfd.py --images-root detecting_faces_data --manifest detecting_faces_data/manifest.csv --model models/scrfd_10g_kps.onnx --output-dir results_scrfd
python face_detection/accuracy_report.py --results-dir results_scrfd --variants scrfd_10g_kps
```

At the default 0.5 confidence threshold, SCRFD-10G-KPS clearly outperforms
both BlazeFace variants on this sample: 96.8% accuracy / 100% precision /
95.0% recall (n=411), vs. `full_range`'s 81.0% accuracy / 99.5% precision /
70.5% recall. It also had zero false positives on `no_person` images
(`full_range` had 1).

## Findings (500-image sample: 166 no_person / 167 one_person / 167 multiple_people)

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
