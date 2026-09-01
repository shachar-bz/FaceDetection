# MediaPipe Face Detector evaluation

Evaluates the MediaPipe Face Detector task (BlazeFace `short_range` and
`full_range` variants) against the 500-image sample in
`mediapipe-face-detection-images/`, to pick a model, confidence threshold,
and crop margin for the face-embedding pipeline.

## Setup

```
pip install -r requirements.txt
python scripts/download_face_detector_models.py   # -> models/*.tflite
```

## Scripts

- `face_detection/detect.py` — runs both model variants over every image in
  the manifest, writes per-image detection counts/scores/boxes to
  `output/detections_<variant>.csv`.
- `face_detection/summarize.py` — turns a detections CSV into recall /
  false-positive stats, broken down by brightness, framing, and orientation.
- `face_detection/threshold_sweep.py` — re-thresholds a low-confidence
  (`--min-confidence 0.1`) detection run to show the recall vs.
  false-positive trade-off across confidence cutoffs without re-running
  the model.
- `face_detection/visualize_margins.py` — for a list of images, draws the
  raw bounding box plus square crops expanded by several margin ratios, as
  a side-by-side contact sheet, to eyeball crop quality.

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
