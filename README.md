# Face Detection and Identification Study

## Overview

This repository evaluates a pipeline that identifies people from a closed, predefined database of known individuals.

The study has two stages. The first compares face detection models on how reliably they report whether faces are present in an image and how many. The second compares face embedding models, matching strategies, and decision thresholds on how accurately they name a person from the reference database.

The final selected configuration is packaged as a standalone pipeline in [pipeline_a_resnet50_webface600k/](pipeline_a_resnet50_webface600k/).

---

## 1. Face Detection Evaluation

### Goal

Determine which detector most reliably reports the number of faces present in an image. Identity was not evaluated at this stage.

### Models

| Model | Implementation | Notes | Source |
|---|---|---|---|
| BlazeFace `short_range` | MediaPipe Face Detector (`.tflite`) | Tuned for faces within ~2 m of the camera | [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) |
| BlazeFace `full_range` | MediaPipe Face Detector (`.tflite`) | Dense-anchor variant for smaller/farther faces | [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) |
| SCRFD-10G-KPS | InsightFace ONNX (`det_10g.onnx`, from the `buffalo_l` pack), 640×640 input | Also returns 5 facial keypoints, used later for alignment | [InsightFace](https://github.com/deepinsight/insightface) |

All three ran at a confidence threshold of 0.5. Model download scripts are in [downloading_models_scripts/](downloading_models_scripts/).

### Test Cases

The dataset is a curated 450-image sample from Open Images V7 and its MIAP subset, described in [detecting_faces_data/README_HE.md](detecting_faces_data/README_HE.md). Each image was visually verified, and the folder is the authoritative label.

| Group | Images | Contents |
|---|---|---|
| No Faces | 150 | Objects, landscapes, animals, statues, dolls, and illustrations — no real person |
| Single Face | 150 | Exactly one person with a usable visible face |
| Multiple Faces | 150 | Two or more people with at least two visible faces |

The people groups vary deliberately in pose (frontal and profile), tilt, lighting, partial occlusion, and subject distance.

39 images could not be decoded at runtime and were excluded, leaving **411 scored images** (150 / 145 / 116).

### Evaluation

Ground truth is categorical: `0`, `1`, or `2+` faces. Each image was scored by the detected face count against its group:

| Group | Success | Failure |
|---|---|---|
| No Faces | ≤ 1 detection → TN | > 1 detection → FP |
| Single Face | exactly 1 detection → TP | anything else → FN |
| Multiple Faces | > 1 detection → TP | ≤ 1 detection → FN |

Allowing a single detection on a no-face image absorbs known MIAP labelling noise, where "person" boxes were drawn on statues, a helmet, and a light fixture.

Accuracy, precision, and recall were computed overall and per group. See [face_detection/accuracy_report.py](face_detection/accuracy_report.py).

### Results

Per-group accuracy at confidence 0.5:

| Model | No Faces (n=150) | Single Face (n=145) | Multiple Faces (n=116) | Overall accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| BlazeFace `short_range` | 99.3% | 54.5% | 28.4% | 63.5% | 99.1% | 42.9% |
| BlazeFace `full_range` | 99.3% | 77.9% | 61.2% | 81.0% | 99.5% | 70.5% |
| **SCRFD-10G-KPS** | **100.0%** | **95.2%** | **94.8%** | **96.8%** | **100.0%** | **95.0%** |

Confusion counts:

| Model | TP | TN | FP | FN |
|---|---|---|---|---|
| BlazeFace `short_range` | 112 | 149 | 1 | 149 |
| BlazeFace `full_range` | 184 | 149 | 1 | 77 |
| SCRFD-10G-KPS | 248 | 150 | 0 | 13 |

A confidence sweep from 0.1 to 0.7 ([face_detection/threshold_sweep.py](face_detection/threshold_sweep.py)) showed `full_range` beating `short_range` at every threshold. Lowering BlazeFace's threshold to 0.3 roughly doubled recall but raised the false-positive rate on no-face images from 5.4% to ~25%.

### Selected Detector

**SCRFD-10G-KPS.** It leads on every group, had zero false positives, and holds the widest margin on multi-face images — the case both BlazeFace variants handle worst. It also emits the 5 keypoints the embedding models need for alignment, so detection and alignment come from one pass.

---

## 2. Face Identification Evaluation

### Goal

Measure how accurately a detected face can be matched to a person in a closed database, and how often the system correctly refuses to name someone who is not in it.

### Reference Database

| Property | Value |
|---|---|
| People | 234 |
| Reference images | 948 |
| Images per person | 3–5 (mean 4.05) |
| Faces stored | 948 (one per image) |

Images were sampled from the ZebeAI project dataset, organised as `<group>/<person>/<images>` across four groups: public figures, MPs and ministers, veteran/non-serving politicians, and new politicians.

Reference images were filtered to those containing **exactly one detected face**, and capped at 5 per person, so every stored embedding is unambiguously tied to its label.

### Evaluation Set

571 human-reviewed faces across 250 real social-media images from [real_data/](real_data/) — Facebook, Instagram, X, and TikTok (128 still images, 122 video frames).

| Scope | Images | Faces | Known | Unknown |
|---|---|---|---|---|
| `one_person` | 120 | 129 | 60 | 69 |
| `few_people` | 130 | 442 | 77 | 365 |
| `pooled` | 250 | 571 | 137 | 434 |

"Known" means the face belongs to a person in the reference database; the other 434 are distractors that should be rejected. Each labelled face was paired with a detected face by box overlap (IoU ≥ 0.5); all 571 were matched, so no labelled face was left without an embedding.

### Embedding Models

| | ResNet50@WebFace600K | SFace |
|---|---|---|
| Implementation | InsightFace `buffalo_l` (`FaceAnalysis`) | OpenCV `FaceRecognizerSF` (`face_recognition_sface_2021dec.onnx`) |
| Training set | WebFace600K | — |
| Input | 112×112 aligned crop | 112×112 aligned crop |
| Embedding size | 512-d | 128-d |
| Source | [InsightFace](https://github.com/deepinsight/insightface) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |

### Embedding Generation

Detection runs **once per image**, via `FaceAnalysis.get()` (SCRFD-10GF + 5 keypoints). That single detection then feeds both recognizers, each performing its own official alignment — a rotation, scale, and translation of the 5 keypoints onto that model's reference points:

1. **ResNet50@WebFace600K** — alignment and recognition happen inside `FaceAnalysis.get()` itself.
2. **SFace** — the same box, keypoints, and confidence are adapted into the 15-value row `cv2.FaceRecognizerSF.alignCrop()` expects, then passed through `alignCrop()` → `feature()`.

No landmark reordering is needed: OpenCV's `alignCrop` warps to the same ArcFace reference points InsightFace uses. Detecting once guarantees both embeddings for a row always come from the same face. Both are L2-normalized before storage. See [face_embedding/build_face_database.py](face_embedding/build_face_database.py).

---

## 3. Matching Strategies

Similarity is cosine similarity between L2-normalized embeddings. Each strategy scores a query face against **each candidate person** (not a global k-NN over all reference images); the prediction is the highest-scoring person, or `unknown` if that score falls below the threshold.

| Strategy | Score for a candidate person |
|---|---|
| TOP1 | Similarity to that person's single best-matching reference image |
| TOP2 | Mean similarity to that person's 2 closest reference images |
| TOP3 | Mean similarity to that person's 3 closest reference images |
| Centroid | Similarity to the L2-renormalized mean of that person's reference embeddings |

---

## 4. Threshold Evaluation

Thresholds from **0.15 to 0.85 in steps of 0.05** were swept, using the same grid for both models and all four strategies. Every model × strategy × threshold × scope combination was scored (360 rows in [results_identification/identification_metrics.csv](results_identification/identification_metrics.csv)).

Outcomes tracked per face:

| Outcome | Meaning |
|---|---|
| Correct identification | Known face matched to the right person |
| Correct rejection | Unknown face called `unknown` |
| `known_as_unknown` | Known face rejected (recall loss) |
| `unknown_as_known` | Distractor assigned an identity (precision loss) |
| `wrong_identity` | Known face matched to a *different* known person |

The best threshold per model and strategy was chosen by highest F1 on the pooled scope.

### Results — best operating point per configuration

| Model | Strategy | Threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.35 | 0.959 | 0.977 | 0.942 | 0.981 | 0 |
| **ResNet50@WebFace600K** | **TOP2** | **0.30** | **0.960** | 0.963 | 0.956 | 0.981 | 0 |
| ResNet50@WebFace600K | TOP3 | 0.30 | 0.959 | 0.970 | 0.949 | 0.981 | 0 |
| ResNet50@WebFace600K | Centroid | 0.30 | 0.959 | 0.970 | 0.949 | 0.981 | 0 |
| SFace | TOP1 | 0.45 | 0.894 | 0.929 | 0.861 | 0.951 | 0 |
| SFace | TOP2 | 0.45 | 0.898 | 0.966 | 0.839 | 0.955 | 0 |
| SFace | TOP3 | 0.40 | 0.882 | 0.895 | 0.869 | 0.944 | 0 |
| SFace | Centroid | 0.50 | 0.892 | 0.983 | 0.818 | 0.953 | 0 |

### Results — threshold sensitivity (ResNet50@WebFace600K, TOP2, pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→unknown | unknown→known |
|---|---|---|---|---|---|---|
| 0.20 | 0.820 | 0.570 | 0.978 | 0.720 | 2 | 100 |
| 0.25 | 0.953 | 0.856 | 0.956 | 0.903 | 5 | 21 |
| **0.30** | **0.981** | **0.963** | **0.956** | **0.960** | 6 | 5 |
| 0.35 | 0.974 | 0.977 | 0.912 | 0.943 | 12 | 3 |
| 0.40 | 0.962 | 0.975 | 0.861 | 0.915 | 19 | 3 |
| 0.50 | 0.933 | 0.981 | 0.737 | 0.842 | 36 | 2 |
| 0.60 | 0.874 | 1.000 | 0.474 | 0.644 | 72 | 0 |

Every configuration follows this shape: below the optimum, distractors flood in and precision collapses; above it, real matches are silently rejected and recall collapses. SFace needs a threshold roughly 0.15 higher than ResNet to reach its equivalent operating point.

### Results — single-person vs. multi-person images

At each configuration's best pooled threshold:

| Model | Strategy | Threshold | `one_person` F1 | `few_people` F1 |
|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.35 | 0.983 | 0.940 |
| ResNet50@WebFace600K | TOP2 | 0.30 | 0.983 | 0.941 |
| ResNet50@WebFace600K | TOP3 | 0.30 | 0.983 | 0.940 |
| ResNet50@WebFace600K | Centroid | 0.30 | 0.983 | 0.940 |
| SFace | TOP1 | 0.45 | 0.942 | 0.853 |
| SFace | TOP2 | 0.45 | 0.948 | 0.857 |
| SFace | TOP3 | 0.40 | 0.942 | 0.832 |
| SFace | Centroid | 0.50 | 0.938 | 0.855 |

Group photos run 4–13 F1 points below single-person photos for every configuration.

The full sweep across all 15 thresholds, both models, and all four strategies is in [results_identification/RESULTS.md](results_identification/RESULTS.md).

---

## 5. Final Configuration

| Component | Choice |
|---|---|
| Face detection | SCRFD-10G-KPS (via InsightFace `buffalo_l`), confidence 0.5, 640×640 input |
| Face embedding | ResNet50@WebFace600K, 512-d |
| Matching strategy | TOP2 (mean similarity to a person's 2 closest reference images) |
| Threshold | 0.30 cosine similarity |

ResNet50@WebFace600K leads SFace by 6–8 F1 points and is far less sensitive to the strategy chosen — TOP1, TOP2, TOP3, and Centroid land within 0.001 F1 of each other at 0.30. TOP2 was taken as the peak of that flat region; it tolerates one bad reference image per person better than TOP1, without needing three good ones like TOP3.

At 0.30 the errors are balanced (6 known faces rejected, 5 distractors named) and `wrong_identity` is 0 — misnaming one known person as another only appears at thresholds well below the optimum.

This configuration is packaged in [pipeline_a_resnet50_webface600k/](pipeline_a_resnet50_webface600k/), where the model, strategy, and threshold are all defined in [face_pipeline.py](pipeline_a_resnet50_webface600k/face_pipeline.py).

---

## Limitations

- **Detection accuracy is lenient on multi-face images.** Ground truth is categorical (`0`/`1`/`2+`), so a group photo counts as correct whenever more than one face is found — the exact count was not required. The 94.8% multi-face figure is therefore an upper bound on exact-count accuracy.
- **The two stages use different data.** Detection was scored on Open Images; identification on real social-media images. The detection figures do not transfer directly to the identification set.
- **Identification metrics isolate the matching step.** All 571 labelled faces were faces the detector had already found, so detector misses are not reflected in the identification numbers.
- **The threshold depends on the known/unknown mix.** The evaluation set is 24% known faces. A deployment with a very different base rate will shift the precision/recall balance and may need re-tuning.
- **The reference database is domain-specific.** 234 Israeli public figures with 3–5 images each; identity coverage, image quality, and per-person image count all affect where the optimal threshold lands.

---

## Repository Layout

| Path | Contents |
|---|---|
| [detecting_faces_data/](detecting_faces_data/) | 450-image detection dataset, manifest, and curation audit |
| [face_detection/](face_detection/) | Detection runners, accuracy report, threshold sweep, annotation |
| [face_embedding/](face_embedding/) | Builds the reference embedding database (both models) |
| [face_identification/](face_identification/) | Evaluation-set embedding extraction and the identification scorer |
| [downloading_models_scripts/](downloading_models_scripts/) | Model download scripts |
| `results_blazeface/`, `results_scrfd/` | Detection results per model |
| `results_embeddings/`, `results_eval_embeddings/` | Reference and evaluation embeddings with manifests |
| [results_identification/](results_identification/) | Identification metrics and the full write-up |
| [pipeline_a_resnet50_webface600k/](pipeline_a_resnet50_webface600k/) | The final selected configuration, packaged standalone |

## Reproducing

```bash
pip install -r requirements.txt

# Models
python downloading_models_scripts/download_face_detector_models.py
python downloading_models_scripts/download_scrfd_model.py
python downloading_models_scripts/download_recognition_models.py

# Stage 1 — detection
python face_detection/detect.py --images-root detecting_faces_data --manifest detecting_faces_data/manifest.csv --output-dir results_blazeface
python face_detection/accuracy_report.py --results-dir results_blazeface
python face_detection/detect_scrfd.py --images-root detecting_faces_data --manifest detecting_faces_data/manifest.csv --model models/scrfd_10g_kps.onnx --output-dir results_scrfd
python face_detection/accuracy_report.py --results-dir results_scrfd --variants scrfd_10g_kps

# Stage 2 — identification
python face_embedding/build_face_database.py --people-root politicians_images --output-dir results_embeddings
python face_identification/extract_eval_embeddings.py
python face_identification/evaluate_identification.py
```

`buffalo_l` is downloaded and cached automatically by `FaceAnalysis` on first use.
