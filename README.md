# Face Detection and Identification Study

## Overview

This repository evaluates a pipeline that identifies people from a closed, predefined database of known individuals.

The study has two stages. The first compares face detection models on how reliably they report whether faces are present in an image and how many. The second compares face embedding models, matching strategies, and decision thresholds on how accurately they name a person from the reference database.

The final selected configuration is packaged as a standalone pipeline in [pipeline_A_resnet50_webface600k/](pipeline_A_resnet50_webface600k/). The runner-up configuration is packaged the same way in [pipeline_B_sface/](pipeline_B_sface/).

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
| `few_people` | 130 | 442 | 80 | 362 |
| `pooled` | 250 | 571 | 140 | 431 |

"Known" means the face belongs to a person in the reference database; the other 431 are distractors that should be rejected. Each labelled face was paired with a detected face by box overlap (IoU ≥ 0.5); all 571 were matched, so no labelled face was left without an embedding.

Error analysis of an earlier scoring run surfaced five mislabelled faces, which were re-reviewed image-by-image and corrected: four distractors that were in fact known people (one of them confirmed by the source clip's own watermark), and one face labelled with a name whose box contains only the back of a head — no facial pixels, so it was relabelled `unknown`. This moved the split from 137/434 to 140/431. The per-image list is in [results_identification/RESULTS.md](results_identification/RESULTS.md).

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

The three error outcomes, in plain terms:

- **`unknown_as_known`** — a person who isn't in the database at all, but the system says they're a known person anyway.
  Example: a stranger walks in → the system says "That's X." This is a false positive / false accept.
- **`known_as_unknown`** — the person *is* in the database, but the system wasn't confident enough, so it said `unknown`.
  Example: it's really X → but the score is below the threshold → the system says `unknown`. This is a false negative / false reject.
- **`wrong_identity`** — the person *is* in the database, and the system did decide they're known, but it picked the wrong person.
  Example: it's really Y → the system says "That's Z."

The best threshold per model and strategy was chosen by highest F1 on the pooled scope.

### Results — best operating point per configuration

| Model | Strategy | Threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.30 | 0.975 | 0.985 | 0.964 | 0.988 | 0 |
| ResNet50@WebFace600K | TOP2 | 0.30 | 0.978 | 0.993 | 0.964 | 0.990 | 0 |
| **ResNet50@WebFace600K** | **TOP3** | **0.30** | **0.978** | **1.000** | 0.957 | 0.990 | 0 |
| ResNet50@WebFace600K | Centroid | 0.30 | 0.978 | 1.000 | 0.957 | 0.990 | 0 |
| SFace | TOP1 | 0.45 | 0.899 | 0.945 | 0.857 | 0.953 | 0 |
| SFace | TOP2 | 0.45 | 0.903 | 0.983 | 0.836 | 0.956 | 0 |
| SFace | TOP3 | 0.40 | 0.886 | 0.910 | 0.864 | 0.946 | 0 |
| SFace | Centroid | 0.50 | 0.898 | 1.000 | 0.814 | 0.955 | 0 |

All four ResNet strategies peak at the same threshold, 0.30, within 0.004 F1 of each other — TOP2, TOP3, and Centroid within 0.0002. Since F1 cannot separate them, the selection was made on the `unknown_as_known` count instead; see [Final Configuration](#5-final-configuration).

### Results — threshold sensitivity (ResNet50@WebFace600K, TOP3, pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→unknown | unknown→known |
|---|---|---|---|---|---|---|
| 0.20 | 0.893 | 0.701 | 0.971 | 0.814 | 3 | 57 |
| 0.25 | 0.977 | 0.938 | 0.964 | 0.951 | 4 | 8 |
| **0.30** | **0.990** | **1.000** | **0.957** | **0.978** | 6 | 0 |
| 0.35 | 0.977 | 1.000 | 0.907 | 0.951 | 13 | 0 |
| 0.40 | 0.965 | 1.000 | 0.857 | 0.923 | 20 | 0 |
| 0.50 | 0.926 | 1.000 | 0.700 | 0.824 | 42 | 0 |
| 0.60 | 0.853 | 1.000 | 0.400 | 0.571 | 84 | 0 |

Every configuration follows this shape: below the optimum, distractors flood in and precision collapses; above it, real matches are silently rejected and recall collapses. SFace needs a threshold roughly 0.15 higher than ResNet to reach its equivalent operating point.

### Results — single-person vs. multi-person images

At each configuration's best pooled threshold:

| Model | Strategy | Threshold | `one_person` F1 | `few_people` F1 |
|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.30 | 1.000 | 0.955 |
| ResNet50@WebFace600K | TOP2 | 0.30 | 1.000 | 0.962 |
| **ResNet50@WebFace600K** | **TOP3** | **0.30** | **1.000** | **0.961** |
| ResNet50@WebFace600K | Centroid | 0.30 | 1.000 | 0.961 |
| SFace | TOP1 | 0.45 | 0.942 | 0.863 |
| SFace | TOP2 | 0.45 | 0.948 | 0.867 |
| SFace | TOP3 | 0.40 | 0.942 | 0.842 |
| SFace | Centroid | 0.50 | 0.938 | 0.865 |

Group photos run 4–10 F1 points below single-person photos for every configuration. ResNet is perfect on `one_person` at 0.30 under all four strategies, so every error it makes is in a group photo.

The full sweep across all 15 thresholds, both models, and all four strategies is in [results_identification/RESULTS.md](results_identification/RESULTS.md).

---

## 5. Final Configuration

| Component | Choice |
|---|---|
| Face detection | SCRFD-10G-KPS (via InsightFace `buffalo_l`), confidence 0.5, 640×640 input |
| Face embedding | ResNet50@WebFace600K, 512-d |
| Matching strategy | TOP3 (mean similarity to a person's 3 closest reference images) |
| Threshold | 0.30 cosine similarity |

ResNet50@WebFace600K leads SFace by 7–8 F1 points and is far less sensitive to the strategy chosen — TOP1, TOP2, TOP3, and Centroid all peak at 0.30 and land within 0.004 F1 of each other there (TOP2/TOP3/Centroid within 0.0002). On a 571-face set that spread is one or two faces, so F1 cannot pick a winner and the selection was made on error type instead:

| Strategy @ 0.30 | `unknown_as_known` | `known_as_unknown` | F1 |
|---|---|---|---|
| TOP1 | 2 | 5 | 0.975 |
| TOP2 | 1 | 5 | 0.978 |
| **TOP3** | **0** | 6 | 0.978 |
| Centroid | 0 | 6 | 0.978 |

**TOP3 was selected because it produces zero false accepts** — assigning a known identity to someone who is not in the database is the error this system most wants to avoid, since a confidently wrong name is worse than an `unknown` a human can follow up on. Its F1 is effectively tied with TOP2 and Centroid.

TOP3 is preferred over Centroid, which also reaches zero, because **each person in the reference database has only 3–5 images**: a centroid averaged over so few samples is a fragile estimate of someone's appearance, where one atypical photo permanently shifts that person's vector. TOP3 selects the 3 closest reference images per query instead, so an outlier photo is simply not chosen when it does not help. All 234 people have at least 3 images, so TOP3 never falls back to fewer.

At 0.30 only 6 of the 571 faces are scored wrong — all 6 are known faces rejected as `unknown`, no distractor is ever named, and `wrong_identity` is 0. In 4 of the 6 rejections the top-ranked candidate was already the correct person, just below the threshold, so the failure mode is under-confidence rather than confusion. All 6 errors are in group photos.

This configuration is packaged in [pipeline_A_resnet50_webface600k/](pipeline_A_resnet50_webface600k/), where the model, strategy, and threshold are all defined in [face_pipeline.py](pipeline_A_resnet50_webface600k/face_pipeline.py). [pipeline_B_sface/](pipeline_B_sface/) packages the SFace alternative (TOP2 at threshold 0.45) in the same shape, for comparison.

---

## Limitations

- **Detection accuracy is lenient on multi-face images.** Ground truth is categorical (`0`/`1`/`2+`), so a group photo counts as correct whenever more than one face is found — the exact count was not required. The 94.8% multi-face figure is therefore an upper bound on exact-count accuracy.
- **The two stages use different data.** Detection was scored on Open Images; identification on real social-media images. The detection figures do not transfer directly to the identification set.
- **Identification metrics isolate the matching step.** All 571 labelled faces were faces the detector had already found, so detector misses are not reflected in the identification numbers.
- **The threshold depends on the known/unknown mix.** The evaluation set is 25% known faces. A deployment with a very different base rate will shift the precision/recall balance and may need re-tuning.
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
| [pipeline_A_resnet50_webface600k/](pipeline_A_resnet50_webface600k/) | The final selected configuration, packaged standalone |
| [pipeline_B_sface/](pipeline_B_sface/) | The SFace alternative, packaged the same way |
