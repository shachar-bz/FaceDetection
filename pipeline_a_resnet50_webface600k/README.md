# Pipeline A — ResNet50@WebFace600K face identification

Self-contained face-identification pipeline. Point it at a folder of labelled photos of the
people you care about, then hand it any new image and it tells you which of those people
appear in it.

- **Detector:** SCRFD-10GF with 5-point alignment (InsightFace `buffalo_l`)
- **Embedding model:** ResNet50 trained on WebFace600K (512-d, InsightFace `buffalo_l`)
- **Matching strategy:** **TOP2** — a person's score is the mean cosine similarity to their
  2 closest reference images
- **Decision threshold:** **0.30** — anything below this is reported as `unknown`

This folder is standalone: download only this folder, install its `requirements.txt`, and it
runs. Pipeline B (OpenCV SFace) is the alternative; see the comparison at the bottom.

---

## Setup

```bash
cd pipeline_a_resnet50_webface600k
pip install -r requirements.txt
python download_models.py
```

`download_models.py` pulls the `buffalo_l` model pack into `models/` (~275 MB, first run only).

## Step 1 — build your face database

Open [build_face_database.py](build_face_database.py) and set **`PEOPLE_IMAGES_ROOT`** to the
folder holding your labelled photos. The folder that directly contains an image gives that
person's name:

```
my_people/
  Ada Lovelace/
    photo1.jpg
    photo2.jpg
  Alan Turing/
    portrait.png
```

A grouping level above the person folders (`my_people/<group>/<person>/*.jpg`) also works —
only the immediate parent folder names the person.

> **If your photos live in Google Drive / Dropbox / OneDrive:** this is a *local filesystem
> path*, not a web URL. Point it at the locally synced copy of that folder, or download the
> images to disk first.

Then:

```bash
python build_face_database.py
# or, without editing the file:
python build_face_database.py --people-images-root "C:/path/to/my_people"
```

This writes:
- `face_database/face_database.npz` — every reference embedding and the person it belongs to
- `face_database/face_database_manifest.csv` — one row per face that went in (person, source
  image, detection confidence, box), plus any unreadable images

**More reference images per person is better.** TOP2 averages a person's two closest images, so
give each person at least 2–3 varied photos; a person with a single photo still works (the
strategy falls back to that one image) but is matched less reliably.

## Step 2 — identify people in a new image

```bash
python identify_faces.py path/to/new_photo.jpg
python identify_faces.py path/to/new_photo.jpg --annotated-output out/new_photo_annotated.jpg
```

Output:

```
Model: ResNet50@WebFace600K | strategy: TOP2 | threshold: 0.3
Detected 3 face(s) in path/to/new_photo.jpg
  face 0: Ada Lovelace                   similarity=0.5821  box=(120.4, 88.1, 210.7, 205.3)  detection_confidence=0.883
  face 1: unknown                        similarity=0.1904  box=(340.2, 96.6, 421.0, 199.8)  detection_confidence=0.851
  face 2: Alan Turing                    similarity=0.4417  box=(512.9, 74.3, 604.1, 191.2)  detection_confidence=0.874
People identified: Ada Lovelace, Alan Turing
```

To call it from your own code:

```python
from pathlib import Path
from identify_faces import identify_people_in_image, DEFAULT_FACE_DATABASE_PATH

for face in identify_people_in_image(Path("new_photo.jpg"), DEFAULT_FACE_DATABASE_PATH):
    print(face.predicted_person, face.similarity_score, face.bounding_box_xyxy)
```

## Files

| File | What it does |
|---|---|
| [face_pipeline.py](face_pipeline.py) | Shared model, threshold, detection, embedding and matching logic — **all tunable settings live at the top of this file** |
| [download_models.py](download_models.py) | Downloads the `buffalo_l` pack into `models/` |
| [build_face_database.py](build_face_database.py) | Builds `face_database.npz` from your labelled photos |
| [identify_faces.py](identify_faces.py) | Identifies the people in a new image |
| [requirements.txt](requirements.txt) | Pinned dependencies |

## Tuning

Everything is at the top of [face_pipeline.py](face_pipeline.py):

- `IDENTIFICATION_THRESHOLD` (0.30) — **raise it** to reduce false identifications at the cost
  of missing real matches; **lower it** to catch more real matches at the cost of false ones.
  0.30–0.35 is the useful range (see the sweep below); below 0.25 precision collapses.
- `MATCHING_STRATEGY_TOP_K` (2) — 1 = a person's single best image, 3 = their 3 closest. For
  this model the choice barely matters (all within 0.0006 F1).
- `MIN_DETECTION_CONFIDENCE` (0.5) — how confident the detector must be that a region is a face.

---

# Experiment results — ResNet50@WebFace600K

## Setup

- **Reference database:** 234 people.
- **Evaluation set:** 571 human-labelled faces — **137 known** (someone in the reference
  database) and **434 unknown** (distractors). Reported three ways:
  - `one_person` — images with exactly one person (129 faces: 60 known / 69 unknown)
  - `few_people` — images with several people (442 faces: 77 known / 365 unknown)
  - `pooled` — both combined; the tables below are pooled, since that is the operating point
    a deployed threshold actually sees
- **Strategies** (scored per candidate person, not as a global k-NN):
  - **TOP1** — similarity to that person's single best-matching reference image
  - **TOP2 / TOP3** — mean similarity to that person's 2 / 3 closest reference images
  - **Centroid** — similarity to the L2-renormalized mean of all that person's embeddings
- **Threshold grid:** 0.15 → 0.85 in steps of 0.05.
- **`wrong_identity`** = a known face matched to the *wrong* known person, tracked separately
  from known→unknown and unknown→known mistakes because it is a distinct failure mode.

## Headline

**Best F1 ≈ 0.96, at threshold 0.30–0.35.** All four strategies land within 0.0006 F1 of each
other, so for this model the threshold matters and the strategy barely does. `wrong_identity`
is **0 at every strategy's best threshold** — misattributing one known person to another only
happens when the threshold is set well below the optimum.

| Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|
| TOP1 | 0.35 | **0.959** | 0.977 | 0.942 | 0.981 | 0 |
| **TOP2 (this pipeline)** | **0.30** | **0.960** | 0.963 | 0.956 | 0.981 | 0 |
| TOP3 | 0.30 | **0.959** | 0.970 | 0.949 | 0.981 | 0 |
| Centroid | 0.30 | **0.959** | 0.970 | 0.949 | 0.981 | 0 |

Every configuration follows the same shape across the sweep: at **low thresholds** the system
is too permissive — nearly every known face is found (recall ≈ 1.0) but so are hundreds of
unknown faces, dragging precision down toward the known:unknown base rate (≈0.24) and F1 to
~0.4. As the threshold rises precision climbs quickly while recall holds — the useful range
where F1 peaks. Past the peak it becomes **too conservative**, rejecting real matches as
`unknown` (recall collapses), and by 0.80–0.85 it identifies almost nobody. ResNet stabilizes
early, by threshold 0.20–0.25.

## Full threshold sweep

### TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.329 | 0.261 | 0.985 | 0.412 | 135 | 0 | 381 | 2 |
| 0.20 | 0.692 | 0.435 | 0.985 | 0.604 | 135 | 1 | 174 | 1 |
| 0.25 | 0.933 | 0.796 | 0.971 | 0.875 | 133 | 4 | 34 | 0 |
| 0.30 | 0.979 | 0.956 | 0.956 | 0.956 | 131 | 6 | 6 | 0 |
| **0.35** | **0.981** | **0.977** | **0.942** | **0.959 ← best** | 129 | 8 | 3 | 0 |
| 0.40 | 0.967 | 0.976 | 0.883 | 0.927 | 121 | 16 | 3 | 0 |
| 0.45 | 0.956 | 0.975 | 0.839 | 0.902 | 115 | 22 | 3 | 0 |
| 0.50 | 0.939 | 0.972 | 0.766 | 0.857 | 105 | 32 | 3 | 0 |
| 0.55 | 0.925 | 0.980 | 0.701 | 0.817 | 96 | 41 | 2 | 0 |
| 0.60 | 0.891 | 1.000 | 0.547 | 0.708 | 75 | 62 | 0 | 0 |
| 0.65 | 0.860 | 1.000 | 0.416 | 0.588 | 57 | 80 | 0 | 0 |
| 0.70 | 0.813 | 1.000 | 0.219 | 0.359 | 30 | 107 | 0 | 0 |
| 0.75 | 0.776 | 1.000 | 0.066 | 0.123 | 9 | 128 | 0 | 0 |
| 0.80 | 0.762 | 1.000 | 0.007 | 0.015 | 1 | 136 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### TOP2 (pooled) — the strategy this pipeline ships

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.427 | 0.291 | 0.978 | 0.448 | 134 | 0 | 324 | 3 |
| 0.20 | 0.820 | 0.570 | 0.978 | 0.720 | 134 | 2 | 100 | 1 |
| 0.25 | 0.953 | 0.856 | 0.956 | 0.903 | 131 | 5 | 21 | 1 |
| **0.30** | **0.981** | **0.963** | **0.956** | **0.960 ← best (shipped)** | 131 | 6 | 5 | 0 |
| 0.35 | 0.974 | 0.977 | 0.912 | 0.943 | 125 | 12 | 3 | 0 |
| 0.40 | 0.962 | 0.975 | 0.861 | 0.915 | 118 | 19 | 3 | 0 |
| 0.45 | 0.948 | 0.974 | 0.803 | 0.880 | 110 | 27 | 3 | 0 |
| 0.50 | 0.933 | 0.981 | 0.737 | 0.842 | 101 | 36 | 2 | 0 |
| 0.55 | 0.900 | 0.976 | 0.599 | 0.742 | 82 | 55 | 2 | 0 |
| 0.60 | 0.874 | 1.000 | 0.474 | 0.644 | 65 | 72 | 0 | 0 |
| 0.65 | 0.834 | 1.000 | 0.307 | 0.469 | 42 | 95 | 0 | 0 |
| 0.70 | 0.797 | 1.000 | 0.153 | 0.266 | 21 | 116 | 0 | 0 |
| 0.75 | 0.771 | 1.000 | 0.044 | 0.084 | 6 | 131 | 0 | 0 |
| 0.80 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.525 | 0.331 | 0.978 | 0.494 | 134 | 0 | 268 | 3 |
| 0.20 | 0.884 | 0.680 | 0.964 | 0.798 | 132 | 4 | 61 | 1 |
| 0.25 | 0.969 | 0.910 | 0.956 | 0.932 | 131 | 5 | 12 | 1 |
| **0.30** | **0.981** | **0.970** | **0.949** | **0.959 ← best** | 130 | 7 | 4 | 0 |
| 0.35 | 0.972 | 0.976 | 0.905 | 0.939 | 124 | 13 | 3 | 0 |
| 0.40 | 0.960 | 0.975 | 0.854 | 0.910 | 117 | 20 | 3 | 0 |
| 0.45 | 0.942 | 0.973 | 0.781 | 0.866 | 107 | 30 | 3 | 0 |
| 0.50 | 0.925 | 0.980 | 0.701 | 0.817 | 96 | 41 | 2 | 0 |
| 0.55 | 0.893 | 0.987 | 0.562 | 0.716 | 77 | 60 | 1 | 0 |
| 0.60 | 0.858 | 1.000 | 0.409 | 0.580 | 56 | 81 | 0 | 0 |
| 0.65 | 0.818 | 1.000 | 0.241 | 0.388 | 33 | 104 | 0 | 0 |
| 0.70 | 0.781 | 1.000 | 0.088 | 0.161 | 12 | 125 | 0 | 0 |
| 0.75 | 0.762 | 1.000 | 0.007 | 0.015 | 1 | 136 | 0 | 0 |
| 0.80 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.454 | 0.300 | 0.978 | 0.460 | 134 | 0 | 309 | 3 |
| 0.20 | 0.802 | 0.547 | 0.978 | 0.702 | 134 | 2 | 110 | 1 |
| 0.25 | 0.956 | 0.868 | 0.956 | 0.910 | 131 | 5 | 19 | 1 |
| **0.30** | **0.981** | **0.970** | **0.949** | **0.959 ← best** | 130 | 7 | 4 | 0 |
| 0.35 | 0.981 | 0.977 | 0.942 | 0.959 | 129 | 8 | 3 | 0 |
| 0.40 | 0.970 | 0.976 | 0.898 | 0.935 | 123 | 14 | 3 | 0 |
| 0.45 | 0.960 | 0.975 | 0.854 | 0.910 | 117 | 20 | 3 | 0 |
| 0.50 | 0.944 | 0.973 | 0.788 | 0.871 | 108 | 29 | 3 | 0 |
| 0.55 | 0.928 | 0.980 | 0.715 | 0.827 | 98 | 39 | 2 | 0 |
| 0.60 | 0.900 | 0.988 | 0.591 | 0.740 | 81 | 56 | 1 | 0 |
| 0.65 | 0.883 | 1.000 | 0.511 | 0.676 | 70 | 67 | 0 | 0 |
| 0.70 | 0.835 | 1.000 | 0.314 | 0.478 | 43 | 94 | 0 | 0 |
| 0.75 | 0.797 | 1.000 | 0.153 | 0.266 | 21 | 116 | 0 | 0 |
| 0.80 | 0.765 | 1.000 | 0.022 | 0.043 | 3 | 134 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

## Solo vs. group photos, at each strategy's best threshold

Group photos are consistently harder — more distractor faces per image means more chances for
a false match:

| Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|
| TOP1 | 0.35 | 0.983 | 0.940 |
| **TOP2** | **0.30** | **0.983** | **0.941** |
| TOP3 | 0.30 | 0.983 | 0.940 |
| Centroid | 0.30 | 0.983 | 0.940 |

## Takeaways

1. **Threshold 0.30–0.35 is the operating range.** Below it, false identifications flood in;
   above it, real matches get silently rejected.
2. **Strategy choice barely matters for this model** — TOP1/TOP2/TOP3/Centroid are within
   0.0006 F1. TOP2 is shipped because it has the best pooled F1 and the highest recall of the
   four at its best threshold.
3. **Confusing one known person for another is rare and avoidable** — 0 occurrences at the
   recommended threshold.
4. **Expect group photos to score a few F1 points below solo photos**, whatever threshold you
   pick — that is a property of distractor-rich images, not something tuning fixes.

## Compared with pipeline B (OpenCV SFace)

| | Pipeline A (this one) | Pipeline B (SFace) |
|---|---|---|
| Best F1 | **0.960** | 0.898 |
| Threshold | 0.30 | 0.45 |
| Embedding size | 512-d | 128-d |
| Recognition model on disk | ~166 MB | ~37 MB |
| Sensitivity to strategy | negligible (within 0.0006) | noticeable (0.881–0.898) |

**Pipeline A is the accuracy recommendation** — ~6 F1 points ahead and far less sensitive to
configuration. Choose pipeline B only when the smaller, faster recognizer matters more than
those 6 points. Both pipelines download the same `buffalo_l` pack for detection, so pipeline
B's saving is in the recognition model, not the total download.
