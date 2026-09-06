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
cd pipeline_A_resnet50_webface600k
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
  0.30 is where all four strategies peak (see the sweep below); below 0.25 precision collapses.
- `MATCHING_STRATEGY_TOP_K` (2) — 1 = a person's single best image, 3 = their 3 closest. For
  this model the choice barely matters (TOP2/TOP3/Centroid within 0.0002 F1, TOP1 within 0.004).
- `MIN_DETECTION_CONFIDENCE` (0.5) — how confident the detector must be that a region is a face.

---

# Experiment results — ResNet50@WebFace600K

## Setup

- **Reference database:** 234 people.
- **Evaluation set:** 571 human-labelled faces — **140 known** (someone in the reference
  database) and **431 unknown** (distractors). Reported three ways:
  - `one_person` — images with exactly one person (129 faces: 60 known / 69 unknown)
  - `few_people` — images with several people (442 faces: 80 known / 362 unknown)
  - `pooled` — both combined; the tables below are pooled, since that is the operating point
    a deployed threshold actually sees
- **Strategies** (scored per candidate person, not as a global k-NN):
  - **TOP1** — similarity to that person's single best-matching reference image
  - **TOP2 / TOP3** — mean similarity to that person's 2 / 3 closest reference images
  - **Centroid** — similarity to the L2-renormalized mean of all that person's embeddings
- **Threshold grid:** 0.15 → 0.85 in steps of 0.05.
- **`wrong_identity`** = a known face matched to the *wrong* known person, tracked separately
  from known→unknown and unknown→known mistakes because it is a distinct failure mode.
- **Ground-truth revision:** five labels were corrected after reviewing the errors of an earlier
  run — four distractors that were really known people, and one name whose box holds only the
  back of a head (relabelled `unknown`). The split moved from 137/434 to 140/431; see
  [../results_identification/RESULTS.md](../results_identification/RESULTS.md) for the list.

## Headline

**Best F1 ≈ 0.978, at threshold 0.30.** All four strategies now peak at the *same*
threshold, and TOP2/TOP3/Centroid land within 0.0002 F1 of each other there — so for this model
the threshold matters and the strategy barely does. `wrong_identity` is **0 at every strategy's
best threshold** — misattributing one known person to another only happens when the threshold is
set well below the optimum.

| Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|
| TOP1 | 0.30 | 0.975 | 0.985 | 0.964 | 0.988 | 0 |
| **TOP2 (this pipeline)** | **0.30** | **0.978** | 0.993 | 0.964 | 0.990 | 0 |
| TOP3 | 0.30 | 0.978 | 1.000 | 0.957 | 0.990 | 0 |
| Centroid | 0.30 | 0.978 | 1.000 | 0.957 | 0.990 | 0 |

Every configuration follows the same shape across the sweep: at **low thresholds** the system
is too permissive — nearly every known face is found (recall ≈ 1.0) but so are hundreds of
unknown faces, dragging precision down toward the known:unknown base rate (≈0.25) and F1 to
~0.4. As the threshold rises precision climbs quickly while recall holds — the useful range
where F1 peaks. Past the peak it becomes **too conservative**, rejecting real matches as
`unknown` (recall collapses), and by 0.80–0.85 it identifies almost nobody. ResNet stabilizes
early, by threshold 0.25–0.30.

## Full threshold sweep

### TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.336 | 0.268 | 0.993 | 0.422 | 139 | 0 | 378 | 1 |
| 0.20 | 0.701 | 0.448 | 0.993 | 0.618 | 139 | 0 | 170 | 1 |
| 0.25 | 0.942 | 0.820 | 0.979 | 0.892 | 137 | 3 | 30 | 0 |
| **0.30** | **0.988** | **0.985** | **0.964** | **0.975 ← best** | 135 | 5 | 2 | 0 |
| 0.35 | 0.986 | 1.000 | 0.943 | 0.971 | 132 | 8 | 0 | 0 |
| 0.40 | 0.972 | 1.000 | 0.886 | 0.939 | 124 | 16 | 0 | 0 |
| 0.45 | 0.962 | 1.000 | 0.843 | 0.915 | 118 | 22 | 0 | 0 |
| 0.50 | 0.944 | 1.000 | 0.771 | 0.871 | 108 | 32 | 0 | 0 |
| 0.55 | 0.926 | 1.000 | 0.700 | 0.824 | 98 | 42 | 0 | 0 |
| 0.60 | 0.886 | 1.000 | 0.536 | 0.698 | 75 | 65 | 0 | 0 |
| 0.65 | 0.855 | 1.000 | 0.407 | 0.579 | 57 | 83 | 0 | 0 |
| 0.70 | 0.807 | 1.000 | 0.214 | 0.353 | 30 | 110 | 0 | 0 |
| 0.75 | 0.771 | 1.000 | 0.064 | 0.121 | 9 | 131 | 0 | 0 |
| 0.80 | 0.757 | 1.000 | 0.007 | 0.014 | 1 | 139 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### TOP2 (pooled) — the strategy this pipeline ships

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.434 | 0.299 | 0.986 | 0.459 | 138 | 0 | 321 | 2 |
| 0.20 | 0.828 | 0.587 | 0.986 | 0.736 | 138 | 1 | 96 | 1 |
| 0.25 | 0.962 | 0.882 | 0.964 | 0.921 | 135 | 4 | 17 | 1 |
| **0.30** | **0.990** | **0.993** | **0.964** | **0.978 ← best (shipped)** | 135 | 5 | 1 | 0 |
| 0.35 | 0.979 | 1.000 | 0.914 | 0.955 | 128 | 12 | 0 | 0 |
| 0.40 | 0.967 | 1.000 | 0.864 | 0.927 | 121 | 19 | 0 | 0 |
| 0.45 | 0.953 | 1.000 | 0.807 | 0.893 | 113 | 27 | 0 | 0 |
| 0.50 | 0.935 | 1.000 | 0.736 | 0.848 | 103 | 37 | 0 | 0 |
| 0.55 | 0.902 | 1.000 | 0.600 | 0.750 | 84 | 56 | 0 | 0 |
| 0.60 | 0.869 | 1.000 | 0.464 | 0.634 | 65 | 75 | 0 | 0 |
| 0.65 | 0.828 | 1.000 | 0.300 | 0.462 | 42 | 98 | 0 | 0 |
| 0.70 | 0.792 | 1.000 | 0.150 | 0.261 | 21 | 119 | 0 | 0 |
| 0.75 | 0.765 | 1.000 | 0.043 | 0.082 | 6 | 134 | 0 | 0 |
| 0.80 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.532 | 0.341 | 0.986 | 0.506 | 138 | 0 | 265 | 2 |
| 0.20 | 0.893 | 0.701 | 0.971 | 0.814 | 136 | 3 | 57 | 1 |
| 0.25 | 0.977 | 0.938 | 0.964 | 0.951 | 135 | 4 | 8 | 1 |
| **0.30** | **0.990** | **1.000** | **0.957** | **0.978 ← best** | 134 | 6 | 0 | 0 |
| 0.35 | 0.977 | 1.000 | 0.907 | 0.951 | 127 | 13 | 0 | 0 |
| 0.40 | 0.965 | 1.000 | 0.857 | 0.923 | 120 | 20 | 0 | 0 |
| 0.45 | 0.948 | 1.000 | 0.786 | 0.880 | 110 | 30 | 0 | 0 |
| 0.50 | 0.926 | 1.000 | 0.700 | 0.824 | 98 | 42 | 0 | 0 |
| 0.55 | 0.891 | 1.000 | 0.557 | 0.716 | 78 | 62 | 0 | 0 |
| 0.60 | 0.853 | 1.000 | 0.400 | 0.571 | 56 | 84 | 0 | 0 |
| 0.65 | 0.813 | 1.000 | 0.236 | 0.382 | 33 | 107 | 0 | 0 |
| 0.70 | 0.776 | 1.000 | 0.086 | 0.158 | 12 | 128 | 0 | 0 |
| 0.75 | 0.757 | 1.000 | 0.007 | 0.014 | 1 | 139 | 0 | 0 |
| 0.80 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.461 | 0.309 | 0.986 | 0.471 | 138 | 0 | 306 | 2 |
| 0.20 | 0.811 | 0.563 | 0.986 | 0.717 | 138 | 1 | 106 | 1 |
| 0.25 | 0.965 | 0.894 | 0.964 | 0.928 | 135 | 4 | 15 | 1 |
| **0.30** | **0.990** | **1.000** | **0.957** | **0.978 ← best** | 134 | 6 | 0 | 0 |
| 0.35 | 0.986 | 1.000 | 0.943 | 0.971 | 132 | 8 | 0 | 0 |
| 0.40 | 0.976 | 1.000 | 0.900 | 0.947 | 126 | 14 | 0 | 0 |
| 0.45 | 0.965 | 1.000 | 0.857 | 0.923 | 120 | 20 | 0 | 0 |
| 0.50 | 0.949 | 1.000 | 0.793 | 0.884 | 111 | 29 | 0 | 0 |
| 0.55 | 0.930 | 1.000 | 0.714 | 0.833 | 100 | 40 | 0 | 0 |
| 0.60 | 0.898 | 1.000 | 0.586 | 0.739 | 82 | 58 | 0 | 0 |
| 0.65 | 0.877 | 1.000 | 0.500 | 0.667 | 70 | 70 | 0 | 0 |
| 0.70 | 0.830 | 1.000 | 0.307 | 0.470 | 43 | 97 | 0 | 0 |
| 0.75 | 0.792 | 1.000 | 0.150 | 0.261 | 21 | 119 | 0 | 0 |
| 0.80 | 0.760 | 1.000 | 0.021 | 0.042 | 3 | 137 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

## Solo vs. group photos, at each strategy's best threshold

**Every strategy is perfect (F1 = 1.000) on solo photos at 0.30**, so all remaining errors are
in group photos, where more distractor faces per image means more chances for a false match:

| Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|
| TOP1 | 0.30 | 1.000 | 0.955 |
| **TOP2** | **0.30** | **1.000** | **0.962** |
| TOP3 | 0.30 | 1.000 | 0.961 |
| Centroid | 0.30 | 1.000 | 0.961 |

## Errors at the shipped operating point (TOP2, 0.30)

Six of the 571 faces are scored wrong, all of them in group photos:

- **1 × unknown→known** — a campaign poster clipped by the top image border (only mouth and chin
  inside the frame) matched a known person at 0.3128, just 0.013 over the threshold.
- **5 × known→unknown** — real people the system declined to name. In **3 of the 5 the
  top-ranked candidate was already the correct person** (scores 0.204, 0.216, 0.225); the crops
  are small, blurred, or in steep profile, so the score fell short of 0.30. The other two ranked
  a wrong candidate first but stayed below the threshold, so no wrong name was ever asserted.
- **0 × wrong_identity.**

The practical point: when this configuration is unsure it says `unknown` rather than guessing.

## Takeaways

1. **Threshold 0.30 is the operating point.** Below it, false identifications flood in; above
   it, real matches get silently rejected. All four strategies agree on it.
2. **Strategy choice barely matters for this model** — TOP2/TOP3/Centroid are within 0.0002 F1,
   and TOP1 trails by only 0.004. TOP2 is shipped because it has the best pooled F1 and
   tolerates one weak reference image per person better than TOP1.
3. **Confusing one known person for another is rare and avoidable** — 0 occurrences at the
   recommended threshold.
4. **Expect group photos to score below solo photos**, whatever threshold you pick — that is a
   property of distractor-rich images, not something tuning fixes.

## Compared with pipeline B (OpenCV SFace)

| | Pipeline A (this one) | Pipeline B (SFace) |
|---|---|---|
| Best F1 | **0.978** | 0.903 |
| Threshold | 0.30 | 0.45 |
| Embedding size | 512-d | 128-d |
| Recognition model on disk | ~166 MB | ~37 MB |
| Sensitivity to strategy | negligible (within 0.004) | noticeable (0.886–0.903) |

**Pipeline A is the accuracy recommendation** — ~7 F1 points ahead and far less sensitive to
configuration. Choose pipeline B only when the smaller, faster recognizer matters more than
those 7 points. Both pipelines download the same `buffalo_l` pack for detection, so pipeline
B's saving is in the recognition model, not the total download.
