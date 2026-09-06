# Pipeline B — OpenCV SFace face identification

Self-contained face-identification pipeline. Point it at a folder of labelled photos of the
people you care about, then hand it any new image and it tells you which of those people
appear in it.

- **Detector:** SCRFD-10GF with 5 facial keypoints (InsightFace `buffalo_l`, detector only)
- **Embedding model:** OpenCV SFace (128-d), doing its own official `alignCrop()` from those
  keypoints
- **Matching strategy:** **TOP2** — a person's score is the mean cosine similarity to their
  2 closest reference images
- **Decision threshold:** **0.45** — anything below this is reported as `unknown`

This folder is standalone: download only this folder, install its `requirements.txt`, and it
runs. Pipeline A (ResNet50@WebFace600K) is the more accurate alternative; see the comparison
at the bottom.

> **Why a detector from InsightFace?** SFace is a *recognition* model only — it needs a box
> plus 5 facial keypoints to align a face before embedding it. SCRFD supplies exactly those,
> and it is the detector the experiment below was measured with. Only the detector is loaded
> from the pack; its recognition model is never used here.

---

## Setup

```bash
cd pipeline_b_sface
pip install -r requirements.txt
python download_models.py
```

`download_models.py` fetches SFace (~37 MB) from the OpenCV Zoo and the `buffalo_l` pack
(~275 MB, detector only is used) into `models/`. First run only.

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

**More reference images per person is better**, and it matters more here than in pipeline A:
SFace's 128-d embeddings are noisier, so give each person at least 2–3 varied photos. TOP2
averages a person's two closest images; a person with a single photo still works but is
matched less reliably.

## Step 2 — identify people in a new image

```bash
python identify_faces.py path/to/new_photo.jpg
python identify_faces.py path/to/new_photo.jpg --annotated-output out/new_photo_annotated.jpg
```

Output:

```
Model: OpenCV SFace | strategy: TOP2 | threshold: 0.45
Detected 3 face(s) in path/to/new_photo.jpg
  face 0: Ada Lovelace                   similarity=0.6103  box=(120.4, 88.1, 210.7, 205.3)  detection_confidence=0.883
  face 1: unknown                        similarity=0.2871  box=(340.2, 96.6, 421.0, 199.8)  detection_confidence=0.851
  face 2: Alan Turing                    similarity=0.5240  box=(512.9, 74.3, 604.1, 191.2)  detection_confidence=0.874
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
| [face_pipeline.py](face_pipeline.py) | Shared models, threshold, detection, embedding and matching logic — **all tunable settings live at the top of this file** |
| [download_models.py](download_models.py) | Downloads SFace and the SCRFD detector into `models/` |
| [build_face_database.py](build_face_database.py) | Builds `face_database.npz` from your labelled photos |
| [identify_faces.py](identify_faces.py) | Identifies the people in a new image |
| [requirements.txt](requirements.txt) | Pinned dependencies |

## Tuning

Everything is at the top of [face_pipeline.py](face_pipeline.py):

- `IDENTIFICATION_THRESHOLD` (0.45) — **raise it** to reduce false identifications at the cost
  of missing real matches; **lower it** to catch more real matches at the cost of false ones.
  0.45–0.50 is the useful range (see the sweep below). This model is much more
  threshold-sensitive than pipeline A: below 0.40 precision falls apart fast.
- `MATCHING_STRATEGY_TOP_K` (2) — 1 = a person's single best image, 3 = their 3 closest. For
  this model the choice does matter (0.881–0.898 F1); TOP2 is the best of them.
- `MIN_DETECTION_CONFIDENCE` (0.5) — how confident the detector must be that a region is a face.

---

# Experiment results — OpenCV SFace

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

**Best F1 ≈ 0.89–0.90, at threshold 0.40–0.50 depending on strategy.** Unlike pipeline A, the
strategy does matter here — TOP3 is nearly 2 F1 points behind TOP2. `wrong_identity` is
**0 at every strategy's best threshold**; misattributing one known person to another only
happens when the threshold is set well below the optimum.

| Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|
| TOP1 | 0.45 | 0.894 | 0.929 | 0.861 | 0.951 | 0 |
| **TOP2 (this pipeline)** | **0.45** | **0.898** | 0.966 | 0.839 | 0.955 | 0 |
| TOP3 | 0.40 | 0.882 | 0.895 | 0.869 | 0.944 | 0 |
| Centroid | 0.50 | 0.892 | 0.983 | 0.818 | 0.953 | 0 |

Every configuration follows the same shape across the sweep: at **low thresholds** the system
is too permissive — nearly every known face is found (recall ≈ 0.96) but so are essentially
all 434 unknown faces, pinning precision at the known:unknown base rate (≈0.23) and F1 near
0.37. As the threshold rises precision climbs while recall holds — the useful range where F1
peaks. Past the peak it becomes **too conservative**, rejecting real matches as `unknown`
(recall collapses), and by 0.80–0.85 it identifies almost nobody. **SFace's curve is choppier
than pipeline A's and stabilizes much later** — it takes until threshold 0.40 before precision
becomes usable, versus 0.20–0.25 for ResNet, which is why its operating threshold is higher.

## Full threshold sweep

### TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.229 | 0.229 | 0.956 | 0.370 | 131 | 0 | 434 | 6 |
| 0.20 | 0.231 | 0.230 | 0.956 | 0.371 | 131 | 0 | 433 | 6 |
| 0.25 | 0.272 | 0.239 | 0.956 | 0.383 | 131 | 0 | 410 | 6 |
| 0.30 | 0.398 | 0.276 | 0.956 | 0.429 | 131 | 1 | 338 | 5 |
| 0.35 | 0.608 | 0.368 | 0.927 | 0.527 | 127 | 6 | 214 | 4 |
| 0.40 | 0.858 | 0.638 | 0.912 | 0.751 | 125 | 10 | 69 | 2 |
| **0.45** | **0.951** | **0.929** | **0.861** | **0.894 ← best** | 118 | 19 | 9 | 0 |
| 0.50 | 0.942 | 0.982 | 0.774 | 0.865 | 106 | 31 | 2 | 0 |
| 0.55 | 0.923 | 0.979 | 0.693 | 0.812 | 95 | 42 | 2 | 0 |
| 0.60 | 0.898 | 0.988 | 0.584 | 0.734 | 80 | 57 | 1 | 0 |
| 0.65 | 0.872 | 1.000 | 0.467 | 0.637 | 64 | 73 | 0 | 0 |
| 0.70 | 0.837 | 1.000 | 0.321 | 0.486 | 44 | 93 | 0 | 0 |
| 0.75 | 0.795 | 1.000 | 0.146 | 0.255 | 20 | 117 | 0 | 0 |
| 0.80 | 0.767 | 1.000 | 0.029 | 0.057 | 4 | 133 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### TOP2 (pooled) — the strategy this pipeline ships

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.236 | 0.235 | 0.978 | 0.379 | 134 | 0 | 433 | 3 |
| 0.20 | 0.245 | 0.237 | 0.978 | 0.382 | 134 | 0 | 428 | 3 |
| 0.25 | 0.329 | 0.259 | 0.978 | 0.410 | 134 | 0 | 380 | 3 |
| 0.30 | 0.501 | 0.319 | 0.964 | 0.479 | 132 | 3 | 280 | 2 |
| 0.35 | 0.748 | 0.487 | 0.942 | 0.642 | 129 | 8 | 136 | 0 |
| 0.40 | 0.926 | 0.823 | 0.883 | 0.852 | 121 | 16 | 26 | 0 |
| **0.45** | **0.955** | **0.966** | **0.839** | **0.898 ← best (shipped)** | 115 | 22 | 4 | 0 |
| 0.50 | 0.937 | 0.981 | 0.752 | 0.851 | 103 | 34 | 2 | 0 |
| 0.55 | 0.916 | 0.989 | 0.657 | 0.789 | 90 | 47 | 1 | 0 |
| 0.60 | 0.886 | 0.987 | 0.533 | 0.692 | 73 | 64 | 1 | 0 |
| 0.65 | 0.858 | 1.000 | 0.409 | 0.580 | 56 | 81 | 0 | 0 |
| 0.70 | 0.816 | 1.000 | 0.234 | 0.379 | 32 | 105 | 0 | 0 |
| 0.75 | 0.778 | 1.000 | 0.073 | 0.136 | 10 | 127 | 0 | 0 |
| 0.80 | 0.762 | 1.000 | 0.007 | 0.015 | 1 | 136 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.235 | 0.232 | 0.964 | 0.374 | 132 | 0 | 432 | 5 |
| 0.20 | 0.268 | 0.240 | 0.964 | 0.384 | 132 | 0 | 413 | 5 |
| 0.25 | 0.373 | 0.268 | 0.956 | 0.419 | 131 | 1 | 352 | 5 |
| 0.30 | 0.578 | 0.355 | 0.949 | 0.517 | 130 | 5 | 234 | 2 |
| 0.35 | 0.839 | 0.607 | 0.912 | 0.729 | 125 | 11 | 80 | 1 |
| **0.40** | **0.944** | **0.895** | **0.869** | **0.881 ← best** | 119 | 18 | 14 | 0 |
| 0.45 | 0.946 | 0.973 | 0.796 | 0.875 | 109 | 28 | 3 | 0 |
| 0.50 | 0.925 | 0.990 | 0.693 | 0.816 | 95 | 42 | 1 | 0 |
| 0.55 | 0.904 | 0.988 | 0.606 | 0.751 | 83 | 54 | 1 | 0 |
| 0.60 | 0.876 | 1.000 | 0.482 | 0.650 | 66 | 71 | 0 | 0 |
| 0.65 | 0.844 | 1.000 | 0.350 | 0.519 | 48 | 89 | 0 | 0 |
| 0.70 | 0.797 | 1.000 | 0.153 | 0.266 | 21 | 116 | 0 | 0 |
| 0.75 | 0.767 | 1.000 | 0.029 | 0.057 | 4 | 133 | 0 | 0 |
| 0.80 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.236 | 0.234 | 0.971 | 0.377 | 133 | 0 | 432 | 4 |
| 0.20 | 0.254 | 0.238 | 0.971 | 0.382 | 133 | 0 | 422 | 4 |
| 0.25 | 0.338 | 0.260 | 0.971 | 0.410 | 133 | 0 | 374 | 4 |
| 0.30 | 0.489 | 0.312 | 0.956 | 0.470 | 131 | 3 | 286 | 3 |
| 0.35 | 0.706 | 0.444 | 0.934 | 0.602 | 128 | 8 | 159 | 1 |
| 0.40 | 0.897 | 0.735 | 0.890 | 0.805 | 122 | 15 | 44 | 0 |
| 0.45 | 0.949 | 0.915 | 0.869 | 0.891 | 119 | 18 | 11 | 0 |
| **0.50** | **0.953** | **0.983** | **0.818** | **0.892 ← best** | 112 | 25 | 2 | 0 |
| 0.55 | 0.932 | 0.990 | 0.723 | 0.835 | 99 | 38 | 1 | 0 |
| 0.60 | 0.907 | 0.988 | 0.620 | 0.762 | 85 | 52 | 1 | 0 |
| 0.65 | 0.884 | 1.000 | 0.518 | 0.683 | 71 | 66 | 0 | 0 |
| 0.70 | 0.853 | 1.000 | 0.387 | 0.558 | 53 | 84 | 0 | 0 |
| 0.75 | 0.820 | 1.000 | 0.248 | 0.398 | 34 | 103 | 0 | 0 |
| 0.80 | 0.771 | 1.000 | 0.044 | 0.084 | 6 | 131 | 0 | 0 |
| 0.85 | 0.762 | 1.000 | 0.007 | 0.015 | 1 | 136 | 0 | 0 |

## Solo vs. group photos, at each strategy's best threshold

Group photos are consistently harder — more distractor faces per image means more chances for
a false match, and the gap is wider for SFace than for pipeline A:

| Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|
| TOP1 | 0.45 | 0.942 | 0.853 |
| **TOP2** | **0.45** | **0.948** | **0.857** |
| TOP3 | 0.40 | 0.942 | 0.832 |
| Centroid | 0.50 | 0.938 | 0.855 |

## Takeaways

1. **Threshold 0.45–0.50 is the operating range**, notably higher than pipeline A's 0.30–0.35 —
   SFace similarity scores sit higher across the board, so a ResNet-style threshold would let
   in hundreds of false identifications.
2. **Strategy choice matters here**, unlike in pipeline A. TOP2 is shipped as the best pooled
   F1; TOP3 is the weakest by ~1.7 points. Centroid is close behind TOP2 and is the
   precision-favouring choice (0.983 precision at 0.50, at the cost of recall).
3. **This model is threshold-sensitive.** Between 0.30 and 0.45 F1 nearly doubles. If you
   change the threshold, re-check it against your own data rather than nudging it by feel.
4. **Confusing one known person for another is rare and avoidable** — 0 occurrences at the
   recommended threshold.
5. **Expect group photos to score ~9 F1 points below solo photos** — a property of
   distractor-rich images, not something tuning fixes.

## Compared with pipeline A (ResNet50@WebFace600K)

| | Pipeline B (this one) | Pipeline A (ResNet50) |
|---|---|---|
| Best F1 | 0.898 | **0.960** |
| Threshold | 0.45 | 0.30 |
| Embedding size | 128-d | 512-d |
| Recognition model on disk | ~37 MB | ~166 MB |
| Sensitivity to strategy | noticeable (0.881–0.898) | negligible (within 0.0006) |

**Pipeline A is the accuracy recommendation** — ~6 F1 points ahead and far less sensitive to
configuration. Choose pipeline B when the smaller, faster recognizer matters more than those
6 points. Note that both pipelines download the same `buffalo_l` pack for detection, so
pipeline B's disk saving is in the recognition model, not the total download.
