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
cd pipeline_B_sface
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
  this model the choice does matter (0.886–0.903 F1); TOP2 is the best of them.
- `MIN_DETECTION_CONFIDENCE` (0.5) — how confident the detector must be that a region is a face.

---

# Experiment results — OpenCV SFace

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

**Best F1 ≈ 0.89–0.90, at threshold 0.40–0.50 depending on strategy.** Unlike pipeline A, the
strategy does matter here — TOP3 is nearly 2 F1 points behind TOP2. `wrong_identity` is
**0 at every strategy's best threshold**; misattributing one known person to another only
happens when the threshold is set well below the optimum.

| Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|
| TOP1 | 0.45 | 0.899 | 0.945 | 0.857 | 0.953 | 0 |
| **TOP2 (this pipeline)** | **0.45** | **0.903** | 0.983 | 0.836 | 0.956 | 0 |
| TOP3 | 0.40 | 0.886 | 0.910 | 0.864 | 0.946 | 0 |
| Centroid | 0.50 | 0.898 | 1.000 | 0.814 | 0.955 | 0 |

Every configuration follows the same shape across the sweep: at **low thresholds** the system
is too permissive — nearly every known face is found (recall ≈ 0.96) but so are essentially
all 431 unknown faces, pinning precision at the known:unknown base rate (≈0.24) and F1 near
0.38. As the threshold rises precision climbs while recall holds — the useful range where F1
peaks. Past the peak it becomes **too conservative**, rejecting real matches as `unknown`
(recall collapses), and by 0.80–0.85 it identifies almost nobody. **SFace's curve is choppier
than pipeline A's and stabilizes much later** — it takes until threshold 0.40 before precision
becomes usable, versus 0.25–0.30 for ResNet, which is why its operating threshold is higher.

## Full threshold sweep

### TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.235 | 0.235 | 0.957 | 0.377 | 134 | 0 | 431 | 6 |
| 0.20 | 0.236 | 0.235 | 0.957 | 0.378 | 134 | 0 | 430 | 6 |
| 0.25 | 0.277 | 0.245 | 0.957 | 0.390 | 134 | 0 | 407 | 6 |
| 0.30 | 0.403 | 0.283 | 0.957 | 0.436 | 134 | 1 | 335 | 5 |
| 0.35 | 0.615 | 0.377 | 0.929 | 0.536 | 130 | 5 | 210 | 5 |
| 0.40 | 0.862 | 0.648 | 0.907 | 0.756 | 127 | 10 | 66 | 3 |
| **0.45** | **0.953** | **0.945** | **0.857** | **0.899 ← best** | 120 | 20 | 7 | 0 |
| 0.50 | 0.944 | 1.000 | 0.771 | 0.871 | 108 | 32 | 0 | 0 |
| 0.55 | 0.925 | 1.000 | 0.693 | 0.819 | 97 | 43 | 0 | 0 |
| 0.60 | 0.897 | 1.000 | 0.579 | 0.733 | 81 | 59 | 0 | 0 |
| 0.65 | 0.867 | 1.000 | 0.457 | 0.627 | 64 | 76 | 0 | 0 |
| 0.70 | 0.832 | 1.000 | 0.314 | 0.478 | 44 | 96 | 0 | 0 |
| 0.75 | 0.790 | 1.000 | 0.143 | 0.250 | 20 | 120 | 0 | 0 |
| 0.80 | 0.762 | 1.000 | 0.029 | 0.056 | 4 | 136 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### TOP2 (pooled) — the strategy this pipeline ships

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.240 | 0.239 | 0.971 | 0.383 | 136 | 0 | 430 | 4 |
| 0.20 | 0.249 | 0.241 | 0.971 | 0.386 | 136 | 0 | 425 | 4 |
| 0.25 | 0.333 | 0.263 | 0.971 | 0.414 | 136 | 0 | 377 | 4 |
| 0.30 | 0.508 | 0.326 | 0.964 | 0.487 | 135 | 2 | 276 | 3 |
| 0.35 | 0.755 | 0.498 | 0.943 | 0.652 | 132 | 7 | 132 | 1 |
| 0.40 | 0.928 | 0.837 | 0.879 | 0.857 | 123 | 17 | 24 | 0 |
| **0.45** | **0.956** | **0.983** | **0.836** | **0.903 ← best (shipped)** | 117 | 23 | 2 | 0 |
| 0.50 | 0.939 | 1.000 | 0.750 | 0.857 | 105 | 35 | 0 | 0 |
| 0.55 | 0.914 | 1.000 | 0.650 | 0.788 | 91 | 49 | 0 | 0 |
| 0.60 | 0.884 | 1.000 | 0.529 | 0.692 | 74 | 66 | 0 | 0 |
| 0.65 | 0.853 | 1.000 | 0.400 | 0.571 | 56 | 84 | 0 | 0 |
| 0.70 | 0.811 | 1.000 | 0.229 | 0.372 | 32 | 108 | 0 | 0 |
| 0.75 | 0.772 | 1.000 | 0.071 | 0.133 | 10 | 130 | 0 | 0 |
| 0.80 | 0.757 | 1.000 | 0.007 | 0.014 | 1 | 139 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.238 | 0.235 | 0.957 | 0.378 | 134 | 0 | 429 | 6 |
| 0.20 | 0.272 | 0.244 | 0.957 | 0.388 | 134 | 0 | 410 | 6 |
| 0.25 | 0.380 | 0.275 | 0.957 | 0.427 | 134 | 0 | 348 | 6 |
| 0.30 | 0.585 | 0.363 | 0.950 | 0.526 | 133 | 4 | 230 | 3 |
| 0.35 | 0.844 | 0.621 | 0.914 | 0.740 | 128 | 11 | 77 | 1 |
| **0.40** | **0.946** | **0.910** | **0.864** | **0.886 ← best** | 121 | 19 | 12 | 0 |
| 0.45 | 0.948 | 0.991 | 0.793 | 0.881 | 111 | 29 | 1 | 0 |
| 0.50 | 0.923 | 1.000 | 0.686 | 0.814 | 96 | 44 | 0 | 0 |
| 0.55 | 0.902 | 1.000 | 0.600 | 0.750 | 84 | 56 | 0 | 0 |
| 0.60 | 0.870 | 1.000 | 0.471 | 0.641 | 66 | 74 | 0 | 0 |
| 0.65 | 0.839 | 1.000 | 0.343 | 0.511 | 48 | 92 | 0 | 0 |
| 0.70 | 0.792 | 1.000 | 0.150 | 0.261 | 21 | 119 | 0 | 0 |
| 0.75 | 0.762 | 1.000 | 0.029 | 0.056 | 4 | 136 | 0 | 0 |
| 0.80 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.240 | 0.237 | 0.964 | 0.381 | 135 | 0 | 429 | 5 |
| 0.20 | 0.257 | 0.241 | 0.964 | 0.386 | 135 | 0 | 419 | 5 |
| 0.25 | 0.342 | 0.264 | 0.964 | 0.415 | 135 | 0 | 371 | 5 |
| 0.30 | 0.496 | 0.319 | 0.957 | 0.479 | 134 | 2 | 282 | 4 |
| 0.35 | 0.713 | 0.455 | 0.936 | 0.612 | 131 | 7 | 155 | 2 |
| 0.40 | 0.898 | 0.747 | 0.886 | 0.810 | 124 | 16 | 42 | 0 |
| 0.45 | 0.951 | 0.931 | 0.864 | 0.896 | 121 | 19 | 9 | 0 |
| **0.50** | **0.955** | **1.000** | **0.814** | **0.898 ← best** | 114 | 26 | 0 | 0 |
| 0.55 | 0.930 | 1.000 | 0.714 | 0.833 | 100 | 40 | 0 | 0 |
| 0.60 | 0.905 | 1.000 | 0.614 | 0.761 | 86 | 54 | 0 | 0 |
| 0.65 | 0.879 | 1.000 | 0.507 | 0.673 | 71 | 69 | 0 | 0 |
| 0.70 | 0.848 | 1.000 | 0.379 | 0.549 | 53 | 87 | 0 | 0 |
| 0.75 | 0.814 | 1.000 | 0.243 | 0.391 | 34 | 106 | 0 | 0 |
| 0.80 | 0.765 | 1.000 | 0.043 | 0.082 | 6 | 134 | 0 | 0 |
| 0.85 | 0.757 | 1.000 | 0.007 | 0.014 | 1 | 139 | 0 | 0 |

## Solo vs. group photos, at each strategy's best threshold

Group photos are consistently harder — more distractor faces per image means more chances for
a false match:

| Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|
| TOP1 | 0.45 | 0.942 | 0.863 |
| **TOP2** | **0.45** | **0.948** | **0.867** |
| TOP3 | 0.40 | 0.942 | 0.842 |
| Centroid | 0.50 | 0.938 | 0.865 |

## Takeaways

1. **Threshold 0.45 is the operating point for TOP2.** Below 0.40 precision falls apart fast;
   above 0.50 recall drops without buying precision, which is already at 1.000.
2. **Strategy choice does matter here** — 0.886 (TOP3) to 0.903 (TOP2), and each strategy
   peaks at a different threshold (0.40–0.50). Changing one means re-tuning the other.
3. **Confusing one known person for another is rare and avoidable** — 0 occurrences at the
   recommended threshold, though SFace does produce a few at thresholds below 0.40, where
   pipeline A produces almost none.
4. **Expect group photos to score ~8 F1 points below solo photos** — a wider gap than
   pipeline A's, since SFace's weaker embeddings are hit harder by distractor-rich images.

## Compared with pipeline A (ResNet50@WebFace600K)

| | Pipeline B (this one) | Pipeline A (ResNet50) |
|---|---|---|
| Best F1 | 0.903 | **0.978** |
| Threshold | 0.45 | 0.30 |
| Embedding size | 128-d | 512-d |
| Recognition model on disk | ~37 MB | ~166 MB |
| Sensitivity to strategy | noticeable (0.886–0.903) | negligible (within 0.004) |

**Pipeline A is the accuracy recommendation** — ~7 F1 points ahead and far less sensitive to
configuration. This pipeline is the choice when the smaller, faster recognizer matters more
than those 7 points. Both pipelines download the same `buffalo_l` pack for detection, so the
saving here is in the recognition model, not the total download.
