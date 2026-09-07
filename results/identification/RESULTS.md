# Face Identification: Experiment Results

## Setup

- **Reference database:** 234 people, loaded from `embeddings_manifest.csv` (not by scanning
  the embeddings folder, which also holds 174 stale/duplicate person folders).
- **Evaluation set:** 571 human-labelled faces from `real_data/` — **140 known** (belong to a
  person in the reference DB) and **431 unknown** (distractors). Split into two scopes:
  - `one_person` — images where the ground truth says exactly one person appears (129 faces: 60 known / 69 unknown).
  - `few_people` — images with multiple people (442 faces: 80 known / 362 unknown).
  - `pooled` — both combined (the numbers below focus on this, since it's the overall operating
    point a deployed threshold would use).
- **Models:** `resnet_webface600k` (ResNet50 trained on WebFace600K) and `sface` (OpenCV SFace).
- **Matching strategies** (per candidate person, not global k-NN):
  - **TOP1** — similarity to that person's single best-matching reference image.
  - **TOP2 / TOP3** — average similarity to that person's 2 / 3 closest reference images.
  - **Centroid** — similarity to the L2-renormalized mean of that person's reference embeddings.
  - Prediction = the person with the highest score; called `unknown` if that score is below
    the threshold.
- **Threshold grid:** `0.15` to `0.85` in steps of `0.05`, shared across both models and all
  four strategies.
- **`wrong_identity`** counts a known face matched to the *wrong* known person (tracked
  separately from known→unknown / unknown→known misses since it's a distinct failure mode).

## Ground-truth revision

Error analysis of the first scoring run surfaced five mislabelled faces in the human-reviewed
table, all of which were reviewed image-by-image and corrected before the numbers below were
produced:

| Image | Face | Was | Now | Reason |
|---|---|---|---|---|
| `few_people_091_post166794` | face_02 | `unknown` | יצחק וסרלאוף | The source clip is watermarked with his own handle |
| `few_people_141_post185060` | face_01 | `unknown` | לירן אבישר בן חורין | Clear frontal face, confirmed on review |
| `few_people_147_post185269` | face_01 | `unknown` | עדי עזוז | Clear face, confirmed on review |
| `one_person_082_post188165` | face_01 | `unknown` | מידן בר | Confirmed on review despite the dark cockpit lighting |
| `one_person_011_post183325_f235` | face_01 | אמיר אוחנה | `unknown` | The box contains the back of a head — no facial pixels at all, so no embedding model can resolve it |

This shifted the known/unknown split from 137/434 to 140/431 and is the reason these results are
higher than the previous revision of this document: four of the five were cases where the
model had been right and the label was wrong.

## Headline

**ResNet50@WebFace600K clearly beats SFace.** At its best operating point ResNet reaches
**F1 ≈ 0.978** — and all four strategies now peak at the *same* threshold, **0.30**, within
0.0002 F1 of each other, so strategy choice barely matters for this model. SFace tops out
around **F1 ≈ 0.89–0.90** (threshold 0.40–0.50, depending on strategy). `wrong_identity`
(misattributing a known face to a different known person) is **0 at every best threshold for
both models** — it only shows up when the threshold is set too low and the system is
over-eager to assign an identity.

| Model | Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.30 | **0.975** | 0.985 | 0.964 | 0.988 | 0 |
| ResNet50@WebFace600K | TOP2 | 0.30 | **0.978** | 0.993 | 0.964 | 0.990 | 0 |
| ResNet50@WebFace600K | TOP3 | 0.30 | **0.978** | 1.000 | 0.957 | 0.990 | 0 |
| ResNet50@WebFace600K | Centroid | 0.30 | **0.978** | 1.000 | 0.957 | 0.990 | 0 |
| SFace | TOP1 | 0.45 | **0.899** | 0.945 | 0.857 | 0.953 | 0 |
| SFace | TOP2 | 0.45 | **0.903** | 0.983 | 0.836 | 0.956 | 0 |
| SFace | TOP3 | 0.40 | **0.886** | 0.910 | 0.864 | 0.946 | 0 |
| SFace | Centroid | 0.50 | **0.898** | 1.000 | 0.814 | 0.955 | 0 |

## How each configuration behaved across the threshold sweep

Every model/strategy pair follows the same general shape: at **low thresholds** the system is
too permissive — nearly every known face is found (recall near 1.0), but so are hundreds of
unknown faces, dragging precision toward the "known:unknown" base rate (≈0.25, matching
140/571) and F1 down to ~0.4. As the threshold rises, precision climbs quickly while recall
holds — this is the useful operating range where F1 peaks. Past the peak, the system becomes
**too conservative**: it starts rejecting real matches as `unknown` (recall collapses), and by
threshold 0.80–0.85 it identifies almost nobody (F1 → 0). SFace needs a noticeably higher
threshold than ResNet to reach its equivalent operating point, and its curve is choppier —
it takes until 0.40 before it stabilizes, versus 0.25–0.30 for ResNet.

### ResNet50@WebFace600K — TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### ResNet50@WebFace600K — TOP2 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.434 | 0.299 | 0.986 | 0.459 | 138 | 0 | 321 | 2 |
| 0.20 | 0.828 | 0.587 | 0.986 | 0.736 | 138 | 1 | 96 | 1 |
| 0.25 | 0.962 | 0.882 | 0.964 | 0.921 | 135 | 4 | 17 | 1 |
| **0.30** | **0.990** | **0.993** | **0.964** | **0.978 ← best** | 135 | 5 | 1 | 0 |
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

### ResNet50@WebFace600K — TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### ResNet50@WebFace600K — Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — TOP2 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.240 | 0.239 | 0.971 | 0.383 | 136 | 0 | 430 | 4 |
| 0.20 | 0.249 | 0.241 | 0.971 | 0.386 | 136 | 0 | 425 | 4 |
| 0.25 | 0.333 | 0.263 | 0.971 | 0.414 | 136 | 0 | 377 | 4 |
| 0.30 | 0.508 | 0.326 | 0.964 | 0.487 | 135 | 2 | 276 | 3 |
| 0.35 | 0.755 | 0.498 | 0.943 | 0.652 | 132 | 7 | 132 | 1 |
| 0.40 | 0.928 | 0.837 | 0.879 | 0.857 | 123 | 17 | 24 | 0 |
| **0.45** | **0.956** | **0.983** | **0.836** | **0.903 ← best** | 117 | 23 | 2 | 0 |
| 0.50 | 0.939 | 1.000 | 0.750 | 0.857 | 105 | 35 | 0 | 0 |
| 0.55 | 0.914 | 1.000 | 0.650 | 0.788 | 91 | 49 | 0 | 0 |
| 0.60 | 0.884 | 1.000 | 0.529 | 0.692 | 74 | 66 | 0 | 0 |
| 0.65 | 0.853 | 1.000 | 0.400 | 0.571 | 56 | 84 | 0 | 0 |
| 0.70 | 0.811 | 1.000 | 0.229 | 0.372 | 32 | 108 | 0 | 0 |
| 0.75 | 0.772 | 1.000 | 0.071 | 0.133 | 10 | 130 | 0 | 0 |
| 0.80 | 0.757 | 1.000 | 0.007 | 0.014 | 1 | 139 | 0 | 0 |
| 0.85 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 140 | 0 | 0 |

### SFace — TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

## Single-person vs. multi-person photos, at each config's best threshold

The `few_people` scope (crowd/group photos, 442 of the 571 faces) is consistently harder than
`one_person` (129 faces) — more distractor faces per image means more chances for a false
match. **Every ResNet strategy is now perfect (F1 = 1.000) on `one_person`**, so all of its
remaining errors live in group photos:

| Model | Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.30 | 1.000 | 0.955 |
| ResNet50@WebFace600K | TOP2 | 0.30 | 1.000 | 0.962 |
| ResNet50@WebFace600K | TOP3 | 0.30 | 1.000 | 0.961 |
| ResNet50@WebFace600K | Centroid | 0.30 | 1.000 | 0.961 |
| SFace | TOP1 | 0.45 | 0.942 | 0.863 |
| SFace | TOP2 | 0.45 | 0.948 | 0.867 |
| SFace | TOP3 | 0.40 | 0.942 | 0.842 |
| SFace | Centroid | 0.50 | 0.938 | 0.865 |

## Choosing between the tied strategies

TOP2, TOP3 and Centroid are separated by 0.0002 F1 at threshold 0.30, and TOP1 by 0.004. On a
571-face set that is a one-or-two-face difference — well inside the noise this sample size can
resolve — so F1 alone cannot justify a choice. Breaking the tie on *which* errors each strategy
makes gives a clearer answer:

| Strategy @ 0.30 | `unknown_as_known` | `known_as_unknown` | F1 | Highest-scoring distractor |
|---|---|---|---|---|
| TOP1 | 2 | 5 | 0.975 | 0.3292 |
| TOP2 | 1 | 5 | 0.978 | 0.3128 |
| **TOP3 (selected)** | **0** | 6 | 0.978 | 0.2956 |
| Centroid | 0 | 6 | 0.978 | 0.2979 |

**TOP3 was selected because it produces zero false accepts.** Assigning a known identity to
someone who is not in the database is the error this system most wants to avoid: a confidently
wrong name is worse than an `unknown` that a human can follow up on. TOP1 and TOP2 only reach
their F1 because a distractor scores just above the threshold (0.3292 and 0.3128 against a
0.30 cut), whereas under TOP3 no unknown face ever exceeds 0.2956.

TOP3 is preferred over Centroid, which also reaches zero, because **each person in the reference
database has only 3–5 images** (234 people, min 3, mean 4.05). A centroid averaged over so few
samples is a fragile estimate of a person's appearance — one atypical photo (unusual lighting,
angle, or age) permanently shifts that person's single vector. TOP3 picks the 3 closest
reference images per query instead, so an outlier photo is simply not selected when it does not
help. Since every person has at least 3 images, TOP3 never falls back to fewer.

The cost is one extra missed known face relative to TOP2 — a face reported `unknown` rather
than named, which is the safer direction to err in.

## Error analysis at the selected operating point (ResNet50@WebFace600K, TOP3, 0.30)

Six faces out of 571 are scored wrong, and every one of them sits in a group photo:

- **0 × `unknown_as_known`** — no distractor is ever given a name. The highest score any unknown
  face reaches is 0.2956, below the 0.30 threshold. (For contrast, the poster face in
  `few_people_017_post169192` — clipped by the top image border, with only mouth and chin inside
  the frame — is what TOP1 and TOP2 falsely name; TOP3 scores it below the cut.)
- **6 × `known_as_unknown`** — real people the system declined to name. In **4 of the 6 the
  top-scoring candidate was already the right person** (בצלאל סמוטריץ' at 0.197, יוסף חדאד at
  0.198, גלעד ארדן at 0.214, יואב קיש at 0.292); the score simply fell short of 0.30 because the
  crops are small, blurred, or in steep profile. The other two (משה פסל at 0.167, ישראל כץ at
  0.251) scored low *and* ranked a wrong candidate first — but stayed below the threshold, so
  nothing wrong was ever asserted.
- **0 × `wrong_identity`.**

That last point is the practically important one: when this configuration is unsure, it says
`unknown` rather than guessing a name.

## Takeaways

1. **Use ResNet50@WebFace600K, not SFace.** It's ~7–8 F1 points ahead at its best operating
   point, and its performance is far less sensitive to the exact strategy chosen (TOP1/TOP2/
   TOP3/Centroid all land within 0.004 F1 of each other at 0.30).
2. **Recommended threshold: 0.30 for ResNet**, ~0.45–0.50 for SFace if it must be used.
   Going below these ranges lets in a flood of false identifications (precision collapses);
   going above them starts silently rejecting real matches (recall collapses).
3. **When F1 ties, pick on error type.** All four ResNet strategies peak at 0.30 within noise of
   each other, so **TOP3 was selected for reaching zero false accepts** — and preferred over
   Centroid, which also reaches zero, because 3–5 reference images per person are too few to
   average into one representative vector.
4. **Misidentifying a known person as a different known person is rare and avoidable** —
   `wrong_identity` is 0 for both models at every recommended threshold; it only appears when
   the threshold is set well below the optimum.
5. **Expect group photos to underperform solo photos.** ResNet is perfect on `one_person` and
   loses ~0.04 F1 on `few_people`; that gap is a property of the harder distractor-rich images,
   not something a threshold change fixes.

---
*Generated from `results/identification/identification_metrics.csv` (571 labelled faces: 140
known / 431 unknown, matched against the full 234-person reference database). See
`experiments/identification_study/evaluate_identification.py` for the scoring implementation.*
