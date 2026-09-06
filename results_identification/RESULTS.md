# Face Identification: Experiment Results

## Setup

- **Reference database:** 234 people, loaded from `embeddings_manifest.csv` (not by scanning
  the embeddings folder, which also holds 174 stale/duplicate person folders).
- **Evaluation set:** 571 human-labelled faces from `real_data/` — **137 known** (belong to a
  person in the reference DB) and **434 unknown** (distractors). Split into two scopes:
  - `one_person` — images where the ground truth says exactly one person appears (129 faces: 60 known / 69 unknown).
  - `few_people` — images with multiple people (442 faces: 77 known / 365 unknown).
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

## Headline

**ResNet50@WebFace600K clearly beats SFace.** At its best operating point ResNet reaches
**F1 ≈ 0.96** (threshold 0.30–0.35, all four strategies within 0.0006 of each other — strategy
choice barely matters for this model). SFace tops out around **F1 ≈ 0.89–0.90** (threshold
0.40–0.50, depending on strategy). `wrong_identity` (misattributing a known face to a different
known person) is **0 at every best threshold for both models** — it only shows up when the
threshold is set too low and the system is over-eager to assign an identity.

| Model | Strategy | Best threshold | F1 | Precision | Recall | Accuracy | wrong_identity |
|---|---|---|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.35 | **0.959** | 0.977 | 0.942 | 0.981 | 0 |
| ResNet50@WebFace600K | TOP2 | 0.30 | **0.960** | 0.963 | 0.956 | 0.981 | 0 |
| ResNet50@WebFace600K | TOP3 | 0.30 | **0.959** | 0.970 | 0.949 | 0.981 | 0 |
| ResNet50@WebFace600K | Centroid | 0.30 | **0.959** | 0.970 | 0.949 | 0.981 | 0 |
| SFace | TOP1 | 0.45 | 0.894 | 0.929 | 0.861 | 0.951 | 0 |
| SFace | TOP2 | 0.45 | 0.898 | 0.966 | 0.839 | 0.955 | 0 |
| SFace | TOP3 | 0.40 | 0.882 | 0.895 | 0.869 | 0.944 | 0 |
| SFace | Centroid | 0.50 | **0.892** | 0.983 | 0.818 | 0.953 | 0 |

## How each configuration behaved across the threshold sweep

Every model/strategy pair follows the same general shape: at **low thresholds** the system is
too permissive — nearly every known face is found (recall near 1.0), but so are hundreds of
unknown faces, dragging precision toward the "known:unknown" base rate (≈0.24, matching
137/571) and F1 down to ~0.4. As the threshold rises, precision climbs quickly while recall
holds — this is the useful operating range where F1 peaks. Past the peak, the system becomes
**too conservative**: it starts rejecting real matches as `unknown` (recall collapses), and by
threshold 0.80–0.85 it identifies almost nobody (F1 → 0). SFace needs a noticeably higher
threshold than ResNet to reach its equivalent operating point, and its curve is choppier —
it takes until 0.40 before it stabilizes, versus 0.20–0.25 for ResNet.

### ResNet50@WebFace600K — TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### ResNet50@WebFace600K — TOP2 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.427 | 0.291 | 0.978 | 0.448 | 134 | 0 | 324 | 3 |
| 0.20 | 0.820 | 0.570 | 0.978 | 0.720 | 134 | 2 | 100 | 1 |
| 0.25 | 0.953 | 0.856 | 0.956 | 0.903 | 131 | 5 | 21 | 1 |
| **0.30** | **0.981** | **0.963** | **0.956** | **0.960 ← best** | 131 | 6 | 5 | 0 |
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

### ResNet50@WebFace600K — TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### ResNet50@WebFace600K — Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — TOP1 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — TOP2 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.236 | 0.235 | 0.978 | 0.379 | 134 | 0 | 433 | 3 |
| 0.20 | 0.245 | 0.237 | 0.978 | 0.382 | 134 | 0 | 428 | 3 |
| 0.25 | 0.329 | 0.259 | 0.978 | 0.410 | 134 | 0 | 380 | 3 |
| 0.30 | 0.501 | 0.319 | 0.964 | 0.479 | 132 | 3 | 280 | 2 |
| 0.35 | 0.748 | 0.487 | 0.942 | 0.642 | 129 | 8 | 136 | 0 |
| 0.40 | 0.926 | 0.823 | 0.883 | 0.852 | 121 | 16 | 26 | 0 |
| **0.45** | **0.955** | **0.966** | **0.839** | **0.898 ← best** | 115 | 22 | 4 | 0 |
| 0.50 | 0.937 | 0.981 | 0.752 | 0.851 | 103 | 34 | 2 | 0 |
| 0.55 | 0.916 | 0.989 | 0.657 | 0.789 | 90 | 47 | 1 | 0 |
| 0.60 | 0.886 | 0.987 | 0.533 | 0.692 | 73 | 64 | 1 | 0 |
| 0.65 | 0.858 | 1.000 | 0.409 | 0.580 | 56 | 81 | 0 | 0 |
| 0.70 | 0.816 | 1.000 | 0.234 | 0.379 | 32 | 105 | 0 | 0 |
| 0.75 | 0.778 | 1.000 | 0.073 | 0.136 | 10 | 127 | 0 | 0 |
| 0.80 | 0.762 | 1.000 | 0.007 | 0.015 | 1 | 136 | 0 | 0 |
| 0.85 | 0.760 | 0.000 | 0.000 | 0.000 | 0 | 137 | 0 | 0 |

### SFace — TOP3 (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

### SFace — Centroid (pooled)

| Threshold | Accuracy | Precision | Recall | F1 | known→known correct | known→unknown | unknown→known | wrong_identity |
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

## Single-person vs. multi-person photos, at each config's best threshold

The `few_people` scope (crowd/group photos, 442 of the 571 faces) is consistently harder than
`one_person` (129 faces) — more distractor faces per image means more chances for a false
match. At every model/strategy's best pooled threshold, `one_person` F1 is 0.94–0.98 while
`few_people` sits 4–13 points lower:

| Model | Strategy | Threshold | one_person F1 | few_people F1 |
|---|---|---|---|---|
| ResNet50@WebFace600K | TOP1 | 0.35 | 0.983 | 0.940 |
| ResNet50@WebFace600K | TOP2 | 0.30 | 0.983 | 0.941 |
| ResNet50@WebFace600K | TOP3 | 0.30 | 0.983 | 0.940 |
| ResNet50@WebFace600K | Centroid | 0.30 | 0.983 | 0.940 |
| SFace | TOP1 | 0.45 | 0.942 | 0.853 |
| SFace | TOP2 | 0.45 | 0.948 | 0.857 |
| SFace | TOP3 | 0.40 | 0.942 | 0.832 |
| SFace | Centroid | 0.50 | 0.938 | 0.855 |

## Takeaways

1. **Use ResNet50@WebFace600K, not SFace.** It's ~6–8 F1 points ahead at its best operating
   point, and its performance is far less sensitive to the exact strategy chosen (TOP1/TOP2/TOP3/
   Centroid all land within 0.0006 F1 of each other at 0.30).
2. **Recommended threshold: 0.30–0.35 for ResNet**, ~0.45–0.50 for SFace if it must be used.
   Going below these ranges lets in a flood of false identifications (precision collapses);
   going above them starts silently rejecting real matches (recall collapses).
3. **Strategy choice matters much less than threshold choice** for ResNet — any of TOP2/TOP3/
   Centroid at 0.30 is effectively equivalent, so the simplest one (TOP1, or Centroid for
   speed) is a reasonable default.
4. **Misidentifying a known person as a different known person is rare and avoidable** —
   `wrong_identity` is 0 for both models at every recommended threshold; it only appears when
   the threshold is set well below the optimum.
5. **Expect group photos to underperform solo photos** by several F1 points regardless of
   model — this is a property of the harder distractor-rich images, not something a threshold
   change fixes.

---
*Generated from `results_identification/identification_metrics.csv` (571 labelled faces: 137
known / 434 unknown, matched against the full 234-person reference database). See
`face_identification/evaluate_identification.py` for the scoring implementation.*
