# Stage 1 — Classical-CV Defect Baseline (HRIPCB)

Reference-vs-test pipeline on all **693** HRIPCB images. Code:
[`src/traditional_cv.py`](../src/traditional_cv.py),
[`src/evaluate_traditional.py`](../src/evaluate_traditional.py). Notebook:
[`notebooks/01_traditional_baseline.ipynb`](../notebooks/01_traditional_baseline.ipynb).
Numbers below are from `results/traditional_eval_summary.json`
(regenerate: `python -m src.evaluate_traditional`).

## 1. Pipeline & design rationale

```
register_to_template  ->  compute_difference  ->  threshold_and_clean  ->  classify_components
   ORB + RANSAC            max |Δ| over BGR        binarise + morph +        connected comps +
   homography              (colour, not gray)      area filter               nearest-prototype
```

Why this design, and the two findings that shaped it:

1. **Reference differencing, not single-image morphology.** Every defect image
   derives from one of ten clean `PCB_USED` templates *of identical resolution*,
   giving a near-pixel-aligned clean reference — exactly the structure a
   difference baseline needs. ORB+RANSAC registration is essentially perfect:
   **mean inlier ratio 0.99** (min 0.94), so misalignment is *not* a bottleneck.

2. **Colour difference, not grayscale** *(decisive)*. PCB defects are mostly
   copper(orange)-vs-substrate(green) changes. Measured inside ground-truth
   boxes, the grayscale diff is near-zero for the thin classes (`short` p95 ≈ 17,
   `spur` ≈ 17) while the max-over-BGR colour diff lifts the usable signal
   (peaks 35–55). Switching the diff to colour took mAP from **10% → 27%**.

3. **Tight mask for shape, padded box for IoU** *(decisive)*. The diff blob is
   much smaller than the generous human VOC box (a missing-hole core ≈ 700 px²
   vs a ≈ 4 000 px² annotation). Dilating the *mask* to fix IoU destroyed the
   shape cues and collapsed every class to `missing_hole`. The fix: measure
   features on the tight component, expand only the *reported* box by a fixed
   margin. This restored both localisation and classification.

4. **Nearest-prototype classifier, not hand-tuned if/else.** Each class has an
   empirical prototype = median of (log area, aspect, circularity, extent) over
   true-defect components; a component takes the nearest prototype under a
   scale-normalised distance. Extensible (add a class = add a row) and
   data-grounded. The discriminative axes turned out to be **area + aspect +
   circularity** (the Euler-number/holes cue never fired — diff cores are solid).

## 2. Results — all 693 images

**mAP@0.5 = 27.4%**

| class | P | R | F1 | AP@0.5 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| missing_hole | 0.50 | 0.96 | 0.66 | **0.90** | 479 | 487 | 18 |
| spurious_copper | 0.40 | 0.85 | 0.54 | **0.50** | 425 | 652 | 78 |
| short | 0.30 | 0.31 | 0.30 | 0.16 | 152 | 359 | 339 |
| mouse_bite | 0.24 | 0.13 | 0.17 | 0.05 | 66 | 211 | 426 |
| open_circuit | 0.28 | 0.06 | 0.11 | 0.02 | 31 | 80 | 451 |
| spur | 0.17 | 0.05 | 0.08 | 0.02 | 24 | 116 | 464 |

**Speed:** mean **0.79 s/image** (median 0.81, max 0.97) — comfortably real-time
for an offline baseline. **Registration:** mean inlier 0.991, none below 0.3.

## 3. What works, what doesn't, and why

**Works well**
- **Registration & localisation.** 0.99 inlier ratio; defect *regions* are found
  reliably across all classes once the colour diff is used.
- **`missing_hole`** (AP 0.90, recall 0.96): large, round, high-contrast filled
  hole — the cleanest signal. Matches the EDA prediction that it would be easiest.
- **`spurious_copper`** (AP 0.50, recall 0.85): large, elongated, low-circularity
  — a distinctive prototype, so detected reliably (precision suffers, see below).

**Doesn't work — fine-grained classification of small defects**
- **`mouse_bite` / `open_circuit` / `spur`** (AP 0.02–0.05). The defects *are*
  detected, but their shape features overlap almost completely (areas ≈ 80–130 px²,
  similar aspect/circularity), so they are misclassified. The confusion matrix
  shows `spur`→`spurious_copper` (333) and `mouse_bite`→`spurious_copper` (200):
  small elongated blobs get absorbed by the elongated `spurious_copper` prototype.
- **`short`** (AP 0.16): about half are missed (241 of 491 → background) — the
  copper bridge is thin and its colour diff is the weakest of all classes.
- **Precision is low overall** because two "absorber" prototypes (`missing_hole`,
  `spurious_copper`) collect misclassified detections from other classes plus
  ~460 pure-noise/registration-seam false positives.

**Root cause (verified):** not registration (0.99 inlier), not localisation, but
**class discrimination from hand-engineered shape features on tiny, near-identical
diff blobs.** A grid search over threshold / area-filter / feature weights did not
move mAP above ~27% — the overlap is intrinsic to the features, not a tuning miss.

## 4. Honest positioning vs. expectation

The pre-registered expectation was mAP 50–70%. This baseline lands at **27%**,
below that band. The gap is entirely the **fine-grained small-defect
classification** the brief itself flagged as hard (`mouse_bite`/`open_circuit`
expected ~50–65% precision) — and it is worse than hoped because a *single*
rule-based classifier cannot separate four small classes with overlapping shape
statistics. The two easy/medium classes already behave as expected
(`missing_hole` P≈0.50/R≈0.96/AP 0.90). Registration — the brief's suggested
failure mode for a low score — is ruled out.

## 5. Limitations & what Stage 2/3 should fix

1. **Learned features for small-defect discrimination.** The core failure is
   telling `mouse_bite`/`open_circuit`/`spur` apart. A **CNN classifier (Stage 2)**
   on cropped candidate regions should replace the prototype rule and lift the
   three floor classes — keep this pipeline's high-recall region proposals as the
   crop source.
2. **End-to-end detection (Stage 3, YOLO).** Removes the registration dependency
   entirely and learns localisation + classification jointly; expected to fix both
   the `short` misses and the precision problem.
3. **False-positive control.** ~460 seam/noise detections hurt precision; a YOLO
   detector or a learned objectness/region filter would suppress these.
4. **Keep what works.** Colour differencing and ORB registration are cheap,
   interpretable, and could still serve as a fast pre-filter or an explainability
   overlay alongside the learned models.
