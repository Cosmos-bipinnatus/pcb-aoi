# HRIPCB — EDA Summary (Stage 0.4)

Dataset: **HRIPCB** base set (Open Lab on Human–Robot Interaction, Peking
University; Huang & Wei, arXiv:1901.08204, 2019). Pre-rotated augmentation set
**excluded** — augmentation will be done dynamically with `torchvision`.

Source notebook: [`notebooks/00_data_exploration.ipynb`](../notebooks/00_data_exploration.ipynb).
Figures: [`results/figures/`](figures/).

---

## 1. Headline numbers

| metric | value |
|---|---|
| Annotated images | **693** |
| Defect bounding boxes | **2953** |
| Defect classes | 6 |
| Boxes per image | min 1 · **mean 4.26** · max 6 |
| Distinct image resolutions | 10 (one per underlying template) |
| Defect-free templates (`PCB_USED/`) | 10 |
| **Data-quality issues found** | **0** |

---

## 2. Class balance — *excellent*

| class | 中文 | images | boxes |
|---|---|---|---|
| missing_hole | 缺孔 | 115 | 497 |
| mouse_bite | 鼠咬 | 115 | 492 |
| open_circuit | 斷路 | 116 | 482 |
| short | 短路 | 116 | 491 |
| spur | 毛刺 | 115 | 488 |
| spurious_copper | 多餘銅 | 116 | 503 |

Images: **115–116** per class. Boxes: **482–503** per class. This is one of the
most balanced detection datasets you'll find — class-imbalance handling
(reweighting, focal loss, resampling) is **not needed**. All classes are far
above the 50-image floor.

→ Figure: `00_class_distribution.png`

---

## 3. Defect size — *small, and class-dependent*

Bounding-box area in px² (min / median / max):

| class | min | median | max |
|---|---|---|---|
| missing_hole | 1 681 | 4 092 | 19 460 |
| mouse_bite | 800 | 3 723 | 9 516 |
| open_circuit | **625** | **2 434** | 8 924 |
| short | 1 254 | **6 084** | **37 268** |
| spur | 990 | 4 893 | 15 708 |
| spurious_copper | 1 400 | 5 198 | 16 241 |

Images are multi-megapixel (≈ 2240×2016 up to 3056×2464), so a median defect of
~4 000 px² occupies well under **0.1%** of the frame. Key implications:

- **`open_circuit`** is the smallest defect (median 2 434 px², min 625) → hardest
  to detect; most sensitive to downscaling and blur.
- **`short`** is the largest and most variable (median 6 084, max 37 268) → the
  widest size range, easiest to localise but most variable in appearance.
- Aspect ratios spread widely (`mouse_bite`/`spur` are thin slivers) — see
  `00_bbox_size_distribution.png` (width-vs-height scatter).

→ Figures: `00_bbox_size_distribution.png`, `00_boxes_per_image_hist.png`

---

## 4. Data quality — *clean*

All 693 images decode; every annotation parses as valid Pascal-VOC.

| check | result |
|---|---|
| Unreadable images | 0 |
| Missing image file for an XML | 0 |
| XML size ≠ decoded image size | 0 |
| Out-of-bounds boxes | 0 |
| Degenerate boxes (w/h ≤ 0) | 0 |
| Duplicate images (content hash) | 0 |

No cleaning step is required before modelling.

---

## 5. Template structure (relevant to Stage 1)

`PCB_USED/` holds **10 defect-free reference boards** (ids 01, 04–12). Every
defective image filename encodes its template id (`01_missing_hole_03.jpg`), so
each defect image has a **pixel-aligned clean reference**. This is the strongest
structural fact in the dataset for classical CV.

→ Figure: `00_pcb_templates.png` (+ template→class mapping table in the notebook)

---

## 6. Implications for Stage 1 (classical-CV baseline)

1. **A reference-vs-test difference baseline is viable and well-supported.**
   Because every defect image maps to a clean template, a registration +
   image-differencing + morphology pipeline is the natural first baseline (rather
   than pure template matching or single-image morphology). The main engineering
   risk is **alignment**: templates and defect shots are not guaranteed
   pixel-registered, so expect to need feature-based registration
   (ORB/SIFT + homography) before differencing.

2. **Expected difficulty ranking (easy → hard):**
   - *Easier:* `missing_hole` (round, high-contrast, regular) and `short`
     (large, blob-like) — should pop out of a difference image.
   - *Medium:* `spurious_copper`, `spur` — moderate size, but `spur` slivers need
     careful morphology.
   - *Harder:* `mouse_bite` (small edge nibble, low contrast against track edge)
     and especially **`open_circuit`** (smallest, a thin gap) — most likely to be
     missed or confused by a difference baseline.

3. **Resolution matters.** Defects are tiny fractions of multi-MP frames. Any
   downscaling for speed will disproportionately hurt `open_circuit`/`mouse_bite`
   recall — keep full resolution (or tile) in the classical pipeline.

4. **Multiple defects per image (mean 4.26).** The baseline must return *all*
   defect regions per image, not one — connected-component extraction on the
   difference mask, then per-component classification/scoring.

5. **No class-imbalance work needed** at any stage given the balance above.
