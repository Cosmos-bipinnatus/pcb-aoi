# PCB-AOI: Defect Detection on HRIPCB with Traditional CV, Two-stage DL, and End-to-end YOLO

PCB Automated Optical Inspection on the HRIPCB dataset, implemented three ways —
a classical computer-vision baseline, a two-stage (region-proposal + CNN)
detector, and an end-to-end YOLO detector — with a unified evaluation and an
interactive demo.

## Overview

This project implements and compares three approaches to printed-circuit-board
(PCB) surface-defect detection, evaluates them on a single fixed test split, and
wraps them in a Streamlit demo that mirrors the Recipe-based workflow of
production AOI systems. The three methods trace a clear bottleneck progression:
a classical pipeline that localises well but cannot classify fine-grained
defects, a CNN that fixes classification but is bounded by proposal recall, and
an end-to-end detector that removes the proposal dependency entirely.

## Methods

| Stage | Method | Description |
|---|---|---|
| 1 | Traditional CV | Template registration (ORB + homography) → colour difference → morphology → connected components → rule-based (nearest-prototype) classification. Needs a defect-free template. |
| 2 | Two-stage DL (R-CNN style) | Stage-1 region proposals cropped and classified by an ImageNet-pretrained ResNet18. |
| 3 | End-to-end YOLO | YOLOv8n trained to jointly localise and classify, no external proposals. |

## Dataset

**HRIPCB** (Open Lab on Human–Robot Interaction, Peking University). The base set
used here contains **693 images** across **6 defect classes** (`missing_hole`,
`mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`) with **2,953**
Pascal-VOC bounding boxes; the repository's pre-rotated augmentation set is
excluded (augmentation is done at train time instead). All three methods are
evaluated on the **same image-level test split** (`seed=42`, 108 images / 460
boxes).

- Reference / mirror: <https://github.com/Ironbrotherstyle/PCB-DATASET>
- Original paper: Huang & Wei, *A PCB Dataset for Defects Detection and
  Classification*, arXiv:1901.08204, 2019.

## Results

All numbers are on the fixed test split (`seed=42`):

| method | mAP@0.5 | mAP@0.5:0.95 | precision | recall | inference (ms) | model size (MB) | notes |
|---|---|---|---|---|---|---|---|
| Stage 1: Traditional CV | 0.293 | 0.141 | 0.288 | 0.391 | 641.3 | — | difference + morphology; hand-crafted features hit a classification ceiling |
| Stage 2: Two-stage DL (proposals + CNN) | 0.848 | 0.390 | 0.848 | 0.883 | 13.6 | 44.8 | ResNet18 forward (excl. Stage-1 proposal generation); bounded by proposal recall |
| Stage 3: End-to-end YOLO | 0.918 | 0.467 | 0.935 | 0.862 | 5.4 | 6.3 | end-to-end, no proposal dependency; breaks the `short` recall bottleneck |

From Stage 1 to Stage 3, **mAP@0.5 rises from 29.3% to 91.8%** while pure
inference time **accelerates ~119×** (641 ms → 5.4 ms). The single defect class
that the two-stage method could not recover, `short`, goes from AP 19.7% (Stage 1)
→ 36.2% (Stage 2) → **97.5%** (Stage 3), because the end-to-end detector no longer
depends on Stage-1 proposal recall.

Inference time is defined consistently as the model-forward step only (no image
I/O): for Stage 1 the classical pipeline compute, for Stage 2 the ResNet18 forward
(excluding Stage-1 proposal generation), for Stage 3 the ultralytics-reported
inference time.

![Methods side by side](results/figures/04_methods_comparison_side_by_side.png)
![PR curves](results/figures/04_pr_curves.png)
![Speed comparison](results/figures/04_speed_comparison.png)

The full comparison table is in
[`results/comparison_table.md`](results/comparison_table.md); per-stage reports
are in [`results/`](results/) and the walkthrough notebooks in
[`notebooks/`](notebooks/).

## Streamlit Demo

An interactive demo lets you run all three methods on the same board and observe
their differences.

```bash
streamlit run app.py
# or:  ./run_aoi_app.bat   (Windows)
#      ./run_aoi_app.sh    (Linux/macOS)
```

Then open <http://localhost:8501>.

Behaviour differences between the methods in the demo:

- **Stage 1** requires selecting a **Recipe** (defect class → corresponding
  golden template) before inspection, then does template differencing.
- **Stage 2** takes a single image directly. Since the demo has no template, it
  uses an **MSER class-agnostic region proposal** in place of the
  difference-based proposal used elsewhere; precision/recall are therefore lower
  than the full template-based pipeline (this is labelled in the app).
- **Stage 3** takes a single image directly and runs end-to-end.

> The bundled demo images in [`samples/`](samples/) are resized to 1600 px on the
> long side to keep the repository small. The headline metrics in the Results
> section are computed on the full-resolution dataset; see
> `results/comparison_table.md`.

## Installation

Requires **Python 3.11+** and a CUDA-capable GPU for the deep-learning stages.

PyTorch is installed separately so it matches your CUDA version, and
`ultralytics` is installed with `--no-deps` so it cannot overwrite your torch
build. This project was developed on a CUDA 12.8 nightly
(`torch 2.12.0.dev+cu128`, RTX 5070 Ti / Blackwell).

```bash
# 1) install torch/torchvision for your CUDA — see https://pytorch.org
# 2) install ultralytics without its dependencies
pip install ultralytics==8.4.60 --no-deps
pip install py-cpuinfo scipy seaborn polars
# 3) install the remaining requirements
pip install -r requirements.txt
```

See the header of [`requirements.txt`](requirements.txt) for details.

### Getting the dataset

The pipeline code and the demo's bundled samples do not require the full dataset.
To reproduce training/evaluation, fetch the HRIPCB base set into `data/HRIPCB/`
(a sparse checkout avoids downloading the augmentation set):

```bash
cd data
git clone --filter=blob:none --no-checkout https://github.com/Ironbrotherstyle/PCB-DATASET.git HRIPCB
cd HRIPCB
git sparse-checkout init --cone
git sparse-checkout set images Annotations PCB_USED
git checkout master
rm -rf .git           # optional: reclaim the promisor pack
```

Then build the YOLO-format dataset with `python -m src.voc_to_yolo`.

## Project Structure

```
pcb-aoi/
├── app.py                      # Streamlit demo
├── run_aoi_app.bat / .sh       # demo launchers
├── src/
│   ├── hripcb.py               # dataset / VOC parsing utilities
│   ├── traditional_cv.py       # Stage 1 pipeline
│   ├── dataset.py              # crop dataset + image-level splits
│   ├── dl_classifier.py        # ResNet18 model
│   ├── train_classifier.py     # Stage 2 training
│   ├── voc_to_yolo.py          # VOC → YOLO conversion
│   ├── train_yolo.py           # Stage 3 training
│   ├── evaluate_*.py           # per-stage evaluation
│   ├── compare_methods.py      # unified three-method comparison
│   ├── inference_api.py        # unified inference interface (demo)
│   └── viz_*.py, plot_utils.py # visualisation
├── notebooks/                  # 00 EDA → 04 methods comparison
├── tests/                      # pytest suite
├── results/                    # tables, figures, reports, weights
├── samples/                    # bundled demo images + templates
├── data/                       # HRIPCB (git-ignored)
└── models/                     # trained weights
```

## Industrial Context

Industrial AOI systems commonly use commercial libraries such as MVTec HALCON or
Cognex VisionPro, deployed on industrial HMI hardware (15–22 inch touch screens,
integrated with PLC triggers and MES databases). This project uses open-source
OpenCV + PyTorch for reproducibility; the implemented algorithms (template
matching, morphological operations, CNN classification, YOLO detection) are
conceptually equivalent to standard HALCON operators. The Streamlit demo
simulates the Recipe Management workflow of production AOI systems, where
operators load product-specific templates before inspection runs.

## Limitations

- **mAP@0.5:0.95 = 46.7%** (Stage 3): boxes are found (high mAP@0.5) but
  localisation under strict IoU has room to improve, partly due to downsampling
  multi-megapixel images to `imgsz=640`.
- The demo's **Stage 2** uses an MSER class-agnostic proposal, which produces
  more false positives than the full template-based pipeline.
- All three methods are trained and tested on HRIPCB only; **cross-dataset
  generalisation is not validated**.
- Reported inference times are for single-image inference; batch processing would
  reduce per-image cost further.

## Tests

```bash
pytest
# 23 passed
```

Coverage spans the four components: traditional-CV pipeline
(`test_traditional_cv.py`), DL classifier and data splits
(`test_dl_classifier.py`), YOLO conversion/inference (`test_yolo.py`), and the
unified inference API used by the demo (`test_streamlit_pipeline.py`).

## AI Collaboration

This project used an AI assistant for planning, code review, and documentation.
All algorithm design, experimental decisions, and interpretation of results were
directed and verified by the author.

## References

- Huang, W. & Wei, P. *A PCB Dataset for Defects Detection and Classification.*
  arXiv:1901.08204, 2019.
- Jocher, G. et al. *Ultralytics YOLO.* <https://github.com/ultralytics/ultralytics>
- He, K., Zhang, X., Ren, S. & Sun, J. *Deep Residual Learning for Image
  Recognition.* CVPR, 2016.

## License

Released under the [MIT License](LICENSE).
