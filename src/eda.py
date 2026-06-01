"""Exploratory analysis helpers for the HRIPCB base set.

Keeps the heavy lifting (stats + plotting) out of the notebook so
``notebooks/00_data_exploration.ipynb`` stays readable and both the notebook
and ad-hoc scripts share one implementation.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .hripcb import (
    CLASSES,
    Annotation,
    image_path_for,
    iter_annotations,
    template_paths,
)

# Consistent per-class colour mapping used across all figures.
_PALETTE = plt.get_cmap("tab10")
CLASS_COLOR = {c: _PALETTE(i) for i, c in enumerate(CLASSES)}


def load_dataset(root: str | Path) -> list[Annotation]:
    return list(iter_annotations(root))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_stats(anns: list[Annotation]) -> dict:
    """Aggregate counts / size distributions keyed by defect class."""
    img_per_class = Counter()
    box_per_class = Counter()
    boxes_per_image: list[int] = []
    img_sizes = Counter()           # (w, h) -> count
    areas = defaultdict(list)       # class -> [bbox area, ...]
    box_w = defaultdict(list)
    box_h = defaultdict(list)

    for a in anns:
        # an image's class = the (single) class of its objects
        cls = a.objects[0].name if a.objects else "EMPTY"
        img_per_class[cls] += 1
        boxes_per_image.append(a.n_objects)
        img_sizes[(a.width, a.height)] += 1
        for b in a.objects:
            box_per_class[b.name] += 1
            areas[b.name].append(b.area)
            box_w[b.name].append(b.width)
            box_h[b.name].append(b.height)

    return {
        "n_images": len(anns),
        "n_boxes": sum(box_per_class.values()),
        "img_per_class": dict(img_per_class),
        "box_per_class": dict(box_per_class),
        "boxes_per_image": boxes_per_image,
        "img_sizes": dict(img_sizes),
        "areas": {k: v for k, v in areas.items()},
        "box_w": {k: v for k, v in box_w.items()},
        "box_h": {k: v for k, v in box_h.items()},
    }


def quality_checks(anns: list[Annotation], root: str | Path) -> dict:
    """Scan for unreadable images, out-of-bounds boxes, degenerate boxes,
    XML/image size mismatch, and duplicate image files (by content hash)."""
    root = Path(root)
    issues = {
        "unreadable_images": [],
        "size_mismatch": [],         # XML size != decoded image size
        "oob_boxes": [],             # box exceeds image bounds
        "degenerate_boxes": [],      # zero/negative width or height
        "duplicate_images": [],      # same content hash
        "missing_image_file": [],
    }
    hashes: dict[str, str] = {}

    for a in anns:
        img_path = image_path_for(a, root)
        if not img_path.exists():
            issues["missing_image_file"].append(a.filename)
            continue

        # content hash for duplicate detection
        h = hashlib.md5(img_path.read_bytes()).hexdigest()
        if h in hashes:
            issues["duplicate_images"].append((a.filename, hashes[h]))
        else:
            hashes[h] = a.filename

        img = cv2.imread(str(img_path))
        if img is None:
            issues["unreadable_images"].append(a.filename)
            continue
        ih, iw = img.shape[:2]
        if (iw, ih) != (a.width, a.height):
            issues["size_mismatch"].append((a.filename, (a.width, a.height), (iw, ih)))

        for b in a.objects:
            if b.width <= 0 or b.height <= 0:
                issues["degenerate_boxes"].append((a.filename, vars(b)))
            if b.xmin < 0 or b.ymin < 0 or b.xmax > iw or b.ymax > ih:
                issues["oob_boxes"].append((a.filename, vars(b), (iw, ih)))

    issues["summary"] = {k: len(v) for k, v in issues.items() if isinstance(v, list)}
    return issues


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _save(fig, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def plot_class_distribution(stats: dict, out_dir: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    classes = CLASSES
    imgs = [stats["img_per_class"].get(c, 0) for c in classes]
    boxes = [stats["box_per_class"].get(c, 0) for c in classes]
    colors = [CLASS_COLOR[c] for c in classes]

    axes[0].bar(classes, imgs, color=colors)
    axes[0].set_title("Images per class")
    axes[0].set_ylabel("# images")
    for i, v in enumerate(imgs):
        axes[0].text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)

    axes[1].bar(classes, boxes, color=colors)
    axes[1].set_title("Defect bounding boxes per class")
    axes[1].set_ylabel("# boxes")
    for i, v in enumerate(boxes):
        axes[1].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle("HRIPCB class distribution", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "00_class_distribution.png")


def plot_bbox_size_distribution(stats: dict, out_dir: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # boxplot of bbox area (px^2) per class, log scale
    data = [stats["areas"].get(c, [0]) for c in CLASSES]
    bp = axes[0].boxplot(data, labels=CLASSES, showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_alpha(0.7)
    axes[0].set_yscale("log")
    axes[0].set_title("BBox area per class (log scale)")
    axes[0].set_ylabel("area (px$^2$)")

    # scatter of width vs height (sampled) coloured by class
    for c in CLASSES:
        w = np.array(stats["box_w"].get(c, []))
        h = np.array(stats["box_h"].get(c, []))
        axes[1].scatter(w, h, s=8, alpha=0.4, color=CLASS_COLOR[c], label=c)
    axes[1].set_title("BBox width vs height")
    axes[1].set_xlabel("width (px)")
    axes[1].set_ylabel("height (px)")
    axes[1].legend(fontsize=7, markerscale=1.5)

    for ax in (axes[0],):
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle("HRIPCB defect size distribution", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "00_bbox_size_distribution.png")


def plot_boxes_per_image_hist(stats: dict, out_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = stats["boxes_per_image"]
    bins = range(min(vals), max(vals) + 2)
    ax.hist(vals, bins=bins, align="left", color="#4477aa", edgecolor="white")
    ax.set_title("Bounding boxes per image")
    ax.set_xlabel("# boxes in image")
    ax.set_ylabel("# images")
    fig.tight_layout()
    return _save(fig, out_dir, "00_boxes_per_image_hist.png")


def _draw_boxes(img_rgb, ann: Annotation, color=(255, 60, 60), lw=6):
    out = img_rgb.copy()
    for b in ann.objects:
        cv2.rectangle(out, (b.xmin, b.ymin), (b.xmax, b.ymax), color, lw)
    return out


def plot_sample_grid(anns: list[Annotation], root: str | Path, out_dir: str | Path,
                     per_class: int = 2) -> Path:
    root = Path(root)
    by_class = defaultdict(list)
    for a in anns:
        if a.objects:
            by_class[a.objects[0].name].append(a)

    n_cols = per_class
    n_rows = len(CLASSES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, c in enumerate(CLASSES):
        # evenly spaced picks for determinism (no RNG)
        samples = by_class.get(c, [])
        picks = [samples[i * max(1, len(samples)) // (per_class + 1)]
                 for i in range(1, per_class + 1)] if samples else []
        for col in range(n_cols):
            ax = axes[r][col]
            ax.axis("off")
            if col >= len(picks):
                continue
            a = picks[col]
            img = cv2.imread(str(image_path_for(a, root)))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = _draw_boxes(img, a)
            ax.imshow(img)
            ax.set_title(f"{c}  (n={a.n_objects})", fontsize=10)
    fig.suptitle("HRIPCB sample defects (red = ground-truth bbox)",
                 fontweight="bold", y=1.005)
    fig.tight_layout()
    return _save(fig, out_dir, "00_sample_grid_per_class.png")


def plot_templates(root: str | Path, out_dir: str | Path) -> Path:
    paths = template_paths(root)
    n = len(paths)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.set_title(f"{p.stem}  ({w}x{h})", fontsize=9)
    fig.suptitle("PCB_USED — 10 defect-free templates", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "00_pcb_templates.png")
