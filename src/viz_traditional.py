"""Visualisation helpers for the classical-CV baseline (Stage 1).

Produces the step-by-step pipeline figures, per-class result grids, and the
evaluation summary (confusion matrix + precision/recall bars). Kept separate
from the algorithm (``traditional_cv``) and the metrics (``evaluate_traditional``)
so the notebook can stay thin.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .hripcb import CLASSES, parse_voc
from .traditional_cv import run_traditional_pipeline

# consistent per-class BGR colours for drawing boxes
_BGR = {
    "missing_hole": (60, 60, 255),
    "mouse_bite": (0, 200, 255),
    "open_circuit": (0, 255, 0),
    "short": (255, 80, 0),
    "spur": (255, 0, 200),
    "spurious_copper": (0, 140, 255),
}


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def _draw_dets(img, dets, gt=None):
    out = img.copy()
    if gt:
        for (x1, y1, x2, y2) in gt:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 4)  # GT green
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = _BGR.get(d["class"], (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 5)
        cv2.putText(out, f"{d['class']} {d['confidence']:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    return out


def step_by_step(test_path, template_path, xml_path, out_dir, name, config=None):
    """Six-panel intermediate-output strip for one image."""
    out = run_traditional_pipeline(test_path, template_path, config)
    io = out["intermediate_outputs"]
    gt = [(o.xmin, o.ymin, o.xmax, o.ymax) for o in parse_voc(xml_path).objects]

    annotated = _draw_dets(io["warped_test"], out["detections"], gt)

    panels = [
        ("1. test image", _rgb(io["test_img"])),
        ("2. template (PCB_USED)", _rgb(io["template"])),
        ("3. registered test", _rgb(io["warped_test"])),
        ("4. colour diff map", io["diff_map"]),
        ("5. defect mask", io["defect_mask"]),
        ("6. result (GT=green)", _rgb(annotated)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    q = out["registration_quality"]
    fig.suptitle(f"{Path(test_path).name}  —  registration inlier ratio {q:.2f}, "
                 f"{len(out['detections'])} detections", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def per_class_results(samples, out_dir, name, config=None):
    """One representative result per class in a 2x3 grid.

    ``samples``: list of (class, test_path, template_path, xml_path)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (cls, tp, tmpl, xml) in zip(axes.ravel(), samples):
        out = run_traditional_pipeline(tp, tmpl, config)
        gt = [(o.xmin, o.ymin, o.xmax, o.ymax) for o in parse_voc(xml).objects]
        ann = _draw_dets(out["intermediate_outputs"]["warped_test"], out["detections"], gt)
        ax.imshow(_rgb(ann))
        n_hit = sum(1 for d in out["detections"] if d["class"] == cls)
        ax.set_title(f"{cls}  (GT={len(gt)}, pred-{cls}={n_hit})", fontsize=12)
        ax.axis("off")
    fig.suptitle("Classical-CV results — one sample per class "
                 "(green = ground truth, coloured = prediction)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def evaluation_summary(eval_res, out_dir, name):
    """Confusion matrix heatmap + per-class precision/recall/F1 bars."""
    labels = eval_res["confusion_labels"]
    cm = np.array(eval_res["confusion_matrix"], dtype=float)
    # row-normalise for readability
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    ax = axes[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l[:10] for l in labels], rotation=40, ha="right")
    ax.set_yticklabels([l[:12] for l in labels])
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=8, color="black" if cm_norm[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    x = np.arange(len(CLASSES)); wbar = 0.27
    P = [eval_res["per_class"][c]["precision"] for c in CLASSES]
    R = [eval_res["per_class"][c]["recall"] for c in CLASSES]
    F = [eval_res["per_class"][c]["f1"] for c in CLASSES]
    ax.bar(x - wbar, P, wbar, label="precision")
    ax.bar(x, R, wbar, label="recall")
    ax.bar(x + wbar, F, wbar, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title(f"Per-class P / R / F1   (mAP@0.5 = {eval_res['mAP@0.5']*100:.1f}%)")
    ax.legend()

    fig.tight_layout()
    return _save(fig, out_dir, name)


def pipeline_diagram(out_dir, name):
    """Simple block-flow diagram of the pipeline stages."""
    stages = ["test image\n+ template", "feature\nregistration\n(ORB + RANSAC)",
              "colour\ndifference", "threshold +\nmorphology", "connected\ncomponents",
              "rule-based\nclassify", "defect\ndetections"]
    fig, ax = plt.subplots(figsize=(16, 2.6))
    n = len(stages)
    for i, s in enumerate(stages):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.005, 0.25), 1 / n - 0.03, 0.5,
                                    facecolor="#cfe2f3", edgecolor="#2c5f8a", lw=2))
        ax.text(x + (1 / n) / 2 - 0.005, 0.5, s, ha="center", va="center", fontsize=10)
        if i < n - 1:
            ax.annotate("", xy=(x + 1 / n - 0.02, 0.5), xytext=(x + 1 / n - 0.03, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#2c5f8a"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Stage 1 — classical-CV defect pipeline", fontweight="bold", fontsize=13)
    return _save(fig, out_dir, name)
