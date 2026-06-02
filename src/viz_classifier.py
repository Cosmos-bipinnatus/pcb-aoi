"""Stage 2 視覺化:訓練曲線、混淆矩陣、樣本預測、錯誤案例。"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .dataset import PCBDefectCropDataset, get_transforms
from .hripcb import CLASSES


def _save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def training_curves(csv_path, out_dir, name="02_training_curves.png"):
    rows = list(csv.DictReader(open(csv_path)))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], "-o", ms=3, label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], "-o", ms=3, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(ep, [float(r["train_acc"]) for r in rows], "-o", ms=3, label="train")
    ax[1].plot(ep, [float(r["val_acc"]) for r in rows], "-o", ms=3, label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("acc")
    ax[1].set_ylim(0, 1.02); ax[1].legend()
    best = min(rows, key=lambda r: float(r["val_loss"]))
    fig.suptitle(f"Stage 2 訓練曲線 (最佳 val @ epoch {best['epoch']}, "
                 f"val_acc {float(best['val_acc']):.3f})", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def confusion_fig(cm, labels, title, out_dir, name):
    cm = np.array(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l[:11] for l in labels], rotation=40, ha="right")
    ax.set_yticklabels([l[:13] for l in labels])
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="black" if norm[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def sample_crops_grid(out_dir, name="02_sample_crops_per_class.png", per_class=2, seed=42):
    """每類 ``per_class`` 張 GT crops(訓練資料長相),不做預測。"""
    ds = PCBDefectCropDataset("gt", "train")
    by_cls = {c: [] for c in CLASSES}
    for i, s in enumerate(ds.samples):
        if 0 <= s["label"] < len(CLASSES):
            by_cls[CLASSES[s["label"]]].append(i)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(2.4 * per_class, 2.4 * len(CLASSES)))
    for r, c in enumerate(CLASSES):
        picks = rng.choice(by_cls[c], size=min(per_class, len(by_cls[c])), replace=False)
        for col in range(per_class):
            ax = axes[r][col]; ax.axis("off")
            if col < len(picks):
                ax.imshow(ds.patches[int(picks[col])])
                if col == 0:
                    ax.set_ylabel(c, fontsize=10)
                ax.set_title(c if col == 0 else "", fontsize=9, loc="left")
    fig.suptitle("每類典型缺陷 crops(bbox + 30% 上下文 padding)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def _denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def sample_predictions(model, device, out_dir, name="02_sample_predictions.png", n=8, seed=42):
    ds = PCBDefectCropDataset("gt", "test")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6.5))
    for ax, i in zip(axes.ravel(), idxs):
        t, label, meta = ds[int(i)]
        prob = torch.softmax(model(t.unsqueeze(0).to(device)), 1)[0]
        pred = int(prob.argmax()); conf = float(prob[pred])
        ax.imshow(_denorm(t)); ax.axis("off")
        ok = pred == label
        ax.set_title(f"真:{CLASSES[label]}\n預:{CLASSES[pred]} ({conf:.2f})",
                     fontsize=9, color="green" if ok else "red")
    fig.suptitle("Stage 2 樣本預測 (test GT crops;綠=正確 紅=錯誤)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, name)


@torch.no_grad()
def misclassified_examples(model, device, out_dir, name="02_misclassified_examples.png", n=6):
    ds = PCBDefectCropDataset("gt", "test")
    wrong = []
    for i in range(len(ds)):
        t, label, meta = ds[i]
        prob = torch.softmax(model(t.unsqueeze(0).to(device)), 1)[0]
        pred = int(prob.argmax())
        if pred != label:
            wrong.append((i, label, pred, float(prob[pred])))
        if len(wrong) >= n:
            break
    if not wrong:
        # 沒有錯誤(極高 accuracy):放一張說明圖
        fig, ax = plt.subplots(figsize=(8, 2)); ax.axis("off")
        ax.text(0.5, 0.5, "test set 無分類錯誤(accuracy = 100%)", ha="center",
                va="center", fontsize=14)
        return _save(fig, out_dir, name)
    cols = min(n, len(wrong))
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (i, label, pred, conf) in zip(axes, wrong):
        t, _, _ = ds[i]
        ax.imshow(_denorm(t)); ax.axis("off")
        ax.set_title(f"真:{CLASSES[label]}\n預:{CLASSES[pred]} ({conf:.2f})",
                     fontsize=9, color="red")
    fig.suptitle("Stage 2 錯誤案例 (test GT crops)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, name)
