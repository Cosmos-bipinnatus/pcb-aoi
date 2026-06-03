"""Stage 4 比較圖:PR 曲線疊圖、推論速度對比。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import setup_chinese_font

setup_chinese_font()

_LABEL = {"stage1": "Stage 1 傳統 CV", "stage2": "Stage 2 R-CNN 風格",
          "stage3": "Stage 3 YOLOv8n"}
_COLOR = {"stage1": "#d1495b", "stage2": "#edae49", "stage3": "#00798c"}


def _save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def pr_curves(data: dict, out_dir, name="04_pr_curves.png"):
    """三方法 micro-average PR 曲線疊圖。"""
    pr = data["pr_curves"]
    map50 = {r["method"]: r["mAP@0.5"] for r in data["rows"]}
    keys = ["stage1", "stage2", "stage3"]
    methods = [r["method"] for r in data["rows"]]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k, m in zip(keys, methods):
        recall, precision = pr[k]
        ax.plot(recall, precision, color=_COLOR[k], lw=2.2,
                label=f"{_LABEL[k]}  (mAP@0.5 = {map50[m]*100:.1f}%)")
    ax.set_xlabel("Recall(召回率)"); ax.set_ylabel("Precision(精確率)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title("三方法 PR 曲線對比(micro-average,IoU>=0.5)", fontweight="bold")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    return _save(fig, out_dir, name)


def speed_comparison(data: dict, out_dir, name="04_speed_comparison.png"):
    """純推論時間對比(log scale)。"""
    rows = data["rows"]
    labels = [_LABEL[k] for k in ("stage1", "stage2", "stage3")]
    ms = [r["inference_ms"] for r in rows]
    colors = [_COLOR[k] for k in ("stage1", "stage2", "stage3")]
    speedup = data["speedup_s1_to_s3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, ms, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("純推論時間 / 影像(ms,log scale)")
    ax.set_title(f"三方法純推論速度對比 —— Stage 1→3 加速約 {speedup:.0f}×",
                 fontweight="bold")
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_ylim(min(ms) * 0.5, max(ms) * 2)
    return _save(fig, out_dir, name)
