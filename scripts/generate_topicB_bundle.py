"""打包生成作業二主題 B 報告所需的 6 張視覺化圖檔與對照數據。

輸出目標：report_bundle_topicB/
1. fig07_B1_input_crops.png
2. fig08_B1_baseline_output.png
3. fig09_B3_ep5_curve_and_cm.png
4. fig10_B3_ep10_curve_and_cm.png
5. fig11_B3_ep15_curve_and_cm.png
6. fig12_B5_workflow_diagram.png
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp") + "/mpl"
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from src.plot_utils import setup_chinese_font
from src.hripcb import CLASSES
from src.dataset import PCBDefectCropDataset, DEFAULT_ROOT, build_splits
from src.dl_classifier import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_bundle_topicB"
OUT_DIR.mkdir(parents=True, exist_ok=True)

setup_chinese_font()

CLASS_NAMES_ZH = {
    "missing_hole": "缺孔 (missing_hole)",
    "mouse_bite": "鼠咬/囓齒 (mouse_bite)",
    "open_circuit": "開路 (open_circuit)",
    "short": "短路 (short)",
    "spur": "突刺 (spur)",
    "spurious_copper": "假銅斑/殘銅 (spurious_copper)",
}

CLASS_SHORT_ZH = {
    "missing_hole": "缺孔",
    "mouse_bite": "鼠咬",
    "open_circuit": "開路",
    "short": "短路",
    "spur": "突刺",
    "spurious_copper": "假銅斑",
}


def _denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ========================================================================= #
# 圖 7: fig07_B1_input_crops.png (輸入影像裁切塊網格)
# ========================================================================= #
def generate_fig07():
    print("[1/6] 產生 fig07_B1_input_crops.png ...")
    ds = PCBDefectCropDataset("gt", "train", seed=42)
    by_cls = {c: [] for c in CLASSES}
    for i, s in enumerate(ds.samples):
        if 0 <= s["label"] < len(CLASSES):
            by_cls[CLASSES[s["label"]]].append(i)

    rng = np.random.default_rng(42)
    per_class = 3
    n_classes = len(CLASSES)

    fig, axes = plt.subplots(
        per_class, n_classes, figsize=(15, 8.2), dpi=150
    )

    for c_idx, c in enumerate(CLASSES):
        picks = rng.choice(by_cls[c], size=per_class, replace=False)
        for r_idx in range(per_class):
            ax = axes[r_idx, c_idx]
            img_rgb = ds.patches[int(picks[r_idx])]
            ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])

            # 加上漂亮微邊框
            for spine in ax.spines.values():
                spine.set_color("#607D8B")
                spine.set_linewidth(1.2)

            if r_idx == 0:
                ax.set_title(
                    f"{CLASS_SHORT_ZH[c]}\n({c})",
                    fontsize=11,
                    fontweight="bold",
                    pad=8,
                    color="#1A237E",
                )
            if c_idx == 0:
                ax.set_ylabel(f"樣本 {r_idx + 1}", fontsize=10, fontweight="bold", color="#37474F")

    fig.suptitle(
        "圖 7：HRIPCB 六大缺陷類別典型輸入影像裁切塊 (含 30% 上下文 Padding，224×224)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#0D47A1",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    p = OUT_DIR / "fig07_B1_input_crops.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"      -> {p}")


# ========================================================================= #
# 圖 8: fig08_B1_baseline_output.png (原始預設模型曲線 + 測試集樣本預測)
# ========================================================================= #
def generate_fig08():
    print("[2/6] 產生 fig08_B1_baseline_output.png ...")
    csv_path = ROOT / "results" / "training_log_baseline_orig.csv"
    if not csv_path.exists():
        csv_path = ROOT / "results" / "training_log.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) for r in rows]
    va_acc = [float(r["val_acc"]) for r in rows]

    best_idx = int(np.argmin(va_loss))
    best_ep = epochs[best_idx]
    best_vl = va_loss[best_idx]
    best_va = va_acc[best_idx]

    fig = plt.figure(figsize=(16, 7.5), dpi=150)
    gs = GridSpec(2, 3, width_ratios=[1.1, 1.1, 1.8], wspace=0.28, hspace=0.32)

    # 左 1：Loss 曲線
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, tr_loss, "-o", color="#1976D2", ms=3.5, label="Train Loss", lw=1.6)
    ax_loss.plot(epochs, va_loss, "-s", color="#D32F2F", ms=3.5, label="Val Loss", lw=1.6)
    ax_loss.axvline(best_ep, color="#388E3C", linestyle="--", alpha=0.8, lw=1.3)
    ax_loss.scatter([best_ep], [best_vl], color="#388E3C", s=60, zorder=5, label=f"Best Ep{best_ep} ({best_vl:.4f})")
    ax_loss.set_title("基準模型 Loss 收斂曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_loss.set_xlabel("Epoch", fontsize=9.5)
    ax_loss.set_ylabel("Cross Entropy Loss", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=8.5, loc="upper right")

    # 左 2：Accuracy 曲線
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.plot(epochs, [a * 100 for a in tr_acc], "-o", color="#1976D2", ms=3.5, label="Train Acc", lw=1.6)
    ax_acc.plot(epochs, [a * 100 for a in va_acc], "-s", color="#D32F2F", ms=3.5, label="Val Acc", lw=1.6)
    ax_acc.axvline(best_ep, color="#388E3C", linestyle="--", alpha=0.8, lw=1.3)
    ax_acc.set_title("基準模型 Accuracy 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_acc.set_xlabel("Epoch", fontsize=9.5)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax_acc.set_ylim(80, 101.5)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=8.5, loc="lower right")

    # 中間：學習率與早停標註
    ax_lr = fig.add_subplot(gs[:, 1])
    lrs = [float(r["lr"]) for r in rows]
    ax_lr.plot(epochs, lrs, "-^", color="#7B1FA2", ms=4, lw=1.8, label="Learning Rate")
    ax_lr.set_yscale("log")
    ax_lr.set_title("學習率衰減排程與早停事件", fontsize=11, fontweight="bold", color="#1A237E")
    ax_lr.set_xlabel("Epoch", fontsize=9.5)
    ax_lr.set_ylabel("Learning Rate (Log scale)", fontsize=9.5)
    ax_lr.grid(True, linestyle=":", alpha=0.6)
    ax_lr.axvline(best_ep, color="#388E3C", linestyle="--", alpha=0.8, lw=1.3)
    ax_lr.text(
        best_ep + 0.5,
        1e-4 * 0.7,
        f"★ Best Checkpoint\n(Epoch {best_ep})\nVal Loss: {best_vl:.4f}\nVal Acc: {best_va*100:.1f}%",
        fontsize=8.5,
        color="#2E7D32",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#81C784", alpha=0.9),
    )
    ax_lr.text(
        epochs[-1] - 8,
        1.5e-5,
        f"Early Stop @ Ep{epochs[-1]}\n(Patience=10 觸發)",
        fontsize=8.5,
        color="#C62828",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#E57373", alpha=0.9),
    )
    ax_lr.legend(fontsize=8.5, loc="lower left")

    # 右半部：載入模型權重進行 8 筆真實測試集 GT crops 預測
    weights_path = ROOT / "models" / "defect_classifier_baseline_orig.pt"
    if not weights_path.exists():
        weights_path = ROOT / "models" / "defect_classifier.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights_path, device=device)

    ds_test = PCBDefectCropDataset("gt", "test", seed=42)
    # 精選 8 個跨類別典型測試樣本
    rng = np.random.default_rng(101)
    test_indices = rng.choice(len(ds_test), size=8, replace=False)

    # 建立 2x4 小網格放在 gs[:, 2]
    inner_gs = gs[:, 2].subgridspec(2, 4, wspace=0.15, hspace=0.35)
    for idx_box, sample_idx in enumerate(test_indices):
        ax_sub = fig.add_subplot(inner_gs[idx_box // 4, idx_box % 4])
        tensor_crop, label_id, meta = ds_test[int(sample_idx)]
        with torch.no_grad():
            out = model(tensor_crop.unsqueeze(0).to(device))
            prob = torch.softmax(out, dim=1)[0]
            pred_id = int(prob.argmax().item())
            conf = float(prob[pred_id])

        img_np = _denorm(tensor_crop)
        ax_sub.imshow(img_np)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])

        is_correct = (pred_id == label_id)
        border_color = "#2E7D32" if is_correct else "#C62828"
        for spine in ax_sub.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(1.8)

        true_cls_zh = CLASS_SHORT_ZH[CLASSES[label_id]]
        pred_cls_zh = CLASS_SHORT_ZH[CLASSES[pred_id]]
        ax_sub.set_title(
            f"真: {true_cls_zh}\n預: {pred_cls_zh} ({conf*100:.1f}%)",
            fontsize=8.5,
            fontweight="bold",
            color=border_color,
            pad=4,
        )

    fig.suptitle(
        "圖 8：原始基準模型 (Baseline: 31 輪，Early Stopping，Best @ Ep21) 訓練曲線與測試集預測輸出",
        fontsize=13.5,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )
    p = OUT_DIR / "fig08_B1_baseline_output.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"      -> {p}")


# ========================================================================= #
# 混淆矩陣繪製輔助函式
# ========================================================================= #
def _plot_confusion_matrix(ax, cm_list, title, subtitle=None):
    cm = np.array(cm_list, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm))

    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n_classes = len(CLASSES)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    x_labels = [f"{CLASS_SHORT_ZH[c]}\n({c[:4]}.)" for c in CLASSES]
    y_labels = [f"{CLASS_SHORT_ZH[c]} ({c[:5]}.)" for c in CLASSES]
    ax.set_xticklabels(x_labels, fontsize=8.5, rotation=25, ha="right")
    ax.set_yticklabels(y_labels, fontsize=8.5)

    ax.set_xlabel("Predicted Class (預測類別)", fontsize=9.5, fontweight="bold", labelpad=5)
    ax.set_ylabel("Ground Truth (真實類別)", fontsize=9.5, fontweight="bold", labelpad=5)

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=10, fontweight="bold", pad=6, color="#1A237E")

    for i in range(n_classes):
        for j in range(n_classes):
            cnt = int(cm[i, j])
            pct = norm[i, j] * 100
            if cnt > 0:
                text_color = "white" if norm[i, j] > 0.55 else "#1A237E"
                ax.text(
                    j, i, f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=text_color
                )
            else:
                ax.text(j, i, "0", ha="center", va="center", fontsize=7.5, color="#B0BEC5")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Row Normalized Ratio", fontsize=8)


# ========================================================================= #
# 圖 9: fig09_B3_ep5_curve_and_cm.png (Epochs = 5 曲線與混淆矩陣)
# ========================================================================= #
def generate_fig09():
    print("[3/6] 產生 fig09_B3_ep5_curve_and_cm.png ...")
    csv_path = ROOT / "results" / "training_log_ep5.csv"
    eval_json = ROOT / "results" / "classifier_eval_ep5.json"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) * 100 for r in rows]
    va_acc = [float(r["val_acc"]) * 100 for r in rows]

    lm = eval_data["last_model"]["gt_crops"]
    bm = eval_data["best_model"]["gt_crops"]

    fig = plt.figure(figsize=(15.5, 6.2), dpi=150)
    gs = GridSpec(2, 2, figure=fig, top=0.87, bottom=0.10, left=0.06, right=0.96, width_ratios=[1.1, 1.25], wspace=0.25, hspace=0.35)

    # 左上：Loss
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, tr_loss, "-o", color="#1976D2", ms=5, lw=1.8, label="Train Loss")
    ax_loss.plot(epochs, va_loss, "-s", color="#D32F2F", ms=5, lw=1.8, label="Val Loss")
    ax_loss.set_xticks(epochs)
    ax_loss.set_title("設定 1 (Epochs = 5) Loss 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_loss.set_xlabel("Epoch", fontsize=9.5)
    ax_loss.set_ylabel("Loss", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.text(
        4, 0.12, "Best Val: Ep4 (0.0346)\nLast Val: Ep5 (0.0403)",
        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9")
    )

    # 左下：Accuracy
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.plot(epochs, tr_acc, "-o", color="#1976D2", ms=5, lw=1.8, label="Train Acc")
    ax_acc.plot(epochs, va_acc, "-s", color="#D32F2F", ms=5, lw=1.8, label="Val Acc")
    ax_acc.set_xticks(epochs)
    ax_acc.set_title("設定 1 (Epochs = 5) Accuracy 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_acc.set_xlabel("Epoch", fontsize=9.5)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax_acc.set_ylim(88, 101)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=9, loc="lower right")

    # 右側：混淆矩陣 (Last Epoch 5)
    ax_cm = fig.add_subplot(gs[:, 1])
    _plot_confusion_matrix(
        ax_cm,
        lm["confusion_matrix"],
        title=f"設定 1 (Epochs = 5, Last@Ep5) 測試集混淆矩陣 (460 GT Crops)",
        subtitle=f"Acc: {lm['accuracy']*100:.2f}% (456/460) | Macro Recall: {lm['macro_recall']*100:.2f}%  [最佳權重 Best@Ep4: Acc {bm['accuracy']*100:.2f}%, Recall {bm['macro_recall']*100:.2f}%]",
    )

    fig.suptitle(
        "圖 9：設定 1 (Epochs = 5) 訓練曲線與測試集混淆矩陣",
        fontsize=13.5,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )
    p = OUT_DIR / "fig09_B3_ep5_curve_and_cm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"      -> {p}")


# ========================================================================= #
# 圖 10: fig10_B3_ep10_curve_and_cm.png (Epochs = 10 曲線與混淆矩陣)
# ========================================================================= #
def generate_fig10():
    print("[4/6] 產生 fig10_B3_ep10_curve_and_cm.png ...")
    csv_path = ROOT / "results" / "training_log_ep10.csv"
    eval_json = ROOT / "results" / "classifier_eval_ep10.json"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) * 100 for r in rows]
    va_acc = [float(r["val_acc"]) * 100 for r in rows]

    lm = eval_data["last_model"]["gt_crops"]
    bm = eval_data["best_model"]["gt_crops"]

    fig = plt.figure(figsize=(15.5, 6.2), dpi=150)
    gs = GridSpec(2, 2, figure=fig, top=0.87, bottom=0.10, left=0.06, right=0.96, width_ratios=[1.1, 1.25], wspace=0.25, hspace=0.35)

    # 左上：Loss
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, tr_loss, "-o", color="#1976D2", ms=4, lw=1.8, label="Train Loss")
    ax_loss.plot(epochs, va_loss, "-s", color="#D32F2F", ms=4, lw=1.8, label="Val Loss")
    ax_loss.axvline(8, color="#388E3C", linestyle="--", alpha=0.8, lw=1.3)
    ax_loss.set_xticks(epochs)
    ax_loss.set_title("設定 2 (Epochs = 10) Loss 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_loss.set_xlabel("Epoch", fontsize=9.5)
    ax_loss.set_ylabel("Loss", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.text(
        6, 0.12, "Best Val: Ep8 (0.0279)\nLast Val: Ep10 (0.0330)",
        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9")
    )

    # 左下：Accuracy
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.plot(epochs, tr_acc, "-o", color="#1976D2", ms=4, lw=1.8, label="Train Acc")
    ax_acc.plot(epochs, va_acc, "-s", color="#D32F2F", ms=4, lw=1.8, label="Val Acc")
    ax_acc.axvline(8, color="#388E3C", linestyle="--", alpha=0.8, lw=1.3)
    ax_acc.set_xticks(epochs)
    ax_acc.set_title("設定 2 (Epochs = 10) Accuracy 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_acc.set_xlabel("Epoch", fontsize=9.5)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax_acc.set_ylim(88, 101.5)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=9, loc="lower right")

    # 右側：混淆矩陣 (Last Epoch 10)
    ax_cm = fig.add_subplot(gs[:, 1])
    _plot_confusion_matrix(
        ax_cm,
        lm["confusion_matrix"],
        title=f"設定 2 (Epochs = 10, Last@Ep10) 測試集混淆矩陣 (460 GT Crops)",
        subtitle=f"Acc: {lm['accuracy']*100:.2f}% (457/460) | Macro Recall: {lm['macro_recall']*100:.2f}%  [最佳權重 Best@Ep8: Acc {bm['accuracy']*100:.2f}%, Recall {bm['macro_recall']*100:.2f}%]",
    )

    fig.suptitle(
        "圖 10：設定 2 (Epochs = 10) 訓練曲線與測試集混淆矩陣",
        fontsize=13.5,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )
    p = OUT_DIR / "fig10_B3_ep10_curve_and_cm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"      -> {p}")


# ========================================================================= #
# 圖 11: fig11_B3_ep15_curve_and_cm.png (Epochs = 15 曲線與混淆矩陣)
# ========================================================================= #
def generate_fig11():
    print("[5/6] 產生 fig11_B3_ep15_curve_and_cm.png ...")
    csv_path = ROOT / "results" / "training_log_ep15.csv"
    eval_json = ROOT / "results" / "classifier_eval_ep15.json"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) * 100 for r in rows]
    va_acc = [float(r["val_acc"]) * 100 for r in rows]

    lm = eval_data["last_model"]["gt_crops"]
    bm = eval_data["best_model"]["gt_crops"]

    fig = plt.figure(figsize=(15.5, 6.2), dpi=150)
    gs = GridSpec(2, 2, figure=fig, top=0.87, bottom=0.10, left=0.06, right=0.96, width_ratios=[1.1, 1.25], wspace=0.25, hspace=0.35)

    # 左上：Loss，標示 Ep 14 學習率減半
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, tr_loss, "-o", color="#1976D2", ms=4, lw=1.8, label="Train Loss")
    ax_loss.plot(epochs, va_loss, "-s", color="#D32F2F", ms=4, lw=1.8, label="Val Loss")
    ax_loss.axvline(11, color="#388E3C", linestyle="--", alpha=0.8, lw=1.2, label="Best Val (Ep11)")
    ax_loss.axvline(14, color="#E65100", linestyle="-.", alpha=0.9, lw=1.4, label="LR 減半 (Ep14: 1e-4→5e-5)")
    ax_loss.set_xticks(epochs)
    ax_loss.set_title("設定 3 (Epochs = 15) Loss 曲線 (含 LR 排程觸發標記)", fontsize=11, fontweight="bold", color="#1A237E")
    ax_loss.set_xlabel("Epoch", fontsize=9.5)
    ax_loss.set_ylabel("Loss", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.text(
        8.5, 0.12, "Ep14 觸發 ReduceLROnPlateau\n(lr: 1e-4 -> 5e-5)",
        fontsize=8, color="#BF360C", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#FFB74D")
    )

    # 左下：Accuracy
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.plot(epochs, tr_acc, "-o", color="#1976D2", ms=4, lw=1.8, label="Train Acc")
    ax_acc.plot(epochs, va_acc, "-s", color="#D32F2F", ms=4, lw=1.8, label="Val Acc")
    ax_acc.axvline(11, color="#388E3C", linestyle="--", alpha=0.8, lw=1.2)
    ax_acc.axvline(14, color="#E65100", linestyle="-.", alpha=0.9, lw=1.4)
    ax_acc.set_xticks(epochs)
    ax_acc.set_title("設定 3 (Epochs = 15) Accuracy 曲線", fontsize=11, fontweight="bold", color="#1A237E")
    ax_acc.set_xlabel("Epoch", fontsize=9.5)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax_acc.set_ylim(88, 101.5)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=9, loc="lower right")

    # 右側：混淆矩陣 (Last Epoch 15)
    ax_cm = fig.add_subplot(gs[:, 1])
    _plot_confusion_matrix(
        ax_cm,
        lm["confusion_matrix"],
        title=f"設定 3 (Epochs = 15, Last@Ep15) 測試集混淆矩陣 (460 GT Crops)",
        subtitle=f"Acc: {lm['accuracy']*100:.2f}% (457/460) | Macro Recall: {lm['macro_recall']*100:.2f}%  [最佳權重 Best@Ep11: Acc {bm['accuracy']*100:.2f}%, Recall {bm['macro_recall']*100:.2f}%]",
    )

    fig.suptitle(
        "圖 11：設定 3 (Epochs = 15) 訓練曲線與測試集混淆矩陣",
        fontsize=13.5,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )
    p = OUT_DIR / "fig11_B3_ep15_curve_and_cm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"      -> {p}")


# ========================================================================= #
# 圖 12: fig12_B5_workflow_diagram.png (主題 B 四大階段工作流程圖)
# ========================================================================= #
def generate_fig12():
    print("[6/6] 產生 fig12_B5_workflow_diagram.png ...")
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # 4 大階段卡片設定 (x, y, w, h, header_color, bg_color, title, subtitle, bullets, module_info)
    stages = [
        {
            "x": 0.5, "y": 0.65, "w": 3.2, "h": 7.05,
            "header_color": "#1565C0", "bg_color": "#E3F2FD",
            "title": "階段一：Input",
            "subtitle": "資料載入與 VOC 標註解析",
            "bullets": [
                ("資料來源", "北京大學 HRIPCB 資料集"),
                ("影像規格", "693 張高解析度 PCB 圖 (~3k×3k)"),
                ("標註格式", "Pascal-VOC XML 邊界框"),
                ("缺陷總數", "2,953 個真實標註瑕疵"),
                ("涵蓋類別", "6 大類 (缺孔, 鼠咬, 開路,\n短路, 突刺, 假銅斑)"),
                ("金牌模板", "PCB_USED 目錄標準無缺陷板"),
            ],
            "module_info": "核心模組：\nsrc/hripcb.py (L80-L104: parse_voc)\nsrc/hripcb.py (L46-L77: VOC 結構)",
            "tensor_flow": "輸入資料：\nJPG 原始圖 + XML BBox\n[H, W, 3] 影像陣列",
        },
        {
            "x": 4.3, "y": 0.65, "w": 3.4, "h": 7.05,
            "header_color": "#00838F", "bg_color": "#E0F7FA",
            "title": "階段二：Preprocessing",
            "subtitle": "分層切分、上下文裁切與增強",
            "bullets": [
                ("影像級切分", "seed=42 分層防洩漏切分\n70% 訓練 (483 張 / 2051 crops)\n15% 驗證 (102 張 / 442 crops)\n15% 測試 (108 張 / 460 crops)"),
                ("上下文外擴", "BBox 外擴 pad_ratio=0.3\n(引入 30% 鄰近導線背景環境)"),
                ("張量縮放", "Resize 至 224×224 維度"),
                ("訓練增強", "隨機水平翻轉、旋轉 15°、\nColorJitter (明暗對比度調節)"),
                ("標準化", "ImageNet Mean & Std 正規化"),
            ],
            "module_info": "核心模組：\nsrc/dataset.py (L47-L73: build_splits)\nsrc/dataset.py (L100-L111: crop_with_padding)\nsrc/dataset.py (L79-L94: get_transforms)",
            "tensor_flow": "張量維度轉換：\nBGR Patch -> RGB Tensor\n[B, 3, 224, 224] Float32",
        },
        {
            "x": 8.3, "y": 0.65, "w": 3.5, "h": 7.05,
            "header_color": "#2E7D32", "bg_color": "#E8F5E9",
            "title": "階段三：Classification",
            "subtitle": "ResNet18 遷移學習與輪數實驗",
            "bullets": [
                ("骨幹網路", "ResNet18 (IMAGENET1K_V1 預訓練)"),
                ("分類頭設計", "Linear(in_features=512, out=6)"),
                ("微調策略", "全網路端到端梯度更新 (Fine-tuning)"),
                ("優化器設定", "Adam (lr=1e-4, batch_size=64)"),
                ("排程機制", "ReduceLROnPlateau (factor=0.5,\npatience=4 on val loss)"),
                ("主題 B 實驗", "唯一自變數：訓練輪數\n組 A: 5 | 組 B: 10 | 組 C: 15 輪\n基準組: max=50 (EarlyStop @ Ep31)"),
            ],
            "module_info": "核心模組：\nsrc/dl_classifier.py (L15-L36: DefectClassifier)\nsrc/train_classifier.py (L40-L47: set_seed)\nsrc/train_classifier.py (L71-L209: train)",
            "tensor_flow": "特徵抽取與分類：\nBackbone 特徵 -> 512 維向量\n-> 6 類 Logits -> Softmax 機率",
        },
        {
            "x": 12.4, "y": 0.65, "w": 3.2, "h": 7.05,
            "header_color": "#C2185B", "bg_color": "#FCE4EC",
            "title": "階段四：Output",
            "subtitle": "雙軌評估指標與視覺化產出",
            "bullets": [
                ("軌道一評估", "真實標籤 GT Crops 上限評估\n(460 個測試集真實裁切塊)\n- 各類召回率 (Recall)\n- 宏平均 Macro Recall\n- 微平均 Micro Recall / Acc\n- 6×6 測試集混淆矩陣"),
                ("軌道二評估", "端到端 End-to-End 系統評估\n(Stage 1 Proposals + CNN 分類)\n- mAP@0.5 (全系統 84~85%)"),
                ("視覺化圖檔", "fig07~fig12 全套圖表自動匯出"),
            ],
            "module_info": "核心模組：\nsrc/evaluate_classifier.py (L152-L197: GT)\nsrc/evaluate_classifier.py (L203-L277: E2E)\nsrc/viz_classifier.py (L45-L61: 混淆圖)",
            "tensor_flow": "產物輸出：\nJSON 指標彙總、CSV 日誌\n評估報表與高品質視覺化圖檔",
        },
    ]

    for st in stages:
        x, y, w, h = st["x"], st["y"], st["w"], st["h"]
        # 背景卡片
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            fc=st["bg_color"], ec=st["header_color"], lw=1.8, zorder=2
        )
        ax.add_patch(rect)

        # 頂部彩色標題條
        header_h = 1.15
        header_rect = patches.FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            fc=st["header_color"], ec=st["header_color"], lw=1.8, zorder=3
        )
        ax.add_patch(header_rect)

        # 標題條文字
        ax.text(
            x + w / 2, y + h - 0.38, st["title"],
            ha="center", va="center", color="white", fontsize=11.5, fontweight="bold", zorder=4
        )
        ax.text(
            x + w / 2, y + h - 0.78, st["subtitle"],
            ha="center", va="center", color="#ECEFF1", fontsize=9, zorder=4
        )

        # 條列內容
        curr_y = y + h - header_h - 0.22
        for b_title, b_desc in st["bullets"]:
            ax.text(
                x + 0.15, curr_y, f"• {b_title}：",
                ha="left", va="top", fontsize=8.5, fontweight="bold", color="#263238", zorder=4
            )
            lines = b_desc.split("\n")
            desc_offset = 0.20
            for ln in lines:
                ax.text(
                    x + 0.25, curr_y - desc_offset, ln,
                    ha="left", va="top", fontsize=7.8, color="#37474F", zorder=4
                )
                desc_offset += 0.20
            curr_y -= (desc_offset + 0.08)

        # 底部模組與資料流區塊
        box_y = y + 0.12
        box_h = 1.38
        module_box = patches.FancyBboxPatch(
            (x + 0.12, box_y), w - 0.24, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            fc="white", ec=st["header_color"], lw=1.0, alpha=0.95, zorder=3
        )
        ax.add_patch(module_box)

        ax.text(
            x + 0.2, box_y + 1.28, st["tensor_flow"],
            ha="left", va="top", fontsize=7.2, color="#0D47A1", fontweight="bold", zorder=4
        )
        ax.text(
            x + 0.2, box_y + 0.62, st["module_info"],
            ha="left", va="top", fontsize=6.8, color="#37474F", zorder=4
        )

    # 繪製各階段之間的轉接箭頭
    arrow_props = dict(boxstyle="rarrow,pad=0.18", fc="#FFA000", ec="#E65100", lw=1.2)
    for i in range(3):
        x_start = stages[i]["x"] + stages[i]["w"] + 0.05
        x_end = stages[i + 1]["x"] - 0.05
        mid_x = (x_start + x_end) / 2
        mid_y = 4.18
        arrow = patches.FancyBboxPatch(
            (mid_x - 0.22, mid_y - 0.25), 0.44, 0.5,
            boxstyle="rarrow,pad=0.1", fc="#FF8F00", ec="#BF360C", lw=1.2, zorder=5
        )
        ax.add_patch(arrow)

    # 頂部大主標題
    fig.suptitle(
        "圖 12：主題 B（Stage 2 ResNet18 缺陷分類系統）四大階段工作流程與資料流圖",
        fontsize=14.5,
        fontweight="bold",
        color="#0D47A1",
        y=0.97,
    )

    p = OUT_DIR / "fig12_B5_workflow_diagram.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {p}")


if __name__ == "__main__":
    print(f"=== 開始產生主題 B 6 張視覺化圖檔至 {OUT_DIR} ===")
    generate_fig07()
    generate_fig08()
    generate_fig09()
    generate_fig10()
    generate_fig11()
    generate_fig12()
    print("=== 全數圖檔產生完成！ ===")
