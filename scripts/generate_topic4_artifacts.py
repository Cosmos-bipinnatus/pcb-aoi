"""產生作業二主題 4 (CNN Training Epochs 實驗) 之視覺化綜合圖與量化比較表。"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hripcb import CLASSES
from src.plot_utils import setup_chinese_font

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 設定中文字型與環境變數
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp") + "/mpl"
setup_chinese_font()


def load_all_data():
    # 讀取 CSV Logs
    b_log = list(csv.DictReader(open(RESULTS_DIR / "training_log.csv")))
    log5 = list(csv.DictReader(open(RESULTS_DIR / "training_log_ep5.csv")))
    log10 = list(csv.DictReader(open(RESULTS_DIR / "training_log_ep10.csv")))
    log15 = list(csv.DictReader(open(RESULTS_DIR / "training_log_ep15.csv")))

    # 讀取 Eval JSON
    b_eval = json.load(open(RESULTS_DIR / "classifier_eval_summary.json"))
    eval5 = json.load(open(RESULTS_DIR / "classifier_eval_ep5.json"))
    eval10 = json.load(open(RESULTS_DIR / "classifier_eval_ep10.json"))
    eval15 = json.load(open(RESULTS_DIR / "classifier_eval_ep15.json"))

    return {
        "baseline": {"log": b_log, "eval": b_eval, "best_ep": 21, "last_ep": 31},
        "ep5": {"log": log5, "eval": eval5, "best_ep": 4, "last_ep": 5},
        "ep10": {"log": log10, "eval": eval10, "best_ep": 8, "last_ep": 10},
        "ep15": {"log": log15, "eval": eval15, "best_ep": 11, "last_ep": 15},
    }


def plot_summary_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("white")

    # -------------------------------------------------------------
    # 子圖 1: 損失收斂曲線 (Loss Curves)
    # -------------------------------------------------------------
    ax_loss = axes[0, 0]
    b_log = data["baseline"]["log"]
    epochs_b = [int(r["epoch"]) for r in b_log]
    train_loss_b = [float(r["train_loss"]) for r in b_log]
    val_loss_b = [float(r["val_loss"]) for r in b_log]

    ax_loss.plot(epochs_b, train_loss_b, color="#1f77b4", linestyle="--", alpha=0.7, label="Baseline 訓練損失 (Train Loss)")
    ax_loss.plot(epochs_b, val_loss_b, color="#1f77b4", linewidth=2.0, label="Baseline 驗證損失 (Val Loss)")

    # 標示 5, 10, 15 切點與對應實驗組
    cutoffs = [
        (5, data["ep5"]["log"][-1], "#ff7f0e", "組A: Epoch 5 (Val: 0.0403)"),
        (10, data["ep10"]["log"][-1], "#2ca02c", "組B: Epoch 10 (Val: 0.0330)"),
        (15, data["ep15"]["log"][-1], "#d62728", "組C: Epoch 15 (Val: 0.0326)"),
    ]

    for ep_num, log_row, col, lbl in cutoffs:
        vl = float(log_row["val_loss"])
        ax_loss.axvline(x=ep_num, color=col, linestyle=":", linewidth=1.8, alpha=0.85)
        ax_loss.scatter([ep_num], [vl], color=col, s=70, zorder=5)
        ax_loss.annotate(
            f"Ep {ep_num}\n{vl:.4f}",
            (ep_num, vl),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.85),
        )

    # 標示 Baseline 最佳點 (Epoch 21)
    b_best_vl = float(b_log[20]["val_loss"])
    ax_loss.scatter([21], [b_best_vl], color="#9467bd", marker="*", s=160, zorder=6, label=f"Baseline 最佳點 (Ep 21: {b_best_vl:.4f})")
    ax_loss.annotate(
        f"Best Ep 21\n{b_best_vl:.4f}",
        (21, b_best_vl),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#9467bd",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#9467bd", alpha=0.85),
    )

    ax_loss.set_title("(a) 訓練與驗證損失收斂軌跡及實驗組切點", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("訓練輪數 (Epoch)", fontsize=11)
    ax_loss.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax_loss.set_xlim(0, 32)
    ax_loss.set_ylim(-0.02, 0.35)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right", fontsize=9.5)

    # -------------------------------------------------------------
    # 子圖 2: 驗證準確率收斂軌跡 (Validation Accuracy)
    # -------------------------------------------------------------
    ax_acc = axes[0, 1]
    val_acc_b = [float(r["val_acc"]) * 100 for r in b_log]
    train_acc_b = [float(r["train_acc"]) * 100 for r in b_log]

    ax_acc.plot(epochs_b, train_acc_b, color="#3498db", linestyle="--", alpha=0.6, label="Baseline 訓練準確率 (Train Acc)")
    ax_acc.plot(epochs_b, val_acc_b, color="#2980b9", linewidth=2.0, label="Baseline 驗證準確率 (Val Acc)")

    for ep_num, log_row, col, _ in cutoffs:
        va = float(log_row["val_acc"]) * 100
        ax_acc.axvline(x=ep_num, color=col, linestyle=":", linewidth=1.8, alpha=0.85)
        ax_acc.scatter([ep_num], [va], color=col, s=70, zorder=5)
        ax_acc.annotate(
            f"Ep {ep_num}\n{va:.2f}%",
            (ep_num, va),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.85),
        )

    ax_acc.set_title("(b) 驗證準確率收斂曲線 (快速躍升與平原期)", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("訓練輪數 (Epoch)", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xlim(0, 32)
    ax_acc.set_ylim(88, 101)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower right", fontsize=9.5)

    # -------------------------------------------------------------
    # 子圖 3: 測試集各類缺陷召回率 (Per-class Defect Recall)
    # -------------------------------------------------------------
    ax_recall = axes[1, 0]
    classes_display = ["missing\nhole", "mouse\nbite", "open\ncircuit", "short", "spur", "spurious\ncopper"]
    x = np.arange(len(CLASSES))
    width = 0.2

    # 取得各組 Last 模型之 Recall
    rec_ep5 = [data["ep5"]["eval"]["last_model"]["gt_crops"]["per_class"][c]["recall"] * 100 for c in CLASSES]
    rec_ep10 = [data["ep10"]["eval"]["last_model"]["gt_crops"]["per_class"][c]["recall"] * 100 for c in CLASSES]
    rec_ep15 = [data["ep15"]["eval"]["last_model"]["gt_crops"]["per_class"][c]["recall"] * 100 for c in CLASSES]
    rec_b = [data["baseline"]["eval"]["gt_crops"]["per_class"][c]["recall"] * 100 for c in CLASSES]

    r1 = ax_recall.bar(x - 1.5 * width, rec_ep5, width, label="Epochs = 5 (Last)", color="#e67e22")
    r2 = ax_recall.bar(x - 0.5 * width, rec_ep10, width, label="Epochs = 10 (Last)", color="#27ae60")
    r3 = ax_recall.bar(x + 0.5 * width, rec_ep15, width, label="Epochs = 15 (Last)", color="#c0392b")
    r4 = ax_recall.bar(x + 1.5 * width, rec_b, width, label="Baseline (31輪)", color="#2980b9")

    ax_recall.set_title("(c) 測試集 6 類不良品獨立召回率 (Per-class Recall)", fontsize=13, fontweight="bold", pad=10)
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(classes_display, fontsize=10)
    ax_recall.set_ylabel("Recall (%)", fontsize=11)
    ax_recall.set_ylim(94, 102)
    ax_recall.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_recall.legend(loc="lower left", fontsize=9)

    # -------------------------------------------------------------
    # 子圖 4: 全局指標與端到端表現 (Overall & End-to-End Metrics)
    # -------------------------------------------------------------
    ax_macro = axes[1, 1]
    metrics_labels = ["Macro Recall\n(宏平均召回)", "Micro Recall\n(測試集 Acc)", "End-to-End\nmAP@0.5"]
    xm = np.arange(len(metrics_labels))

    m_ep5 = [
        data["ep5"]["eval"]["last_model"]["gt_crops"]["macro_recall"] * 100,
        data["ep5"]["eval"]["last_model"]["gt_crops"]["accuracy"] * 100,
        data["ep5"]["eval"]["last_model"]["end_to_end"]["cnn"]["mAP@0.5"] * 100,
    ]
    m_ep10 = [
        data["ep10"]["eval"]["last_model"]["gt_crops"]["macro_recall"] * 100,
        data["ep10"]["eval"]["last_model"]["gt_crops"]["accuracy"] * 100,
        data["ep10"]["eval"]["last_model"]["end_to_end"]["cnn"]["mAP@0.5"] * 100,
    ]
    m_ep15 = [
        data["ep15"]["eval"]["last_model"]["gt_crops"]["macro_recall"] * 100,
        data["ep15"]["eval"]["last_model"]["gt_crops"]["accuracy"] * 100,
        data["ep15"]["eval"]["last_model"]["end_to_end"]["cnn"]["mAP@0.5"] * 100,
    ]
    m_b = [
        100.0,
        data["baseline"]["eval"]["gt_crops"]["accuracy"] * 100,
        data["baseline"]["eval"]["end_to_end"]["cnn"]["mAP@0.5"] * 100,
    ]

    bm1 = ax_macro.bar(xm - 1.5 * width, m_ep5, width, label="Epochs = 5 (Last)", color="#e67e22")
    bm2 = ax_macro.bar(xm - 0.5 * width, m_ep10, width, label="Epochs = 10 (Last)", color="#27ae60")
    bm3 = ax_macro.bar(xm + 0.5 * width, m_ep15, width, label="Epochs = 15 (Last)", color="#c0392b")
    bm4 = ax_macro.bar(xm + 1.5 * width, m_b, width, label="Baseline (31輪)", color="#2980b9")

    # 為長條圖加上數值標籤
    for b_group in [bm1, bm2, bm3, bm4]:
        for bar in b_group:
            h = bar.get_height()
            ax_macro.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.4,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax_macro.set_title("(d) 系統全局指標與端到端 mAP 比較", fontsize=13, fontweight="bold", pad=10)
    ax_macro.set_xticks(xm)
    ax_macro.set_xticklabels(metrics_labels, fontsize=10)
    ax_macro.set_ylabel("指標百分比 (%)", fontsize=11)
    ax_macro.set_ylim(80, 105)
    ax_macro.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_macro.legend(loc="lower left", fontsize=9)

    fig.suptitle("PCB-AOI Stage 2: CNN 訓練輪數 (Epochs = 5, 10, 15 vs Baseline) 綜合實驗評估分析", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()

    out_file = FIG_DIR / "02_epochs_comparison_summary.png"
    fig.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"成功輸出高解析度綜合對比圖: {out_file}")


def format_table_and_markdown(data):
    # 定義數值格式化輔助函式
    def fmt_num(v_last, v_best, is_pct=False, decimals=2, loss_dec=4, best_ep=None):
        if is_pct:
            s_last = f"{v_last * 100:.{decimals}f}%"
            s_best = f"{v_best * 100:.{decimals}f}%"
        else:
            s_last = f"{v_last:.{loss_dec}f}"
            s_best = f"{v_best:.{loss_dec}f}"
        if s_last == s_best or best_ep is None:
            return s_last
        return f"{s_last} (Best@Ep{best_ep}: {s_best})"

    rows = []

    # 1. 實驗組 A (Epochs = 5)
    d5 = data["ep5"]
    last_log5, best_log5 = d5["log"][-1], d5["log"][d5["best_ep"] - 1]
    last_m5, best_m5 = d5["eval"]["last_model"], d5["eval"]["best_model"]
    row5 = {
        "group": "實驗組 A (`Epochs = 5`)",
        "train_loss": fmt_num(float(last_log5["train_loss"]), float(best_log5["train_loss"]), loss_dec=4, best_ep=d5["best_ep"]),
        "val_loss": fmt_num(float(last_log5["val_loss"]), float(best_log5["val_loss"]), loss_dec=4, best_ep=d5["best_ep"]),
        "val_acc": fmt_num(float(last_log5["val_acc"]), float(best_log5["val_acc"]), is_pct=True, decimals=2, best_ep=d5["best_ep"]),
        "macro_recall": fmt_num(last_m5["gt_crops"]["macro_recall"], best_m5["gt_crops"]["macro_recall"], is_pct=True, decimals=2, best_ep=d5["best_ep"]),
        "micro_recall": fmt_num(last_m5["gt_crops"]["accuracy"], best_m5["gt_crops"]["accuracy"], is_pct=True, decimals=2, best_ep=d5["best_ep"]),
        "e2e_map": fmt_num(last_m5["end_to_end"]["cnn"]["mAP@0.5"], best_m5["end_to_end"]["cnn"]["mAP@0.5"], is_pct=True, decimals=2, best_ep=d5["best_ep"]),
    }
    for c in CLASSES:
        row5[c] = fmt_num(
            last_m5["gt_crops"]["per_class"][c]["recall"],
            best_m5["gt_crops"]["per_class"][c]["recall"],
            is_pct=True,
            decimals=2,
            best_ep=d5["best_ep"],
        )
    rows.append(row5)

    # 2. 實驗組 B (Epochs = 10)
    d10 = data["ep10"]
    last_log10, best_log10 = d10["log"][-1], d10["log"][d10["best_ep"] - 1]
    last_m10, best_m10 = d10["eval"]["last_model"], d10["eval"]["best_model"]
    row10 = {
        "group": "實驗組 B (`Epochs = 10`)",
        "train_loss": fmt_num(float(last_log10["train_loss"]), float(best_log10["train_loss"]), loss_dec=4, best_ep=d10["best_ep"]),
        "val_loss": fmt_num(float(last_log10["val_loss"]), float(best_log10["val_loss"]), loss_dec=4, best_ep=d10["best_ep"]),
        "val_acc": fmt_num(float(last_log10["val_acc"]), float(best_log10["val_acc"]), is_pct=True, decimals=2, best_ep=d10["best_ep"]),
        "macro_recall": fmt_num(last_m10["gt_crops"]["macro_recall"], best_m10["gt_crops"]["macro_recall"], is_pct=True, decimals=2, best_ep=d10["best_ep"]),
        "micro_recall": fmt_num(last_m10["gt_crops"]["accuracy"], best_m10["gt_crops"]["accuracy"], is_pct=True, decimals=2, best_ep=d10["best_ep"]),
        "e2e_map": fmt_num(last_m10["end_to_end"]["cnn"]["mAP@0.5"], best_m10["end_to_end"]["cnn"]["mAP@0.5"], is_pct=True, decimals=2, best_ep=d10["best_ep"]),
    }
    for c in CLASSES:
        row10[c] = fmt_num(
            last_m10["gt_crops"]["per_class"][c]["recall"],
            best_m10["gt_crops"]["per_class"][c]["recall"],
            is_pct=True,
            decimals=2,
            best_ep=d10["best_ep"],
        )
    rows.append(row10)

    # 3. 實驗組 C (Epochs = 15)
    d15 = data["ep15"]
    last_log15, best_log15 = d15["log"][-1], d15["log"][d15["best_ep"] - 1]
    last_m15, best_m15 = d15["eval"]["last_model"], d15["eval"]["best_model"]
    row15 = {
        "group": "實驗組 C (`Epochs = 15`)",
        "train_loss": fmt_num(float(last_log15["train_loss"]), float(best_log15["train_loss"]), loss_dec=4, best_ep=d15["best_ep"]),
        "val_loss": fmt_num(float(last_log15["val_loss"]), float(best_log15["val_loss"]), loss_dec=4, best_ep=d15["best_ep"]),
        "val_acc": fmt_num(float(last_log15["val_acc"]), float(best_log15["val_acc"]), is_pct=True, decimals=2, best_ep=d15["best_ep"]),
        "macro_recall": fmt_num(last_m15["gt_crops"]["macro_recall"], best_m15["gt_crops"]["macro_recall"], is_pct=True, decimals=2, best_ep=d15["best_ep"]),
        "micro_recall": fmt_num(last_m15["gt_crops"]["accuracy"], best_m15["gt_crops"]["accuracy"], is_pct=True, decimals=2, best_ep=d15["best_ep"]),
        "e2e_map": fmt_num(last_m15["end_to_end"]["cnn"]["mAP@0.5"], best_m15["end_to_end"]["cnn"]["mAP@0.5"], is_pct=True, decimals=2, best_ep=d15["best_ep"]),
    }
    for c in CLASSES:
        row15[c] = fmt_num(
            last_m15["gt_crops"]["per_class"][c]["recall"],
            best_m15["gt_crops"]["per_class"][c]["recall"],
            is_pct=True,
            decimals=2,
            best_ep=d15["best_ep"],
        )
    rows.append(row15)

    # 4. 基準對照組 (Baseline: max_epochs=50, Early Stopping @ 31)
    db = data["baseline"]
    last_log_b, best_log_b = db["log"][-1], db["log"][db["best_ep"] - 1]
    eval_b = db["eval"]
    row_b = {
        "group": "基準對照組 (`Baseline`, max=50, 實際 31 輪)",
        "train_loss": fmt_num(float(last_log_b["train_loss"]), float(best_log_b["train_loss"]), loss_dec=4, best_ep=db["best_ep"]),
        "val_loss": fmt_num(float(last_log_b["val_loss"]), float(best_log_b["val_loss"]), loss_dec=4, best_ep=db["best_ep"]),
        "val_acc": fmt_num(float(last_log_b["val_acc"]), float(best_log_b["val_acc"]), is_pct=True, decimals=2, best_ep=db["best_ep"]),
        "macro_recall": "100.00%",
        "micro_recall": f"{eval_b['gt_crops']['accuracy']*100:.2f}%",
        "e2e_map": f"{eval_b['end_to_end']['cnn']['mAP@0.5']*100:.2f}%",
    }
    for c in CLASSES:
        row_b[c] = f"{eval_b['gt_crops']['per_class'][c]['recall']*100:.2f}%"
    rows.append(row_b)

    # 輸出 CSV
    csv_fields = [
        "Training Epochs",
        "Train Loss",
        "Val Loss",
        "Val Accuracy (%)",
        "Macro Recall (%)",
        "Micro Recall / Test Acc (%)",
        "missing_hole Recall (%)",
        "mouse_bite Recall (%)",
        "open_circuit Recall (%)",
        "short Recall (%)",
        "spur Recall (%)",
        "spurious_copper Recall (%)",
        "End-to-End mAP@0.5 (%)",
    ]

    csv_path = RESULTS_DIR / "topic4_epochs_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        for r in rows:
            writer.writerow([
                r["group"].replace("`", ""),
                r["train_loss"],
                r["val_loss"],
                r["val_acc"],
                r["macro_recall"],
                r["micro_recall"],
                r["missing_hole"],
                r["mouse_bite"],
                r["open_circuit"],
                r["short"],
                r["spur"],
                r["spurious_copper"],
                r["e2e_map"],
            ])
    print(f"成功輸出量化指標 CSV: {csv_path}")

    # 輸出 Markdown
    md_lines = [
        "# 作業二主題 4：CNN 訓練輪數（Training Epochs）對分類性能與召回率影響量化總表",
        "",
        "> **表格數值呈現規則**：以該組最後一輪（`Last Epoch`）的實測值為主數據；若該組內 `Best Val Epoch` 數值不同，則於括號內附註 `(Best@EpX: 數值)`。",
        "",
        "| 實驗組別設定 (`Training Epochs`) | Train Loss | Val Loss | Val Accuracy (%) | Macro Recall (%) | Micro Recall / Test Acc (%) | missing_hole Recall (%) | mouse_bite Recall (%) | open_circuit Recall (%) | short Recall (%) | spur Recall (%) | spurious_copper Recall (%) | End-to-End mAP@0.5 (%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        line = (
            f"| {r['group']} | {r['train_loss']} | {r['val_loss']} | {r['val_acc']} | "
            f"{r['macro_recall']} | {r['micro_recall']} | {r['missing_hole']} | "
            f"{r['mouse_bite']} | {r['open_circuit']} | {r['short']} | {r['spur']} | "
            f"{r['spurious_copper']} | {r['e2e_map']} |"
        )
        md_lines.append(line)

    md_lines.extend([
        "",
        "---",
        "",
        "## 各組混淆矩陣誤判案例深入分析（Misclassification Case Breakdown）",
        "",
        "在測試集（Test Split）460 個真實缺陷塊（GT Crops）下，各設定之混淆矩陣分類誤判案例如下：",
        "",
        "### 1. 實驗組 A (`Epochs = 5`)",
        "- **最後一輪（Last Epoch 5）**：共 4 筆誤判（測試集準確率 **99.13%**, 456/460）：",
        "  1. `missing_hole`（缺孔）被誤判為 `short`（短路）：**1 筆**",
        "  2. `spur`（微短路突刺）被誤判為 `mouse_bite`（邊緣凹折囓齒）：**2 筆**",
        "  3. `spurious_copper`（假銅斑）被誤判為 `short`（短路）：**1 筆**",
        "- **最佳驗證輪（Best@Ep4）**：共 1 筆誤判（測試集準確率 **99.78%**, 459/460）：",
        "  1. `spur` 被誤判為 `mouse_bite`：**1 筆**",
        "",
        "### 2. 實驗組 B (`Epochs = 10`)",
        "- **最後一輪（Last Epoch 10）**：共 3 筆誤判（測試集準確率 **99.35%**, 457/460）：",
        "  1. `missing_hole` 被誤判為 `short`：**1 筆**",
        "  2. `spurious_copper` 被誤判為 `short`：**2 筆**",
        "- **最佳驗證輪（Best@Ep8）**：共 2 筆誤判（測試集準確率 **99.57%**, 458/460）：",
        "  1. `missing_hole` 被誤判為 `short`：**1 筆**",
        "  2. `spurious_copper` 被誤判為 `short`：**1 筆**",
        "",
        "### 3. 實驗組 C (`Epochs = 15`)",
        "- **最後一輪（Last Epoch 15）**：共 3 筆誤判（測試集準確率 **99.35%**, 457/460）：",
        "  1. `missing_hole` 被誤判為 `short`：**1 筆**",
        "  2. `open_circuit`（開路）被誤判為 `mouse_bite`：**1 筆**",
        "  3. `spur` 被誤判為 `mouse_bite`：**1 筆**",
        "- **最佳驗證輪（Best@Ep11）**：共 2 筆誤判（測試集準確率 **99.57%**, 458/460）：",
        "  1. `missing_hole` 被誤判為 `short`：**1 筆**",
        "  2. `mouse_bite` 被誤判為 `spur`：**1 筆**",
        "",
        "### 4. 基準對照組 (`Baseline`, max=50, Early Stopping @ 31 輪，最佳權重 @ Ep21)",
        "- **最佳驗證權重（Best@Ep21）**：**0 筆誤判**（測試集準確率 **100.00%**, 460/460，全部正分類）。",
        "",
        "---",
        "",
        "## 核心工程洞察與結論（Key Engineering Insights）",
        "",
        "1. **極速收斂性（Rapid Convergence via Transfer Learning）**：",
        "   - 受惠於 ResNet18 在 ImageNet 上的預訓練特徵表示，模型在 **Epoch 4~5** 時驗證集準確率即已躍升至 **99.32% ~ 99.77%**，測試集 Macro Recall 超過 **99.14%**。",
        "   - 不良品召回率在短短 5 輪內便全面突破 97.4%，展現微調（Fine-tuning）在工業外觀瑕疵分類上的極高收斂效率。",
        "",
        "2. **形態相近類別的判別邊界（Morphological Feature Ambiguity）**：",
        "   - 誤判高度集中於「邊緣形態相反」的缺陷對：`spur`（突起）與 `mouse_bite`（咬痕）。兩者皆位於銅箔導線邊緣且面積微小，早期輪數較易因空間細節關注不足產生偶發混淆。",
        "   - 另一類為 `spurious_copper` 與 `short`，兩者皆為額外的異常銅斑，當假銅斑恰好貼近相鄰兩條導線時，在特徵抽象層次極易與短路橋接重疊。",
        "",
        "3. **候選框召回率瓶頸與系統邊界效應（Stage 1 Proposal Bottleneck）**：",
        "   - 三組設定在 End-to-End 系統評估中的 mAP@0.5 分別為 **84.94% (Ep5)**、**85.39% (Ep10)**、**84.47% (Ep15)**，與訓練 31 輪 Baseline 的 **84.75%** 完全處於同一水準（差異僅在 ±0.6% 範圍內）。",
        "   - 充分驗證了：**一旦 CNN 分類器在第 5 輪達到 ~99% 準確率後，進一步增加訓練輪數對整體 AOI 檢測系統的端到端指標並無顯著增益；系統總體表現被嚴格箝制於 Stage 1 傳統演算法對 `short` 缺陷候選框的低召回率（僅 ~43% AP），這正是促使專案邁向 Stage 3 端到端 YOLO 的關鍵工程動機！**",
    ])

    md_path = RESULTS_DIR / "topic4_epochs_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"成功輸出量化指標 Markdown: {md_path}")


def main():
    print("載入所有實驗數據...")
    data = load_all_data()
    print("開始繪製高解析度綜合對比圖...")
    plot_summary_figure(data)
    print("開始彙整量化指標總表與分析報告...")
    format_table_and_markdown(data)
    print("全數任務執行完成！")


if __name__ == "__main__":
    main()
