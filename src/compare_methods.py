"""Stage 4:三方法統一比較(同 test split,seed=42)。

不重新訓練,只整合 Stage 1/2/3 既有結果,並補齊比較所需但前面沒算的量:
mAP@0.5:0.95、純推論時間、PR 曲線資料。輸出:

* results/comparison_table.csv / .md  —— 三列比較表
* results/intermediate/comparison_data.json —— per-class 與 PR 曲線資料(供畫圖)

純推論時間定義(統一只計「模型 forward 那段」,不含載圖):
* Stage 1:傳統 pipeline 的 register + difference + threshold + classify
* Stage 2:ResNet18 forward(**不含 Stage 1 proposal 生成**)
* Stage 3:5.4 ms(ultralytics 純推論)

    python -m src.compare_methods
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import build_splits, crop_with_padding, get_transforms
from .dl_classifier import load_model, DEFAULT_WEIGHTS
from .evaluate_classifier import iou, voc_ap
from .hripcb import CLASSES, get_template_for_test, parse_voc
from .traditional_cv import run_traditional_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"
RESULTS = ROOT / "results"
YOLO_BEST = RESULTS / "yolo_training" / "run1" / "weights" / "best.pt"
IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.5 .. 0.95

# Stage 3 純推論時間(ultralytics 內計,固定;見 yolo_report.md)
STAGE3_INFER_MS = 5.4


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #
def _map_at(records, thr, classes=CLASSES) -> float:
    """單一 IoU 門檻下的 mAP(per-class VOC AP 平均)。"""
    aps = []
    for cls in classes:
        preds, n_gt = [], 0
        for rec in records:
            cls_gts = [bb for c, bb in rec["gts"] if c == cls]
            n_gt += len(cls_gts)
            matched = [False] * len(cls_gts)
            for dc, dbb, conf in sorted(rec["dets"], key=lambda d: d[2], reverse=True):
                if dc != cls:
                    continue
                best, bj = 0.0, -1
                for j, gbb in enumerate(cls_gts):
                    if matched[j]:
                        continue
                    v = iou(dbb, gbb)
                    if v > best:
                        best, bj = v, j
                tp = best >= thr and bj >= 0
                if tp:
                    matched[bj] = True
                preds.append((conf, tp))
        if not n_gt:
            continue
        preds.sort(key=lambda x: x[0], reverse=True)
        tp_arr = np.array([1 if t else 0 for _, t in preds])
        if tp_arr.size == 0:
            aps.append(0.0); continue
        tp_cum, fp_cum = np.cumsum(tp_arr), np.cumsum(1 - tp_arr)
        rec_c = tp_cum / n_gt
        prec_c = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(voc_ap(rec_c, prec_c))
    return float(np.mean(aps)) if aps else 0.0


def _map_5095(records) -> float:
    return float(np.mean([_map_at(records, t) for t in IOU_THRS]))


def _micro_pr(records, thr=0.5):
    """跨類別 micro-average 的 PR 曲線點(供三方法疊圖)。"""
    all_dets, total_gt = [], 0
    for rec in records:
        gts = rec["gts"]; total_gt += len(gts)
        matched = [False] * len(gts)
        for dc, dbb, conf in sorted(rec["dets"], key=lambda d: d[2], reverse=True):
            best, bj = 0.0, -1
            for j, (gc, gbb) in enumerate(gts):
                if matched[j] or gc != dc:
                    continue
                v = iou(dbb, gbb)
                if v > best:
                    best, bj = v, j
            tp = best >= thr and bj >= 0
            if tp:
                matched[bj] = True
            all_dets.append((conf, tp))
    all_dets.sort(key=lambda x: x[0], reverse=True)
    tp_arr = np.array([1 if t else 0 for _, t in all_dets])
    if tp_arr.size == 0 or total_gt == 0:
        return [0.0], [1.0]
    tp_cum, fp_cum = np.cumsum(tp_arr), np.cumsum(1 - tp_arr)
    recall = (tp_cum / total_gt).tolist()
    precision = (tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)).tolist()
    return recall, precision


def _macro_pr_at_conf0(records, classes=CLASSES):
    """conf 門檻 0(全部 detections)下的 macro 平均 precision / recall。"""
    Ps, Rs = [], []
    for cls in classes:
        tp = fp = fn = 0
        for rec in records:
            cls_gts = [bb for c, bb in rec["gts"] if c == cls]
            matched = [False] * len(cls_gts)
            for dc, dbb, _ in sorted(rec["dets"], key=lambda d: d[2], reverse=True):
                if dc != cls:
                    continue
                best, bj = 0.0, -1
                for j, gbb in enumerate(cls_gts):
                    if matched[j]:
                        continue
                    v = iou(dbb, gbb)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5 and bj >= 0:
                    matched[bj] = True; tp += 1
                else:
                    fp += 1
            fn += matched.count(False)
        Ps.append(tp / (tp + fp) if tp + fp else 0.0)
        Rs.append(tp / (tp + fn) if tp + fn else 0.0)
    return float(np.mean(Ps)), float(np.mean(Rs))


# --------------------------------------------------------------------------- #
# 跑 Stage 1 + Stage 2 over test split(取 records + 計時純推論)
# --------------------------------------------------------------------------- #
def _run_s1_s2(device):
    items = build_splits(DATA)["test"]
    cnn = load_model(DEFAULT_WEIGHTS, device)
    tf = get_transforms("test")

    s1_records, s2_records = [], []
    s1_infer_s = 0.0          # register+difference+threshold+classify 累計
    cnn_forward_s = 0.0       # ResNet18 forward 累計(不含 Stage 1)
    n = len(items)

    for it in items:
        img_path = it["image"]
        tmpl = get_template_for_test(Path(img_path).name, DATA / "PCB_USED")
        out = run_traditional_pipeline(img_path, tmpl)
        t = out["timings"]
        s1_infer_s += t["register"] + t["difference"] + t["threshold"] + t["classify"]
        dets = out["detections"]
        gts = [(o.name, (o.xmin, o.ymin, o.xmax, o.ymax))
               for o in parse_voc(it["xml"]).objects]
        s1_records.append({"gts": gts,
                           "dets": [(d["class"], d["bbox"], d["confidence"]) for d in dets]})

        # Stage 2:對同一批 proposals 用 CNN 分類,單獨計 forward 時間
        img = cv2.imread(img_path)
        if dets:
            batch = torch.stack([
                tf(Image.fromarray(cv2.cvtColor(
                    crop_with_padding(img, d["bbox"], 0.3), cv2.COLOR_BGR2RGB)))
                for d in dets]).to(device)
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                prob = torch.softmax(cnn(batch), 1)
            if device != "cpu":
                torch.cuda.synchronize()
            cnn_forward_s += time.perf_counter() - t0
            conf, idx = prob.max(1)
            s2 = [(CLASSES[int(idx[i])], dets[i]["bbox"], float(conf[i]))
                  for i in range(len(dets))]
        else:
            s2 = []
        s2_records.append({"gts": gts, "dets": s2})

    return {
        "s1_records": s1_records, "s2_records": s2_records,
        "s1_infer_ms": s1_infer_s / n * 1000,
        "s2_infer_ms": cnn_forward_s / n * 1000,
        "n_images": n,
    }


# --------------------------------------------------------------------------- #
# Stage 3:YOLO 在 test 上的 detections(供 PR)
# --------------------------------------------------------------------------- #
def _run_s3(device):
    from ultralytics import YOLO
    model = YOLO(str(YOLO_BEST))
    items = build_splits(DATA)["test"]
    records = []
    for it in items:
        r = model.predict(it["image"], imgsz=640, device=device, conf=0.001, verbose=False)[0]
        names = r.names
        dets = [(names[int(b.cls)], tuple(b.xyxy[0].cpu().numpy().tolist()), float(b.conf))
                for b in r.boxes]
        gts = [(o.name, (o.xmin, o.ymin, o.xmax, o.ymax))
               for o in parse_voc(it["xml"]).objects]
        records.append({"gts": gts, "dets": dets})
    return records


# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  跑 Stage 1+2 over test split ...")
    s12 = _run_s1_s2(device)

    # 既有官方數字(test split,seed=42),用於填表並交叉驗證
    clf = json.loads((RESULTS / "classifier_eval_summary.json").read_text())
    yolo = json.loads((RESULTS / "yolo_eval_summary.json").read_text())
    s1_off = clf["end_to_end"]["stage1"]["mAP@0.5"]
    s2_off = clf["end_to_end"]["cnn"]["mAP@0.5"]

    # 自算 mAP@0.5 交叉驗證(必須對得上既有數字)
    s1_map50 = _map_at(s12["s1_records"], 0.5)
    s2_map50 = _map_at(s12["s2_records"], 0.5)
    print(f"交叉驗證 Stage1 mAP@0.5: 自算 {s1_map50:.4f} vs 既有 {s1_off:.4f}")
    print(f"交叉驗證 Stage2 mAP@0.5: 自算 {s2_map50:.4f} vs 既有 {s2_off:.4f}")
    for name, a, b in [("S1", s1_map50, s1_off), ("S2", s2_map50, s2_off)]:
        if abs(a - b) > 0.02:
            raise SystemExit(f"!! {name} mAP@0.5 自算與既有差異過大({a:.3f} vs {b:.3f}),停下")

    s1_p, s1_r = _macro_pr_at_conf0(s12["s1_records"])
    s2_p, s2_r = _macro_pr_at_conf0(s12["s2_records"])
    s1_map5095 = _map_5095(s12["s1_records"])
    s2_map5095 = _map_5095(s12["s2_records"])

    print("跑 Stage 3 YOLO predict over test split ...")
    s3_records = _run_s3(device)
    s3_map50_self = _map_at(s3_records, 0.5)
    print(f"交叉驗證 Stage3 mAP@0.5: 自算 {s3_map50_self:.4f} vs 官方 {yolo['mAP@0.5']:.4f}")

    # PR 曲線資料(三方法 micro PR)
    pr = {
        "stage1": _micro_pr(s12["s1_records"]),
        "stage2": _micro_pr(s12["s2_records"]),
        "stage3": _micro_pr(s3_records),
    }

    import os
    s2_size = round(os.path.getsize(DEFAULT_WEIGHTS) / 1e6, 1)
    s3_size = round(os.path.getsize(YOLO_BEST) / 1e6, 1)

    rows = [
        {"method": "Stage 1:傳統 CV", "mAP@0.5": round(s1_off, 4),
         "mAP@0.5:0.95": round(s1_map5095, 4), "precision": round(s1_p, 4),
         "recall": round(s1_r, 4), "inference_ms": round(s12["s1_infer_ms"], 1),
         "model_size_mb": "N/A",
         "notes": "差影+形態學;hand-crafted feature 分類撞牆"},
        {"method": "Stage 2:R-CNN 風格(proposals+CNN)", "mAP@0.5": round(s2_off, 4),
         "mAP@0.5:0.95": round(s2_map5095, 4), "precision": round(s2_p, 4),
         "recall": round(s2_r, 4), "inference_ms": round(s12["s2_infer_ms"], 1),
         "model_size_mb": s2_size,
         "notes": "ResNet18 forward(不含 Stage1 proposal 生成);受 proposal recall 綁住"},
        {"method": "Stage 3:YOLOv8n 端到端", "mAP@0.5": round(yolo["mAP@0.5"], 4),
         "mAP@0.5:0.95": round(yolo["mAP@0.5:0.95"], 4),
         "precision": None, "recall": None, "inference_ms": STAGE3_INFER_MS,
         "model_size_mb": s3_size,
         "notes": "端到端,無 proposal 依賴;突破 short recall 瓶頸"},
    ]
    # Stage 3 的 P/R 用自算 macro(與 S1/S2 同方法)
    s3_p, s3_r = _macro_pr_at_conf0([{"gts": r["gts"],
                                      "dets": [d for d in r["dets"] if d[2] >= 0.25]}
                                     for r in s3_records])
    rows[2]["precision"] = round(s3_p, 4)
    rows[2]["recall"] = round(s3_r, 4)

    _write_table(rows)

    per_class = {
        "stage1": {c: clf["end_to_end"]["stage1"]["per_class"][c]["ap"] for c in CLASSES},
        "stage2": {c: clf["end_to_end"]["cnn"]["per_class"][c]["ap"] for c in CLASSES},
        "stage3": {c: yolo["per_class_ap50"][c] for c in CLASSES},
    }
    (RESULTS / "intermediate").mkdir(parents=True, exist_ok=True)
    (RESULTS / "intermediate" / "comparison_data.json").write_text(json.dumps(
        {"rows": rows, "per_class_ap50": per_class, "pr_curves": pr,
         "speedup_s1_to_s3": round(s12["s1_infer_ms"] / STAGE3_INFER_MS, 1)}, indent=2))
    print(f"\n加速倍率 Stage1→Stage3(純推論): {s12['s1_infer_ms']/STAGE3_INFER_MS:.1f}x")
    print(f"寫入 {RESULTS/'comparison_table.csv'} / .md / intermediate/comparison_data.json")


def _write_table(rows):
    import csv
    cols = ["method", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall",
            "inference_ms", "model_size_mb", "notes"]
    with open(RESULTS / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    # markdown
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (RESULTS / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
