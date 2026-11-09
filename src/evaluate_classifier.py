"""Stage 2 分類器評估。

兩種評估,缺一不可 —— 對比才能看出「CNN 分類能力強,但受限於 Stage 1 的
proposal 品質」這個 narrative:

1. **GT crops**(理想上限):用 ground-truth bbox 切 crop 分類,衡量 CNN 純分類
   能力(accuracy / per-class P-R-F1 / confusion)。
2. **End-to-end**(實戰):對 test 影像跑 Stage 1 pipeline 取得 proposals,
   *同一批 bbox* 分別用 Stage 1 rule-based 分類器與 Stage 2 CNN 分類,各算
   mAP@0.5。同樣的 proposals、只差分類器,直接對比 CNN 帶來的增益。

    python -m src.evaluate_classifier
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import (PCBDefectCropDataset, build_splits, collate_crops,
                      crop_with_padding, get_transforms)
from .dl_classifier import load_model, DEFAULT_WEIGHTS
from .evaluate_traditional import iou, voc_ap
from .hripcb import CLASSES, get_template_for_test, parse_voc
from .traditional_cv import run_traditional_pipeline
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"
IOU_THRESH = 0.5


# --------------------------------------------------------------------------- #
# 共用:per-class P/R/F1 + mAP(沿用 Stage 1 的 VOC 比對邏輯)
# --------------------------------------------------------------------------- #
def compute_detection_metrics(records: list[dict], classes=CLASSES) -> dict:
    """records: 每張影像 ``{"gts": [(cls, bbox)], "dets": [(cls, bbox, conf)]}``。"""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    preds = defaultdict(list)   # class -> [(conf, is_tp)]
    n_gt = defaultdict(int)

    for rec in records:
        gts = rec["gts"]
        for c, _ in gts:
            n_gt[c] += 1
        dets = sorted(rec["dets"], key=lambda d: d[2], reverse=True)
        for cls in classes:
            cls_gts = [bb for c, bb in gts if c == cls]
            matched = [False] * len(cls_gts)
            for dc, dbb, conf in dets:
                if dc != cls:
                    continue
                best, bj = 0.0, -1
                for j, gbb in enumerate(cls_gts):
                    if matched[j]:
                        continue
                    v = iou(dbb, gbb)
                    if v > best:
                        best, bj = v, j
                is_tp = best >= IOU_THRESH and bj >= 0
                if is_tp:
                    matched[bj] = True; tp[cls] += 1
                else:
                    fp[cls] += 1
                preds[cls].append((conf, is_tp))
            fn[cls] += matched.count(False)

    per_class, aps = {}, {}
    for cls in classes:
        TP, FP, FN = tp[cls], fp[cls], fn[cls]
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        pr = sorted(preds[cls], key=lambda x: x[0], reverse=True)
        if pr and n_gt[cls]:
            tp_arr = np.array([1 if t else 0 for _, t in pr])
            tp_cum = np.cumsum(tp_arr); fp_cum = np.cumsum(1 - tp_arr)
            rec_c = tp_cum / n_gt[cls]
            prec_c = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = voc_ap(rec_c, prec_c)
        else:
            ap = 0.0
        aps[cls] = ap
        per_class[cls] = {"precision": P, "recall": R, "f1": F1, "ap": ap,
                          "tp": TP, "fp": FP, "fn": FN}
    return {"mAP@0.5": float(np.mean([aps[c] for c in classes])), "per_class": per_class}


def confusion_from_records(records: list[dict], classes=CLASSES) -> dict:
    """class-agnostic localization 比對下的混淆矩陣(含 background 列/欄)。"""
    labels = list(classes) + ["background"]
    li = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for rec in records:
        gts = rec["gts"]
        dets = sorted(rec["dets"], key=lambda d: d[2], reverse=True)
        gmatched = [False] * len(gts)
        dmatched = [False] * len(dets)
        for di, (dc, dbb, _) in enumerate(dets):
            best, bj = 0.0, -1
            for j, (_, gbb) in enumerate(gts):
                if gmatched[j]:
                    continue
                v = iou(dbb, gbb)
                if v > best:
                    best, bj = v, j
            if best >= IOU_THRESH and bj >= 0:
                gmatched[bj] = True; dmatched[di] = True
                cm[li[gts[bj][0]], li[dc]] += 1
        for di, (dc, _, _) in enumerate(dets):
            if not dmatched[di]:
                cm[li["background"], li[dc]] += 1
        for j, m in enumerate(gmatched):
            if not m:
                cm[li[gts[j][0]], li["background"]] += 1
    return {"matrix": cm.tolist(), "labels": labels}


# --------------------------------------------------------------------------- #
# CNN 批次推論
# --------------------------------------------------------------------------- #
class _CropClassifier:
    def __init__(self, model, device, pad_ratio=0.3):
        self.model, self.device = model, device
        self.tf = get_transforms("test")
        self.pad_ratio = pad_ratio

    @torch.no_grad()
    def predict_patches(self, patches_rgb: list[np.ndarray], batch=128):
        """回傳 (class_idx array, confidence array)。"""
        if not patches_rgb:
            return np.array([], dtype=int), np.array([])
        cls_out, conf_out = [], []
        for i in range(0, len(patches_rgb), batch):
            chunk = patches_rgb[i:i + batch]
            x = torch.stack([self.tf(Image.fromarray(p)) for p in chunk]).to(self.device)
            prob = torch.softmax(self.model(x), dim=1)
            conf, idx = prob.max(dim=1)
            cls_out.append(idx.cpu().numpy()); conf_out.append(conf.cpu().numpy())
        return np.concatenate(cls_out), np.concatenate(conf_out)


# --------------------------------------------------------------------------- #
# 1. GT crops 評估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_gt_crops(model, device, split="test") -> dict:
    ds = PCBDefectCropDataset("gt", split)
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_crops)
    y_true, y_pred = [], []
    for imgs, labels, _ in ld:
        out = model(imgs.to(device))
        y_pred.extend(out.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        per_class[c] = {"precision": float(P), "recall": float(R), "f1": float(F1),
                        "support": int(cm[i].sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    recalls = [per_class[c]["recall"] for c in CLASSES]
    precisions = [per_class[c]["precision"] for c in CLASSES]
    f1s = [per_class[c]["f1"] for c in CLASSES]
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    total_tp = sum(cm[i, i] for i in range(n))
    total_fn = sum(cm[i, :].sum() - cm[i, i] for i in range(n))
    denom = total_tp + total_fn
    micro_recall = float(total_tp / denom) if denom > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": len(y_true),
    }


# --------------------------------------------------------------------------- #
# 2. End-to-end 評估(同一批 Stage 1 proposals,對比 rule-based vs CNN)
# --------------------------------------------------------------------------- #
def evaluate_end_to_end(model, device, config=None, split="test", pad_ratio=0.3) -> dict:
    clf = _CropClassifier(model, device, pad_ratio)
    items = build_splits(DATA)[split]

    cache_path = ROOT / "results" / "intermediate" / "stage1_test_proposals.json"
    use_cache = (config is None and split == "test")
    proposals_cache = None
    if use_cache and cache_path.exists():
        try:
            proposals_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"載入 Stage 1 proposals 快取: {cache_path}")
        except Exception as e:
            print(f"快取載入失敗，將重新執行 Stage 1 pipeline: {e}")
            proposals_cache = None

    new_cache = {} if (use_cache and proposals_cache is None) else None

    stage1_records, cnn_records = [], []
    sample_dets = []   # 供視覺化:保留少數影像的 proposals(兩種分類)
    for k, it in enumerate(items):
        img_path = it["image"]
        img_name = Path(img_path).name
        gts = [(o.name, (o.xmin, o.ymin, o.xmax, o.ymax)) for o in parse_voc(it["xml"]).objects]

        if proposals_cache is not None and (img_name in proposals_cache or img_path in proposals_cache):
            raw_dets = proposals_cache.get(img_name, proposals_cache.get(img_path))
            dets = [
                {
                    "bbox": tuple(d["bbox"]),
                    "class": d["class"],
                    "confidence": float(d["confidence"]),
                }
                for d in raw_dets
            ]
        else:
            tmpl = get_template_for_test(img_name, DATA / "PCB_USED")
            out = run_traditional_pipeline(img_path, tmpl, config)
            dets = out["detections"]
            if new_cache is not None:
                new_cache[img_name] = [
                    {
                        "bbox": list(d["bbox"]),
                        "class": d["class"],
                        "confidence": float(d["confidence"]),
                    }
                    for d in dets
                ]

        img = cv2.imread(img_path)
        patches = [cv2.cvtColor(crop_with_padding(img, d["bbox"], pad_ratio), cv2.COLOR_BGR2RGB)
                   for d in dets]
        cnn_idx, cnn_conf = clf.predict_patches(patches)

        s1 = [(d["class"], tuple(d["bbox"]), float(d["confidence"])) for d in dets]
        cnn = [(CLASSES[cnn_idx[i]], tuple(dets[i]["bbox"]), float(cnn_conf[i]))
               for i in range(len(dets))]
        stage1_records.append({"gts": gts, "dets": s1})
        cnn_records.append({"gts": gts, "dets": cnn})

        if k < 6:
            sample_dets.append({"image": img_path, "gts": gts, "stage1": s1, "cnn": cnn})

    if new_cache is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(new_cache, indent=2), encoding="utf-8")
        print(f"Stage 1 test proposals 已快取至: {cache_path}")

    return {
        "stage1": compute_detection_metrics(stage1_records),
        "cnn": compute_detection_metrics(cnn_records),
        "stage1_confusion": confusion_from_records(stage1_records),
        "cnn_confusion": confusion_from_records(cnn_records),
        "n_images": len(items),
        "sample_dets": sample_dets,
    }


# --------------------------------------------------------------------------- #
def _print_gt(res):
    print(f"\n[GT crops] accuracy = {res['accuracy']*100:.1f}%  (n={res['n']})")
    print(f"  macro_recall = {res.get('macro_recall', 0.0)*100:.2f}%  "
          f"micro_recall = {res.get('micro_recall', 0.0)*100:.2f}%  "
          f"macro_f1 = {res.get('macro_f1', 0.0)*100:.2f}%")
    print(f"{'class':<16}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    for c in CLASSES:
        m = res["per_class"][c]
        print(f"{c:<16}{m['precision']*100:>6.1f}{m['recall']*100:>7.1f}"
              f"{m['f1']*100:>7.1f}{m['support']:>9}")


def _print_e2e(res):
    s1, cnn = res["stage1"]["mAP@0.5"], res["cnn"]["mAP@0.5"]
    print(f"\n[End-to-end, test split, 同一批 Stage 1 proposals] n={res['n_images']} 影像")
    print(f"  Stage 1 rule-based  mAP@0.5 = {s1*100:.1f}%")
    print(f"  Stage 2 CNN         mAP@0.5 = {cnn*100:.1f}%   (Δ {(cnn-s1)*100:+.1f})")
    print(f"{'class':<16}{'S1 AP':>8}{'CNN AP':>8}")
    for c in CLASSES:
        print(f"{c:<16}{res['stage1']['per_class'][c]['ap']*100:>8.1f}"
              f"{res['cnn']['per_class'][c]['ap']*100:>8.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 缺陷分類器評估")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="模型權重 (或最佳權重) 路徑")
    parser.add_argument("--weights-last", type=str, default=None, help="最後一輪權重路徑 (可選，同時評估 best 與 last)")
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "classifier_eval_summary.json"), help="輸出 JSON 路徑")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = Path(args.weights)
    out_path = Path(args.out)

    if args.weights_last is not None:
        weights_last_path = Path(args.weights_last)
        print(f"=== 雙權重評估模式 ===")
        print(f"1. 評估 Last 模型: {weights_last_path}  (device={device})")
        model_last = load_model(weights_last_path, device)
        gt_last = evaluate_gt_crops(model_last, device, "test")
        _print_gt(gt_last)
        e2e_last = evaluate_end_to_end(model_last, device, split="test")
        _print_e2e(e2e_last)

        print(f"\n2. 評估 Best 模型: {weights_path}  (device={device})")
        model_best = load_model(weights_path, device)
        gt_best = evaluate_gt_crops(model_best, device, "test")
        _print_gt(gt_best)
        e2e_best = evaluate_end_to_end(model_best, device, split="test")
        _print_e2e(e2e_best)

        last_model_data = {
            "weights": str(weights_last_path),
            "gt_crops": gt_last,
            "end_to_end": {k: v for k, v in e2e_last.items() if k != "sample_dets"},
        }
        best_model_data = {
            "weights": str(weights_path),
            "gt_crops": gt_best,
            "end_to_end": {k: v for k, v in e2e_best.items() if k != "sample_dets"},
        }
        out = {
            "gt_crops": gt_last,
            "end_to_end": {k: v for k, v in e2e_last.items() if k != "sample_dets"},
            "last_model": last_model_data,
            "best_model": best_model_data,
        }
        sample_dets_data = e2e_last["sample_dets"]
    else:
        print(f"載入模型 {weights_path}  device={device}")
        model = load_model(weights_path, device)
        gt = evaluate_gt_crops(model, device, "test")
        _print_gt(gt)
        e2e = evaluate_end_to_end(model, device, split="test")
        _print_e2e(e2e)
        out = {
            "gt_crops": gt,
            "end_to_end": {k: v for k, v in e2e.items() if k != "sample_dets"},
        }
        sample_dets_data = e2e["sample_dets"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n寫入 {out_path}")

    # 若輸出目標為預設的 classifier_eval_summary.json，同時存 intermediate/classifier_eval.json (含 sample_dets)
    if out_path.name == "classifier_eval_summary.json":
        res_dir = ROOT / "results"
        (res_dir / "intermediate").mkdir(parents=True, exist_ok=True)
        inter_path = res_dir / "intermediate" / "classifier_eval.json"
        inter_path.write_text(
            json.dumps({**out, "sample_dets": sample_dets_data}, indent=2),
            encoding="utf-8"
        )
        print(f"寫入 {inter_path}")
