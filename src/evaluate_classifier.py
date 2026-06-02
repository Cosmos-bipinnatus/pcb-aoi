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
        per_class[c] = {"precision": P, "recall": R, "f1": F1, "support": int(cm[i].sum())}
    return {"accuracy": acc, "per_class": per_class,
            "confusion_matrix": cm.tolist(), "n": len(y_true)}


# --------------------------------------------------------------------------- #
# 2. End-to-end 評估(同一批 Stage 1 proposals,對比 rule-based vs CNN)
# --------------------------------------------------------------------------- #
def evaluate_end_to_end(model, device, config=None, split="test", pad_ratio=0.3) -> dict:
    clf = _CropClassifier(model, device, pad_ratio)
    items = build_splits(DATA)[split]

    stage1_records, cnn_records = [], []
    sample_dets = []   # 供視覺化:保留少數影像的 proposals(兩種分類)
    for k, it in enumerate(items):
        img_path = it["image"]
        tmpl = get_template_for_test(Path(img_path).name, DATA / "PCB_USED")
        out = run_traditional_pipeline(img_path, tmpl, config)
        dets = out["detections"]
        gts = [(o.name, (o.xmin, o.ymin, o.xmax, o.ymax)) for o in parse_voc(it["xml"]).objects]

        img = cv2.imread(img_path)
        patches = [cv2.cvtColor(crop_with_padding(img, d["bbox"], pad_ratio), cv2.COLOR_BGR2RGB)
                   for d in dets]
        cnn_idx, cnn_conf = clf.predict_patches(patches)

        s1 = [(d["class"], d["bbox"], d["confidence"]) for d in dets]
        cnn = [(CLASSES[cnn_idx[i]], dets[i]["bbox"], float(cnn_conf[i]))
               for i in range(len(dets))]
        stage1_records.append({"gts": gts, "dets": s1})
        cnn_records.append({"gts": gts, "dets": cnn})

        if k < 6:
            sample_dets.append({"image": img_path, "gts": gts, "stage1": s1, "cnn": cnn})

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(DEFAULT_WEIGHTS, device)
    print(f"載入模型 {DEFAULT_WEIGHTS}  device={device}")

    gt = evaluate_gt_crops(model, device, "test")
    _print_gt(gt)
    e2e = evaluate_end_to_end(model, device, split="test")
    _print_e2e(e2e)

    out = {"gt_crops": gt, "end_to_end": {k: v for k, v in e2e.items() if k != "sample_dets"}}
    res_dir = ROOT / "results"
    (res_dir / "intermediate").mkdir(parents=True, exist_ok=True)
    # 完整(含 sample_dets)存 intermediate(gitignore),精簡 summary 存 results(追蹤)
    (res_dir / "intermediate" / "classifier_eval.json").write_text(
        json.dumps({**out, "sample_dets": e2e["sample_dets"]}, indent=2))
    (res_dir / "classifier_eval_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n寫入 {res_dir/'classifier_eval_summary.json'}")
