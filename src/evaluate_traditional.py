"""Evaluate the classical-CV baseline on the full HRIPCB set.

Runs :func:`src.traditional_cv.run_traditional_pipeline` over every image,
matches predictions to Pascal-VOC ground truth at IoU >= 0.5, and reports:

* per-class precision / recall / F1 (class-aware VOC matching)
* mAP@0.5 (mean of per-class average precision)
* a localization-based confusion matrix (incl. background row/col)
* timing and registration-quality statistics

Run as a script to evaluate the whole dataset and cache results to
``results/intermediate/traditional_eval.json`` (git-ignored)::

    python -m src.evaluate_traditional
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .hripcb import CLASSES, get_template_for_test, iter_annotations, image_path_for
from .traditional_cv import run_traditional_pipeline

IOU_THRESH = 0.5


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point (VOC2010+) average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------- #
# core evaluation
# --------------------------------------------------------------------------- #
def evaluate_dataset(root: str | Path, config: dict = None, max_images: int = None,
                     progress: bool = True, stride: int = 1) -> dict:
    root = Path(root)
    pcb_dir = root / "PCB_USED"

    # accumulators
    gt_count = defaultdict(int)                      # class -> #GT
    preds_by_class = defaultdict(list)              # class -> [(conf, is_tp), ...]
    # class-aware counts for P/R/F1
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    # confusion matrix labels: classes + 'background'
    labels = CLASSES + ["background"]
    lidx = {c: i for i, c in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)  # [gt, pred]

    timings_total = defaultdict(float)
    reg_quality = []
    per_image = []

    anns = list(iter_annotations(root))
    if stride > 1:                       # stratified subsample (folders are per-class)
        anns = anns[::stride]
    if max_images:
        anns = anns[:max_images]
    n = len(anns)

    for k, ann in enumerate(anns):
        img_path = image_path_for(ann, root)
        tmpl_path = get_template_for_test(ann.filename, pcb_dir)
        out = run_traditional_pipeline(str(img_path), tmpl_path, config)
        dets = out["detections"]
        reg_quality.append(out["registration_quality"])
        for kk, vv in out["timings"].items():
            timings_total[kk] += vv

        gts = [(o.name, (o.xmin, o.ymin, o.xmax, o.ymax)) for o in ann.objects]
        for cls, _ in gts:
            gt_count[cls] += 1

        # ---- class-aware matching for P/R/F1 + AP ----
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        for cls in CLASSES:
            cls_gts = [bb for c, bb in gts if c == cls]
            matched = [False] * len(cls_gts)
            for d in dets_sorted:
                if d["class"] != cls:
                    continue
                best_iou, best_j = 0.0, -1
                for j, gbb in enumerate(cls_gts):
                    if matched[j]:
                        continue
                    v = iou(d["bbox"], gbb)
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_iou >= IOU_THRESH and best_j >= 0
                if is_tp:
                    matched[best_j] = True
                    tp[cls] += 1
                else:
                    fp[cls] += 1
                preds_by_class[cls].append((d["confidence"], is_tp))
            fn[cls] += matched.count(False)

        # ---- class-agnostic matching for confusion matrix ----
        gt_matched = [False] * len(gts)
        det_matched = [False] * len(dets_sorted)
        for di, d in enumerate(dets_sorted):
            best_iou, best_j = 0.0, -1
            for j, (_, gbb) in enumerate(gts):
                if gt_matched[j]:
                    continue
                v = iou(d["bbox"], gbb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= IOU_THRESH and best_j >= 0:
                gt_matched[best_j] = True
                det_matched[di] = True
                confusion[lidx[gts[best_j][0]], lidx[d["class"]]] += 1
        # unmatched detections -> predicted X but gt background
        for di, d in enumerate(dets_sorted):
            if not det_matched[di]:
                confusion[lidx["background"], lidx[d["class"]]] += 1
        # unmatched gts -> missed (gt X, pred background)
        for j, m in enumerate(gt_matched):
            if not m:
                confusion[lidx[gts[j][0]], lidx["background"]] += 1

        per_image.append({
            "filename": ann.filename,
            "n_gt": len(gts),
            "n_det": len(dets),
            "reg_quality": out["registration_quality"],
            "time": out["timings"]["total"],
        })

        if progress and (k + 1) % 50 == 0:
            print(f"  [{k+1}/{n}] evaluated", flush=True)

    # ---- per-class metrics + AP ----
    per_class = {}
    aps = {}
    for cls in CLASSES:
        TP, FP, FN = tp[cls], fp[cls], fn[cls]
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # AP from PR curve
        preds = sorted(preds_by_class[cls], key=lambda x: x[0], reverse=True)
        n_gt = gt_count[cls]
        if preds and n_gt:
            tp_arr = np.array([1 if is_tp else 0 for _, is_tp in preds])
            fp_arr = 1 - tp_arr
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            rec = tp_cum / n_gt
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = voc_ap(rec, prec)
        else:
            ap = 0.0
        aps[cls] = ap
        per_class[cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "ap": ap, "tp": TP, "fp": FP, "fn": FN, "n_gt": n_gt,
        }

    mAP = float(np.mean([aps[c] for c in CLASSES]))
    times = [p["time"] for p in per_image]

    return {
        "n_images": n,
        "mAP@0.5": mAP,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": labels,
        "timing": {
            "mean_s_per_image": float(np.mean(times)),
            "median_s_per_image": float(np.median(times)),
            "max_s_per_image": float(np.max(times)),
            "stage_totals_s": dict(timings_total),
        },
        "registration": {
            "mean_inlier_ratio": float(np.mean(reg_quality)),
            "min_inlier_ratio": float(np.min(reg_quality)),
            "frac_below_0.3": float(np.mean(np.array(reg_quality) < 0.3)),
        },
        "per_image": per_image,
    }


def _print_summary(res: dict) -> None:
    print("\n================ Traditional CV baseline — HRIPCB ================")
    print(f"images: {res['n_images']}   mAP@0.5: {res['mAP@0.5']*100:.1f}%")
    print(f"{'class':<16}{'P':>7}{'R':>7}{'F1':>7}{'AP':>7}{'TP':>6}{'FP':>6}{'FN':>6}")
    for c in CLASSES:
        m = res["per_class"][c]
        print(f"{c:<16}{m['precision']*100:>6.1f}{m['recall']*100:>7.1f}"
              f"{m['f1']*100:>7.1f}{m['ap']*100:>7.1f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")
    t = res["timing"]
    print(f"\nspeed: mean {t['mean_s_per_image']:.2f}s/img  median {t['median_s_per_image']:.2f}s  max {t['max_s_per_image']:.2f}s")
    r = res["registration"]
    print(f"registration: mean inlier {r['mean_inlier_ratio']:.3f}  min {r['min_inlier_ratio']:.3f}  frac<0.3 {r['frac_below_0.3']*100:.1f}%")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data" / "HRIPCB"
    max_images = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.perf_counter()
    print(f"evaluating {'all' if not max_images else max_images} images in {root} ...")
    res = evaluate_dataset(root, max_images=max_images)
    res["wall_time_s"] = time.perf_counter() - t0
    _print_summary(res)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir = results_dir / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "traditional_eval.json"
    out_path.write_text(json.dumps(res, indent=2))

    # compact, version-controlled summary (no per-image records)
    summary = {k: v for k, v in res.items() if k != "per_image"}
    summary["config"] = {**__import__("src.traditional_cv", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG}
    (results_dir / "traditional_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {out_path}  (wall {res['wall_time_s']:.1f}s)")
    print(f"wrote {results_dir / 'traditional_eval_summary.json'}")
