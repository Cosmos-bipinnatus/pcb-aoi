"""評估訓練好的 YOLOv8 偵測器 (Stage 3),在 test split 上。

    python -m src.evaluate_yolo

報告 mAP@0.5、mAP@0.5:0.95、per-class AP、推論速度,並輸出與 Stage 1 / Stage 2
對比所需的精簡 summary。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .hripcb import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
RESULTS = ROOT / "results"
DEFAULT_BEST = RESULTS / "yolo_training" / "run1" / "weights" / "best.pt"


def evaluate(weights: str | Path = DEFAULT_BEST, imgsz: int = 640) -> dict:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))

    # 在 test split 上驗證
    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=imgsz,
                        device=device, verbose=False)

    # per-class AP@0.5(metrics.box.ap50 依 names 順序;用 ap_class_index 對應)
    names = model.names
    per_class = {c: 0.0 for c in CLASSES}
    ap50 = metrics.box.ap50          # 每個出現類別的 AP@0.5
    ap = metrics.box.ap              # AP@0.5:0.95
    per_class_5095 = {c: 0.0 for c in CLASSES}
    for i, ci in enumerate(metrics.box.ap_class_index):
        cname = names[int(ci)]
        if cname in per_class:
            per_class[cname] = float(ap50[i])
            per_class_5095[cname] = float(ap[i])

    # 推論速度:對 test 影像實測 ms/影像
    test_imgs = sorted((ROOT / "data" / "yolo" / "images" / "test").glob("*.jpg"))
    sample = [str(p) for p in test_imgs[:40]]
    model.predict(sample[:4], imgsz=imgsz, device=device, verbose=False)  # warmup
    t0 = time.perf_counter()
    model.predict(sample, imgsz=imgsz, device=device, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    ms_per_img = (time.perf_counter() - t0) / len(sample) * 1000

    out = {
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "per_class_ap50": per_class,
        "per_class_ap5095": per_class_5095,
        "ms_per_image": float(ms_per_img),
        "imgsz": imgsz,
        "weights": str(weights),
    }
    return out


def _print(res):
    print(f"\n================ Stage 3 YOLOv8n — test split ================")
    print(f"mAP@0.5      = {res['mAP@0.5']*100:.1f}%")
    print(f"mAP@0.5:0.95 = {res['mAP@0.5:0.95']*100:.1f}%")
    print(f"{'class':<16}{'AP@0.5':>9}{'AP@.5:.95':>11}")
    for c in CLASSES:
        print(f"{c:<16}{res['per_class_ap50'][c]*100:>9.1f}{res['per_class_ap5095'][c]*100:>11.1f}")
    print(f"\n推論速度: {res['ms_per_image']:.1f} ms/影像 (imgsz={res['imgsz']})")


if __name__ == "__main__":
    res = evaluate()
    _print(res)
    (RESULTS / "yolo_eval_summary.json").write_text(json.dumps(res, indent=2))
    print(f"\n寫入 {RESULTS / 'yolo_eval_summary.json'}")
