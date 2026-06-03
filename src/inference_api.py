"""三方法統一推論介面(供 Streamlit demo 使用)。

三方法都吃單張 BGR 影像,回傳統一格式的 dict:

    {
      "detections": [{"bbox": (x1,y1,x2,y2), "class": str, "confidence": float}, ...],
      "inference_ms": float,     # 純運算時間,不含載圖 / 編碼
      "method": str,
      "note": str,               # 給 demo 顯示的說明(可空)
    }

* Stage 1:傳統 CV,**需 template**(模擬產線 AOI 的 Recipe 流程)。
* Stage 2:R-CNN 風格。完整 Stage 2 的 proposals 來自 Stage 1 差影、需 template;
  為了讓 demo 能「直接吃單張圖」,此處改用 **MSER class-agnostic region
  proposal**(不需 template)+ ResNet18 分類。**此為 demo 專用的近似**,召回與
  精度低於有 template 的差影版本(見 note 與 README 說明)。
* Stage 3:YOLOv8n 端到端,本就不需 template。
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import crop_with_padding, get_transforms
from .dl_classifier import load_model, DEFAULT_WEIGHTS
from .hripcb import CLASSES
from .traditional_cv import (register_to_template, compute_difference,
                             threshold_and_clean, classify_components, DEFAULT_CONFIG)

ROOT = Path(__file__).resolve().parents[1]
YOLO_BEST = ROOT / "results" / "yolo_training" / "run1" / "weights" / "best.pt"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 模型快取(lazy)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _cnn():
    return load_model(DEFAULT_WEIGHTS, _DEVICE)


@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_BEST))


def _as_bgr(img) -> np.ndarray:
    """接受路徑或 np.ndarray(BGR),回傳 BGR ndarray。"""
    if isinstance(img, (str, Path)):
        out = cv2.imread(str(img))
        if out is None:
            raise FileNotFoundError(f"無法讀取影像: {img}")
        return out
    return img


# --------------------------------------------------------------------------- #
# Stage 1:傳統 CV(需 template)
# --------------------------------------------------------------------------- #
def run_stage1(test_img, template_img, conf_threshold: float | None = None,
               config: dict = None) -> dict:
    test = _as_bgr(test_img)
    tmpl = _as_bgr(template_img)
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    t0 = time.perf_counter()
    warped, _ = register_to_template(test, tmpl, cfg["register_method"])
    diff = compute_difference(warped, tmpl, cfg["blur_ksize"])
    mask = threshold_and_clean(diff, cfg["threshold"], cfg["morph_open_ksize"],
                               cfg["morph_close_ksize"], cfg["min_area"],
                               cfg["max_area"], cfg["dilate_ksize"])
    dets = classify_components(mask, warped, cfg["bbox_pad"])
    infer_ms = (time.perf_counter() - t0) * 1000

    detections = [{"bbox": tuple(int(v) for v in d["bbox"]), "class": d["class"],
                   "confidence": float(d["confidence"])} for d in dets]
    if conf_threshold:
        detections = [d for d in detections if d["confidence"] >= conf_threshold]
    return {"detections": detections, "inference_ms": infer_ms,
            "method": "Stage 1: 傳統 CV (差影 + 形態學)",
            "note": "需 golden template(模擬產線 AOI Recipe)。"}


# --------------------------------------------------------------------------- #
# Stage 2:class-agnostic proposal(MSER)+ CNN
# --------------------------------------------------------------------------- #
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def _nms(boxes, iou_thr=0.3):
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))  # 小框優先(缺陷偏小)
    keep = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep


def mser_proposals(img_bgr, min_area=40, max_area=15000, max_boxes=120) -> list:
    """class-agnostic 候選框(MSER 穩定區域 → bbox → 尺寸過濾 → NMS)。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, _ = mser.detectRegions(gray)
    boxes = []
    H, W = gray.shape
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        if min_area <= w * h <= max_area and w >= 6 and h >= 6:
            boxes.append((x, y, x + w, y + h))
    boxes = _nms(boxes, 0.3)
    return boxes[:max_boxes]


def run_stage2(test_img, conf_threshold: float = 0.5) -> dict:
    test = _as_bgr(test_img)
    model = _cnn()
    tf = get_transforms("test")

    t0 = time.perf_counter()
    boxes = mser_proposals(test)
    detections = []
    if boxes:
        crops = [tf(Image.fromarray(cv2.cvtColor(
            crop_with_padding(test, b, 0.3), cv2.COLOR_BGR2RGB))) for b in boxes]
        batch = torch.stack(crops).to(_DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(batch), 1)
        conf, idx = prob.max(1)
        for b, ci, cf in zip(boxes, idx.cpu().numpy(), conf.cpu().numpy()):
            if cf >= conf_threshold:
                detections.append({"bbox": tuple(int(v) for v in b),
                                   "class": CLASSES[int(ci)], "confidence": float(cf)})
    infer_ms = (time.perf_counter() - t0) * 1000
    return {"detections": detections, "inference_ms": infer_ms,
            "method": "Stage 2: R-CNN 風格 (proposals + CNN)",
            "note": "demo 無 template,改用 MSER class-agnostic proposal 取代差影 "
                    "proposal;精度/召回低於有 template 的完整版本。"}


# --------------------------------------------------------------------------- #
# Stage 3:YOLOv8n 端到端
# --------------------------------------------------------------------------- #
def run_stage3(test_img, conf_threshold: float = 0.25) -> dict:
    test = _as_bgr(test_img)
    model = _yolo()
    device = 0 if _DEVICE == "cuda" else "cpu"
    r = model.predict(test, imgsz=640, device=device, conf=conf_threshold, verbose=False)[0]
    names = r.names
    detections = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        detections.append({"bbox": (int(x1), int(y1), int(x2), int(y2)),
                           "class": names[int(box.cls)], "confidence": float(box.conf)})
    infer_ms = float(r.speed.get("inference", 0.0))    # ultralytics 純推論 ms
    return {"detections": detections, "inference_ms": infer_ms,
            "method": "Stage 3: YOLOv8n (端到端)",
            "note": "端到端,免 template。"}
