"""把偵測結果(bbox + class + confidence)畫到影像上,供 demo 顯示。

每類固定一色,與 Stage 3 的 viz_yolo 風格一致。
"""
from __future__ import annotations

import cv2
import numpy as np

# 每類固定 BGR 顏色(與 viz_yolo 一致)
_BGR = {
    "missing_hole": (60, 60, 255), "mouse_bite": (0, 200, 255),
    "open_circuit": (0, 255, 0), "short": (255, 80, 0),
    "spur": (255, 0, 200), "spurious_copper": (0, 140, 255),
}


def draw_detections(img_bgr: np.ndarray, detections: list[dict],
                    show_conf: bool = True) -> np.ndarray:
    """在 BGR 影像上畫框與標籤,回傳 **RGB** ndarray(供 Streamlit 直接顯示)。"""
    out = img_bgr.copy()
    h = out.shape[0]
    lw = max(2, round(h / 400))               # 線寬隨影像大小
    fs = max(0.5, h / 1400)                    # 字體大小
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _BGR.get(d["class"], (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        label = d["class"] + (f" {d['confidence']:.2f}" if show_conf else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        ytxt = max(0, y1 - 4)
        cv2.rectangle(out, (x1, ytxt - th - 4), (x1 + tw, ytxt), color, -1)
        cv2.putText(out, label, (x1, ytxt - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), max(1, lw - 1), cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
