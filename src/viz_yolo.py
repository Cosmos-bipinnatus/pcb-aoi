"""Stage 3 視覺化:訓練曲線、PR、混淆矩陣、樣本偵測、三方法並列對比。"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import crop_with_padding, get_transforms
from .dl_classifier import load_model, DEFAULT_WEIGHTS
from .hripcb import CLASSES, get_template_for_test, parse_voc
from .traditional_cv import run_traditional_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"

# 每類固定 BGR 顏色
_BGR = {
    "missing_hole": (60, 60, 255), "mouse_bite": (0, 200, 255),
    "open_circuit": (0, 255, 0), "short": (255, 80, 0),
    "spur": (255, 0, 200), "spurious_copper": (0, 140, 255),
}


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def _draw(img, dets, gt=None):
    out = img.copy()
    if gt:
        for (x1, y1, x2, y2) in gt:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
    for cls, (x1, y1, x2, y2), conf in dets:
        c = _BGR.get(cls, (0, 0, 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 5)
        cv2.putText(out, f"{cls} {conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    return out


# --------------------------------------------------------------------------- #
# 複製 ultralytics 自動產的圖
# --------------------------------------------------------------------------- #
def copy_training_plots(run_dir, out_dir) -> list[Path]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "results.png": "03_training_curves.png",
        "BoxPR_curve.png": "03_pr_curve.png",          # ultralytics 8.4 命名
        "PR_curve.png": "03_pr_curve.png",             # 舊版相容
        "confusion_matrix.png": "03_confusion_matrix.png",
    }
    saved = []
    for src, dst in mapping.items():
        sp = run_dir / src
        if sp.exists():
            shutil.copy(sp, out_dir / dst); saved.append(out_dir / dst)
    return saved


# --------------------------------------------------------------------------- #
# YOLO 樣本偵測
# --------------------------------------------------------------------------- #
def _yolo_dets(model, img_path, imgsz=640, device=0, conf=0.25):
    r = model.predict(img_path, imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
    names = r.names
    dets = []
    for b in r.boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        dets.append((names[int(b.cls)], tuple(xyxy), float(b.conf)))
    return dets


def sample_predictions(model, out_dir, n=8, imgsz=640, device=0,
                       name="03_sample_predictions.png"):
    imgs = sorted((ROOT / "data" / "yolo" / "images" / "test").glob("*.jpg"))
    # 平均取樣涵蓋不同類別
    picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, n).astype(int)]
    fig, axes = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 6))
    for ax, ip in zip(axes.ravel(), picks):
        img = cv2.imread(str(ip))
        dets = _yolo_dets(model, str(ip), imgsz, device)
        ax.imshow(_rgb(_draw(img, dets))); ax.axis("off")
        ax.set_title(f"{ip.stem[:18]} ({len(dets)} dets)", fontsize=9)
    fig.suptitle("Stage 3 YOLOv8n 偵測結果(test set)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------- #
# 三方法並列對比(Stage 1 / 2 / 3)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def three_methods_comparison(yolo_model, out_dir, image_names=None, imgsz=640, device=0,
                             name="03_three_methods_comparison.png"):
    """對選定 test 影像並列 Stage 1 / Stage 2 / Stage 3 的偵測結果。"""
    test_items = {Path(p).name: p for p in
                  (ROOT / "data" / "yolo" / "images" / "test").glob("*.jpg")}
    if image_names is None:
        # 預設挑一張 short(展示被救起)+ 一張 missing_hole
        cand = [n for n in test_items if "short" in n][:1] + \
               [n for n in test_items if "missing_hole" in n][:1]
        image_names = cand or list(test_items)[:2]

    cnn = load_model(DEFAULT_WEIGHTS, device)
    tf = get_transforms("test")
    from PIL import Image

    rows = len(image_names)
    fig, axes = plt.subplots(rows, 3, figsize=(21, 6.2 * rows))
    if rows == 1:
        axes = axes[None, :]

    for r, iname in enumerate(image_names):
        img_path = test_items[iname]
        img = cv2.imread(str(img_path))
        xml = DATA / "Annotations" / _cls_dir(iname) / (Path(iname).stem + ".xml")
        gt = [(o.xmin, o.ymin, o.xmax, o.ymax) for o in parse_voc(xml).objects] \
            if xml.exists() else None

        # Stage 1
        tmpl = get_template_for_test(iname, DATA / "PCB_USED")
        out1 = run_traditional_pipeline(str(img_path), tmpl)
        s1 = [(d["class"], d["bbox"], d["confidence"]) for d in out1["detections"]]

        # Stage 2:對 Stage 1 proposals 用 CNN 重分類
        s2 = []
        for d in out1["detections"]:
            patch = cv2.cvtColor(crop_with_padding(img, d["bbox"], 0.3), cv2.COLOR_BGR2RGB)
            prob = torch.softmax(cnn(tf(Image.fromarray(patch)).unsqueeze(0).to(device)), 1)[0]
            s2.append((CLASSES[int(prob.argmax())], d["bbox"], float(prob.max())))

        # Stage 3:YOLO
        s3 = _yolo_dets(yolo_model, str(img_path), imgsz, device)

        for c, (title, dets) in enumerate([
                ("Stage 1:傳統 CV", s1),
                ("Stage 2:Stage1 proposals + CNN", s2),
                ("Stage 3:YOLOv8n 端到端", s3)]):
            axes[r][c].imshow(_rgb(_draw(img, dets, gt)))
            axes[r][c].set_title(f"{title}  ({len(dets)} dets)", fontsize=12)
            axes[r][c].axis("off")

    fig.suptitle("三方法偵測對比(綠 = ground truth)", fontweight="bold", fontsize=15)
    fig.tight_layout()
    return _save(fig, out_dir, name)


_CLS_DIR_MAP = {
    "missing_hole": "Missing_hole", "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit", "short": "Short", "spur": "Spur",
    "spurious_copper": "Spurious_copper",
}


def _cls_dir(filename: str) -> str:
    """由檔名推出類別資料夾名(01_missing_hole_03.jpg -> Missing_hole)。"""
    cls = "_".join(Path(filename).stem.split("_")[1:-1])   # 去掉前綴 id 與末尾編號
    return _CLS_DIR_MAP[cls]
