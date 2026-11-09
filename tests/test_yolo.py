"""Stage 3 YOLO 的輕量測試(不含訓練)。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.hripcb import CLASSES
from src.voc_to_yolo import voc_to_yolo_lines, YOLO_DIR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"
HAS_DATA = DATA.exists() and any((DATA / "PCB_USED").glob("*"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="HRIPCB dataset 不存在")

# 已下載的 YOLO 預訓練權重(訓練啟動時下載到專案根目錄)
YOLO_PT = ROOT / "yolov8n.pt"
HAS_PT = YOLO_PT.exists()
needs_pt = pytest.mark.skipif(not HAS_PT, reason="yolov8n.pt 尚未下載")


# --------------------------------------------------------------------------- #
# 1. VOC → YOLO 的 cxcywh 計算正確(手算驗證)
# --------------------------------------------------------------------------- #
def test_voc_to_yolo_math(tmp_path):
    xml = tmp_path / "t.xml"
    # 影像 200x100,一個 bbox (50,20)-(150,60) -> cx=100/200=0.5, cy=40/100=0.4,
    # w=100/200=0.5, h=40/100=0.4
    xml.write_text(
        "<annotation><size><width>200</width><height>100</height><depth>3</depth></size>"
        "<object><name>short</name><bndbox>"
        "<xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax>"
        "</bndbox></object></annotation>")
    lines = voc_to_yolo_lines(xml)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == str(CLASSES.index("short"))   # class idx
    cx, cy, w, h = map(float, parts[1:])
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.4) < 1e-6
    assert abs(w - 0.5) < 1e-6 and abs(h - 0.4) < 1e-6


# --------------------------------------------------------------------------- #
# 2. data.yaml 格式正確、含 6 類、可被 ultralytics 讀
# --------------------------------------------------------------------------- #
@needs_data
def test_data_yaml_valid():
    yaml_path = YOLO_DIR / "data.yaml"
    assert yaml_path.exists(), "請先執行 python -m src.voc_to_yolo"
    try:
        import yaml
        cfg = yaml.safe_load(yaml_path.read_text())
        assert set(cfg["names"].values()) == set(CLASSES)
        assert len(cfg["names"]) == 6
        for split in ("train", "val", "test"):
            assert (YOLO_DIR / cfg[split]).exists()
    except ImportError:
        content = yaml_path.read_text(encoding="utf-8")
        for c in CLASSES:
            assert c in content
        for split in ("train", "val", "test"):
            assert (YOLO_DIR / f"images/{split}").exists()


# --------------------------------------------------------------------------- #
# 3. YOLO 標註與影像數量一致(每張影像對應一個 label 檔)
# --------------------------------------------------------------------------- #
@needs_data
def test_yolo_images_labels_aligned():
    for split in ("train", "val", "test"):
        imgs = list((YOLO_DIR / "images" / split).glob("*.jpg"))
        labels = list((YOLO_DIR / "labels" / split).glob("*.txt"))
        assert len(imgs) == len(labels) > 0
        # 每張影像都有同名 label
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in labels}
        assert img_stems == lbl_stems


# --------------------------------------------------------------------------- #
# 4. YOLO 模型載入 OK
# --------------------------------------------------------------------------- #
@needs_pt
def test_yolo_model_loads():
    from ultralytics import YOLO
    model = YOLO(str(YOLO_PT))
    assert model is not None


# --------------------------------------------------------------------------- #
# 5. 單張推論不 crash + 端到端速度 < 200ms/影像(GPU)
# --------------------------------------------------------------------------- #
@needs_data
@needs_pt
def test_yolo_inference_smoke_and_speed():
    import torch
    from ultralytics import YOLO
    model = YOLO(str(YOLO_PT))
    device = 0 if torch.cuda.is_available() else "cpu"
    imgs = sorted((YOLO_DIR / "images" / "test").glob("*.jpg"))[:5]
    assert imgs, "test 影像不存在"
    paths = [str(p) for p in imgs]
    model.predict(paths[:1], imgsz=640, device=device, verbose=False)  # warmup
    t0 = time.perf_counter()
    res = model.predict(paths, imgsz=640, device=device, verbose=False)
    per_img = (time.perf_counter() - t0) / len(paths)
    assert len(res) == len(paths)               # 不 crash、每張都有結果
    budget = 0.2 if device != "cpu" else 2.0
    assert per_img < budget, f"推論過慢: {per_img*1000:.0f}ms/img (device={device})"
