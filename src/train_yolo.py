"""訓練 YOLOv8 nano 端到端偵測器 (Stage 3)。

    python -m src.train_yolo

繞過 Stage 1+2 的兩階段 proposal 瓶頸,直接學習定位 + 分類。預訓練權重
``yolov8n.pt`` 首次執行會自動下載(~6 MB)。
"""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
PROJECT = ROOT / "results" / "yolo_training"

CONFIG = {
    "epochs": 100,
    "imgsz": 640,        # 影像 ~3000x1586 會 downsample;若 mAP 不佳改 1280
    "batch": 16,         # RTX 5070 Ti 16GB;OOM 則降到 8
    "patience": 20,      # early stopping
    "name": "run1",
}


def train(config: dict = None) -> dict:
    cfg = {**CONFIG, **(config or {})}
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=device,
        patience=cfg["patience"],
        project=str(PROJECT),
        name=cfg["name"],
        save=True,
        plots=True,
        seed=42,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    print(f"\n訓練完成 -> {save_dir}")
    print(f"最佳權重 -> {save_dir / 'weights' / 'best.pt'}")
    return {"save_dir": str(save_dir), "best": str(save_dir / "weights" / "best.pt")}


if __name__ == "__main__":
    train()
