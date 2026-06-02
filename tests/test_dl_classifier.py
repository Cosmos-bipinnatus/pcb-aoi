"""Stage 2 CNN 分類器的單元 / sanity 測試。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from src.dataset import build_splits, PCBDefectCropDataset, collate_crops
from src.dl_classifier import build_model, DefectClassifier
from src.hripcb import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"
HAS_DATA = DATA.exists() and any((DATA / "PCB_USED").glob("*"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="HRIPCB dataset 不存在")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 1. split 為 image-level、無洩漏、比例正確
# --------------------------------------------------------------------------- #
@needs_data
def test_split_is_image_level_no_leak():
    sp = build_splits(DATA, seed=42)
    imgs = {k: {it["image"] for it in v} for k, v in sp.items()}
    assert imgs["train"] & imgs["val"] == set()
    assert imgs["train"] & imgs["test"] == set()
    assert imgs["val"] & imgs["test"] == set()
    total = sum(len(v) for v in imgs.values())
    assert total == 693
    # 約 70/15/15
    assert 0.66 < len(imgs["train"]) / total < 0.74


@needs_data
def test_split_is_deterministic():
    a = build_splits(DATA, seed=42)["test"]
    b = build_splits(DATA, seed=42)["test"]
    assert [x["image"] for x in a] == [x["image"] for x in b]


# --------------------------------------------------------------------------- #
# 2. Dataset 取出的 tensor 形狀與 label 正確
# --------------------------------------------------------------------------- #
@needs_data
def test_dataset_item_shape_and_label():
    ds = PCBDefectCropDataset("gt", "val", max_samples=8)
    tensor, label, meta = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert 0 <= label < len(CLASSES)
    assert meta["class_name"] in CLASSES


# --------------------------------------------------------------------------- #
# 3. 模型 forward 輸出 (batch, 6)
# --------------------------------------------------------------------------- #
def test_model_forward_shape():
    model = build_model(num_classes=6, pretrained=False).eval()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 6)


# --------------------------------------------------------------------------- #
# 4. ImageNet 預訓練權重確實載入(非隨機初始化)
# --------------------------------------------------------------------------- #
def test_pretrained_weights_loaded():
    pre = DefectClassifier(pretrained=True)
    rnd = DefectClassifier(pretrained=False)
    w_pre = pre.backbone.conv1.weight
    w_rnd = rnd.backbone.conv1.weight
    # 預訓練 conv1 與隨機初始化應明顯不同;且預訓練權重 std 在合理範圍
    assert not torch.allclose(w_pre, w_rnd)
    assert 0.03 < w_pre.std().item() < 0.5


# --------------------------------------------------------------------------- #
# 5. 訓練 1 個 epoch 不 crash、且 loss 為有限值(小子集 sanity)
# --------------------------------------------------------------------------- #
@needs_data
def test_train_one_epoch_runs():
    ds = PCBDefectCropDataset("gt", "train", max_samples=24)
    ld = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_crops)
    model = build_model(num_classes=6, pretrained=True, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    last = None
    for imgs, labels, _ in ld:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(imgs), labels)
        loss.backward(); opt.step()
        last = loss.item()
    assert last is not None and torch.isfinite(torch.tensor(last))


# --------------------------------------------------------------------------- #
# 6. 推論速度合理(< 50ms/影像 on GPU;CPU 放寬)
# --------------------------------------------------------------------------- #
def test_inference_speed():
    model = build_model(num_classes=6, pretrained=False, device=DEVICE).eval()
    x = torch.randn(16, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        model(x)  # warmup
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        per_img = (time.perf_counter() - t0) / 16
    budget = 0.05 if DEVICE == "cuda" else 0.5
    assert per_img < budget, f"推論過慢: {per_img*1000:.1f}ms/img (device={DEVICE})"
