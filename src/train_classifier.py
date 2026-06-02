"""訓練 Stage 2 缺陷分類器(ResNet18)。

    python -m src.train_classifier

設定:Adam(lr=1e-4)、CrossEntropy、ReduceLROnPlateau(on val loss)、
early stopping(patience=10)、最多 50 epochs、batch 64。全程記錄 train/val
loss + accuracy 到 results/training_log.csv,最佳權重存 models/defect_classifier.pt。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PCBDefectCropDataset, collate_crops
from .dl_classifier import build_model, save_model

ROOT = Path(__file__).resolve().parents[1]
LOG_CSV = ROOT / "results" / "training_log.csv"
WEIGHTS = ROOT / "models" / "defect_classifier.pt"

CONFIG = {
    "lr": 1e-4,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 10,          # early stopping(on val loss)
    "freeze_backbone": False,
    "num_workers": 0,        # Windows + 大圖快取,主行程載入較穩
    "seed": 42,
}


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / total, correct / total


def train(config: dict = None, device: str = None) -> dict:
    cfg = {**CONFIG, **(config or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    train_ds = PCBDefectCropDataset("gt", "train", seed=cfg["seed"])
    val_ds = PCBDefectCropDataset("gt", "val", seed=cfg["seed"])
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], collate_fn=collate_crops)
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_crops)

    model = build_model(num_classes=6, freeze_backbone=cfg["freeze_backbone"],
                        pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=4)

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    best_val, best_epoch, patience_ctr = float("inf"), -1, 0
    print(f"device={device}  train={len(train_ds)} crops  val={len(val_ds)} crops")

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _run_epoch(model, train_ld, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_ld, criterion, device, None)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0
        rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "lr": lr_now, "sec": dt})
        print(f"epoch {epoch:2d}  train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.1e} | {dt:.1f}s")

        if va_loss < best_val - 1e-4:
            best_val, best_epoch, patience_ctr = va_loss, epoch, 0
            save_model(model, WEIGHTS, extra={"epoch": epoch, "val_loss": va_loss,
                                              "val_acc": va_acc, "config": cfg})
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"early stopping @ epoch {epoch}(best epoch {best_epoch})")
                break

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                          "val_loss", "val_acc", "lr", "sec"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n最佳 val_loss {best_val:.4f} @ epoch {best_epoch}  ->  {WEIGHTS}")
    print(f"訓練記錄 -> {LOG_CSV}")
    return {"best_epoch": best_epoch, "best_val_loss": best_val,
            "log": rows, "weights": str(WEIGHTS)}


if __name__ == "__main__":
    train()
