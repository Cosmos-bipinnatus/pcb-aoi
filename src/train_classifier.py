"""訓練 Stage 2 缺陷分類器(ResNet18)。

    python -m src.train_classifier

設定:Adam(lr=1e-4)、CrossEntropy、ReduceLROnPlateau(on val loss)、
early stopping(patience=10)、最多 50 epochs、batch 64。全程記錄 train/val
loss + accuracy 到 results/training_log.csv,最佳權重存 models/defect_classifier.pt。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import time

import numpy as np
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


def set_seed(seed: int = 42):
    """鎖定隨機種子以確保實驗完全可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train(
    config: dict = None,
    device: str = None,
    weights_path: str | Path = WEIGHTS,
    weights_last_path: str | Path = None,
    log_path: str | Path = LOG_CSV,
) -> dict:
    cfg = {**CONFIG, **(config or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(weights_path)
    log_path = Path(log_path)
    if weights_last_path is not None:
        weights_last_path = Path(weights_last_path)

    # 鎖定隨機種子 (在建立 Dataset, DataLoader, Model 之前呼叫)
    set_seed(cfg["seed"])

    train_ds = PCBDefectCropDataset("gt", "train", seed=cfg["seed"])
    val_ds = PCBDefectCropDataset("gt", "val", seed=cfg["seed"])

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_ld = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_crops,
        generator=g,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_crops,
    )

    model = build_model(
        num_classes=6,
        freeze_backbone=cfg["freeze_backbone"],
        pretrained=True,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_last_path is not None:
        weights_last_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    best_val, best_epoch, patience_ctr = float("inf"), -1, 0
    last_epoch, last_val_loss, last_val_acc = 0, float("nan"), 0.0
    print(f"device={device}  train={len(train_ds)} crops  val={len(val_ds)} crops")

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _run_epoch(model, train_ld, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_ld, criterion, device, None)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0
        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": lr_now,
            "sec": dt,
        })
        print(
            f"epoch {epoch:2d}  train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val_loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.1e} | {dt:.1f}s"
        )

        last_epoch = epoch
        last_val_loss = va_loss
        last_val_acc = va_acc

        if va_loss < best_val - 1e-4:
            best_val, best_epoch, patience_ctr = va_loss, epoch, 0
            save_model(
                model,
                weights_path,
                extra={
                    "epoch": epoch,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "config": cfg,
                },
            )
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"early stopping @ epoch {epoch}(best epoch {best_epoch})")
                break

    # 若指定最後一輪權重儲存路徑，將當前最新權重存入
    if weights_last_path is not None:
        save_model(
            model,
            weights_last_path,
            extra={
                "epoch": last_epoch,
                "val_loss": last_val_loss,
                "val_acc": last_val_acc,
                "config": cfg,
            },
        )
        print(f"最後一輪權重 (epoch {last_epoch}) -> {weights_last_path}")

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\n最佳 val_loss {best_val:.4f} @ epoch {best_epoch}  ->  {weights_path}")
    print(f"訓練記錄 -> {log_path}")
    res = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "log": rows,
        "weights": str(weights_path),
    }
    if weights_last_path is not None:
        res["weights_last"] = str(weights_last_path)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Stage 2 缺陷分類器 (ResNet18)")
    parser.add_argument("--epochs", type=int, default=None, help="訓練輪數 (覆寫 CONFIG['max_epochs'])")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS), help="最佳權重輸出路徑")
    parser.add_argument("--weights-last", type=str, default=None, help="最後一輪權重輸出路徑")
    parser.add_argument("--log", type=str, default=str(LOG_CSV), help="訓練日誌 CSV 輸出路徑")
    parser.add_argument("--no-early-stop", action="store_true", help="關閉 early stopping (跑滿指定 epochs)")
    args = parser.parse_args()

    cfg_override = {}
    if args.epochs is not None:
        cfg_override["max_epochs"] = args.epochs
    if args.no_early_stop:
        epochs = args.epochs if args.epochs is not None else CONFIG["max_epochs"]
        cfg_override["patience"] = epochs + 1
    elif args.patience is not None:
        cfg_override["patience"] = args.patience

    train(
        config=cfg_override,
        weights_path=args.weights,
        weights_last_path=args.weights_last,
        log_path=args.log,
    )
