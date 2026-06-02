"""HRIPCB 缺陷 patch 分類資料集 (Stage 2)。

核心架構:沿用 R-CNN 的想法 —— Stage 1 的 region proposals 與 ground-truth
bounding box 都可作為 crop 來源,交給 Stage 2 的 CNN 做精細分類。

* ``mode="gt"``       : 從 VOC ground-truth bbox 切 crops(訓練 / 上限評估用)
* ``mode="proposal"`` : 從 Stage 1 region proposals 切 crops(end-to-end 評估用)

train/val/test 以 **image 為單位** 切分(避免同一張圖的多個 bbox 跨越 split
造成資料洩漏),並對每一類分別切分以維持類別平衡。固定 ``seed=42``。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .hripcb import CLASSES, parse_voc

# ImageNet 預訓練模型的標準正規化參數
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

# 影像 / 標註所在的類別資料夾名(首字母大寫)
_CLASS_DIR = {
    "missing_hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
}

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "HRIPCB"


# --------------------------------------------------------------------------- #
# image-level、依類別分層的 train/val/test 切分
# --------------------------------------------------------------------------- #
def build_splits(root: str | Path = DEFAULT_ROOT, seed: int = 42,
                 ratios=(0.70, 0.15, 0.15)) -> dict[str, list[dict]]:
    """回傳 ``{"train": [...], "val": [...], "test": [...]}``。

    每筆為 ``{"image", "xml", "class"}``。切分以 image 為單位、依類別分層,
    同一隨機種子保證可重現。"""
    root = Path(root)
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for cls, cdir in _CLASS_DIR.items():
        img_dir = root / "images" / cdir
        ann_dir = root / "Annotations" / cdir
        images = sorted(img_dir.glob("*.jpg"))
        rng.shuffle(images)

        n = len(images)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        bounds = {"train": images[:n_tr],
                  "val": images[n_tr:n_tr + n_va],
                  "test": images[n_tr + n_va:]}
        for split, imgs in bounds.items():
            for ip in imgs:
                xml = ann_dir / (ip.stem + ".xml")
                splits[split].append({"image": str(ip), "xml": str(xml), "class": cls})
    return splits


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def get_transforms(split: str):
    """訓練用資料增強;val/test 僅 resize + normalize。"""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# crop 工具
# --------------------------------------------------------------------------- #
def crop_with_padding(img: np.ndarray, bbox, pad_ratio: float = 0.3) -> np.ndarray:
    """以 bbox 為中心、四周外擴 ``pad_ratio`` 比例的上下文,切出 BGR patch。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:                      # 退化情形,回傳 1x1 避免崩潰
        patch = img[max(0, y1):y1 + 1, max(0, x1):x1 + 1]
    return patch


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PCBDefectCropDataset(Dataset):
    """PCB 缺陷 patch 分類資料集。

    Parameters
    ----------
    mode : "gt" | "proposal"
        crop 來源 —— ground-truth bbox 或 Stage 1 region proposals。
    split : "train" | "val" | "test"
    transform : callable, optional
        未指定時依 ``split`` 自動選用 :func:`get_transforms`。
    pad_ratio : float
        bbox 周圍補的上下文比例(CNN 需看周圍環境)。
    root : dataset 根目錄。
    proposals : dict, optional
        ``mode="proposal"`` 時使用,格式 ``{image_path: [{"bbox", "confidence",
        "class"(optional)}, ...]}``。每個 proposal 產生一個 crop。
    seed : split 隨機種子。
    """

    def __init__(self, mode: str, split: str, transform=None, pad_ratio: float = 0.3,
                 root: str | Path = DEFAULT_ROOT, proposals: dict = None, seed: int = 42,
                 max_samples: int = None):
        assert mode in ("gt", "proposal"), f"未知 mode: {mode}"
        assert split in ("train", "val", "test"), f"未知 split: {split}"
        self.mode = mode
        self.split = split
        self.pad_ratio = pad_ratio
        self.root = Path(root)
        self.transform = transform if transform is not None else get_transforms(split)

        split_items = build_splits(root, seed)[split]
        self.samples: list[dict] = []   # 每筆: {image, bbox, label, class_name, confidence}

        if mode == "gt":
            for it in split_items:
                ann = parse_voc(it["xml"])
                for o in ann.objects:
                    self.samples.append({
                        "image": it["image"],
                        "bbox": (o.xmin, o.ymin, o.xmax, o.ymax),
                        "label": CLASSES.index(o.name),
                        "class_name": o.name,
                        "confidence": 1.0,
                    })
        else:  # proposal
            if proposals is None:
                raise ValueError("mode='proposal' 需要傳入 proposals 字典 "
                                 "(image_path -> [{'bbox', ...}])")
            split_images = {it["image"] for it in split_items}
            for img_path, props in proposals.items():
                if str(img_path) not in split_images:
                    continue
                for p in props:
                    self.samples.append({
                        "image": str(img_path),
                        "bbox": tuple(p["bbox"]),
                        "label": CLASSES.index(p["class"]) if p.get("class") in CLASSES else -1,
                        "class_name": p.get("class", "?"),
                        "confidence": float(p.get("confidence", 1.0)),
                    })

        if max_samples is not None:          # 測試用:限制樣本數以加速
            self.samples = self.samples[:max_samples]

        # 預切所有 patch 進記憶體:每張大圖只讀一次(依影像分組),
        # 切完的 patch 很小(~100x100),整個 split 僅數十 MB,訓練時免重讀大圖。
        self.patches: list[np.ndarray] = [None] * len(self.samples)
        order = sorted(range(len(self.samples)), key=lambda i: self.samples[i]["image"])
        cur_path, cur_img = None, None
        for i in order:
            s = self.samples[i]
            if s["image"] != cur_path:
                cur_path, cur_img = s["image"], cv2.imread(s["image"])
            patch = crop_with_padding(cur_img, s["bbox"], self.pad_ratio)
            self.patches[i] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        tensor = self.transform(Image.fromarray(self.patches[idx]))
        meta = {"image": s["image"], "bbox": s["bbox"],
                "class_name": s["class_name"], "confidence": s["confidence"]}
        return tensor, s["label"], meta

    def class_counts(self) -> dict[str, int]:
        out = {c: 0 for c in CLASSES}
        for s in self.samples:
            if 0 <= s["label"] < len(CLASSES):
                out[CLASSES[s["label"]]] += 1
        return out


def collate_crops(batch):
    """保留 meta 為 list 的 collate(預設 collate 會破壞 bbox tuple)。"""
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return imgs, labels, metas
