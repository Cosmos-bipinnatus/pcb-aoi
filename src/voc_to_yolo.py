"""VOC → YOLO 格式轉換 (Stage 3)。

將 HRIPCB 的 Pascal-VOC XML 標註轉成 YOLO txt(每行 ``class cx cy w h``,皆
normalize 到 0–1),並建立 ultralytics 需要的目錄結構與 ``data.yaml``::

    data/yolo/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        data.yaml

split 沿用 :func:`src.dataset.build_splits`(image-level、依類別分層、``seed=42``),
與 Stage 2 完全一致 —— 確保三個 stage 在相同 test 影像上可直接比較。

    python -m src.voc_to_yolo
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .dataset import build_splits
from .hripcb import CLASSES, parse_voc

ROOT = Path(__file__).resolve().parents[1]
SRC_DATA = ROOT / "data" / "HRIPCB"
YOLO_DIR = ROOT / "data" / "yolo"
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def voc_to_yolo_lines(xml_path: str | Path) -> list[str]:
    """把一個 VOC XML 轉成 YOLO 標註行(normalized cx cy w h)。"""
    ann = parse_voc(xml_path)
    W, H = ann.width, ann.height
    lines = []
    for o in ann.objects:
        cx = (o.xmin + o.xmax) / 2 / W
        cy = (o.ymin + o.ymax) / 2 / H
        w = (o.xmax - o.xmin) / W
        h = (o.ymax - o.ymin) / H
        lines.append(f"{_CLASS_IDX[o.name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def build_yolo_dataset(seed: int = 42, clean: bool = True) -> dict:
    """建立完整 data/yolo 結構並回傳統計。"""
    if clean and YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for split in ("train", "val", "test"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = build_splits(SRC_DATA, seed=seed)
    stats = {}
    for split, items in splits.items():
        n_box = 0
        for it in items:
            img_src = Path(it["image"])
            shutil.copy(img_src, YOLO_DIR / "images" / split / img_src.name)
            lines = voc_to_yolo_lines(it["xml"])
            n_box += len(lines)
            (YOLO_DIR / "labels" / split / (img_src.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""))
        stats[split] = {"images": len(items), "boxes": n_box}

    write_data_yaml()
    return stats


def write_data_yaml() -> Path:
    yaml_path = YOLO_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    # path 用絕對路徑(forward slash),ultralytics 在 Windows 亦可讀
    content = (
        f"path: {YOLO_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names}\n"
    )
    yaml_path.write_text(content)
    return yaml_path


if __name__ == "__main__":
    print("轉換 VOC → YOLO ...")
    stats = build_yolo_dataset()
    for split, s in stats.items():
        print(f"  {split:5}: {s['images']:4} 影像, {s['boxes']:5} boxes")
    print(f"data.yaml -> {YOLO_DIR / 'data.yaml'}")
