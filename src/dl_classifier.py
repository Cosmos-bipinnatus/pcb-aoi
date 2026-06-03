"""Stage 2 缺陷分類模型:ResNet18 + ImageNet 預訓練的遷移學習。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .hripcb import CLASSES

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "defect_classifier.pt"


class DefectClassifier(nn.Module):
    """ResNet18 backbone + 6 類分類 head。

    以 ImageNet 預訓練權重初始化,將最後 fc 換成 6 類輸出。預設微調整個網路;
    ``freeze_backbone=True`` 時凍結 backbone、只訓練 fc(資料少時較穩)。
    """

    def __init__(self, num_classes: int = 6, freeze_backbone: bool = False,
                 pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 6, freeze_backbone: bool = False,
                pretrained: bool = True, device: str = None) -> DefectClassifier:
    model = DefectClassifier(num_classes, freeze_backbone, pretrained)
    if device:
        model = model.to(device)
    return model


def save_model(model: nn.Module, path: str | Path = DEFAULT_WEIGHTS, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "classes": CLASSES}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path: str | Path = DEFAULT_WEIGHTS, device: str = "cpu") -> DefectClassifier:
    if isinstance(device, int):              # torch.load 不接受 int 作 map_location
        device = f"cuda:{device}"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DefectClassifier(num_classes=len(payload.get("classes", CLASSES)), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model
