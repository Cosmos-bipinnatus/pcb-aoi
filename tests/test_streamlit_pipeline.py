"""Stage 5 inference_api 測試(只測三方法統一介面,不測 Streamlit UI)。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.hripcb import CLASSES

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
MANIFEST_PATH = SAMPLES / "manifest.json"
CNN_PT = ROOT / "models" / "defect_classifier.pt"
YOLO_PT = ROOT / "results" / "yolo_training" / "run1" / "weights" / "best.pt"

HAS_SAMPLES = MANIFEST_PATH.exists()
needs_samples = pytest.mark.skipif(not HAS_SAMPLES, reason="samples/ 不存在")
needs_cnn = pytest.mark.skipif(not CNN_PT.exists(), reason="CNN 權重不存在")
needs_yolo = pytest.mark.skipif(not YOLO_PT.exists(), reason="YOLO 權重不存在")


def _first_sample():
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    s = m["samples"][0]
    return str(SAMPLES / s["image"]), str(SAMPLES / s["template"])


def _check_result_format(result: dict):
    assert {"detections", "inference_ms", "method"} <= set(result)
    assert isinstance(result["inference_ms"], float) and result["inference_ms"] >= 0
    assert isinstance(result["method"], str) and result["method"]
    assert isinstance(result["detections"], list)
    for d in result["detections"]:
        assert {"bbox", "class", "confidence"} <= set(d)
        assert len(d["bbox"]) == 4 and all(isinstance(v, int) for v in d["bbox"])
        assert d["class"] in CLASSES
        assert 0.0 <= d["confidence"] <= 1.0


@needs_samples
def test_run_stage1_format():
    from src.inference_api import run_stage1
    img, tmpl = _first_sample()
    _check_result_format(run_stage1(img, tmpl))


@needs_samples
@needs_cnn
def test_run_stage2_format():
    from src.inference_api import run_stage2
    img, _ = _first_sample()
    res = run_stage2(img, conf_threshold=0.5)
    _check_result_format(res)
    assert "note" in res            # demo 用 class-agnostic,應有說明


@needs_samples
@needs_yolo
def test_run_stage3_format_and_speed():
    from src.inference_api import run_stage3
    img, _ = _first_sample()
    run_stage3(img)                 # warmup(載入模型 + CUDA init)
    res = run_stage3(img, conf_threshold=0.25)
    _check_result_format(res)
    assert res["inference_ms"] < 1000.0, f"純推論過慢: {res['inference_ms']:.0f}ms"
