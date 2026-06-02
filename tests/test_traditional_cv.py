"""Unit + smoke tests for the classical-CV baseline (src/traditional_cv.py)."""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.traditional_cv import (
    register_to_template,
    compute_difference,
    threshold_and_clean,
    classify_components,
    run_traditional_pipeline,
)
from src.hripcb import get_template_for_test

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HRIPCB"
PCB_USED = DATA / "PCB_USED"
HAS_DATA = DATA.exists() and any(PCB_USED.glob("*"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="HRIPCB dataset not present")


def _synthetic_board(w=400, h=300):
    """A green 'PCB' with a few bright copper rectangles."""
    img = np.full((h, w, 3), (30, 110, 40), np.uint8)  # BGR greenish substrate
    cv2.rectangle(img, (40, 40), (360, 70), (40, 160, 200), -1)   # copper trace
    cv2.rectangle(img, (40, 120), (360, 150), (40, 160, 200), -1)
    cv2.circle(img, (200, 220), 18, (60, 60, 60), -1)            # a hole (dark)
    return img


# --------------------------------------------------------------------------- #
# 1. registration on identical image -> near-identity homography
# --------------------------------------------------------------------------- #
@needs_data
def test_register_identity_on_same_image():
    f = sorted(glob.glob(str(DATA / "images" / "Missing_hole" / "*.jpg")))[0]
    img = cv2.imread(f)
    warped, H = register_to_template(img, img, method="orb")
    # homography should be ~identity (up to scale)
    Hn = H / H[2, 2]
    assert np.allclose(Hn, np.eye(3), atol=1e-2), f"H not identity:\n{Hn}"
    assert warped.shape == img.shape


# --------------------------------------------------------------------------- #
# 2. difference of an image with itself ~ 0
# --------------------------------------------------------------------------- #
def test_difference_of_same_image_is_near_zero():
    img = _synthetic_board()
    diff = compute_difference(img, img, blur_ksize=3)
    assert diff.shape == img.shape[:2]
    assert diff.max() <= 5, f"self-difference too large: max={diff.max()}"


# --------------------------------------------------------------------------- #
# 3. threshold_and_clean on pure noise -> ~0 surviving components
# --------------------------------------------------------------------------- #
def test_threshold_clean_suppresses_noise():
    rng = np.random.default_rng(0)
    # salt noise well below the area filter; morphology + area filter should kill it
    noise = (rng.random((300, 400)) > 0.97).astype(np.uint8) * 255
    mask = threshold_and_clean(noise, threshold=30, morph_open_ksize=3,
                               morph_close_ksize=5, min_area=50, max_area=50000,
                               dilate_ksize=0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    assert n - 1 <= 1, f"expected ~0 components from noise, got {n - 1}"


# --------------------------------------------------------------------------- #
# 4. classify a constructed 'missing hole' blob correctly
# --------------------------------------------------------------------------- #
def test_classify_missing_hole_shape():
    # a large, round, solid blob -> missing_hole prototype
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(mask, (150, 150), 17, 255, -1)   # area ~900, aspect ~1, high circularity
    dets = classify_components(mask, bbox_pad=10)
    assert len(dets) == 1
    assert dets[0]["class"] == "missing_hole", f"got {dets[0]['class']}"
    assert 0.0 <= dets[0]["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# 5. classify an elongated blob as a non-round defect (not missing_hole)
# --------------------------------------------------------------------------- #
def test_classify_elongated_not_missing_hole():
    mask = np.zeros((300, 300), np.uint8)
    cv2.rectangle(mask, (60, 145), (240, 158), 255, -1)  # long thin bar
    dets = classify_components(mask, bbox_pad=5)
    assert len(dets) == 1
    # an elongated bar must not be classified as the round 'missing_hole'
    assert dets[0]["class"] != "missing_hole"


# --------------------------------------------------------------------------- #
# 6. bbox padding expands and stays within image bounds
# --------------------------------------------------------------------------- #
def test_classify_bbox_padding_clipped():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (10, 10), 6, 255, -1)   # near the corner
    dets = classify_components(mask, bbox_pad=20)
    x1, y1, x2, y2 = dets[0]["bbox"]
    assert x1 == 0 and y1 == 0           # clipped at origin
    assert x2 <= 100 and y2 <= 100       # clipped at far edge


# --------------------------------------------------------------------------- #
# 7. full pipeline smoke test on 5 real images
# --------------------------------------------------------------------------- #
@needs_data
def test_pipeline_smoke_on_5_images():
    files = sorted(glob.glob(str(DATA / "images" / "*" / "*.jpg")))[::140][:5]
    assert files, "no images found"
    for f in files:
        tmpl = get_template_for_test(os.path.basename(f), PCB_USED)
        out = run_traditional_pipeline(f, tmpl)
        assert "detections" in out and isinstance(out["detections"], list)
        assert 0.0 <= out["registration_quality"] <= 1.0
        for d in out["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            assert x1 < x2 and y1 < y2
            assert d["class"] in {
                "missing_hole", "mouse_bite", "open_circuit",
                "short", "spur", "spurious_copper",
            }


# --------------------------------------------------------------------------- #
# 8. speed: a real image runs comfortably under 10 s
# --------------------------------------------------------------------------- #
@needs_data
def test_pipeline_speed_under_10s():
    f = sorted(glob.glob(str(DATA / "images" / "Missing_hole" / "*.jpg")))[0]
    tmpl = get_template_for_test(os.path.basename(f), PCB_USED)
    t0 = time.perf_counter()
    run_traditional_pipeline(f, tmpl)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"pipeline too slow: {elapsed:.2f}s"
