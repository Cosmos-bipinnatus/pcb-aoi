"""Classical-CV PCB defect baseline.

Pipeline (per defect image, against its defect-free template):

    register_to_template  ->  compute_difference  ->  threshold_and_clean
                                                            -> classify_components

The HRIPCB structure makes a *reference-vs-test* approach natural: every defect
image is derived from one of ten clean ``PCB_USED`` templates of identical
resolution, so after homography registration a pixel-wise difference exposes the
introduced defects, which are then segmented (morphology + connected
components) and labelled by a small, extensible rule-based classifier.

Defect classes (see :mod:`src.hripcb`): missing_hole, mouse_bite, open_circuit,
short, spur, spurious_copper.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .hripcb import CLASSES

# --------------------------------------------------------------------------- #
# Default configuration (overridable per-call via run_traditional_pipeline)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "register_method": "orb",
    "blur_ksize": 3,
    "threshold": 22,
    "morph_open_ksize": 3,
    "morph_close_ksize": 5,
    "dilate_ksize": 0,    # keep mask tight so shape features stay discriminative
    "bbox_pad": 16,       # expand only the reported bbox toward VOC annotation scale
    "min_area": 40,
    "max_area": 50000,
}


# --------------------------------------------------------------------------- #
# 1. Registration
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _register(test_img: np.ndarray, template_img: np.ndarray, method: str = "orb"):
    """Internal: returns (warped_test, H, quality) where quality is the RANSAC
    inlier ratio in [0, 1]. Falls back to identity if matching is too weak."""
    h, w = template_img.shape[:2]
    g_test = _to_gray(test_img)
    g_tmpl = _to_gray(template_img)

    method = method.lower()
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"unknown registration method: {method!r} (use 'orb' or 'sift')")

    kp1, des1 = detector.detectAndCompute(g_test, None)
    kp2, des2 = detector.detectAndCompute(g_tmpl, None)

    identity = np.eye(3, dtype=np.float64)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # cannot match -> assume already aligned (same-resolution capture)
        warped = cv2.warpPerspective(test_img, identity, (w, h))
        return warped, identity, 0.0

    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        warped = cv2.warpPerspective(test_img, identity, (w, h))
        return warped, identity, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        warped = cv2.warpPerspective(test_img, identity, (w, h))
        return warped, identity, 0.0

    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0
    warped = cv2.warpPerspective(test_img, H, (w, h))
    return warped, H, inlier_ratio


def register_to_template(test_img: np.ndarray, template_img: np.ndarray,
                         method: str = "orb") -> tuple[np.ndarray, np.ndarray]:
    """Register ``test_img`` onto ``template_img`` via feature matching +
    homography. Returns ``(warped_test_img, homography_matrix)``.

    ORB is the default (fast, no licensing concerns); SIFT is patent-free since
    OpenCV 4.4 and available via ``method='sift'``. The RANSAC inlier ratio is
    used as a registration-quality signal by :func:`run_traditional_pipeline`.
    """
    warped, H, _ = _register(test_img, template_img, method)
    return warped, H


# --------------------------------------------------------------------------- #
# 2. Difference
# --------------------------------------------------------------------------- #
def compute_difference(warped_test: np.ndarray, template: np.ndarray,
                       blur_ksize: int = 3) -> np.ndarray:
    """Difference between the aligned test image and the template.

    PCB defects are mostly copper(orange)-vs-substrate(green) changes that barely
    move grayscale luminance but are strong in colour, so the difference is taken
    as the **maximum absolute difference across the B/G/R channels** rather than
    on luminance. A small Gaussian blur suppresses sub-pixel-misalignment noise,
    and the warped border (black region the homography leaves uncovered) is
    masked out.

    Returns a uint8 single-channel difference map.
    """
    if warped_test.ndim == 2:                      # grayscale inputs (tests)
        a = warped_test.astype(np.int16)
        b = template.astype(np.int16)
        diff = np.abs(a - b).astype(np.uint8)
        valid_src = warped_test
    else:
        a = warped_test.astype(np.int16)
        b = template.astype(np.int16)
        diff = np.abs(a - b).max(axis=2).astype(np.uint8)  # max over B/G/R
        valid_src = _to_gray(warped_test)

    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize | 1  # force odd
        diff = cv2.GaussianBlur(diff, (k, k), 0)

    # mask out the warped border (pixels the test image does not cover)
    valid = (valid_src > 0).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))  # shave registration seam
    diff = cv2.bitwise_and(diff, diff, mask=valid)
    return diff


# --------------------------------------------------------------------------- #
# 3. Threshold + morphology + area filter
# --------------------------------------------------------------------------- #
def threshold_and_clean(diff_map: np.ndarray, threshold: int = 30,
                        morph_open_ksize: int = 3, morph_close_ksize: int = 5,
                        min_area: int = 50, max_area: int = 50000,
                        dilate_ksize: int = 15) -> np.ndarray:
    """Binarise the difference map, denoise with morphological opening then
    closing, optionally dilate, and drop connected components outside
    ``[min_area, max_area]``. Returns a uint8 binary mask (0/255).

    The optional dilation grows the (typically tight) difference blobs toward the
    scale of the human VOC annotations and merges fragments belonging to one
    defect; component features are still measured on this grown mask.
    """
    _, binary = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)

    if morph_open_ksize and morph_open_ksize >= 1:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)
    if morph_close_ksize and morph_close_ksize >= 1:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)
    if dilate_ksize and dilate_ksize >= 1:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        binary = cv2.dilate(binary, kd)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, n):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cleaned[labels == i] = 255
    return cleaned


# --------------------------------------------------------------------------- #
# 4. Component features + rule-based classification
# --------------------------------------------------------------------------- #
def _component_features(comp_mask: np.ndarray) -> dict:
    """Shape descriptors for a single connected component (0/255 mask)."""
    cnts, hier = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    area = int(cv2.countNonZero(comp_mask))
    if not cnts:
        return dict(area=area, w=0, h=0, aspect=1.0, extent=0.0,
                    solidity=0.0, circularity=0.0, holes=0)

    # largest external contour
    outer = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(outer)
    aspect = max(w, h) / max(1, min(w, h))
    extent = area / max(1, w * h)

    hull = cv2.convexHull(outer)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    perim = cv2.arcLength(outer, True)
    circularity = (4 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
    circularity = min(circularity, 1.0)

    # holes = inner contours (hierarchy parent != -1)
    holes = 0
    if hier is not None:
        holes = int(np.sum(hier[0][:, 3] != -1))

    return dict(area=area, w=int(w), h=int(h), aspect=float(aspect),
                extent=float(extent), solidity=float(solidity),
                circularity=float(circularity), holes=int(holes))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# Data-driven nearest-prototype classifier.
#
# Each class has a prototype shape vector — the *empirical median* of
# (log10 area, aspect, circularity, extent) over true-defect components,
# measured on the training images during calibration (see
# notebooks/01_traditional_baseline.ipynb / the EDA probe). A component is
# labelled by the nearest prototype under a scale-normalised distance.
#
# This keeps the system extensible (add a class == add a prototype row) and
# data-grounded, rather than hand-weighting six overlapping if/else rules.
_PROTO_FEATURES = ("log_area", "aspect", "circularity", "extent")

PROTOTYPES = {
    #                  log_area  aspect  circ   extent
    "missing_hole":    (2.90,    1.12,   0.93,  0.77),
    "mouse_bite":      (1.93,    2.14,   0.73,  0.68),
    "open_circuit":    (2.11,    1.20,   0.93,  0.89),
    "short":           (2.27,    1.77,   0.84,  0.90),
    "spur":            (1.91,    2.41,   0.70,  0.79),
    "spurious_copper": (2.74,    4.70,   0.40,  0.80),
}

# Per-feature spread across the prototypes (used to normalise distances) and a
# weight emphasising the more discriminative axes (area, aspect, circularity).
_FEATURE_SCALE = {"log_area": 0.39, "aspect": 1.24, "circularity": 0.19, "extent": 0.08}
_FEATURE_WEIGHT = {"log_area": 1.0, "aspect": 1.0, "circularity": 1.0, "extent": 0.4}
_SOFTMAX_T = 0.7  # temperature for converting distances -> confidences


def _feature_vector(f: dict) -> dict:
    area = max(f["area"], 1)
    return {
        "log_area": math.log10(area),
        "aspect": f["aspect"],
        "circularity": f["circularity"],
        "extent": f["extent"],
    }


def _classify_features(f: dict) -> tuple[str, float, dict]:
    """Return (class, confidence, per-class scores) for one component."""
    fv = _feature_vector(f)
    dists = {}
    for cls, proto in PROTOTYPES.items():
        d2 = 0.0
        for k, pv in zip(_PROTO_FEATURES, proto):
            z = (fv[k] - pv) / _FEATURE_SCALE[k]
            d2 += _FEATURE_WEIGHT[k] * z * z
        dists[cls] = math.sqrt(d2)
    # softmax over negative distance -> confidence
    exps = {c: math.exp(-d / _SOFTMAX_T) for c, d in dists.items()}
    z = sum(exps.values()) or 1.0
    probs = {c: e / z for c, e in exps.items()}
    cls = min(dists, key=dists.get)
    return cls, probs[cls], probs


def classify_components(defect_mask: np.ndarray, test_img: np.ndarray = None,
                        bbox_pad: int = 16) -> list[dict]:
    """Connected-component analysis on ``defect_mask``; classify each component
    into one of the six defect classes with a small rule-based system.

    Shape features are measured on the tight component (so elongation /
    concavity / circularity stay discriminative); the reported bounding box is
    expanded by ``bbox_pad`` pixels on every side toward the (more generous)
    VOC annotation scale and clipped to the image.

    Each detection: ``{bbox: (x1,y1,x2,y2), class: str, confidence: float,
    area: int, features: dict}``. ``confidence`` is the winning rule's share of
    the total score (a soft, comparable ranking signal for mAP)."""
    H, W = defect_mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    detections = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        f = _component_features(comp)

        cls, conf, _ = _classify_features(f)

        x1 = max(0, x - bbox_pad); y1 = max(0, y - bbox_pad)
        x2 = min(W, x + w + bbox_pad); y2 = min(H, y + h + bbox_pad)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class": cls,
            "confidence": float(conf),
            "area": area,
            "features": f,
        })
    return detections


# --------------------------------------------------------------------------- #
# 5. Orchestrator
# --------------------------------------------------------------------------- #
def run_traditional_pipeline(test_img_path: str, template_img_path: str,
                             config: dict = None) -> dict:
    """Load -> register -> diff -> threshold -> classify.

    Returns ``{detections, registration_quality, timings, intermediate_outputs}``.
    ``intermediate_outputs`` carries the stage images for visualisation.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    timings = {}

    t0 = time.perf_counter()
    test_img = cv2.imread(str(test_img_path))
    template = cv2.imread(str(template_img_path))
    if test_img is None:
        raise FileNotFoundError(f"cannot read test image: {test_img_path}")
    if template is None:
        raise FileNotFoundError(f"cannot read template image: {template_img_path}")
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    warped, H, quality = _register(test_img, template, cfg["register_method"])
    timings["register"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    diff = compute_difference(warped, template, cfg["blur_ksize"])
    timings["difference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask = threshold_and_clean(diff, cfg["threshold"], cfg["morph_open_ksize"],
                               cfg["morph_close_ksize"], cfg["min_area"], cfg["max_area"],
                               cfg["dilate_ksize"])
    timings["threshold"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    detections = classify_components(mask, warped, cfg["bbox_pad"])
    timings["classify"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())

    return {
        "detections": detections,
        "registration_quality": float(quality),
        "timings": timings,
        "intermediate_outputs": {
            "test_img": test_img,
            "template": template,
            "warped_test": warped,
            "homography": H,
            "diff_map": diff,
            "defect_mask": mask,
        },
    }
