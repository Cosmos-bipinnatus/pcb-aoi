"""PCB-AOI 互動 demo(Streamlit)。

在瀏覽器上比較三種 PCB 缺陷檢測方法,並模擬產線 AOI 的 Recipe 切換流程:
Stage 1 需先選 golden template(Recipe),Stage 2/3 直接吃單張圖。

啟動:  streamlit run app.py   (或執行 run_aoi_app.bat / run_aoi_app.sh)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src import inference_api as api
from src.viz_inference import draw_detections
from src.hripcb import CLASSES, CLASS_ZH

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MANIFEST = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="PCB-AOI Demo", page_icon="🔍", layout="wide")


def _read_bgr(path):
    return cv2.imread(str(path))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔍 PCB-AOI Demo")
    st.caption("HRIPCB 缺陷檢測 · 三方法比較")

    method = st.radio(
        "選擇檢測方法",
        ["Stage 1: 傳統 CV (差影+形態學)",
         "Stage 2: R-CNN 風格 (proposals + CNN)",
         "Stage 3: YOLOv8n (端到端)"],
        index=2,
    )
    stage = int(method.split(":")[0].split()[-1])

    recipe_cls = None
    if stage == 1:
        st.markdown("**缺陷類別 (Recipe)**")
        recipe_cls = st.selectbox(
            "對應該類的 golden template",
            CLASSES, format_func=lambda c: f"{c} ({CLASS_ZH[c]})")
        st.caption(f"→ 使用 template: `{Path(MANIFEST['recipes'][recipe_cls]).name}`")

    st.markdown("---")
    st.markdown("**進階設定**")
    conf = st.slider("Confidence threshold (Stage 2/3 適用)", 0.0, 1.0, 0.25, 0.05)

    st.markdown("---")
    st.markdown("**業界對應說明**")
    st.info(
        "真實 AOI 系統中,Stage 1 流程稱為 **Recipe Management**,工程師會依產品"
        "料號切換對應 golden template 後再檢測。Stage 2/3 為端到端方法,免 template。")


# --------------------------------------------------------------------------- #
# 主畫面
# --------------------------------------------------------------------------- #
st.title("PCB-AOI:HRIPCB 缺陷檢測示範")

c_src, c_btn = st.columns([3, 1])
with c_src:
    source = st.radio("輸入來源", ["選擇範例", "上傳 PCB 圖"], horizontal=True)
    test_bgr, test_name, sample_meta = None, None, None
    if source == "選擇範例":
        names = [s["name"] for s in MANIFEST["samples"]]
        pick = st.selectbox("選擇範例", names)
        sample_meta = next(s for s in MANIFEST["samples"] if s["name"] == pick)
        test_bgr = _read_bgr(SAMPLES / sample_meta["image"])
        test_name = Path(sample_meta["image"]).name
    else:
        up = st.file_uploader("上傳 PCB 圖", type=["jpg", "jpeg", "png"])
        if up is not None:
            arr = np.frombuffer(up.read(), np.uint8)
            test_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            test_name = up.name

with c_btn:
    st.write("")
    st.write("")
    run = st.button("🚀 執行檢測", use_container_width=True, type="primary")

# Stage 1 的 template 解析:範例優先用自帶 template,否則用 sidebar Recipe
if stage == 1:
    if source == "選擇範例" and sample_meta is not None:
        template_rel = sample_meta["template"]
        st.caption(f"Stage 1 將以範例自帶 template 差影:`{Path(template_rel).name}`")
    else:
        template_rel = MANIFEST["recipes"][recipe_cls] if recipe_cls else None
else:
    template_rel = None


def _run_inference(bgr):
    if stage == 1:
        tmpl = _read_bgr(SAMPLES / template_rel)
        return api.run_stage1(bgr, tmpl, conf_threshold=None)
    if stage == 2:
        return api.run_stage2(bgr, conf_threshold=conf)
    return api.run_stage3(bgr, conf_threshold=conf)


if run and test_bgr is not None:
    with st.spinner("檢測中..."):
        result = _run_inference(test_bgr)
        annotated = draw_detections(test_bgr, result["detections"])
        st.session_state["result"] = {
            "result": result, "annotated": annotated,
            "input_rgb": cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB),
            "name": test_name,
        }
elif run and test_bgr is None:
    st.warning("請先選擇範例或上傳一張 PCB 圖。")


# --------------------------------------------------------------------------- #
# 結果區
# --------------------------------------------------------------------------- #
if "result" in st.session_state:
    state = st.session_state["result"]
    result = state["result"]
    st.markdown("---")
    st.subheader(f"檢測結果 — {result['method']}")
    if result.get("note"):
        st.caption("ℹ️ " + result["note"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**輸入影像**")
        st.image(state["input_rgb"], use_container_width=True)
    with col2:
        st.markdown("**標註結果**")
        st.image(state["annotated"], use_container_width=True)

    dets = result["detections"]
    confs = [d["confidence"] for d in dets]
    by_cls = Counter(d["class"] for d in dets)
    m1, m2, m3 = st.columns(3)
    m1.metric("偵測數量", len(dets))
    m2.metric("平均信心度", f"{(sum(confs)/len(confs)):.3f}" if confs else "—")
    m3.metric("純推論時間", f"{result['inference_ms']:.1f} ms")

    if by_cls:
        st.markdown("**各類偵測數量**")
        st.table({"類別": list(by_cls.keys()),
                  "中文": [CLASS_ZH[c] for c in by_cls],
                  "數量": list(by_cls.values())})

    # 下載
    d1, d2 = st.columns(2)
    ok, png = cv2.imencode(".png", cv2.cvtColor(state["annotated"], cv2.COLOR_RGB2BGR))
    d1.download_button("⬇️ 下載標註圖", png.tobytes(),
                       file_name=f"annotated_{state['name']}.png", mime="image/png")
    meta = {"input": state["name"], "method": result["method"],
            "inference_ms": result["inference_ms"], "detections": dets}
    d2.download_button("⬇️ 下載 JSON metadata",
                       json.dumps(meta, ensure_ascii=False, indent=2),
                       file_name=f"result_{state['name']}.json", mime="application/json")
else:
    st.info("選擇方法與輸入,按「執行檢測」開始。")
