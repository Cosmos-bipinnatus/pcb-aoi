# Stage 3 — YOLOv8n 端到端偵測器(HRIPCB)

用單一網路同時學定位 + 分類,繞過 Stage 1+2 的兩階段 proposal 瓶頸。
程式:[`src/voc_to_yolo.py`](../src/voc_to_yolo.py)、[`src/train_yolo.py`](../src/train_yolo.py)、
[`src/evaluate_yolo.py`](../src/evaluate_yolo.py)。
Notebook:[`notebooks/03_yolo_detection.ipynb`](../notebooks/03_yolo_detection.ipynb)。
數據:`results/yolo_eval_summary.json`。

## 1. 訓練設定

| 項目 | 設定 |
|---|---|
| 模型 | YOLOv8n(`yolov8n.pt` 預訓練,3.0M 參數,8.2 GFLOPs) |
| 標註 | VOC → YOLO txt(normalized `cx cy w h`),split 沿用 Stage 2 的 image-level、`seed=42`(train 483 / val 102 / test 108 張) |
| imgsz | 640(原圖 ~3000×1586 會 downsample) |
| batch / epochs | 16 / 100(`patience=20` early stopping) |
| optimizer | ultralytics 預設(auto:SGD,lr0=0.01,warmup 3 epochs) |
| 增強 | mosaic、HSV、flip、translate、scale(ultralytics 預設) |

**訓練結果:** 健康收斂 —— **最佳 @ epoch 79**,early stopping @ epoch 99,
總耗時 **~9 分鐘**(RTX 5070 Ti,torch nightly cu128)。曲線見
`results/figures/03_training_curves.png`。

> torch 版本全程未被 ultralytics 動到:仍為 `2.12.0.dev20260311+cu128`
> (以 `--no-deps` 安裝 ultralytics 並手動補裝小依賴)。

## 2. 評估結果(test split,108 張影像 / 460 instances)

| 指標 | 值 |
|---|---|
| **mAP@0.5** | **91.8%** |
| **mAP@0.5:0.95** | **46.7%** |
| 純推論 | 5.4 ms/影像(另 pre 1.5 + post 2.2 ms) |

**per-class AP:**

| class | 中文 | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|
| missing_hole | 缺孔 | 98.3 | 57.1 |
| short | 短路 | **97.5** | 49.2 |
| spur | 毛刺 | 92.6 | 44.2 |
| mouse_bite | 鼠咬 | 88.9 | 43.4 |
| spurious_copper | 多餘銅 | 87.3 | 45.9 |
| open_circuit | 斷路 | 86.1 | 40.4 |

## 3. 三方法對比(相同 test split,seed=42)

| 方法 | mAP@0.5 | 推論時間 | 備註 |
|---|---|---|---|
| Stage 1:傳統 CV | 27.4%* | 0.79 s/影像 | hand-crafted feature 分類撞牆 |
| Stage 2:R-CNN 風格(proposals + CNN) | 84.8% | Stage1 + CNN(<5 ms/crop) | proposal recall 綁住 short |
| **Stage 3:YOLOv8n 端到端** | **91.8%** | **~9 ms/影像** | 端到端,無 proposal 依賴 |

(*Stage 1 為全資料集數字;在此 test split 上為 29.3%。)

**per-class AP@0.5 三方法並排:**

| class | Stage 1 | Stage 2 (e2e) | Stage 3 (YOLO) |
|---|---|---|---|
| missing_hole | 86.1 | 93.9 | 98.3 |
| mouse_bite | 2.6 | 94.8 | 88.9 |
| open_circuit | 5.9 | 94.1 | 86.1 |
| **short** | **19.7** | **36.2** | **97.5** |
| spur | 0.4 | 91.2 | 92.6 |
| spurious_copper | 61.0 | 98.3 | 87.3 |

## 4. `short` 提升分析(專案 narrative 核心)

**`short`:19.7%(Stage 1)→ 36.2%(Stage 2)→ 97.5%(Stage 3)。**

* **Stage 1** 用 colour diff,但 short(細銅橋)的差異訊號最弱,proposal 階段就
  漏抓約一半 → AP 僅 19.7%。
* **Stage 2** 的 CNN 在 GT crops 上分 short 是 100%,但 end-to-end 只拿得到
  Stage 1 給的 proposals,漏抓的 short 無從分類 → 受 recall 綁住,僅 36.2%。
* **Stage 3** YOLO 端到端直接學習定位,不依賴外部 proposals,直接在原圖上找到
  short → AP 躍升到 97.5%。

這條曲線清楚說明整個專案的瓶頸轉移:**分類能力(Stage 1→2 解決)→ proposal
recall(Stage 2→3 由端到端架構解決)**。

## 5. 推論時間與部署考量

* 純推論 **5.4 ms/影像**(GPU),加前後處理約 **9 ms**,遠快於 Stage 1 的
  0.79 s(差影 + 形態學 + 連通元件在 CPU 上慢)。
* YOLOv8n 僅 3M 參數、6.3 MB 權重,適合產線即時 AOI 部署(可進一步 TensorRT /
  ONNX 量化)。
* 逐張 Python API 呼叫(含載圖)實測 ~133 ms/影像;批次推論或串流可攤提此 overhead。

## 6. 失敗案例與限制

* **mAP@0.5:0.95 僅 46.7%**:框雖找得到(mAP@0.5 高),但定位精度在嚴格 IoU 下
  普通,與 imgsz=640 對多 MP 影像的 downsample 有關。若要更精準的框,可試
  **imgsz=1280**(代價是速度)。
* `open_circuit` / `spurious_copper`(AP 86–87%)略低於其餘類,可能與小尺寸 +
  下採樣有關。
* 整體已超越兩階段法,且端到端、速度快 —— 對 Stage 4(三方法比較)而言素材齊備。

## 7. Stage 4 素材

三個 stage 已在**相同 test split**上評估完畢,數據與圖檔皆備:
`traditional_eval_summary.json`、`classifier_eval_summary.json`、
`yolo_eval_summary.json`,以及 `03_three_methods_comparison.png`(同圖三方法並列)。
