# AGENTS.md：PCB-AOI 系統開發與實驗規範手冊

本文件為 **HRIPCB 工業 AOI 缺陷檢測與分類系統** 之核心協作開發與實驗規範指南。所有參與本專案開發、重構、維護及實驗評估之 AI Agent 與研發人員，在進行任何程式碼更動與實驗執行前，必須嚴格閱讀並恪遵本規範。

---

## 1. 專案概覽與四階段架構（Project Overview）

本專案基於北京大學 **HRIPCB** 資料集（包含 693 張高解析度 PCB 影像、2,953 個標註瑕疵，涵蓋 `missing_hole`、`mouse_bite`、`open_circuit`、`short`、`spur`、`spurious_copper` 6 大缺陷類別），實作了從傳統電腦視覺至現代深度學習的演進系統。系統展示了明確的技術瓶頸突破鏈：**傳統方法定位精準但分類受限 → 兩階段分類器提升分類能力但受限於候選框召回率 → 端到端 YOLO 徹底擺脫候選框依賴**。

整體專案架構劃分為五個階段（Stage 0 至 Stage 4）：

```
pcb-aoi/
├── src/
│   ├── hripcb.py               # [Stage 0] 資料解析與 VOC 工具
│   ├── eda.py                  # [Stage 0] 探索性資料分析
│   ├── plot_utils.py           # [Stage 0] 中文字型與繪圖工具
│   ├── traditional_cv.py       # [Stage 1] 傳統 CV 基準演算法
│   ├── evaluate_traditional.py # [Stage 1] 傳統 CV 基準評估
│   ├── viz_traditional.py      # [Stage 1] 傳統 CV 流程可視化
│   ├── dataset.py              # [Stage 2] 裁切資料集與分層切分
│   ├── dl_classifier.py        # [Stage 2] ResNet18 分類器架構
│   ├── train_classifier.py     # [Stage 2] 分類器訓練迴圈
│   ├── evaluate_classifier.py  # [Stage 2] 分類器雙軌評估
│   ├── viz_classifier.py       # [Stage 2] 分類器曲線與錯誤分析
│   ├── voc_to_yolo.py          # [Stage 3] VOC 轉 YOLO 格式工具
│   ├── train_yolo.py           # [Stage 3] YOLOv8n 訓練腳本
│   ├── evaluate_yolo.py        # [Stage 3] YOLOv8n 驗證評估
│   ├── viz_yolo.py             # [Stage 3] YOLOv8n 偵測可視化
│   ├── compare_methods.py      # [Stage 4] 跨方法統一比較評估
│   ├── viz_compare.py          # [Stage 4] 跨方法比較視覺化圖表
│   ├── inference_api.py        # [Stage 4] 統一推論介面 (Demo 後端)
│   └── viz_inference.py        # [Stage 4] 推論結果框選渲染
├── app.py                      # Streamlit AOI Recipe 互動展示系統
├── tests/                      # pytest 測試套件
├── results/                    # 訓練紀錄、評估報告與圖表輸出
├── models/                     # 訓練權重檔案
└── data/                       # HRIPCB 與 YOLO 衍生資料集
```

### 各階段模組職責說明

| 階段 | 核心技術與目標 | 核心原始碼檔案 | 關鍵輸出產物 |
|---|---|---|---|
| **Stage 0：資料解析與探索** | 解析 Pascal-VOC 標註，統計缺陷長寬比、面積分佈、類別平衡度，並視覺化無缺陷模板（`PCB_USED`）。 | [`src/hripcb.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/hripcb.py)<br>[`src/eda.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/eda.py)<br>[`src/plot_utils.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/plot_utils.py) | `results/eda_summary.md`<br>`results/figures/00_*.png` |
| **Stage 1：傳統 CV 基準** | 需金牌模板（Golden Template）。ORB/SIFT 特徵匹配 + RANSAC 單應性對位；高斯模糊後求取 BGR 最大差影；形態學開閉運算與連通元件提取候選框；形狀特徵之原型規則分類。 | [`src/traditional_cv.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/traditional_cv.py)<br>[`src/evaluate_traditional.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_traditional.py)<br>[`src/viz_traditional.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_traditional.py) | `results/traditional_eval_summary.json`<br>`results/traditional_baseline_report.md`<br>`results/figures/01_*.png` |
| **Stage 2：R-CNN 風格兩階段分類器** | 候選區域外擴 30% 上下文裁切（`pad_ratio=0.3`），輸入 ImageNet 預訓練之 ResNet18 遷移學習分類器。具備雙軌評估：(1) GT crops 上限評估 (2) End-to-end 同候選框比對。 | [`src/dataset.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py)<br>[`src/dl_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dl_classifier.py)<br>[`src/train_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py)<br>[`src/evaluate_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py)<br>[`src/viz_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_classifier.py) | `models/defect_classifier.pt`<br>`results/training_log.csv`<br>`results/classifier_eval_summary.json`<br>`results/dl_classifier_report.md`<br>`results/figures/02_*.png` |
| **Stage 3：YOLOv8n 端到端物件偵測** | VOC 標註轉換為 YOLO 格式；訓練 YOLOv8n 一步到位同時輸出邊界框與類別，擺脫兩階段候選框召回率瓶頸（特別是 `short` 短路缺陷）。 | [`src/voc_to_yolo.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/voc_to_yolo.py)<br>[`src/train_yolo.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_yolo.py)<br>[`src/evaluate_yolo.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_yolo.py)<br>[`src/viz_yolo.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_yolo.py) | `data/yolo/`<br>`results/yolo_training/run1/`<br>`results/yolo_eval_summary.json`<br>`results/yolo_report.md`<br>`results/figures/03_*.png` |
| **Stage 4：跨方法統一評估與推論 API** | 在固定的相同測試集（`seed=42`）上統一對齊指標（mAP@0.5, mAP@0.5:0.95, 純推論延遲）；提供給 Streamlit Recipe 介面調用之統一推論 API。 | [`src/compare_methods.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/compare_methods.py)<br>[`src/viz_compare.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_compare.py)<br>[`src/inference_api.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/inference_api.py)<br>[`src/viz_inference.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_inference.py) | `results/comparison_table.csv`<br>`results/comparison_table.md`<br>`results/figures/04_*.png` |

---

## 2. 當前任務目標：作業二「主題 4（CNN Training Epochs 實驗）」專項規範

本專案現階段的開發與研究焦點聚焦於 **Stage 2（ResNet18 缺陷分類器）** 的超參數敏感度實驗，主題為：**「探討訓練輪數（Training Epochs）對 CNN 分類性能、收斂穩定性與測試集不良品召回率的影響」**。

AI Agent 在進行任何程式碼擴充、修改或自動化實驗時，**必須絕對嚴格遵守以下五條鐵律**：

### 鐵律一：指定修改參數與三組設定（Controlled Independent Variable & Epoch Sets）
- **唯一自變數（Independent Variable）**：本實驗唯一允許調整的超參數為 **`Training Epochs`（訓練總輪數）**。
- **三組指定實驗設定**：
  - **實驗組 A**：`epochs = 5`
  - **實驗組 B**：`epochs = 10`
  - **實驗組 C**：`epochs = 15`
- **基準對照組（Baseline）**：
  - 專案原始預設設定（`max_epochs = 50` 搭配 `patience = 10` 之 Early Stopping）作為全局基準對照，原始紀錄應完整保留。
- **其餘超參數嚴格凍結**：學習率（`lr = 1e-4`）、優化器（`Adam`）、Batch Size（`batch_size = 64`）、排程器（`ReduceLROnPlateau(factor=0.5, patience=4)`）、外擴裁切比例（`pad_ratio = 0.3`）以及輸入維度（`224x224`）皆必須維持不變。

### 鐵律二：嚴格固定實驗控制條件（Reproducibility & Data Integrity）
1. **資料切分絕對一致（No Split Drift）**：
   - 必須完全沿用 [`src/dataset.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py) 中的 `build_splits(seed=42)`。
   - 切分比例為 **70% 訓練集、15% 驗證集、15% 測試集**。
   - 必須堅持以 **影像為單位（Image-level）** 依類別分層切分，嚴防同一張影像之不同缺陷 bbox 跨 split 分配所造成的嚴重資料洩漏（Data Leakage）。
2. **全局隨機種子完全固化（Deterministic Random Seeds）**：
   - 每次執行訓練與評估時，必須在腳本最前端設定完整的隨機種子控制代碼：
     ```python
     import random
     import numpy as np
     import torch

     def set_seed(seed: int = 42):
         random.seed(seed)
         np.random.seed(seed)
         torch.manual_seed(seed)
         torch.cuda.manual_seed_all(seed)
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
     ```
3. **初始權重嚴格隔離（Zero Weight Carry-Over）**：
   - 三組實驗（Epoch 5, 10, 15）必須**各自從完全相同的初始狀態重新開始訓練**。
   - 骨幹網路統一載入標準 `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` 預訓練權重。
   - 最後的全連接分類層（`fc`）必須在相同的隨機種子初始化下生成。
   - **嚴禁**接續前一組訓練後留存的模型權重繼續微調（嚴禁任何形式的 Weight Carry-over）。
4. **測試集嚴禁洩漏調參（Strict Test Split Sequestration）**：
   - 驗證集（`val` split）僅用於訓練過程中的早停判斷與損失監控。
   - **嚴禁用測試集（`test` split）反覆調試、挑選超參數或挑選最優 Checkpoint**；測試集僅能在每組訓練完全終止後，作為最終指標結算的唯一依據。

### 鐵律三：必記錄的量化指標（Mandatory Quantitative Metrics）
所有實驗必須輸出結構化的量化指標，缺一不可：
1. **訓練過程監控指標**：
   - 每輪之 `Train Loss` 與 `Val Loss`
   - 每輪之 `Train Accuracy` 與 `Val Accuracy`
   - 該輪學習率 `lr` 與單輪耗時 `sec`
2. **測試集核心指標（Ground-Truth Crops 評估）**：
   - 在高可靠度測試集真實標籤（GT Crops）下，將 6 類缺陷視為正類（One-vs-Rest），計算並輸出：
     - **6 類個別不良品召回率（Per-class Recall）**：`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
     - **宏平均召回率（Macro-average Recall）**：$\frac{1}{6}\sum_{c=1}^6 \text{Recall}_c$
     - **微平均召回率（Micro-average Recall）**：$\frac{\sum \text{TP}_c}{\sum (\text{TP}_c + \text{FN}_c)}$
     - **測試集分類準確率（Overall Test Accuracy）**
3. **系統級 End-to-End 指標（選配/對照）**：
   - 使用 Stage 1 提取之候選框經 Stage 2 CNN 重分類後的整體 `mAP@0.5` 與各類別 AP。

### 鐵律四：檔案與權重隔離原則（File & Checkpoint Isolation Policy）
為確保專案原有主線成果不受破壞，所有實驗組之產出必須進行檔案命名隔離：
1. **保護主線成果**：不得直接覆蓋既有的 `models/defect_classifier.pt`、`results/training_log.csv` 及 `results/classifier_eval_summary.json`。
2. **實驗檔案命名規則**：以 `_ep{5,10,15}` 後綴區隔：
   - 訓練紀錄 CSV：`results/training_log_ep{5,10,15}.csv`
   - 權重檔案：
     - 最後一輪權重（Last Epoch）：`models/defect_classifier_ep{5,10,15}_last.pt`
     - 最佳驗證損失權重（Best Val Loss Epoch）：`models/defect_classifier_ep{5,10,15}_best.pt`
   - 評估結果 JSON：`results/classifier_eval_ep{5,10,15}.json`（需同時載明 Last 與 Best 權重的評估數據）
   - 訓練曲線圖表：`results/figures/02_training_curves_ep{5,10,15}.png`
3. **對照分析彙整表**：
   - 實驗結束後，應彙整產生綜合對照表（Markdown 與 CSV），記載不同 Epoch 設定（5 vs 10 vs 15 vs 原始 Baseline）下的 Val Loss、Val Acc、Macro Recall、Micro Recall 以及各類缺陷召回率之變化趨勢。

### 鐵律五：保持向後相容（Backward Compatibility & API Integrity）
1. **介面相容性**：修改 [`src/train_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py)、[`src/evaluate_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py) 或 [`src/viz_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_classifier.py) 支援 CLI 參數（例如 `--epochs`, `--weights-out`, `--log-out`）時，各函式與 CLI 的**預設參數值（Default Values）必須保持指向原始預設路徑**。
2. **防破壞機制**：
   - 嚴禁破壞 [`src/compare_methods.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/compare_methods.py) 對 `DEFAULT_WEIGHTS` 的引用。
   - 嚴禁破壞 [`src/inference_api.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/inference_api.py) 及 Streamlit [`app.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/app.py) 的推論邏輯。
   - 任何改動後必須保證 `pytest` 單元測試全數 PASS。

---

## 3. 常用執行指令與驗證流程（Commands & Workflows）

所有模組均應使用 Python 模組方式（`python -m src.<module>`）於專案根目錄下調用。

### 標準執行指令對照表

| 階段 / 功能 | 執行指令 | 核心輸入 / 前置條件 | 主要產出路徑 | 說明 |
|---|---|---|---|---|
| **環境檢查** | `pytest` | 虛擬環境需安裝 requirements | 終端測試結果 (23 passed) | 驗證全階段演算法與推論 API 正確性 |
| **Stage 0 EDA** | `python -m src.eda` | `data/HRIPCB/` 資料夾 | `results/figures/00_*.png`<br>`results/eda_summary.md` | 產生缺陷統計與模板可視化圖表 |
| **Stage 1 評估** | `python -m src.evaluate_traditional` | `data/HRIPCB/` (含 `PCB_USED`) | `results/traditional_eval_summary.json` | 執行傳統 CV 基準評估並計算 mAP |
| **Stage 1 視覺化** | `python -m src.viz_traditional` | `data/HRIPCB/`, 中文字型 | `results/figures/01_*.png` | 產生差影與連通元件提取過程圖 |
| **Stage 2 預設訓練** | `python -m src.train_classifier` | `data/HRIPCB/` (Pascal VOC) | `models/defect_classifier.pt`<br>`results/training_log.csv` | 原始預設訓練 (max_epochs=50, patience=10) |
| **Stage 2 預設評估** | `python -m src.evaluate_classifier` | `models/defect_classifier.pt` | `results/classifier_eval_summary.json` | 評估 GT crops accuracy 與 End-to-end mAP |
| **Stage 2 預設視覺化** | `python -m src.viz_classifier` | `results/training_log.csv` | `results/figures/02_*.png` | 繪製訓練曲線、混淆矩陣與預測範例 |
| **Stage 3 資料轉換** | `python -m src.voc_to_yolo` | `data/HRIPCB/` | `data/yolo/data.yaml` | 將 VOC 標註轉換為 YOLO txt 格式 |
| **Stage 3 YOLO 訓練** | `python -m src.train_yolo` | `data/yolo/data.yaml` | `results/yolo_training/run1/weights/best.pt` | 訓練 YOLOv8n 端到端檢測器 |
| **Stage 3 YOLO 評估** | `python -m src.evaluate_yolo` | `results/yolo_training/run1/` | `results/yolo_eval_summary.json` | 評估 YOLOv8n 測試集指標 |
| **Stage 3 YOLO 視覺化** | `python -m src.viz_yolo` | YOLO 權重與測試圖 | `results/figures/03_*.png` | 繪製 YOLO 檢測框與置信度熱區 |
| **Stage 4 跨方法比較** | `python -m src.compare_methods` | Stage 1/2/3 評估產物 | `results/comparison_table.csv`<br>`results/comparison_table.md` | 整合產出三方法統一評估對照表 |
| **Stage 4 比較圖繪製** | `python -m src.viz_compare` | `results/intermediate/` 資料 | `results/figures/04_*.png` | 繪製三方法 PR 曲線疊圖與推論延遲長條圖 |
| **互動 Demo 系統** | `streamlit run app.py` | 需已產出三階段模型權重 | 瀏覽器開啟 `http://localhost:8501` | 啟動模擬產線 Recipe 之 AOI 檢測介面 |

---

### 作業二主題 4 專項執行命令規範範例

為滿足主題 4 之參數隔離原則，擴充後的 `train_classifier.py` 與 `evaluate_classifier.py` 應支援指定 epoch 參數調用：

```bash
# 1. 執行 5 個 Epoch 實驗
python -m src.train_classifier --epochs 5 --weights models/defect_classifier_ep5.pt --log results/training_log_ep5.csv
python -m src.evaluate_classifier --weights models/defect_classifier_ep5.pt --out results/classifier_eval_ep5.json

# 2. 執行 10 個 Epoch 實驗
python -m src.train_classifier --epochs 10 --weights models/defect_classifier_ep10.pt --log results/training_log_ep10.csv
python -m src.evaluate_classifier --weights models/defect_classifier_ep10.pt --out results/classifier_eval_ep10.json

# 3. 執行 15 個 Epoch 實驗
python -m src.train_classifier --epochs 15 --weights models/defect_classifier_ep15.pt --log results/training_log_ep15.csv
python -m src.evaluate_classifier --weights models/defect_classifier_ep15.pt --out results/classifier_eval_ep15.json
```

---

## 4. AI Agent 操作規範與邊界安全準則

1. **嚴禁破壞性修改**：若非使用者明確要求，不可逕自刪除現有的 baseline 紀錄、報表與歷史 figure 圖檔。
2. **維持文檔與註解完整性**：修改任何 Python 模組時，必須保留原有註解、Docstrings，若新增功能或參數需補齊對應之繁體中文型別標註與說明。
3. **驗證先行**：任何涉及核心邏輯或函式簽名的修改，完成後必須優先執行 `pytest`。若單元測試失敗，必須立即修復至全數通過。
4. **透明化報告**：每次進行實驗或程式碼更動後，向使用者回報具體的數值變動（如 Train/Val Loss、各類缺陷之 Recall 差異）、產出檔案路徑以及相容性測試結果。

