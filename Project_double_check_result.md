Analyzed hripcb.py:L1-100
Analyzed hripcb.py:L101-144
Analyzed classifier_eval_summary.json:L1-367
Analyzed training_log.csv:L1-32
Ran search for casting in codebase
Analyzed dataset.py:L1-160
Analyzed dataset.py:L161-216
Analyzed voc_to_yolo.py:L1-60
Analyzed compare_methods.py:L1-60
Ran check build_splits in compare_methods.py
Ran check build_splits in evaluate_traditional.py
Analyzed evaluate_traditional.py:L1-60
Ran check build_splits in evaluate_classifier.py
Analyzed dl_classifier.py:L1-63
Analyzed train_classifier.py:L1-118
Analyzed evaluate_classifier.py:L1-120
Analyzed evaluate_classifier.py:L121-256
Ran inspect 02_dl_classification.ipynb cells · Exited with code 1
Created inspect_nb.py
Ran run inspect_nb.py
Created dump_nb.py
Ran dump 02_dl_classification.ipynb content
Analyzed viz_classifier.py:L1-60
Created check_all_nbs.py
Ran run check_all_nbs.py

這是一份針對目前的本地專案 `pcb-aoi` 所完成的**檔案完整性檢查**與**系統架構分析交叉驗證（Double Check）**報告。本次檢查全程僅進行靜態程式碼審查與檔案讀取，**未修改任何程式碼，亦未啟動任何訓練程序**。

---

### 任務 1：專案目錄與檔案完整性盤點

#### 1. 專案目錄結構與既有檔案清單
經掃描專案工作區，目前的目錄樹與檔案分佈如下：

- **專案根目錄**：
  - 設定檔與測試環境：
    - [`requirements.txt`](file:///c:/Users/user/Documents/Projects/pcb-aoi/requirements.txt)：存在（定義相依套件，標頭特別強調 PyTorch CUDA 12.8 nightly 與 `ultralytics` 須以 `--no-deps` 安裝以防覆蓋 CUDA 環境）。
    - `pyproject.toml`：**不存在**於本地。
    - [`pytest.ini`](file:///c:/Users/user/Documents/Projects/pcb-aoi/pytest.ini)：存在。
    - [`.gitignore`](file:///c:/Users/user/Documents/Projects/pcb-aoi/gitignore)：存在（已將 `data/` 中的資料集、`results/intermediate/` 等設為忽略）。
  - 專案說明與授權：
    - [`README.md`](file:///c:/Users/user/Documents/Projects/pcb-aoi/README.md)：存在（詳細闡述三階段架構演進、指標數據與 Streamlit 執行說明）。
    - [`LICENSE`](file:///c:/Users/user/Documents/Projects/pcb-aoi/LICENSE)：存在（MIT License）。
  - 應用程式與腳本：
    - [`app.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/app.py)：存在（Streamlit 工業 AOI 配方管理與缺陷檢測展示系統）。
    - [`run_aoi_app.bat`](file:///c:/Users/user/Documents/Projects/pcb-aoi/run_aoi_app.bat) / [`run_aoi_app.sh`](file:///c:/Users/user/Documents/Projects/pcb-aoi/run_aoi_app.sh)：存在（Demo 啟動指令）。
- **`src/` 原始碼目錄**：
  - 包含 `__init__.py` 與 **19 個 Python 核心模組**（涵蓋資料處理、三階段演算法、評估與視覺化）。
- **`notebooks/` 互動筆記本目錄**：
  - 包含 5 個 Jupyter Notebooks：
    - `00_data_exploration.ipynb`
    - `01_traditional_baseline.ipynb`
    - `02_dl_classification.ipynb`
    - `03_yolo_detection.ipynb`
    - `04_methods_comparison.ipynb`
- **`models/` 模型權重目錄**：
  - 包含 `.gitkeep` 以及作者已訓練好的 Stage 2 權重檔 [`defect_classifier.pt`](file:///c:/Users/user/Documents/Projects/pcb-aoi/models/defect_classifier.pt)（大小約 42.7 MB / 44,800,971 bytes）。
- **`results/` 實驗數據與報表目錄**：
  - 包含量化報表與日誌：`classifier_eval_summary.json`、`training_log.csv`、`comparison_table.csv`、`comparison_table.md`、`traditional_eval_summary.json`、`yolo_eval_summary.json`，以及對應的 Markdown 報告（如 `dl_classifier_report.md`、`traditional_baseline_report.md`、`yolo_report.md`、`eda_summary.md`）。
  - 子目錄 `results/figures/`：存放 24 張高解析度視覺化圖檔（包含 EDA、混淆矩陣、PR 曲線、訓練曲線、方法對比圖等）。
  - 子目錄 `results/yolo_training/run1/weights/`：包含 Stage 3 YOLOv8n 的最佳權重 [`best.pt`](file:///c:/Users/user/Documents/Projects/pcb-aoi/results/yolo_training/run1/weights/best.pt)（大小約 6.0 MB / 6,257,706 bytes）。
- **`samples/` 內建範例目錄**：
  - 包含 `manifest.json`、`templates/`（標準樣板 PCB 圖）與 `test_images/`（1600px 縮圖），供 Streamlit Demo 在無完整資料集時亦能離線展示。
- **`tests/` 單元測試目錄**：
  - 包含 `test_traditional_cv.py`、`test_dl_classifier.py`、`test_yolo.py` 與 `test_streamlit_pipeline.py`。

---

#### 2. 資料集目錄（`data/HRIPCB/` 與 `data/yolo/`）檢查
- **本地存在狀態**：
  - `data/HRIPCB/`（含 `images/`、`Annotations/`、`PCB_USED/`）：**本地不存在**。
  - `data/yolo/`：**本地不存在**。
  - 現況下 `data/` 目錄內僅有說明文件 [`data/README.md`](file:///c:/Users/user/Documents/Projects/pcb-aoi/data/README.md)，本地目前**沒有**任何完整 HRIPCB 原始影像與標註檔（影像數量：0，標註檔數量：0）。
- **下載腳本或指引說明**：
  - 專案內**沒有自動下載的 Python 腳本**（如 `download.py`）。
  - 作者在 [`README.md`](file:///c:/Users/user/Documents/Projects/pcb-aoi/README.md#L124-L134) 的 *Getting the dataset* 區塊提供了手動下載指引，使用 Git Sparse Checkout 僅抓取所需資料夾以避開額外的旋轉增強集：
    ```bash
    cd data
    git clone --filter=blob:none --no-checkout https://github.com/Ironbrotherstyle/PCB-DATASET.git HRIPCB
    cd HRIPCB
    git sparse-checkout init --cone
    git sparse-checkout set images Annotations PCB_USED
    git checkout master
    rm -rf .git
    ```
  - 下載完成後，執行 `python -m src.voc_to_yolo` 即可自動生成 `data/yolo/` 格式。

---

#### 3. 既有基準數據讀取與摘要
專案作者已經預先跑好並留存了完整的基準數據，關鍵檔案解析如下：

##### (1) `results/classifier_eval_summary.json` 數據摘要
- **GT Crops 評估（純 CNN 分類能力上限，460 個測試集 Ground-Truth 裁切）**：
  - **Overall Accuracy**: **100.0%** (1.0)
  - 6 個類別（`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`）的 **Precision、Recall、F1-score 均達 1.000 (100%)**。
  - 混淆矩陣呈現完美的 6×6 對角線（無任何誤判，樣本數分別為 74, 77, 72, 78, 77, 82，總計 460 個 patch）。
- **End-to-End 評估（在 108 張測試影像上，以 Stage 1 proposals 作為輸入）**：
  - **Stage 1 規則分類器**：mAP@0.5 = **29.30%** (0.2930)
  - **Stage 2 ResNet18 分類器**：mAP@0.5 = **84.75%** (0.8475)，**相較 Stage 1 顯著提昇 +55.45%**。
  - **各類別 AP@0.5 比較（Stage 1 vs. Stage 2）**：
    - `missing_hole`: 86.09% → **93.91%**
    - `mouse_bite`: 2.65% → **94.78%**（巨幅改善）
    - `open_circuit`: 5.92% → **94.10%**（巨幅改善）
    - `short`: 19.70% → **36.20%**（改善有限，主因在於 Stage 1 的候選框召回率不足，為系統瓶頸）
    - `spur`: 0.43% → **91.25%**（巨幅改善）
    - `spurious_copper`: 61.01% → **98.27%**

##### (2) `results/training_log.csv` 數據摘要
- **訓練長度**：共記錄 **31 個 epoch**（設定上限為 50 epoch）。
- **最佳驗證表現**：發生在 **Epoch 21**，取得最佳指標：
  - `val_loss`: **0.01112**
  - `val_acc`: **99.77%** (0.9977)
  - `train_loss`: **0.00467**
  - `train_acc`: **99.90%** (0.9990)
- **早停機制觸發（Early Stopping）**：
  - 由於設定 `patience=10`，自 Epoch 21 起連續 10 個 epoch 驗證損失皆未能在門檻內改善（`va_loss < best_val - 1e-4`），程式於 Epoch 31 觸發 early stopping 終止。
- **動態學習率調整（`ReduceLROnPlateau`）**：
  - Epoch 1 ~ 13: `lr = 1e-4`
  - Epoch 14 ~ 25: `lr = 5e-5`
  - Epoch 26 ~ 30: `lr = 2.5e-5`
  - Epoch 31: `lr = 1.25e-5`
  - 平均單一 epoch 耗時約 8.5 ~ 9.8 秒。

---

### 任務 2：Double Check 既有 `src/` 架構分析是否完全正確

經逐行交叉核對 `src/` 相關原始碼與各階段評估實作，針對 5 項技術判定的驗證結果如下：

#### 判定 A（資料集與任務性質）
- **核驗結果**：**完全正確**。
- **程式碼證實**：
  - 在 [`src/hripcb.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/hripcb.py#L21-L28) 中明確宣告六類標籤：`CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]`。
  - 經全文檢索，專案內完全沒有「鑄件（Casting Product）」、「良/不良二元分類（OK/NG）」相關之任何代碼或資料結構。本專案核心確實專注於 HRIPCB 印刷電路板之六類表面缺陷定位與多類別分類。

#### 判定 B（防資料洩漏切分機制）
- **核驗結果**：**核心判定完全正確，並發掘一項隱藏細節**。
- **程式碼證實**：
  - [`src/dataset.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L47-L73) 的 `build_splits` 預設 `seed=42` 與 `ratios=(0.70, 0.15, 0.15)`。它透過遍歷 6 個類別資料夾分別對檔案做隨機打亂，再依比例劃分，確實為「**以影像（Image-level）為單位進行類別分層切分**」，避免了同一張 PCB 影像上不同缺陷 crop 跨分流洩漏的問題。
  - **跨階段共用切分**：
    - Stage 2：[`src/train_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L63-L64) 透過 `PCBDefectCropDataset` 使用此切分；
    - Stage 3：[`src/voc_to_yolo.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/voc_to_yolo.py#L52) 呼叫 `build_splits` 生成 YOLO 專用的 `train/val/test`；
    - Stage 4：[`src/compare_methods.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/compare_methods.py#L146) 直接使用 `build_splits(DATA)["test"]`（108 張圖）進行統一比較。
  - **隱藏細節補充**：
    - 單獨執行 [`src/evaluate_traditional.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_traditional.py#L1-L15) 時，其腳本預設是針對 HRIPCB **全資料集（693 張圖）** 進行評估；
    - 只有在 `src/evaluate_classifier.py`（的 `evaluate_end_to_end`）與 `src/compare_methods.py` 中，Stage 1 的表現才被收斂至相同的固定 test split（108 張圖）進行公平對比。

#### 判定 C（Stage 2 ResNet18 訓練與覆寫機制）
- **核驗結果**：**完全正確，且存在「重新訓練即破壞基準模型」的風險**。
- **程式碼證實**：
  - 模型本體：[`src/dl_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dl_classifier.py#L22-L37) 載入 `ResNet18_Weights.IMAGENET1K_V1`，並置換 `self.backbone.fc = nn.Linear(in_features, 6)`。
  - 邊界襯墊：[`src/dataset.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L100-L111) 之 `crop_with_padding` 預設 `pad_ratio=0.3`。
  - 訓練超參數：[`src/train_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L26-L34) 採用 Adam (`lr=1e-4`)、`nn.CrossEntropyLoss()`、`ReduceLROnPlateau(..., factor=0.5, patience=4)` 與 `patience=10`。
  - **覆寫機制**：[`src/train_classifier.py` 第 95-98 行](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L95-L98)：
    ```python
    if va_loss < best_val - 1e-4:
        best_val, best_epoch, patience_ctr = va_loss, epoch, 0
        save_model(model, WEIGHTS, ...)
    ```
    此處的 `WEIGHTS` 硬編碼為單一檔案 `models/defect_classifier.pt`。
  - **重要警示**：程式碼中**沒有**任何備份、版本標籤或防止覆寫的安全防護。一旦執行 `python -m src.train_classifier`，只要任何 epoch 的 `val_loss` 優於初始值，就會直接覆蓋專案原本保存好的預訓練權重。

#### 判定 D（隨機性控制缺口）
- **核驗結果**：**完全正確**。
- **程式碼證實**：
  - [`src/train_classifier.py` 第 61 行](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L61) 僅呼叫了 `torch.manual_seed(cfg["seed"])`。
  - 遺漏了：
    1. Python 內建的 `random.seed(cfg["seed"])`；
    2. NumPy 的 `np.random.seed(cfg["seed"])`；
    3. CuDNN 確定性設定：未設置 `torch.backends.cudnn.deterministic = True` 與 `torch.backends.cudnn.benchmark = False`；
    4. CUDA 多卡種子：未設置 `torch.cuda.manual_seed_all`。
  - 配合 `get_transforms("train")` 中使用的 `RandomHorizontalFlip`、`RandomRotation(15)`、`ColorJitter` 等隨機增強，若未全域鎖定隨機性，在不同硬體環境或 PyTorch 實作版本下重新訓練，將存在數值無法 100% bit-exact 重現的風險。

#### 判定 E（雙軌評估設計）
- **核驗結果**：**完全正確**。
- **程式碼證實**：
  - [`src/evaluate_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py) 確實包含雙重設計：
    1. [`evaluate_gt_crops`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py#L151-L176)：純粹以 Ground-Truth BBox 切出的 patch 作為輸入，用以驗證 CNN 的分類能力天花板，輸出 Overall Accuracy、6 類別個別 Precision / Recall / F1，以及 6×6 混淆矩陣。
    2. [`evaluate_end_to_end`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py#L181-L216)：在測試集影像上先執行 Stage 1 的影像處理流程以擷取 Region Proposals，再將這同一批 Proposal 同時餵給「傳統規則分類器」與「ResNet18 分類器」進行比對，計算 mAP@0.5。其混淆矩陣為 7×7（6 缺陷類 + 1 類 background，包含定位匹配的 FP 與 FN）。
  - **隱藏細節補充**：`evaluate_end_to_end` 在執行時會即時呼叫 `run_traditional_pipeline` 重新跑傳統差分流程，因此本地必須具備 `data/HRIPCB`（含 test images 與 `PCB_USED` 範本圖）才能順利執行。

---

### 任務 3：檢查 `notebooks/` 的額外脈絡

經檢查 [`notebooks/02_dl_classification.ipynb`](file:///c:/Users/user/Documents/Projects/pcb-aoi/notebooks/02_dl_classification.ipynb)（共 17 個 cells）以及其他筆記本，發現以下架構與邏輯特性：

1. **Notebook 定位為「純展示與成果報告」**：
   - 該 Notebook **完全沒有重新執行模型訓練或推論計算**。
   - 它並未重複實作任何不在 `src/` 中的演算法或實驗邏輯，其定位是作為技術成果展示、實驗敘事導覽（Walkthrough）與報告閱覽。

2. **如何呼叫 `src/` 模組**：
   - 程式碼僅以 `from src.hripcb import CLASSES, CLASS_ZH` 取得類別清單與中文字典。
   - 量化數據部分：直接透過 `json.loads` 讀取先前由 `src/evaluate_classifier.py` 產生的 [`results/classifier_eval_summary.json`](file:///c:/Users/user/Documents/Projects/pcb-aoi/results/classifier_eval_summary.json)，並使用 `pandas.DataFrame` 整理成對比表格。

3. **視覺化呈現方式**：
   - Notebook 中的所有圖表（包括訓練曲線、GT 混淆矩陣、End-to-End 混淆矩陣、樣本預測圖、失敗案例分析圖等），均直接透過 `IPython.display.Image` 引用在 [`results/figures/`](file:///c:/Users/user/Documents/Projects/pcb-aoi/results/figures/) 目錄下既有的靜態 PNG 檔案。
   - 所有繪圖運算與分析邏輯實際上已完整實作在 [`src/viz_classifier.py`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_classifier.py) 中，並非寫在 Notebook 內部。

4. **隱藏細節與其他 Notebooks 脈絡**：
   - 在 `02_dl_classification.ipynb` 的 Cell 01 中，包含一段路徑回退邏輯：
     ```python
     if not (ROOT / "src").exists():
         ROOT = Path(r"D:\ash-portfolio\pcb-aoi")
     ```
     這透露了作者最初的本地專案開發路徑為 `D:\ash-portfolio\pcb-aoi`。
   - 專案中的其餘筆記本（`00_data_exploration.ipynb` 至 `04_methods_comparison.ipynb`）同樣遵循一致的架構規範：**運算與繪圖邏輯封裝於 `src/`，筆記本端專注於讀取 `results/` 產出與圖檔展示**。

---

### 總結

1. 專案的目錄結構、預先跑好的評估數據（`classifier_eval_summary.json`、`training_log.csv` 等）以及模型權重（`defect_classifier.pt` 與 YOLO 的 `best.pt`）皆**完整存在**。
2. 完整訓練資料集（`data/HRIPCB/` 與 `data/yolo/`）目前未下載於本地，若需復現或重新跑端到端評估，需透過 `git sparse-checkout` 指引獲取。
3. 您先前針對 `src/` 的技術判定（A 至 E）**完全正確且精準**，同時進一步確認了：
   - **`train_classifier.py` 的覆寫特性**：啟動訓練會直接蓋掉既有權重；
   - **隨機性缺口**：缺少全域隨機種子與 CuDNN 確定性設定；
   - **Notebooks 角色**：全系列筆記本均為讀取 `results/` 的輕量展示本，未脫離 `src/` 定義之架構。