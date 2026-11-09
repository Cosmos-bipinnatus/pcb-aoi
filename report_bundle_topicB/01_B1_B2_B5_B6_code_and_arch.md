# 作業二主題 B：環境、程式碼修改、系統架構與資料流對照報告 (B-1, B-2, B-5, B-6)

本文件完整整理國立清華大學工業工程與工程管理學系 **作業二主題 B（Stage 2: ResNet18 CNN 缺陷分類器，Epochs = 5, 10, 15 實驗）** 中，報告章節 **B-1（環境與資料集統計）**、**B-2（程式碼修改對照）**、**B-5（流程圖與函式對照）** 與 **B-6（三大核心技術與資料流）** 所需之全部精確數據、程式碼片段與行號對照。

---

## 一、 B-1: 軟硬體實驗環境與資料集精確統計

### 1.1 實體運算環境規格（實機檢測數據）

所有實驗均於本地工作站實體環境執行，各項軟硬體規格如下表所示：

| 項目 | 規格與版本 | 詳細說明 / 備註 |
|---|---|---|
| **作業系統 (OS)** | Windows 11 Home (64-bit) | Build 10.0.26300 (SP0) |
| **中央處理器 (CPU)** | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics | 8 核心 / 16 執行緒，主頻 3.8 GHz (Turbo 5.1 GHz) |
| **系統主記憶體 (RAM)** | 16 GB DDR5 (實測可用 15.25 GB) | 雙通道高速記憶體 |
| **圖形處理器 (GPU)** | NVIDIA GeForce RTX 4050 Laptop GPU | 6.00 GB GDDR6 專屬視訊記憶體 (VRAM) |
| **平行運算架構** | CUDA 12.4 / cuDNN 9.1 | 硬體支援 Tensor Core 加速 |
| **Python 直譯器** | Python 3.11.14 (64-bit) | MSC v.1944 AMD64 架構 |
| **深度學習框架** | PyTorch 2.6.0+cu124 | 搭配 torchvision 0.21.0+cu124 |
| **電腦視覺函式庫** | OpenCV 4.11.0 (`opencv-python`) | 負責影像載入、單應性矩陣變換與形態學運算 |
| **科學計算與繪圖** | NumPy 2.2.3 / Matplotlib 3.10.0 / Pandas | 搭配中文字型 `Microsoft JhengHei` (微軟正黑體) |

---

### 1.2 HRIPCB 資料集切分統計（`seed=42`, ratios = 70% / 15% / 15%）

本實驗基於北京大學 **HRIPCB** 工業缺陷檢測資料集，共收錄 **693 張高解析度 PCB 影像**（解析度約 $3000 \times 3000$ 像素）與 **2,953 個**以 Pascal-VOC XML 標註之缺陷邊界框（Bounding Box）。

#### 嚴格防洩漏切分原則（Image-level Stratified Splitting）
1. **影像級切分（No Split Drift）**：切分嚴格以「**影像檔案**」為不可分割之基本單位。若某張 PCB 影像上標註有 4 個不同瑕疵，此 4 個瑕疵之裁切塊必全部歸入同一集合，嚴防同一影像因基板紋理相似造成測試集特徵洩漏（Data Leakage）。
2. **類別分層平衡（Stratification）**：對 6 大類別資料夾分別固定 `seed=42` 進行獨立隨機洗牌，精確維持 70% 訓練集、15% 驗證集、15% 測試集之樣本分佈比例。

#### 資料集各集合與 6 大類別樣本分佈表

| 缺陷類別 (Class Name) | 中文缺陷名稱 | 訓練集 (Train, 70%) | 驗證集 (Val, 15%) | 測試集 (Test, 15%) | 全資料集總計 (Total) |
|---|---|---|---|---|---|
| `missing_hole` | 缺孔 | 347 | 76 | 74 | **497** |
| `mouse_bite` | 鼠咬 / 邊緣囓齒 | 339 | 76 | 77 | **492** |
| `open_circuit` | 開路 | 337 | 73 | 72 | **482** |
| `short` | 短路 | 341 | 72 | 78 | **491** |
| `spur` | 突刺 | 338 | 73 | 77 | **488** |
| `spurious_copper` | 假銅斑 / 殘銅 | 349 | 72 | 82 | **503** |
| **缺陷裁切塊總數 (Crops)** | **所有缺陷合計** | **2,051** | **442** | **460** | **2,953** |
| **影像總數 (Images)** | **PCB 原圖張數** | **483 張 (69.7%)** | **102 張 (14.7%)** | **108 張 (15.6%)** | **693 張 (100.0%)** |

> **統計檢核**：
> - 訓練集比例：$2051 / 2953 = 69.45\%$（影像 $483 / 693 = 69.70\%$）
> - 驗證集比例：$442 / 2953 = 14.97\%$（影像 $102 / 693 = 14.72\%$）
> - 測試集比例：$460 / 2953 = 15.58\%$（影像 $108 / 693 = 15.58\%$）
> 各集合內 6 類缺陷分佈均勻對稱，測試集擁有高達 460 個獨立真實瑕疵塊，具備高度統計顯著性。

---

## 二、 B-2: 修改前 vs. 修改後程式碼完整對照

為了落實專案規範手冊《AGENTS.md》之鐵律一（唯一自變數為 Epochs）、鐵律二（隨機種子固化與零權重殘留）、鐵律三（記錄宏平均與微平均召回率）以及鐵律四（檔案與權重隔離原則），專案針對 `src/train_classifier.py` 與 `src/evaluate_classifier.py` 進行了關鍵擴充。

### 2.1 `src/train_classifier.py` 核心修改對照

#### 修改重點：
1. **全域隨機種子深度固化**：新增 `set_seed()` 函式，同時鎖定 Python、NumPy、PyTorch CPU/CUDA 以及 cuDNN 行為，並使用專屬 `torch.Generator` 控制 DataLoader 洗牌。
2. **多輪數實驗與權重隔離介面**：擴充 `train()` 函式支援指定輪數、自訂日誌與權重輸出路徑，並在訓練結束時自動儲存最後一輪權重（`_last.pt`）與最佳驗證損失權重（`_best.pt`）。
3. **CLI 指令列參數擴充**：支援 `--epochs`, `--weights`, `--weights-last`, `--log`, `--no-early-stop` 等參數，兼顧腳本調度與預設行為的完全向後相容。

#### 程式碼片段精確比對（附確切行號）：

```python
# ===========================================================================
# 檔案：src/train_classifier.py
# ===========================================================================

# ---------------------------------------------------------------------------
# 修改點 1：全域隨機種子深層控制函式 (新增於 L40-L47)
# ---------------------------------------------------------------------------
# [修改前]：
# 僅在 train() 中簡單呼叫 torch.manual_seed(cfg["seed"])，未固定 numpy 與 cudnn 行為。

# [修改後 (L40-L47)]：
def set_seed(seed: int = 42):
    """鎖定隨機種子以確保實驗完全可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 修改點 2：train() 簽名與 DataLoader 確定性洗牌 (L71-L107)
# ---------------------------------------------------------------------------
# [修改前 (L58-L68)]：
def train(config: dict = None, device: str = None) -> dict:
    cfg = {**CONFIG, **(config or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    train_ds = PCBDefectCropDataset("gt", "train", seed=cfg["seed"])
    val_ds = PCBDefectCropDataset("gt", "val", seed=cfg["seed"])
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], collate_fn=collate_crops)
    ...

# [修改後 (L71-L107)]：
def train(
    config: dict = None,
    device: str = None,
    weights_path: str | Path = WEIGHTS,
    weights_last_path: str | Path = None,
    log_path: str | Path = LOG_CSV,
) -> dict:
    cfg = {**CONFIG, **(config or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(weights_path)
    log_path = Path(log_path)
    if weights_last_path is not None:
        weights_last_path = Path(weights_last_path)

    # 鎖定隨機種子 (在建立 Dataset, DataLoader, Model 之前呼叫)
    set_seed(cfg["seed"])

    train_ds = PCBDefectCropDataset("gt", "train", seed=cfg["seed"])
    val_ds = PCBDefectCropDataset("gt", "val", seed=cfg["seed"])

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_ld = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_crops,
        generator=g,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_crops,
    )

# ---------------------------------------------------------------------------
# 修改點 3：雙權重隔離儲存 (Last Epoch vs. Best Val Loss) (L175-L189)
# ---------------------------------------------------------------------------
# [修改前]：僅在 val_loss 創歷史新低時覆蓋 models/defect_classifier.pt，無法評估最後一輪權重。

# [修改後 (L175-L189)]：
    # 若指定最後一輪權重儲存路徑，將當前最新權重存入
    if weights_last_path is not None:
        save_model(
            model,
            weights_last_path,
            extra={
                "epoch": last_epoch,
                "val_loss": last_val_loss,
                "val_acc": last_val_acc,
                "config": cfg,
            },
        )
        print(f"最後一輪權重 (epoch {last_epoch}) -> {weights_last_path}")

# ---------------------------------------------------------------------------
# 修改點 4：CLI 命令列引數解析與早停覆寫 (L212-L236)
# ---------------------------------------------------------------------------
# [修改前]：
if __name__ == "__main__":
    train()

# [修改後 (L212-L236)]：
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Stage 2 缺陷分類器 (ResNet18)")
    parser.add_argument("--epochs", type=int, default=None, help="訓練輪數 (覆寫 CONFIG['max_epochs'])")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS), help="最佳權重輸出路徑")
    parser.add_argument("--weights-last", type=str, default=None, help="最後一輪權重輸出路徑")
    parser.add_argument("--log", type=str, default=str(LOG_CSV), help="訓練日誌 CSV 輸出路徑")
    parser.add_argument("--no-early-stop", action="store_true", help="關閉 early stopping (跑滿指定 epochs)")
    args = parser.parse_args()

    cfg_override = {}
    if args.epochs is not None:
        cfg_override["max_epochs"] = args.epochs
    if args.no_early_stop:
        epochs = args.epochs if args.epochs is not None else CONFIG["max_epochs"]
        cfg_override["patience"] = epochs + 1
    elif args.patience is not None:
        cfg_override["patience"] = args.patience

    train(
        config=cfg_override,
        weights_path=args.weights,
        weights_last_path=args.weights_last,
        log_path=args.log,
    )
```

---

### 2.2 `src/evaluate_classifier.py` 核心修改對照

#### 修改重點：
1. **量化指標全面升級**：在 `evaluate_gt_crops()` 中，新增宏平均召回率（`macro_recall`）、微平均召回率（`micro_recall`）、宏平均精確率（`macro_precision`）、宏平均 F1（`macro_f1`），並在各類別記錄中補充 `tp`, `fp`, `fn`。
2. **端到端評估候選框快取加速**：在 `evaluate_end_to_end()` 中引入 Stage 1 候選框快取機制，大幅減省重複運算，同時保持推論資料完全一致。
3. **支援雙權重同測 CLI 介面**：支援 `--weights`、`--weights-last` 與 `--out` 參數，可一次性自動比對並匯出最後一輪與最佳驗證輪之完整指標 JSON。

#### 程式碼片段精確比對（附確切行號）：

```python
# ===========================================================================
# 檔案：src/evaluate_classifier.py
# ===========================================================================

# ---------------------------------------------------------------------------
# 修改點 1：evaluate_gt_crops() 增加 Macro / Micro Recall 與 TP/FP/FN (L170-L197)
# ---------------------------------------------------------------------------
# [修改前 (L170-L175)]：
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        per_class[c] = {"precision": P, "recall": R, "f1": F1, "support": int(cm[i].sum())}
    return {"accuracy": acc, "per_class": per_class,
            "confusion_matrix": cm.tolist(), "n": len(y_true)}

# [修改後 (L170-L197)]：
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        per_class[c] = {"precision": float(P), "recall": float(R), "f1": float(F1),
                        "support": int(cm[i].sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    recalls = [per_class[c]["recall"] for c in CLASSES]
    precisions = [per_class[c]["precision"] for c in CLASSES]
    f1s = [per_class[c]["f1"] for c in CLASSES]
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    total_tp = sum(cm[i, i] for i in range(n))
    total_fn = sum(cm[i, :].sum() - cm[i, i] for i in range(n))
    denom = total_tp + total_fn
    micro_recall = float(total_tp / denom) if denom > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": len(y_true),
    }

# ---------------------------------------------------------------------------
# 修改點 2：CLI 擴充雙模型自動切換評估 (L302-L355)
# ---------------------------------------------------------------------------
# [修改前]：固定載入 DEFAULT_WEIGHTS 並覆寫 results/classifier_eval_summary.json。

# [修改後 (L302-L355)]：
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 缺陷分類器評估")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="模型權重 (或最佳權重) 路徑")
    parser.add_argument("--weights-last", type=str, default=None, help="最後一輪權重路徑 (可選，同時評估 best 與 last)")
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "classifier_eval_summary.json"), help="輸出 JSON 路徑")
    args = parser.parse_args()

    ...
    if args.weights_last is not None:
        # 自動執行雙權重評估流程，輸出包含 last_model 與 best_model 結構
        ...
```

---

## 三、 B-5 & B-6: 函式位置、起訖行號與資料流對照表

本節對應專案目前的實際程式碼，完整條列各階段的核心類別與函式名稱、檔案路徑、起訖行號、輸入/輸出結構與資料流（Data Flow）。

### 3.1 四大階段函式與模組真實對照表

| 系統階段 | 技術 / 模組名稱 | 類別 / 函式名稱 | 檔案路徑與確切行號 | 輸入資料格式 (Input) | 輸出資料格式 (Output) |
|---|---|---|---|---|---|
| **Input** | VOC 標註解析 | `parse_voc` | [`src/hripcb.py: L80-L104`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/hripcb.py#L80-L104) | XML 檔案路徑 (`str/Path`) | `VOCRecord` 物件 (含 `objects` 列表、寬高) |
| **Input** | 缺陷標註結構 | `class BBox` / `class Annotation` | [`src/hripcb.py: L46-L77`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/hripcb.py#L46-L77) | 原始整數座標 $(x_1, y_1, x_2, y_2)$ | 具備面積、寬高比、中心座標之資料類別 |
| **Input** | 測試金牌模板對應 | `get_template_for_test` | [`src/hripcb.py: L129-L144`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/hripcb.py#L129-L144) | 測試圖檔名 (`01_missing_hole_01.jpg`) | 對應之無缺陷模板路徑 (`01.jpg`) |
| **Preprocessing** | 影像級分層切分 | `build_splits` | [`src/dataset.py: L47-L73`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L47-L73) | 資料根目錄、隨機種子 (`seed=42`)、比例 | `dict`: `{"train": [...], "val": [...], "test": [...]}` |
| **Preprocessing** | 資料增強與標準化 | `get_transforms` | [`src/dataset.py: L79-L94`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L79-L94) | 集合名稱 (`split="train"/"val"/"test"`) | `torchvision.transforms.Compose` 物件 |
| **Preprocessing** | 30% 上下文外擴裁切 | `crop_with_padding` | [`src/dataset.py: L100-L111`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L100-L111) | 全圖陣列 `[H, W, 3]`、`bbox`、`pad_ratio=0.3` | 外擴後之局部 BGR 陣列 `[h_pad, w_pad, 3]` |
| **Preprocessing** | 缺陷裁切塊資料集 | `class PCBDefectCropDataset` | [`src/dataset.py: L117-L208`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L117-L208) | `mode="gt"`, `split`, `pad_ratio`, `seed` | PyTorch Dataset (`__getitem__` 產出 Tensor) |
| **Preprocessing** | 批次打包保留 Meta | `collate_crops` | [`src/dataset.py: L211-L216`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L211-L216) | 樣本 List `[(tensor, label, meta), ...]` | `(imgs_tensor, labels_tensor, metas_list)` |
| **Classification** | ResNet18 分類架構 | `class DefectClassifier` | [`src/dl_classifier.py: L15-L36`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dl_classifier.py#L15-L36) | 輸入張量 `[B, 3, 224, 224]` | 6 類未正規化 Logits `[B, 6]` |
| **Classification** | 模型建構與裝置分配 | `build_model` | [`src/dl_classifier.py: L39-L44`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dl_classifier.py#L39-L44) | 類別數 (6)、`freeze_backbone`、`device` | 載入 ImageNet 權重並移至 GPU 之模型 |
| **Classification** | 權重儲存與載入 | `save_model` / `load_model` | [`src/dl_classifier.py: L47-L63`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dl_classifier.py#L47-L63) | 模型權重檔案路徑 (`.pt`) | 字典檔 (`state_dict`, `classes`, metadata) |
| **Classification** | 全域種子固化 | `set_seed` | [`src/train_classifier.py: L40-L47`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L40-L47) | 整數隨機種子 (`seed=42`) | 固定亂數環境狀態 |
| **Classification** | 單輪前向與反向傳播 | `_run_epoch` | [`src/train_classifier.py: L50-L68`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L50-L68) | 模型、DataLoader、Loss 函式、優化器 | 平均損失 (`float`)、分類準確率 (`float`) |
| **Classification** | 主訓練迴圈與排程 | `train` | [`src/train_classifier.py: L71-L209`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/train_classifier.py#L71-L209) | 超參數配置、權重路徑、日誌路徑 | 訓練紀錄列表、最佳驗證輪、權重儲存 |
| **Output** | GT 裁切塊上限評估 | `evaluate_gt_crops` | [`src/evaluate_classifier.py: L152-L197`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py#L152-L197) | 訓練好之模型、運算裝置、`split="test"` | 測試集 Acc、宏/微平均 Recall、各類指標、CM |
| **Output** | 端到端系統評估 | `evaluate_end_to_end` | [`src/evaluate_classifier.py: L203-L277`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/evaluate_classifier.py#L203-L277) | 模型、裝置、Stage 1 配置 | Stage 1 vs. Stage 2 CNN 之 mAP@0.5 與 AP |
| **Output** | 訓練曲線繪製 | `training_curves` | [`src/viz_classifier.py: L27-L42`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_classifier.py#L27-L42) | `training_log.csv` 路徑、輸出目錄 | 包含 Loss 與 Acc 之雙子圖 PNG |
| **Output** | 混淆矩陣視覺化 | `confusion_fig` | [`src/viz_classifier.py: L45-L61`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/viz_classifier.py#L45-L61) | 混淆矩陣陣列、類別標籤、標題 | 熱區圖與數值標註之 PNG |

---

### 3.2 三大核心技術與張量資料流（Data Flow Analysis）

```
[原始 PCB 影像: 3000x3000x3 BGR] 
             │
             ├── (1) 幾何邊界外擴：crop_with_padding(pad_ratio=0.3)
             ▼
[外擴局部裁切塊: ~120x120x3 BGR]
             │
             ├── (2) 色彩轉換與增強：cvtColor(BGR2RGB) -> PIL Image -> transforms
             ▼
[標準化輸入張量: [B, 3, 224, 224] Float32, Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225]]
             │
             ├── (3) 卷積特徵抽取：ResNet18 Backbone (conv1 -> layer1-4 -> avgpool)
             ▼
[全局特徵向量: [B, 512, 1, 1] -> 展平為 [B, 512]]
             │
             ├── (4) 類別全連接預測：Linear(512, 6)
             ▼
[未正規化 Logits: [B, 6]]
             │
             ├── (5) 機率分佈歸一化：torch.softmax(dim=1)
             ▼
[缺陷預測機率與類別: Probabilities [B, 6], Predicted_Class = argmax]
```

#### 技術一：外擴 30% 上下文裁切機制（Context Padding Algorithm）
- **工程動機**：PCB 缺陷（例如微小短路 `short`、導線微突刺 `spur` 或邊緣囓齒 `mouse_bite`）若僅貼齊標註邊界緊密裁切，CNN 將完全喪失周圍綠漆基板與鄰近銅箔導線的空間拓撲關係，極易將導線正常邊緣誤判為缺陷。
- **演算法公式**：
  $$\text{pad}_w = \lfloor (x_2 - x_1) \times 0.3 \rfloor, \quad \text{pad}_h = \lfloor (y_2 - y_1) \times 0.3 \rfloor$$
  $$x_1' = \max(0, x_1 - \text{pad}_w), \quad y_1' = \max(0, y_1 - \text{pad}_h)$$
  $$x_2' = \min(W, x_2 + \text{pad}_w), \quad y_2' = \min(H, y_2 + \text{pad}_h)$$
- **保護機制**：在 [`src/dataset.py: L109-L111`](file:///c:/Users/user/Documents/Projects/pcb-aoi/src/dataset.py#L109-L111) 設計退化保護，若遇極端裁切邊界為 0，自動回傳 $1\times 1$ 安全像素陣列，防止張量維度潰縮。

#### 技術二：ResNet18 遷移學習與動態排程機制（Transfer Learning & Scheduler）
- **骨幹初始化**：載入 ImageNet-1K V1 預訓練權重，保留卷積層對邊緣、紋理與角點之泛化抽取能力。
- **分類頭替換**：移除 ImageNet 原始 1,000 類全連接層，將最後一層 `model.backbone.fc` 替換為 `nn.Linear(512, 6)`，並在固定種子下重新初始化。
- **動態學習率排程**：採用 `ReduceLROnPlateau(factor=0.5, patience=4)` 監控驗證集損失。當損失連續 4 輪未下降時，學習率自動衰減 50%（如 Epoch 15 實驗中在第 14 輪由 $1\times 10^{-4}$ 衰減至 $5\times 10^{-5}$），確保模型在平坦鞍點處進行精細權重收斂。

#### 技術三：雙軌驗證體系（Dual-Track Evaluation Benchmark）
- **軌道一（GT Crops 上限評估）**：輸入 460 個測試集真實裁切塊，排除前端檢測器的定位干擾，專注於量化 ResNet18 分類器本身的判別能力上限（本實驗核心：各類 Recall 突破 97.4% ~ 100%）。
- **軌道二（End-to-End 系統評估）**：使用 Stage 1 特徵差影法提取真實候選框（Region Proposals），再交由 CNN 重新分類。該指標真實反映整合系統於產線運作時之綜合 mAP@0.5（約 84.5% ~ 85.8%），並直接揭露 Stage 1 對 `short` 缺陷之召回率瓶頸。

