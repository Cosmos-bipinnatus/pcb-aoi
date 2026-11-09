# 作業二主題 4：CNN 訓練輪數（Training Epochs）對分類性能與召回率影響量化總表

> **表格數值呈現規則**：以該組最後一輪（`Last Epoch`）的實測值為主數據；若該組內 `Best Val Epoch` 數值不同，則於括號內附註 `(Best@EpX: 數值)`。

| 實驗組別設定 (`Training Epochs`) | Train Loss | Val Loss | Val Accuracy (%) | Macro Recall (%) | Micro Recall / Test Acc (%) | missing_hole Recall (%) | mouse_bite Recall (%) | open_circuit Recall (%) | short Recall (%) | spur Recall (%) | spurious_copper Recall (%) | End-to-End mAP@0.5 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 實驗組 A (`Epochs = 5`) | 0.0211 (Best@Ep4: 0.0216) | 0.0403 (Best@Ep4: 0.0346) | 99.32% (Best@Ep4: 99.77%) | 99.14% (Best@Ep4: 99.78%) | 99.13% (Best@Ep4: 99.78%) | 98.65% (Best@Ep4: 100.00%) | 100.00% | 100.00% | 100.00% | 97.40% (Best@Ep4: 98.70%) | 98.78% (Best@Ep4: 100.00%) | 84.93% (Best@Ep4: 85.12%) |
| 實驗組 B (`Epochs = 10`) | 0.0330 (Best@Ep8: 0.0068) | 0.0330 (Best@Ep8: 0.0279) | 98.87% (Best@Ep8: 99.55%) | 99.37% (Best@Ep8: 99.57%) | 99.35% (Best@Ep8: 99.57%) | 98.65% | 100.00% | 100.00% | 100.00% | 100.00% | 97.56% (Best@Ep8: 98.78%) | 85.40% (Best@Ep8: 85.79%) |
| 實驗組 C (`Epochs = 15`) | 0.0077 (Best@Ep11: 0.0096) | 0.0326 (Best@Ep11: 0.0203) | 99.32% (Best@Ep11: 99.55%) | 99.33% (Best@Ep11: 99.56%) | 99.35% (Best@Ep11: 99.57%) | 98.65% | 100.00% (Best@Ep11: 98.70%) | 98.61% (Best@Ep11: 100.00%) | 100.00% | 98.70% (Best@Ep11: 100.00%) | 100.00% | 84.47% (Best@Ep11: 85.50%) |
| 基準對照組 (`Baseline`, max=50, 實際 31 輪) | 0.0052 (Best@Ep21: 0.0047) | 0.0169 (Best@Ep21: 0.0111) | 99.77% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 84.75% |

---

## 各組混淆矩陣誤判案例深入分析（Misclassification Case Breakdown）

在測試集（Test Split）460 個真實缺陷塊（GT Crops）下，各設定之混淆矩陣分類誤判案例如下：

### 1. 實驗組 A (`Epochs = 5`)
- **最後一輪（Last Epoch 5）**：共 4 筆誤判（測試集準確率 **99.13%**, 456/460）：
  1. `missing_hole`（缺孔）被誤判為 `short`（短路）：**1 筆**
  2. `spur`（微短路突刺）被誤判為 `mouse_bite`（邊緣凹折囓齒）：**2 筆**
  3. `spurious_copper`（假銅斑）被誤判為 `short`（短路）：**1 筆**
- **最佳驗證輪（Best@Ep4）**：共 1 筆誤判（測試集準確率 **99.78%**, 459/460）：
  1. `spur` 被誤判為 `mouse_bite`：**1 筆**

### 2. 實驗組 B (`Epochs = 10`)
- **最後一輪（Last Epoch 10）**：共 3 筆誤判（測試集準確率 **99.35%**, 457/460）：
  1. `missing_hole` 被誤判為 `short`：**1 筆**
  2. `spurious_copper` 被誤判為 `short`：**2 筆**
- **最佳驗證輪（Best@Ep8）**：共 2 筆誤判（測試集準確率 **99.57%**, 458/460）：
  1. `missing_hole` 被誤判為 `short`：**1 筆**
  2. `spurious_copper` 被誤判為 `short`：**1 筆**

### 3. 實驗組 C (`Epochs = 15`)
- **最後一輪（Last Epoch 15）**：共 3 筆誤判（測試集準確率 **99.35%**, 457/460）：
  1. `missing_hole` 被誤判為 `short`：**1 筆**
  2. `open_circuit`（開路）被誤判為 `mouse_bite`：**1 筆**
  3. `spur` 被誤判為 `mouse_bite`：**1 筆**
- **最佳驗證輪（Best@Ep11）**：共 2 筆誤判（測試集準確率 **99.57%**, 458/460）：
  1. `missing_hole` 被誤判為 `short`：**1 筆**
  2. `mouse_bite` 被誤判為 `spur`：**1 筆**

### 4. 基準對照組 (`Baseline`, max=50, Early Stopping @ 31 輪，最佳權重 @ Ep21)
- **最佳驗證權重（Best@Ep21）**：**0 筆誤判**（測試集準確率 **100.00%**, 460/460，全部正分類）。

---

## 核心工程洞察與結論（Key Engineering Insights）

1. **極速收斂性（Rapid Convergence via Transfer Learning）**：
   - 受惠於 ResNet18 在 ImageNet 上的預訓練特徵表示，模型在 **Epoch 4~5** 時驗證集準確率即已躍升至 **99.32% ~ 99.77%**，測試集 Macro Recall 超過 **99.14%**。
   - 不良品召回率在短短 5 輪內便全面突破 97.4%，展現微調（Fine-tuning）在工業外觀瑕疵分類上的極高收斂效率。

2. **形態相近類別的判別邊界（Morphological Feature Ambiguity）**：
   - 誤判高度集中於「邊緣形態相反」的缺陷對：`spur`（突起）與 `mouse_bite`（咬痕）。兩者皆位於銅箔導線邊緣且面積微小，早期輪數較易因空間細節關注不足產生偶發混淆。
   - 另一類為 `spurious_copper` 與 `short`，兩者皆為額外的異常銅斑，當假銅斑恰好貼近相鄰兩條導線時，在特徵抽象層次極易與短路橋接重疊。

3. **候選框召回率瓶頸與系統邊界效應（Stage 1 Proposal Bottleneck）**：
   - 三組設定在 End-to-End 系統評估中的 mAP@0.5 分別為 **84.94% (Ep5)**、**85.39% (Ep10)**、**84.47% (Ep15)**，與訓練 31 輪 Baseline 的 **84.75%** 完全處於同一水準（差異僅在 ±0.6% 範圍內）。
   - 充分驗證了：**一旦 CNN 分類器在第 5 輪達到 ~99% 準確率後，進一步增加訓練輪數對整體 AOI 檢測系統的端到端指標並無顯著增益；系統總體表現被嚴格箝制於 Stage 1 傳統演算法對 `short` 缺陷候選框的低召回率（僅 ~43% AP），這正是促使專案邁向 Stage 3 端到端 YOLO 的關鍵工程動機！**