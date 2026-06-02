# Stage 2 — CNN 精細分類器(HRIPCB)

用 ImageNet 預訓練的 **ResNet18** 解決 Stage 1 撞牆的精細分類問題,並沿用
Stage 1 的 region proposals 作為 crop 來源(R-CNN 風格的兩階段合作)。

程式:[`src/dataset.py`](../src/dataset.py)、[`src/dl_classifier.py`](../src/dl_classifier.py)、
[`src/train_classifier.py`](../src/train_classifier.py)、
[`src/evaluate_classifier.py`](../src/evaluate_classifier.py)。
Notebook:[`notebooks/02_dl_classification.ipynb`](../notebooks/02_dl_classification.ipynb)。
數據來源:`results/classifier_eval_summary.json`(重現:先 `python -m src.train_classifier`
再 `python -m src.evaluate_classifier`)。

## 1. 訓練設定

| 項目 | 設定 |
|---|---|
| 架構 | ResNet18(`IMAGENET1K_V1` 預訓練)+ 6 類 fc head,微調整個網路 |
| 輸入 | bbox + 30% 上下文 padding → resize 224×224,ImageNet normalize |
| split | image-level 70/15/15、依類別分層、`seed=42`(train 483 / val 102 / test 108 張影像;GT crops 2051 / 442 / 460) |
| 增強 | random horizontal flip / rotation ±15° / colour jitter(僅訓練) |
| optimizer | Adam,lr=1e-4 |
| loss / scheduler | CrossEntropy / ReduceLROnPlateau(on val loss) |
| early stopping | patience=10(on val loss),batch 64,最多 50 epochs |

**訓練結果:** 收斂健康、無 overfitting —— 最佳 **val_loss 0.0111 @ epoch 21**
(val_acc **0.998**),early stopping @ epoch 31,每 epoch ~9s(RTX 5070 Ti)。
曲線見 `results/figures/02_training_curves.png`。

## 2. 評估結果(兩種,對比是重點)

### 2.1 GT crops(分類能力上限)— test set 460 crops
**Accuracy = 100.0%**,六類 precision / recall / F1 皆 **100%**。

> 註:GT crops 100% 顯示 ResNet18 在「乾淨、已正確框出」的 crop 上分辨六類缺陷
> 幾乎無難度(crop 含 30% 上下文,類別在視覺上區別明確)。這是分類的**上限**;
> 實戰指標應看 end-to-end。test set 完美也代表此測試對 GT crops 偏容易。

### 2.2 End-to-end(實戰)— test set 108 張影像,同一批 Stage 1 proposals

| 分類器(共用 Stage 1 proposals) | mAP@0.5 |
|---|---|
| Stage 1 rule-based | 29.3% |
| **Stage 2 CNN** | **84.8%**(Δ **+55.4**) |

(Stage 1 全資料集 baseline 為 27.4%;在此 test split 上為 29.3%,一致。)

**per-class AP@0.5 對比:**

| class | 中文 | Stage 1 AP | CNN AP | Δ |
|---|---|---|---|---|
| missing_hole | 缺孔 | 86.1 | 93.9 | +7.8 |
| mouse_bite | 鼠咬 | 2.6 | **94.8** | **+92.2** |
| open_circuit | 斷路 | 5.9 | **94.1** | **+88.2** |
| short | 短路 | 19.7 | 36.2 | +16.5 |
| spur | 毛刺 | 0.4 | **91.2** | **+90.8** |
| spurious_copper | 多餘銅 | 61.0 | 98.3 | +37.3 |

## 3. 分析:哪裡好、哪裡仍受限

**CNN 徹底解決了 Stage 1 的分類撞牆。** Stage 1 AP 僅 2–6% 的三個小缺陷
(`mouse_bite`/`open_circuit`/`spur`)——形狀特徵幾乎無法區分——被 CNN 全部拉到
**90%+**。這證明問題確實出在「hand-crafted feature 無法分辨」,而非偵測或定位。

**唯一仍偏低的 `short`(36.2)不是分類問題,而是 proposal 瓶頸。** Stage 1 在
proposal 階段就漏抓約一半的 short(其 colour diff 最弱),CNN 再強也無法分類
「不存在的 proposal」。GT crops 上 short 仍是 100% —— 證明 CNN 會分,只是沒拿到
candidate。這正是兩階段架構的本質限制:**end-to-end 上限被 Stage 1 的 recall 綁住**。

**GT crops 100% vs end-to-end 84.8% 的差距** 即 proposal 品質的代價:漏偵測
(尤其 short)、定位 IoU 不足、以及多餘 proposals 被分到某類造成的 FP。

## 4. 與 Stage 1 對比小結

| 指標 | Stage 1(傳統 CV) | Stage 2(CNN end-to-end) |
|---|---|---|
| mAP@0.5 | 27.4% | **84.8%** |
| 小缺陷 AP | 2–6% | 91–95% |
| 推論速度 | ~0.8 s/影像 | Stage 1 proposals + CNN 分類(CNN <5 ms/crop on GPU) |

## 5. 限制與 Stage 3(YOLO)該補強

1. **proposal recall 是天花板(最關鍵)。** end-to-end 被 Stage 1 的偵測 recall
   綁死,`short` 是最明顯受害者。Stage 3 的 **YOLO 端到端偵測**直接學習定位 +
   分類,不依賴 Stage 1 proposals,應可解決 short 的漏抓並進一步推高 mAP。
2. **FP 控制。** Stage 1 仍會丟出雜訊/seam proposals,被 CNN 強制分到六類之一造成
   FP。YOLO 的 objectness 分支或加入「background」類可抑制。
3. **GT crops 評估偏容易。** 100% 顯示需要更難的測試(如跨 template 泛化、加噪);
   但 end-to-end 84.8% 已是可信的實戰指標。
4. **保留兩階段價值。** 「Stage 1 proposals + CNN」可解釋性高、訓練資料需求低,
   可作為 YOLO 的對照組與 ablation。
