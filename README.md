# pcb-aoi
PCB Automated Optical Inspection — traditional CV baseline,
deep-learning classifier, and YOLO detector on the HRIPCB dataset.

(Work in progress)

## Streamlit 互動 Demo

在瀏覽器上比較三種 PCB 缺陷檢測方法,並模擬產線 AOI 的 Recipe 切換流程。

**啟動方式**:
- Windows:雙擊 `run_aoi_app.bat`
- Linux / macOS:`bash run_aoi_app.sh`
- 或直接:`streamlit run app.py`

啟動後瀏覽器開啟 `http://localhost:8501`,左側選方法、選範例或上傳 PCB 圖,按「執行檢測」。

**三方法的 demo 行為差異**:
- **Stage 1 傳統 CV**:需先選 **Recipe**(缺陷類別 → 對應 golden template),再對測試圖差影 + 形態學。
- **Stage 2 R-CNN 風格**:直接吃單張圖。因 demo 無 template,改用 **MSER class-agnostic proposal** + ResNet18 分類取代差影 proposal —— 精度與召回低於有 template 的完整版本(此差異於 app 與本說明中誠實標註)。
- **Stage 3 YOLOv8n**:直接吃單張圖,端到端偵測,免 template。

**業界對應**:Stage 1 的「選 template 再檢測」對應真實產線 AOI 的 **Recipe Management** —— 工程師會依產品料號切換對應的 golden template 後再檢測,是標準作業流程(standard workflow)。Stage 2/3 的端到端方法則免去此步驟。

> **環境注意**:本專案 torch 為 RTX 5070 Ti(Blackwell)專用的 nightly cu128 build,ultralytics 以 `pip install ultralytics --no-deps` 安裝以免依賴解析覆蓋 torch。請**勿**直接 `pip install -r requirements.txt` 覆蓋既有 torch nightly。
