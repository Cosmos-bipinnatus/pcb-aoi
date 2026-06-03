| method | mAP@0.5 | mAP@0.5:0.95 | precision | recall | inference_ms | model_size_mb | notes |
|---|---|---|---|---|---|---|---|
| Stage 1:傳統 CV | 0.293 | 0.141 | 0.2883 | 0.3908 | 641.3 | N/A | 差影+形態學;hand-crafted feature 分類撞牆 |
| Stage 2:R-CNN 風格(proposals+CNN) | 0.8475 | 0.3899 | 0.8475 | 0.8834 | 13.6 | 44.8 | ResNet18 forward(不含 Stage1 proposal 生成);受 proposal recall 綁住 |
| Stage 3:YOLOv8n 端到端 | 0.9178 | 0.4672 | 0.9351 | 0.8616 | 5.4 | 6.3 | 端到端,無 proposal 依賴;突破 short recall 瓶頸 |
