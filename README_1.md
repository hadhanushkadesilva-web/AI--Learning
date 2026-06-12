# AI Learning Projects

My self-study portfolio while working through a 9-section AI/ML roadmap
(MSc Mathematical Statistics, University of Stockholm).

## 🎯 YOLOv8 Object Detection
Real-time object detection demo using pretrained YOLOv8.

📖 **View notebook:**
- [Open in Colab](https://colab.research.google.com/github/hadhanushkadesilva-web/AI--Learning/blob/main/yolov8_coco128.ipynb) (recommended — interactive)
- [View on nbviewer](https://nbviewer.org/github/hadhanushkadesilva-web/AI--Learning/blob/main/yolov8_coco128.ipynb) (static rendered view)
- [Raw GitHub link](https://github.com/hadhanushkadesilva-web/AI--Learning/blob/main/yolov8_coco128.ipynb)

### What it does
Loads YOLOv8-nano (pretrained on COCO, 80 classes) and runs inference on a sample
Madrid street scene. Successfully detects:
- 🚌 Bus (confidence: 0.87)
- 🚶 Person × 4 (confidence: 0.35 – 0.86)
- 🛑 Stop sign (partial, confidence: 0.26)

### Stack
Python • PyTorch • Ultralytics YOLOv8 • Google Colab (T4 GPU)

### What I learned
Pretrained models + transfer learning are how most production CV works.
Building an inference pipeline taught me how YOLO's architecture
(backbone → neck → head + Non-Max Suppression) flows from input to detections.

---

🤖 Built while learning object detection theory (IoU, NMS, mAP, YOLO).
