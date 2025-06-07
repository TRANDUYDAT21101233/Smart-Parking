# Parking Management using YOLO

## 📌 Overview

This project aims to build an intelligent **parking management system** using deep learning for vehicle detection. The core objective is to **detect and monitor parked vehicles** in real-time video streams or surveillance footage.

We initially adopted **YOLO** (You Only Look Once) for object detection due to its speed and efficiency. However, adapting YOLO to our **custom dataset** required additional steps, especially due to inaccuracies in default model predictions.

## 🔧 Technologies & Libraries

- Python 3.12
- [YOLOv11](https://github.com/ultralytics/ultralytics)
- OpenCV
- PyTorch
- SAHI (Slicing Aided Hyper Inference)
- Supervision (for post-processing & visualization)

## ⚠️ Problem Faced

When testing with the pre-trained `yolo11.pt` model on our parking lot video data, we found that **YOLO misclassified or failed to detect** vehicles correctly. Issues included:
- False positives (detecting non-vehicles as cars)
- Missed detections in low-light or occluded areas

Alternative attempts using **SAHI** and **Supervision** post-processing were also unsatisfactory, with no significant improvement.

## ✅ Our Solution: Transfer Learning

To address this, we applied **Transfer Learning** using YOLO:
- We collected and annotated a **custom dataset** of vehicles in the actual parking lot scenario.
- We fine-tuned the YOLO model (YOLOv11) using this dataset to better align with the domain-specific characteristics of our environment (e.g., camera angle, vehicle types).
- After training, the model showed **significant improvements** in detection accuracy.

We then **integrated** the fine-tuned model back into YOLO's detection pipeline, maintaining YOLO's real-time performance while achieving **higher precision**.

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/ultralytics/ultralytics.git
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run detection:

```bash
python main.py
```

## 📊 Results

Pre-trained YOLO model (original): ~5% mAP on our dataset

Fine-tuned YOLO (transfer learning): ~85% mAP

Real-time FPS maintained: ~20–25 FPS (depending on hardware)

## 📷 Demo

Detection of parked cars with custom-trained YOLO model

## 📚 References

YOLOv11: https://github.com/ultralytics/ultralytics

SAHI: https://github.com/obss/sahi

Supervision: https://github.com/roboflow/supervision
