# 🌿 LeafLens — Multi-Crop Leaf Disease Detection

> **AI Blend for Engineers (24ESQ2119-B) · AI-Powered Robotic System for Early Plant Disease Detection in Greenhouses**
> Saintgits College of Engineering (Autonomous) · Project ID: 24AB046

---

## 👥 Team

| Name | Branch |
|------|--------|
| Amy B Chiramel | EEE |
| Hannah Maria Jacob | CSB |
| Aiben Koshy Cherian | RBA |
| Boby Joseph | EEE |
| Christy Roy | RBA |

**Supervisor:** Dr. Preena Prasad

---

## 🔍 What is LeafLens?

**LeafLens** is a cross-platform desktop GUI application that detects diseases in leaf images of five crops — **potato, tomato, strawberry, apple, and corn** — using deep-learning models exported to ONNX format.

Upload one or more photos of a leaf (or grab frames from a webcam / ESP32-CAM), choose the crop, click **Analyze**, and LeafLens returns:
- The predicted disease class with a confidence score
- A colour-coded badge: 🟢 Confident / 🟡 Uncertain / 🔴 Unknown
- Symptom description and suggested action
- A probability bar chart for all classes
- History log with thumbnail images stored in a local SQLite database

---

## 🌱 Supported Crops & Models

| Crop | Architecture | Classes | Test Accuracy | Test Macro-F1 |
|------|-------------|---------|--------------|---------------|
| 🍎 Apple | EfficientNet-B0 | 4 | **100.0%** | 1.000 |
| 🌽 Corn | EfficientNet-B0 | 4 | **98.97%** | 0.985 |
| 🥔 Potato | MobileNet-V2 | 3 | — (re-run pending) | — |
| 🍓 Strawberry | EfficientNet-B0 | 2 | **100.0%** | 1.000 |
| 🍅 Tomato | EfficientNet-B0 | 10 | **99.85%** | 0.998 |

**Overall: 2,445 / 2,451 test images correct (99.76%)**

### Disease Classes Detected

| Crop | Classes |
|------|---------|
| Apple | Apple Scab · Black Rot · Cedar Apple Rust · Healthy |
| Corn | Cercospora Leaf Spot (Gray Leaf Spot) · Common Rust · Northern Leaf Blight · Healthy |
| Potato | Early Blight · Late Blight · Healthy |
| Strawberry | Leaf Scorch · Healthy |
| Tomato | Bacterial Spot · Early Blight · Late Blight · Leaf Mold · Septoria Leaf Spot · Spider Mites · Target Spot · Yellow Leaf Curl Virus · Mosaic Virus · Healthy |

---

## 🏗️ Project Architecture

```
Greenhouse camera (ESP32-CAM / webcam / uploaded image)
        |
        v
   LeafLens GUI (leaflens.py · PySide6)
        |
        +-- Image Quality Check (sharpness · leaf area %)
        +-- Optional Background Removal (HSV mask)
        +-- Preprocessing (RGB -> resize 224x224 -> normalize -> NCHW float32)
        |
        v
   ONNX Runtime Inference (CPU or CUDA GPU)
        |
        +-- Softmax with Temperature Scaling (calibrated confidence)
        +-- Multi-frame Averaging (up to 5 frames + mirror flip)
        |
        v
   Result Panel (label · confidence · probability bars · disease guide)
        |
        +-- SQLite History DB + Feedback folder for dataset collection
```

---

## 📦 Installation

### Prerequisites
- Python 3.9 or later
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Install dependencies

```bash
pip install -r "requirements (2).txt"
# For GPU inference, replace onnxruntime with:
pip install onnxruntime-gpu
```

### Requirements

```
PySide6>=6.5
onnxruntime>=1.16        # use onnxruntime-gpu if you have NVIDIA GPU + CUDA
opencv-python-headless>=4.8
numpy
Pillow
```

---

## 🚀 Running LeafLens

```bash
# Default: looks for models in ./models/
python leaflens.py

# Custom models folder
python leaflens.py --models "D:/my_models"
```

### Models folder layout

LeafLens expects the trained ONNX models and a `deploy_config.json` in the models folder:

```
models/
├── apple_disease_model.onnx
├── corn_disease_model.onnx
├── potato_disease_model.onnx
├── strawberry_disease_model.onnx
├── tomato_disease_model.onnx
└── deploy_config.json
```

Each JSON object in `deploy_config.json` contains crop name, class names, image size, normalization parameters, calibration temperature, and confidence thresholds.

---

## 🖥️ Application Features

| Feature | Details |
|---------|---------|
| **Multi-frame averaging** | Up to 5 frames + horizontal mirror -> averaged softmax |
| **Image quality check** | Laplacian blur score + leaf-area % (HSV-based mask) |
| **Confidence thresholds** | Configurable: Confident >= 85% · Uncertain 50-84% · Unknown < 50% |
| **Input sources** | File upload · drag & drop · Webcam · ESP32-CAM (HTTP capture) |
| **Background removal** | Experimental HSV-based leaf mask (off by default) |
| **GPU inference** | Automatic if CUDA is available via ONNX Runtime |
| **History** | SQLite database with thumbnail, crop, label, confidence, status |
| **Feedback collection** | Mark predictions correct/wrong -> saves images to labelled folders |
| **Export** | History exportable to CSV |
| **Disease guide** | Built-in symptom description and suggested action per disease |
| **Settings** | Fully configurable: model folder, thresholds, camera, GPU, feedback dir |

---

## 🧠 Model Training

Five separate Jupyter notebooks train one model per crop. Each notebook follows this pipeline:

1. **Dataset**: PlantVillage (Hughes & Salathe, 2016) — 54,304 total leaf images across 38 classes; 23 classes used across 5 crops
2. **Split**: Train / Validation / Test (no leakage between sets)
3. **Backbone**: EfficientNet-B0 (Apple, Corn, Strawberry, Tomato) · MobileNet-V2 (Potato) — pretrained on ImageNet via PyTorch/torchvision
4. **Training**: Transfer learning; class-weighted cross-entropy; data augmentation (random flip, rotate, colour jitter)
5. **Evaluation**: Accuracy, macro-F1, confusion matrix on held-out test set
6. **Calibration**: Temperature scaling fitted on validation logits
7. **Export**: PyTorch checkpoint (.pt) -> ONNX (.onnx) for deployment

### Training notebooks

| Notebook | Crop | Epochs | Best val acc |
|----------|------|--------|-------------|
| apple_disease_training.ipynb | Apple | 19 | 100.0% |
| corn_disease_training.ipynb | Corn | 34 | 98.96% |
| potato_disease_training.ipynb | Potato | — | — |
| strawberry_disease_training.ipynb | Strawberry | 15 | 100.0% |
| tomato_disease_training.ipynb | Tomato | 36 | 100.0% |

---

## 📂 Repository Structure

```
Plant_Disease_detection_system/
|
+-- leaflens.py                      # Main desktop application (PySide6 + ONNX Runtime, 1716 lines)
+-- requirements (2).txt             # Python dependencies
+-- README.md                        # This file
|
+-- models/                          # Exported ONNX models + deploy_config.json
|   +-- apple_disease_model.onnx     (~15.3 MB)
|   +-- corn_disease_model.onnx      (~15.3 MB)
|   +-- potato_disease_model.onnx    (~8.9 MB, MobileNetV2)
|   +-- strawberry_disease_model.onnx (~15.3 MB)
|   +-- tomato_disease_model.onnx    (~15.3 MB)
|   +-- deploy_config.json
|
+-- training notebooks/              # Jupyter notebooks — one per crop
|   +-- apple_disease_training.ipynb
|   +-- corn_disease_training.ipynb
|   +-- potato_disease_training.ipynb
|   +-- strawberry_disease_training.ipynb
|   +-- tomato_disease_training.ipynb
|
+-- Training_outcomes/               # Saved checkpoints, logs, and plots per crop
|   +-- apple_model/    (best_model.pt, checkpoint.pt, train_log.csv, confusion_matrix.png, training_curves.png)
|   +-- corn_model/
|   +-- potato_model/
|   +-- strawberry_model/
|   +-- tomato_model/
|
+-- literature review/
+-- AI_Blend_Project_Report.docx
+-- Interim_Evaluation_Report (2).docx
```

---

## 📊 Results

| Crop | Classes | Test Images | Correct | Errors | Accuracy | Macro-F1 |
|------|---------|------------|---------|--------|----------|----------|
| Potato | 3 | 216 | 216 | 0 | 100.00% | — |
| Apple | 4 | 318 | 318 | 0 | 100.00% | 1.000 |
| Corn | 4 | 387 | 383 | 4 | 98.97% | 0.985 |
| Strawberry | 2 | 157 | 157 | 0 | 100.00% | 1.000 |
| Tomato | 10 | 1,373 | 1,371 | 2 | 99.85% | 0.998 |
| **All crops** | **23** | **2,451** | **2,445** | **6** | **99.76%** | — |

> **Note:** Results are on PlantVillage test images (controlled background). Tomato test covers 7 of 10 classes. Potato model is being re-run with the updated notebook template.

---

## ⚠️ Known Limitations

1. **PlantVillage background bias**: Controlled-background training images may not generalise to real field conditions
2. **Tomato split**: Three classes absent from test set (Bacterial Spot, Leaf Mold, Target Spot) — re-split pending
3. **Potato model**: Older notebook version; being re-run with updated template and calibration
4. **Over-sharp confidence**: Fitted temperatures < 1 sharpen outputs excessively — set to 1.0 until proper refit on real-camera data
5. **No crop-identification gate**: Crop must be selected manually
6. **Square resize without cropping**: No leaf-crop step before resize

---

## 🔮 Future Scope

- Real-camera evaluation dataset (30-50 photos per class per crop)
- Automatic crop identification and non-leaf image rejection
- Leaf bounding-box crop before resize
- Grad-CAM visualisations to verify model attention on lesions
- ESP32-CAM + robotic arm integration (PID joint control)
- ROS2/Gazebo simulation of the robotic platform
- Field evaluation on PlantDoc dataset
- Fine-tuning with user-confirmed feedback images

---

## 🤖 Robotic Platform Status

A five-axis 3D-printed robotic arm is designed to carry the ESP32-CAM for automated leaf imaging:

- ✅ SolidWorks design (base, shoulder, elbow, pitch, roll)
- ✅ 3D printed
- ⏳ ESP32-CAM integration
- ⏳ Potentiometer feedback + PID control
- ⏳ ROS2/Gazebo simulation

---

## 🌍 SDG Alignment

| SDG | Relevance |
|-----|-----------|
| SDG 2 – Zero Hunger | Early low-cost disease screening protects crop yield and food supply |
| SDG 9 – Industry, Innovation and Infrastructure | Combines deep learning, desktop app, and 3D-printed robotic arm into a low-cost agri-tech prototype |
| SDG 12 – Responsible Consumption and Production | Targeted intervention reduces crop loss and unnecessary pesticide use |

---

## 🔗 Links

- **GitHub**: https://github.com/Christy-Roy-tech/-Al-Powered-Robotic-System-for-Early-Plant-Disease-Detection-in-Greenhouses
- **Dataset**: PlantVillage — D.P. Hughes & M. Salathe, 2016 (available on Kaggle)

---

*Saintgits College of Engineering (Autonomous), Kottukulam Hills, Pathamuttom P.O., Kottayam - 686 532, Kerala*
*October 2025*
