# 🦺 Hard Hat Safety Detection

A real-time object detection system that identifies whether workers are wearing hard hats, built with **YOLOv11 (Ultralytics)**.

The project includes:
- A live webcam detection system with an audio alarm.
- A Streamlit web application supporting image, video, and webcam input.

This project was built as a hands-on exercise covering the complete object detection pipeline:

- Dataset preparation
- Annotation conversion
- Model training
- Evaluation
- Error analysis
- Deployment

---

# 🎯 What It Does

The system detects two classes in real time:

- **head** → Worker without a hard hat (safety violation)
- **helmet** → Worker wearing a hard hat (safe)

## Features

- Detects hard hat violations using **YOLO11s**.
- Triggers an audio alarm when a worker without a helmet is detected for several consecutive frames.
- Automatically stops the alarm when a helmet is detected again.
- Provides a Streamlit web interface supporting:
  - Image upload
  - Video upload
  - Live webcam detection

---

# 📊 Dataset & Results

The model was trained on the **Hard Hat Detection dataset** from Kaggle containing approximately:

- **5,000 annotated images**
- **Pascal VOC annotation format**

The dataset was converted into YOLO format and split into:

- **70% Training**
- **20% Validation**
- **10% Testing**

---

## Dataset Modification

The original dataset contained three classes:

- person
- head
- helmet

After training analysis, the **person** class was removed because of severe class imbalance:

- Person annotations: ~750
- Helmet annotations: ~19,000

The model failed to learn the person class effectively:

- Precision: 0%
- Recall: 0%

Therefore, the final model was trained only on:

- **head**
- **helmet**

---

# 📈 Final Model Performance

**Model:** YOLO11s  
**Training:** 50 epochs  
**Image Size:** 640×640  


| Class | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| head | 0.907 | 0.895 | 0.943 | 0.626 |
| helmet | 0.940 | 0.917 | 0.957 | 0.628 |
| Overall | 0.923 | 0.906 | 0.950 | 0.627 |

---

The dataset was also supplemented with manually labeled real-world false detection examples (mainly close-range indoor images) to improve robustness beyond the original construction-site imagery.

---

# 🛠️ Pipeline Overview

## Dataset Preparation

### `prepare_dataset.py`

Responsible for:

- Converting Pascal VOC XML annotations into YOLO format.
- Splitting the dataset into:
  - Training
  - Validation
  - Testing

---

## Model Training

Training was performed on:

- Google Colab
- Tesla T4 GPU
- Ultralytics YOLO11s

Training configuration:

- Epochs: 50
- Image size: 640×640

---

## Real-Time Detection

### `head_alarm_detection.py`

Provides:

- Live webcam detection.
- Missing helmet detection.
- Audio alarm during safety violations.

---

## Web Application

### `app.py`

A Streamlit application supporting:

- Image detection
- Video detection
- Webcam detection

---

# 🚀 Running Locally

## 1. Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install ultralytics opencv-python streamlit streamlit-webrtc av
```

---

# 🎥 Real-Time Webcam Detection With Alarm

Run:

```bash
python3 head_alarm_detection.py
```

Press:

```
q
```

to quit.

---

# 🌐 Streamlit Web Application

Run:

```bash
streamlit run app.py
```

The application supports:

- Image upload with detection
- Video upload with downloadable annotated output
- Live browser webcam detection

---

# ⚠️ Known Limitations

The model performs best on images similar to its training distribution:

- Medium-distance views
- Outdoor industrial environments

Performance may decrease on out-of-distribution inputs such as:

- Close-range indoor images
- Selfie-style webcam views

---

The Streamlit webcam tab provides visual detection only:

- Bounding boxes
- Detection labels

The audio alarm is not supported in Streamlit because of threading limitations in `streamlit-webrtc`.

For the complete experience with real-time audio alarms, run:

```bash
python3 head_alarm_detection.py
```

---

# 📁 Project Structure

```text
yolo_hardhat_detection/
│
├── app.py
├── head_alarm_detection.py
├── prepare_dataset.py
├── data.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── best_final_model.pt
│   └── Initial_model.pt
│
└── assets/
    └── alarm.mp3
```

---

# 🙏 Acknowledgments

Dataset:

**Hard Hat Detection by Andrew Mvd (Kaggle)**

Built with:

**Ultralytics YOLO11**