Readme · MD
🦺 Hard Hat Safety Detection

A real-time object detection system that identifies whether workers are wearing hard hats, built with YOLOv11 (Ultralytics). Includes a live webcam alarm system and a Streamlit web app supporting image, video, and webcam input.

This project was built as a hands-on exercise in the full object detection pipeline: dataset preparation, format conversion, training, evaluation, error analysis, and deployment.

🎯 What it does
Detects two classes in real time: head (no hard hat — safety violation) and helmet (hard hat worn — safe)
Triggers an audio alarm when a worker without a hard hat is detected for several consecutive frames, and automatically stops the moment a hard hat is detected again
Provides a Streamlit web interface supporting image upload, video upload, and live webcam detection
📊 Dataset & Results

Trained on the Hard Hat Detection dataset (Kaggle, 5,000 annotated images, Pascal VOC format), converted to YOLO format and split 70/20/10 into train/val/test.

The original dataset includes a person class, which was dropped after training revealed severe class imbalance (only ~750 annotations vs. ~19,000 for helmet) caused the model to fail to learn it entirely (0% precision/recall). The final model trains on head and helmet only.

Final model performance (YOLO11s, 50 epochs):

Class	Precision	Recall	mAP50	mAP50-95
head	0.907	0.895	0.943	0.626
helmet	0.940	0.917	0.957	0.628
Overall	0.923	0.906	0.950	0.627

The dataset was also supplemented with manually labeled real-world false-detection examples (close-range indoor images) to improve robustness beyond the original dataset's construction-site imagery.

🛠️ Pipeline overview
prepare_dataset.py — converts Pascal VOC XML annotations to YOLO format and splits the dataset into train/val/test
Training — run on Google Colab (T4 GPU) using ultralytics YOLO11s, 50 epochs, 640px images
head_alarm_detection.py — real-time webcam detection with an audio alarm for no-helmet violations
app.py — Streamlit web app for image/video/webcam detection
🚀 Running locally
Setup
bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python streamlit streamlit-webrtc av
Real-time webcam detection with alarm
bash
python3 head_alarm_detection.py

Press q to quit.

Streamlit web app
bash
streamlit run app.py

Supports image upload, video upload (with downloadable annotated output), and live browser webcam detection.

⚠️ Known limitations
The model performs best on imagery similar to its training distribution (medium-distance, outdoor/industrial settings). Performance may degrade on out-of-distribution inputs, such as close-range indoor selfie-style webcam shots.
The Streamlit webcam tab provides visual detection only (bounding boxes + on-frame text) — the audio alarm is not supported there due to a threading limitation in streamlit-webrtc. Use head_alarm_detection.py locally for the full audio alarm experience.
In order to try the detction and the alarm real sound , run this file : python3 yolo_hardhat_detection/head_alarm_detection.py

📁 Project structure
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
│   └── best_final_model.pt
|   └── Initial_model.pt
│
├── assets/
│   └── alarm.mp3
│

🙏 Acknowledgments
Dataset: Hard Hat Detection by Andrew Mvd (Kaggle)
Built with Ultralytics YOLO11