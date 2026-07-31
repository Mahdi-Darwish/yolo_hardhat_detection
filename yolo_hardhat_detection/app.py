import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import tempfile
import os
import base64
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from head_alarm_detection import annotate_frame

st.set_page_config(page_title="Hard Hat Safety Detection", layout="wide")

@st.cache_resource
def get_model():
    return YOLO('models/best_final_model.pt')

model = get_model()

def play_browser_alarm():
    with open("assets/alarm.mp3", "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(
        f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>""",
        height=0, width=0
    )

st.title("🦺 Hard Hat Safety Detection")
st.write("YOLO11-based worker safety detection — identifies workers with and without hard hats.")

tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Video", "🔴 Webcam (live)"])

# Image Tab
with tab1:
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="img")
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, no_helmet = annotate_frame(model, frame.copy())
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detection result", use_column_width=True)

        if no_helmet:
            st.error("🚨 No helmet detected — safety violation")
            play_browser_alarm()
        else:
            st.success("✅ All detected workers wearing helmets")

# Video Tab
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'], key="vid")
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        no_helmet_frame_count = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, no_helmet = annotate_frame(model, frame)
            if no_helmet:
                no_helmet_frame_count += 1
            out.write(annotated)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()

        st.video(out_path)

        if no_helmet_frame_count > 0:
            st.error(f"🚨 No helmet detected in {no_helmet_frame_count} / {total_frames} frames")
            play_browser_alarm()
        else:
            st.success("✅ No safety violations detected in this video")

        with open(out_path, 'rb') as f:
            st.download_button("Download annotated video", f, file_name="annotated_output.mp4")

# Webcam Tab
with tab3:
    st.write("Live detection using your browser's webcam. Click **Start** below.")
    st.info("ℹ️ This tab shows live visual detection (bounding boxes + labels). "
            "For the full experience with an audio alarm, run `python3 head_alarm_detection.py` locally.")

    class HardHatProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, no_helmet = annotate_frame(model, img)

            # visual on-frame alert — this DOES work, since it's drawn directly onto the video frame
            if no_helmet:
                cv2.putText(annotated, "NO HELMET DETECTED!", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(key="hardhat-webcam", video_processor_factory=HardHatProcessor)