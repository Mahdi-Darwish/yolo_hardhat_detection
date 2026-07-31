import cv2
from ultralytics import YOLO
import subprocess
import time

CLASS_COLORS = {'head': (0, 0, 255), 'helmet': (0, 255, 0)}  # BGR


def annotate_frame(model, frame):
    """Runs YOLO detection on one frame, draws boxes, returns (annotated_frame, no_helmet_found)"""
    results = model(frame, verbose=False)
    no_helmet_found = False

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if class_name == 'head':
                no_helmet_found = True

    return frame, no_helmet_found


class RealtimeYOLODetectionWithAlerts:
    def __init__(self, model_path='models/best_final_model.pt', alarm_path='assets/alarm.mp3'):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.alarm_path = alarm_path
        self.alert_active = False
        self.no_helmet_streak = 0
        self.NO_HELMET_THRESHOLD = 5
        self.alarm_process = None  # NEW: keep a handle to the running sound process

    def start_alarm(self):
        # NEW: Popen starts playback WITHOUT blocking the rest of the code
        self.alarm_process = subprocess.Popen(['afplay', self.alarm_path])
        self.alert_active = True

    def stop_alarm(self):
        # NEW: forcibly kill the sound the instant a helmet is detected again
        if self.alarm_process is not None and self.alarm_process.poll() is None:
            self.alarm_process.terminate()
        self.alarm_process = None
        self.alert_active = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Retrieval Error, Exiting..")
                break

            start_time = time.time()
            annotated, no_helmet = annotate_frame(self.model, frame)

            if no_helmet:
                self.no_helmet_streak += 1
            else:
                self.no_helmet_streak = 0
                if self.alert_active:
                    self.stop_alarm()  # helmet is back on — cut the alarm immediately

            if self.no_helmet_streak >= self.NO_HELMET_THRESHOLD:
                if not self.alert_active:
                    # first time crossing the threshold — start the alarm
                    self.start_alarm()
                elif self.alarm_process is not None and self.alarm_process.poll() is not None:
                    # NEW: alarm finished playing on its own, but still no helmet — restart it
                    self.start_alarm()

            if self.alert_active:
                cv2.putText(annotated, "NO HELMET DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Real-time YOLO Detection with Alerts', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = RealtimeYOLODetectionWithAlerts()
    detector.run()