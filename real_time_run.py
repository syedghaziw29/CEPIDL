import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

# Load the trained model
model = tf.keras.models.load_model("lstm_Bicep_curls_model.keras")

# Define class labels
class_labels = {0: "neutral", 1: "bicep_curl", 2: "wrong"}

# Initialize MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Smoothing parameters
angle_smoothing = deque(maxlen=5)


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def make_landmark_timestep(results):
    """Convert pose landmarks to a flat list of selected features."""
    landmarks = results.pose_landmarks.landmark
    relevant_landmarks = [
        mpPose.PoseLandmark.NOSE,
        mpPose.PoseLandmark.LEFT_EYE,
        mpPose.PoseLandmark.RIGHT_EYE,
        mpPose.PoseLandmark.LEFT_EAR,
        mpPose.PoseLandmark.RIGHT_EAR,
        mpPose.PoseLandmark.LEFT_SHOULDER,
        mpPose.PoseLandmark.RIGHT_SHOULDER,
        mpPose.PoseLandmark.LEFT_ELBOW,
        mpPose.PoseLandmark.RIGHT_ELBOW,
        mpPose.PoseLandmark.LEFT_WRIST,
        mpPose.PoseLandmark.RIGHT_WRIST,
        mpPose.PoseLandmark.LEFT_PINKY,
        mpPose.PoseLandmark.RIGHT_PINKY,
        mpPose.PoseLandmark.LEFT_INDEX,
        mpPose.PoseLandmark.RIGHT_INDEX,
        mpPose.PoseLandmark.LEFT_THUMB,
        mpPose.PoseLandmark.RIGHT_THUMB,
        mpPose.PoseLandmark.LEFT_HIP,
        mpPose.PoseLandmark.RIGHT_HIP,
    ]
    # Extract x, y, z, and visibility for each relevant landmark
    return [val for lm_id in relevant_landmarks for val in (
        landmarks[lm_id].x,
        landmarks[lm_id].y,
        landmarks[lm_id].z,
        landmarks[lm_id].visibility)]


class BicepCurlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bicep Curl AI Trainer")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #f0f4f8;")  # Light gray background

        # GUI Components
        self.layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #0078D7; border-radius: 10px;")  # Blue border
        self.layout.addWidget(self.video_label)

        # Feedback label
        self.feedback_label = QLabel("Feedback: Initializing...", self)
        self.feedback_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.feedback_label.setStyleSheet("color: #0078D7; padding: 10px;")
        self.layout.addWidget(self.feedback_label)

        # Close button
        self.close_button = QPushButton("Close", self)
        self.close_button.setFont(QFont("Arial", 14))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        self.setLayout(self.layout)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms per frame

        # Other Variables
        self.lm_list = []
        self.rep_counter = 0
        self.arms_extended = True
        self.label = "neutral"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.feedback_label.setText("Error: Unable to access webcam.")
            return

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            self.lm_list.append(lm)
            if len(self.lm_list) == 20:  # Sequence length for LSTM
                lm_array = np.expand_dims(np.array(self.lm_list), axis=0)
                prediction = model.predict(lm_array)
                predicted_class = np.argmax(prediction[0])
                self.label = class_labels.get(predicted_class, "unknown")
                self.lm_list = []

            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            angle_smoothing.append(elbow_angle)
            smoothed_angle = np.mean(angle_smoothing)

            # Repetition logic
            if self.label == "bicep_curl":
                if smoothed_angle <= 30:  # Fully curled position
                    self.arms_extended = False
                elif not self.arms_extended and smoothed_angle >= 170:  # Fully extended position
                    self.rep_counter += 1
                    self.arms_extended = True

            # Draw landmarks
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Draw feedback
        feedback_text = f"Label: {self.label} | Reps: {self.rep_counter}"
        self.feedback_label.setText(feedback_text)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        q_img = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def close_application(self):
        self.cap.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BicepCurlApp()
    window.show()
    sys.exit(app.exec_())
