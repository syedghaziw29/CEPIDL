import cv2
import mediapipe as mp
import pandas as pd
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "wrong_bicep_final"
no_of_frames = 1000  

# Full list of landmarks for the upper body
LANDMARKS_TO_USE = [
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

def make_landmark_timestep(results):
    """Extract specific pose landmarks for the overhead press."""
    c_lm = []
    if results.pose_landmarks:
        for idx in LANDMARKS_TO_USE:
            lm = results.pose_landmarks.landmark[idx]
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    """Draw landmarks and connections on the frame."""
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

# Check if the file exists
csv_file = label + ".csv"
file_exists = os.path.exists(csv_file)

while len(lm_list) < no_of_frames:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        lm = make_landmark_timestep(results)
        if lm:
            lm_list.append(lm)
        frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        print("Error: Failed to capture image.")
        break

# Convert the list to a DataFrame
df = pd.DataFrame(lm_list)

# Append data to the CSV file
df.to_csv(csv_file, mode='a', index=False, header=not file_exists)

# Release resources
cap.release()
cv2.destroyAllWindows()
