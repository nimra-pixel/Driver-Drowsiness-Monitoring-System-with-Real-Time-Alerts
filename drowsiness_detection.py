import cv2
import mediapipe as mp
import simpleaudio as sa
import time
import os

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks (left + right)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Alarm function
def play_alarm():
    try:
        wave_obj = sa.WaveObject.from_wave_file("mixkit-classic-alarm-995.wav")
        wave_obj.play()
    except Exception as e:
        print(f"[WARN] Alarm sound failed: {e}")

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(landmarks, eye_points, w, h):
    p1 = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
    p2 = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
    p3 = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))
    p4 = (int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h))
    p5 = (int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h))
    p6 = (int(landmarks[eye_points[5]].x * w), int(landmarks[eye_points[5]].y * h))

    ear = (abs(p2[1] - p6[1]) + abs(p3[1] - p5[1])) / (2.0 * abs(p1[0] - p4[0]))
    return ear

# Open webcam
cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

frame_counter = 0
alert_on = False
capture_count = 0
paused = False  # New flag for pause/resume

print("[INFO] Starting webcam... Press 'q' to quit. 'p' to pause, 'r' to resume.")

while True:
    if not paused:  # Only process frames when not paused
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1

                    if frame_counter >= CONSEC_FRAMES:
                        if not alert_on:
                            alert_on = True
                            play_alarm()

                            # Save drowsy image
                            capture_count += 1
                            filename = f"drowsy_capture_{capture_count}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"[ALERT] Drowsiness detected! Saved: {filename}")

                            paused = True  # Pause after capture

                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.rectangle(frame, (30, 30), (w-30, h-30), (0, 0, 255), 4)
                else:
                    frame_counter = 0
                    alert_on = False
                    cv2.putText(frame, "Awake - Monitoring", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.rectangle(frame, (30, 30), (w-30, h-30), (0, 255, 0), 3)

        cv2.imshow("Drowsiness Detection", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('p'):  # Pause
        paused = True
        print("[INFO] Detection Paused.")
    elif key == ord('r'):  # Resume
        paused = False
        frame_counter = 0
        alert_on = False
        print("[INFO] Detection Resumed.")

cap.release()
cv2.destroyAllWindows()
