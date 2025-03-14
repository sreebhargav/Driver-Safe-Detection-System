'''AI-Powered Real-Time Drowsiness Detection System'''

# Import required libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame  # For sound alerts
import time
import dlib
import cv2
import smtplib
from email.mime.text import MIMEText
from sklearn.linear_model import SGDClassifier

# --- Email & SMS Alert Configurations ---
SENDER_EMAIL = "sreebhargavbalusu@gmail.com"
SENDER_PASSWORD = "ugyt npcs tqba lvau"
CARETAKER_EMAIL = "sreebhargavbalusu@gmail.com"
CARETAKER_SMS_EMAIL = "6692518266@tmomail.net"  # T-Mobile SMS Gateway

# Function to send Email & SMS alerts
def send_alert(subject, body, recipient):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
    except Exception as e:
        pass  # Avoid crashing due to email failures

def send_email_alert():
    send_alert("🚨 Drowsiness Alert!", "Drowsiness detected! Please check immediately.", CARETAKER_EMAIL)

def send_sms_alert():
    send_alert("🚨 Drowsiness Alert!", "Drowsiness detected! Please check immediately.", CARETAKER_SMS_EMAIL)

# --- Sound Alert Setup ---
pygame.mixer.init()
pygame.mixer.music.load('audio/alert_final.wav')
pygame.mixer.music.set_volume(1.0)

# --- Drowsiness Detection Configuration ---
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Adjusted for better detection
EYE_ASPECT_RATIO_CONSEC_FRAMES = 30  # Faster response time
COUNTER = 0
ALARM_ON = False

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Get indexes for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- AI-Powered Learning (No Pretrained Model Needed) ---
model = SGDClassifier(loss="log_loss")
is_trained = False  # Track if AI has been trained

# Start webcam capture
video_capture = cv2.VideoCapture(0)
time.sleep(2)  # Allow the camera to adjust

while True:
    # Capture video frame
    ret, frame = video_capture.read()
    if not ret:
        break  # Stop if the camera feed fails

    frame = cv2.flip(frame, 1)  # Mirror effect for better experience
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection

    # Detect faces
    faces = detector(gray, 0)

    # Process detected faces
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # AI model training
        label = 1 if ear < EYE_ASPECT_RATIO_THRESHOLD else 0  # 1 = DROWSY, 0 = ALERT
        model.partial_fit([[ear]], [label], classes=[0, 1])
        is_trained = True

        # AI Prediction
        drowsy_prediction = model.predict([[ear]])[0]

        # Drowsiness detection logic
        if drowsy_prediction == 1:
            COUNTER += 1

            # Trigger alert if eyes have been closed for enough frames
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)  # Play alert sound
                    send_email_alert()  # Send Email Alert
                    send_sms_alert()  # Send SMS Alert
                    ALARM_ON = True

                cv2.putText(frame, "DROWSY! WAKE UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            pygame.mixer.music.stop()  # Stop alarm if eyes are open
            COUNTER = 0
            ALARM_ON = False

    # Display the video feed
    cv2.imshow('Drowsiness Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup on exit
video_capture.release()
cv2.destroyAllWindows()

