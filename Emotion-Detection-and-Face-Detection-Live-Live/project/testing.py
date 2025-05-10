import os
import pygame
import time
import threading
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from datetime import datetime

# ------------------------------
# Load the face detection model
# ------------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ------------------------------
# Load the emotion recognition model
# ------------------------------
model_path = 'emotion_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
classifier = load_model(model_path)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------------
# Audio setup for all emotions
# ------------------------------
pygame.mixer.init()
emotion_sounds = {
    'Angry': 'angry.mp3',
    'Disgust': 'disgust.mp3',
    'Fear': 'fear.mp3',
    'Happy': 'smile.mp3',
    'Neutral': 'neutral.mp3',
    'Sad': 'sad.mp3',
    'Surprise': 'surprise.mp3'
}
currently_playing = None
lock = threading.Lock()

def play_sound(file):
    with lock:
        global currently_playing
        if currently_playing != file:
            try:
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                currently_playing = file
            except Exception as e:
                print(f"Error playing sound: {e}")

# ------------------------------
# Create screenshot folder
# ------------------------------
os.makedirs("screenshots", exist_ok=True)

# ------------------------------
# Open webcam and detect emotion
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi, verbose=0)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)

            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Play sound for detected emotion
            if label in emotion_sounds:
                threading.Thread(target=play_sound, args=(emotion_sounds[label],)).start()

            # Take screenshot for any detected emotion
            filename = os.path.join("screenshots", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{label}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot taken for {label} and saved as {filename}")
        else:
            cv2.putText(frame, 'No Face Found', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
