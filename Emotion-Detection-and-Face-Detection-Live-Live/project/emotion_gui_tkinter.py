import tkinter as tk
from PIL import Image, ImageTk
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# ------------------------------
# Load the face detection model
# ------------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ------------------------------
# Load the emotion recognition model
# ------------------------------
model_path = 'emotion_model.h5'
classifier = load_model(model_path)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create window
window = tk.Tk()
window.title("Emotion Detector Dashboard")

# Label for video feed
video_label = tk.Label(window)
video_label.pack()

# Label for emotion
emotion_label = tk.Label(window, text="Emotion: N/A", font=("Helvetica", 16))
emotion_label.pack()

# Create a Text widget to display emotion probabilities
emotion_probs_text = tk.Text(window, height=10, width=40, font=("Helvetica", 12))
emotion_probs_text.pack()

# Capture and update frame
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert frame to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 6)

        detected_emotions = []  # List to store detected emotions and their probabilities

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion probabilities
                preds = classifier.predict(roi, verbose=0)[0]
                emotion_probabilities = {class_labels[i]: preds[i] for i in range(len(class_labels))}

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Store the detected emotion and its probability
                detected_emotions = [(emotion, f"{prob:.2f}") for emotion, prob in emotion_probabilities.items()]

        # Convert frame to ImageTk format
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update the emotion probabilities display
        emotion_probs_text.delete(1.0, tk.END)  # Clear the previous text
        for emotion, prob in detected_emotions:
            emotion_probs_text.insert(tk.END, f"{emotion}: {prob}\n")
    
    video_label.after(10, update_frame)

update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
