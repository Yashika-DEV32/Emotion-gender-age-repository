import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
from keras.models import load_model

# File paths
log_file = "emotion_log.csv"
user_profile_file = "user_profiles.csv"
photo_log_file = "emotion_photos_log.csv"

# Create folders
os.makedirs('emotion_photos', exist_ok=True)
os.makedirs('gender_model', exist_ok=True)
os.makedirs('age_model', exist_ok=True)

# Ensure CSVs have correct columns
def initialize_csv(file_path, columns):
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in columns):
            raise ValueError("Incorrect columns")
    except:
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)

initialize_csv(log_file, ["Timestamp", "Emotion", "Gender", "Age"])
initialize_csv(user_profile_file, ["UserID", "Emotion", "Gender", "Age", "Count"])
initialize_csv(photo_log_file, ["Filename", "Timestamp", "Emotion", "Gender", "Age"])

# Load models
try:
    emotion_model = load_model("emotion_model.h5", compile=False)
    st.success("Emotion model loaded.")
except Exception as e:
    st.error(f"Failed to load emotion model: {e}")
    emotion_model = None

try:
    gender_net = cv2.dnn.readNetFromCaffe(
        "gender_model/deploy_gender.prototxt",
        "gender_model/gender_net.caffemodel"
    )
    st.success("Gender model loaded.")
except Exception as e:
    st.error(f"Failed to load gender model: {e}")
    gender_net = None

try:
    age_net = cv2.dnn.readNetFromCaffe(
        "age_model/deploy_age.prototxt",
        "age_model/age_net.caffemodel"
    )
    st.success("Age model loaded.")
except Exception as e:
    st.error(f"Failed to load age model: {e}")
    age_net = None

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("Emotion, Gender & Age Detection Dashboard")
FRAME_WINDOW = st.image([])

# Sidebar
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body {background-color: #2e2e2e; color: white;}</style>", unsafe_allow_html=True)

time_range = st.sidebar.selectbox("Show data from last...", ["1 minute", "5 minutes", "All time"])
start_time, end_time = st.sidebar.slider(
    "Custom Time Range",
    value=(datetime.now() - pd.Timedelta(minutes=30), datetime.now()),
    min_value=datetime(2025, 1, 1),
    max_value=datetime.now()
)

user_id = st.sidebar.text_input("Enter User ID", value="user_1")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False

if st.button("Start Camera"):
    st.session_state.run = True

if st.button("Stop Camera"):
    st.session_state.run = False

# Filter helper
def filter_df_by_time(df, time_range):
    if time_range == "All time":
        return df
    minutes = int(time_range.split()[0])
    now = datetime.now()
    return df[df['Timestamp'] >= now - pd.Timedelta(minutes=minutes)]

# Camera loop
if st.session_state.run and emotion_model and gender_net and age_net:
    cap = cv2.VideoCapture(0)

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Emotion prediction
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)[..., np.newaxis]
            emotion_preds = emotion_model.predict(roi, verbose=0)
            emotion = class_labels[np.argmax(emotion_preds)]

            # Gender prediction
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[np.argmax(gender_preds)]

            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[np.argmax(age_preds)]

            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

            # Log emotion
            with open(log_file, "a") as f:
                f.write(f"{timestamp},{emotion},{gender},{age}\n")

            # Update user profile
            df_user = pd.read_csv(user_profile_file)
            match = (
                (df_user['UserID'] == user_id) &
                (df_user['Emotion'] == emotion) &
                (df_user['Gender'] == gender) &
                (df_user['Age'] == age)
            )
            if match.any():
                df_user.loc[match, 'Count'] = df_user.loc[match, 'Count'].astype(int) + 1
            else:
                new_row = pd.DataFrame({
                    "UserID": [user_id],
                    "Emotion": [emotion],
                    "Gender": [gender],
                    "Age": [age],
                    "Count": [1]
                })
                df_user = pd.concat([df_user, new_row], ignore_index=True)
            df_user.to_csv(user_profile_file, index=False)

            # Save image and log
            filename = f"{emotion}_{gender}_{age}_{timestamp_str}.jpg"
            filepath = os.path.join("emotion_photos", filename)
            cv2.imwrite(filepath, face_img)
            with open(photo_log_file, "a") as f:
                f.write(f"{filename},{timestamp},{emotion},{gender},{age}\n")

            # Annotate frame
            label = f"{emotion}, {gender}, {age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)

    cap.release()

# --------------- ANALYTICS SECTION ---------------
try:
    df_preview = pd.read_csv(log_file, nrows=5, on_bad_lines='skip')
    if df_preview.shape[1] == 4:
        df = pd.read_csv(log_file, names=["Timestamp", "Emotion", "Gender", "Age"], header=0, on_bad_lines='skip')
    elif df_preview.shape[1] == 3:
        df = pd.read_csv(log_file, names=["Timestamp", "Emotion", "Gender"], header=0, on_bad_lines='skip')
        df["Age"] = "Unknown"
    else:
        raise ValueError("Unexpected column format")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df_filtered = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

    st.subheader("Emotion Frequency")
    st.bar_chart(df_filtered['Emotion'].value_counts())

    st.subheader("Emotion Trends Over Time")
    emotion_time = df_filtered.groupby([pd.Grouper(key='Timestamp', freq='1min'), 'Emotion']).size().unstack(fill_value=0)
    st.line_chart(emotion_time)

    st.subheader("Gender Distribution")
    st.bar_chart(df_filtered['Gender'].value_counts())

    st.subheader("Age Distribution")
    st.bar_chart(df_filtered['Age'].value_counts())

    st.subheader("Recent Emotion Log")
    st.dataframe(df_filtered.tail(10))

    if not df_filtered.empty:
        top_emotion = df_filtered['Emotion'].value_counts().idxmax()
        st.subheader(f"Most Frequent Emotion: {top_emotion}")
    else:
        st.info("No data available for the selected time range.")
except Exception as e:
    st.error(f"Failed to load emotion log: {e}")

# --------------- USER PROFILE ---------------
st.subheader("User Emotion Profile")
try:
    df_user = pd.read_csv(user_profile_file)
    st.dataframe(df_user[df_user['UserID'] == user_id])
except Exception as e:
    st.warning(f"User profile not available: {e}")
