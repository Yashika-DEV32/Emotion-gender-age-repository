# ... (same imports and setup code as before)
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
from keras.models import load_model

# File paths
log_file = "emotion_log.csv"
user_profile_file = "user_profiles.csv"
photo_log_file = "emotion_photos_log.csv"

# Create folders
os.makedirs('emotion_photos', exist_ok=True)
os.makedirs('profile_pictures', exist_ok=True)
os.makedirs('gender_model', exist_ok=True)
os.makedirs('age_model', exist_ok=True)

# Initialize CSVs
def initialize_csv(file_path, columns):
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in columns):
            raise ValueError("Incorrect columns")
    except:
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)

initialize_csv(log_file, ["Timestamp", "UserID", "UserName", "Emotion", "Gender", "Age"])
initialize_csv(user_profile_file, ["UserID", "Name", "Emotion", "Gender", "Age", "Count", "Profile_Picture"])
initialize_csv(photo_log_file, ["Filename", "Timestamp", "UserID", "Emotion", "Gender", "Age"])

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

# Config
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(20-24)' ,'(25-32)', '(38-43)', '(48-53)', '(60-100)']
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if 'run' not in st.session_state:
    st.session_state.run = False

# UI
st.title("Emotion, Gender & Age Detection Dashboard")
FRAME_WINDOW = st.image([])

# Sidebar
st.sidebar.subheader("User Profile")
user_name = st.sidebar.text_input("Enter your name:")
user_profile_picture = st.sidebar.file_uploader("Upload your profile picture", type=["jpg", "png", "jpeg"])
user_id = st.sidebar.text_input("Enter User ID", value="user_1")

if user_name and user_profile_picture:
    img = Image.open(user_profile_picture)
    profile_picture_path = f"profile_pictures/{user_id}_profile_pic.jpg"
    img.save(profile_picture_path)

    df_user = pd.read_csv(user_profile_file)
    if user_id in df_user['UserID'].values:
        df_user.loc[df_user['UserID'] == user_id, 'Name'] = user_name
        df_user.loc[df_user['UserID'] == user_id, 'Profile_Picture'] = profile_picture_path
    else:
        new_row = pd.DataFrame({
            "UserID": [user_id],
            "Name": [user_name],
            "Emotion": ["Neutral"],
            "Gender": ["Unknown"],
            "Age": ["Unknown"],
            "Count": [0],
            "Profile_Picture": [profile_picture_path]
        })
        df_user = pd.concat([df_user, new_row], ignore_index=True)
    df_user.to_csv(user_profile_file, index=False)

if user_name:
    st.sidebar.write(f"Hello, {user_name}!")
    if os.path.exists(f"profile_pictures/{user_id}_profile_pic.jpg"):
        st.sidebar.image(f"profile_pictures/{user_id}_profile_pic.jpg", use_container_width=True)
else:
    st.sidebar.write("Please enter your name and upload a profile picture.")

if st.sidebar.button("Start Camera"):
    st.session_state.run = True
if st.sidebar.button("Stop Camera"):
    st.session_state.run = False

# Filters
st.sidebar.markdown("### ⏳ Filters")
time_duration = st.sidebar.selectbox(
    "Select Time Duration:",
    ["All Time", "1 Minute Ago", "5 Minutes Ago", "10 Minutes Ago", "30 Minutes Ago", "1 Hour Ago"]
)

month_filter = st.sidebar.selectbox(
    "Select Month (Optional):",
    ["None", "February", "March", "April", "May", "June", "July"]
)

selected_user = st.sidebar.text_input("Filter by User ID", "")

# Camera
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
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)[..., np.newaxis]
            emotion_preds = emotion_model.predict(roi, verbose=0)
            emotion = class_labels[np.argmax(emotion_preds)]

            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
            gender_net.setInput(blob)
            gender = gender_list[np.argmax(gender_net.forward())]

            age_net.setInput(blob)
            age = age_list[np.argmax(age_net.forward())]

            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

            with open(log_file, "a") as f:
                f.write(f"{timestamp},{user_id},{user_name},{emotion},{gender},{age}\n")

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
                    "Name": [user_name],
                    "Emotion": [emotion],
                    "Gender": [gender],
                    "Age": [age],
                    "Count": [1],
                    "Profile_Picture": [""]
                })
                df_user = pd.concat([df_user, new_row], ignore_index=True)
            df_user.to_csv(user_profile_file, index=False)

            filename = f"{emotion}_{gender}_{age}_{timestamp_str}.jpg"
            filepath = os.path.join("emotion_photos", filename)
            cv2.imwrite(filepath, face_img)
            with open(photo_log_file, "a") as f:
                f.write(f"{filename},{timestamp},{user_id},{emotion},{gender},{age}\n")

            label = f"{emotion}, {gender}, {age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)
    cap.release()

# --------------- ANALYTICS SECTION ---------------
st.header("📊 Emotion Analytics")

try:
    df = pd.read_csv(log_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Time filter
    now = datetime.now()
    if time_duration == "All Time":
        start_time = datetime.min
    elif time_duration == "1 Minute Ago":
        start_time = now - timedelta(minutes=1)
    elif time_duration == "5 Minutes Ago":
        start_time = now - timedelta(minutes=5)
    elif time_duration == "10 Minutes Ago":
        start_time = now - timedelta(minutes=10)
    elif time_duration == "30 Minutes Ago":
        start_time = now - timedelta(minutes=30)
    elif time_duration == "1 Hour Ago":
        start_time = now - timedelta(hours=1)
    end_time = now

    df_filtered = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

    # Month filter
    if month_filter != "None":
        month_number = datetime.strptime(month_filter, "%B").month
        df_filtered = df_filtered[df_filtered['Timestamp'].dt.month == month_number]

    # User filter
    if selected_user:
        df_filtered = df_filtered[df_filtered['UserID'] == selected_user]

    if df_filtered.empty:
        st.warning("No data found in the selected filters.")
    else:
        st.subheader("📌 Emotion Log Table")
        st.dataframe(df_filtered)

        st.subheader("📌 Emotion Frequency")
        st.bar_chart(df_filtered['Emotion'].value_counts())

        st.subheader("📌 Gender Distribution")
        gender_counts = df_filtered['Gender'].value_counts()
        gender_fig = plt.figure(figsize=(6, 4))
        gender_counts.plot(kind='bar', color=['#85C1AE', '#FF6F61'])
        plt.title("Gender Distribution")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        st.pyplot(gender_fig)

        st.subheader("📌 Age Distribution")
        st.bar_chart(df_filtered['Age'].value_counts())

        st.subheader("📌 Emotion Trend Over Time")
        emotion_time = df_filtered.groupby([pd.Grouper(key='Timestamp', freq='1min'), 'Emotion']).size().unstack(fill_value=0)
        st.line_chart(emotion_time)

        st.subheader("⭐ Most Frequent Emotion with Gender")
        most_common = df_filtered['Emotion'].value_counts().idxmax()
        most_common_gender = df_filtered[df_filtered['Emotion'] == most_common]['Gender'].value_counts().idxmax()
        count = df_filtered['Emotion'].value_counts().max()
        st.success(f"Most frequent emotion: **{most_common}** ({count} times), Gender: **{most_common_gender}**")

        # Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, index=False, sheet_name='Emotion Log')
        excel_buffer.seek(0)

        st.download_button(
            label="📥 Download Emotion Log as Excel",
            data=excel_buffer,
            file_name="emotion_log_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

except Exception as e:
    st.error(f"Error in analytics: {e}")
