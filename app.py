import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

# Website Settings
st.set_page_config(page_title="Emotion Detector", page_icon="😎")

st.title("Vamshi Emotion Detector 📸")
st.write("Take a selfie below to check your mood!")

# Colors for Boxes
colors = {
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'angry': (0, 0, 255),
    'neutral': (255, 255, 255),
    'surprise': (0, 255, 255)
}

# Camera Button
img_file_buffer = st.camera_input("Click to Scan Face")

if img_file_buffer is not None:
    # Photo process cheyadam
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    try:
        with st.spinner('Analyzing... Wait cheyandi...'):
            # AI Checking
            result = DeepFace.analyze(cv2_img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            region = result[0]['region']

            # Box Draw Cheyadam
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            color = colors.get(emotion, (255, 255, 255))

            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), color, 4)
            cv2.rectangle(cv2_img, (x, y-50), (x+w, y), color, -1)
            cv2.putText(cv2_img, emotion.upper(), (x+10, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # Chupinchadam
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption=f"Mood: {emotion.upper()}", use_column_width=True)

            if emotion == 'happy':
                st.balloons()

    except:

        st.error("Face sarigga kanipinchadam ledu. Malli try cheyandi.")
