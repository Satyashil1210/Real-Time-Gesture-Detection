import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from collections import defaultdict

st.set_page_config(page_title="RT-Gesture3D", layout="wide")

st.title("🎯 Real-Time Gesture Detection System")
st.subheader("Hand Gesture Recognition using MediaPipe & YOLOv8")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Gesture Recognition Logic
def get_gesture(landmarks):
    """Simple gesture recognition based on landmarks"""
    if landmarks is None:
        return "No Hand"
    
    # Get finger positions
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    palm = landmarks[0]
    
    # Simple heuristics
    fingers_up = [
        thumb_tip.y < landmarks[3].y,
        index_tip.y < landmarks[6].y,
        middle_tip.y < landmarks[10].y,
        ring_tip.y < landmarks[14].y,
        pinky_tip.y < landmarks[18].y
    ]
    
    fingers_count = sum(fingers_up)
    
    if fingers_count == 0:
        return "Fist"
    elif fingers_count == 1 and fingers_up[0]:
        return "Thumbs Up"
    elif fingers_count == 1 and fingers_up[1]:
        return "Point"
    elif fingers_count == 2 and fingers_up[1] and fingers_up[2]:
        return "Victory"
    elif fingers_count == 5:
        return "Open Palm"
    else:
        return f"Custom ({fingers_count} fingers)"

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Live Demo", "Project Info", "How to Use"])

with tab1:
    st.subheader("📹 Live Gesture Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Enable your webcam below:**")
        picture = st.camera_input("Capture gesture")
        
        if picture is not None:
            # Process the image
            img_array = np.frombuffer(picture.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Flip for selfie view
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            
            # MediaPipe processing
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get gesture
                    gesture = get_gesture(hand_landmarks.landmark)
                    
                    # Put text on image
                    cv2.putText(
                        img,
                        gesture,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2
                    )
            else:
                cv2.putText(
                    img,
                    "No hand detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            st.image(img, channels="BGR", use_column_width=True)
    
    with col2:
        st.info("### 🎯 How it works:")
        st.write("""
        1. Click "Take a photo" button
        2. Allow camera access
        3. Show your hand gesture
        4. System detects gesture in real-time
        
        **Gestures detected:**
        - 👊 Fist
        - ☝️ Point
        - ✌️ Victory
        - 👍 Thumbs Up
        - ✋ Open Palm
        """)

with tab2:
    st.markdown("""
    ### 📋 Project Overview
    A production-grade real-time hand gesture recognition system with:
    - ✅ Static gesture recognition (current)
    - 🔄 Dynamic gesture recognition (upcoming)
    - 📊 Modular ML-ready architecture
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### 🚀 Features")
        st.write("""
        - Real-time hand detection via MediaPipe
        - Landmark-based gesture classification
        - Live avatar rendering
        - CSV-based gesture registry
        """)
    
    with col2:
        st.info("### 📁 Project Structure")
        st.write("""
        - `src/inference/` - Prediction pipeline
        - `src/detection/` - MediaPipe wrappers
        - `src/processing/` - Feature extraction
        - `datasets/` - Gesture registry
        """)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Language", "Python 3.10")
    with col2:
        st.metric("Vision", "MediaPipe")
    with col3:
        st.metric("ML", "TensorFlow Lite")
    with col4:
        st.metric("Framework", "Streamlit")

with tab3:
    st.markdown("""
    ### 💻 How to use this app:
    
    1. **Live Demo Tab**
       - Click "Take a photo" to capture gesture
       - System automatically detects hand and predicts gesture
       - Shows landmarks overlay
    
    2. **Gestures Supported**
       - Fist (all fingers closed)
       - Point (only index finger up)
       - Victory (index + middle finger up)
       - Thumbs Up (only thumb up)
       - Open Palm (all fingers up)
    
    3. **For Full Feature Access**
       - Clone repo: `git clone https://github.com/Satyashil1210/Real-Time-Gesture-Detection.git`
       - Run locally: `python src/inference/live_gesture_demo.py`
       - Full ML pipeline with temporal gesture recognition
    """)
    
    st.info("""
    🔗 **GitHub Repository**: [Real-Time-Gesture-Detection](https://github.com/Satyashil1210/Real-Time-Gesture-Detection)
    """)