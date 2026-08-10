import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RT-Gesture3D",
    page_icon="🖐️",
    layout="wide",
)

st.title("🎯 Real-Time Gesture Detection System")
st.caption("Live hand gesture recognition using MediaPipe + Streamlit WebRTC")


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def get_gesture(landmarks):
    if landmarks is None:
        return "No Hand"

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    fingers_up = [
        thumb_tip.y < landmarks[3].y,
        index_tip.y < landmarks[6].y,
        middle_tip.y < landmarks[10].y,
        ring_tip.y < landmarks[14].y,
        pinky_tip.y < landmarks[18].y,
    ]

    count = sum(fingers_up)

    if count == 0:
        return "Fist"

    if count == 1 and fingers_up[0]:
        return "Thumbs Up"

    if count == 1 and fingers_up[1]:
        return "Point"

    if count == 2 and fingers_up[1] and fingers_up[2]:
        return "Victory"

    if count == 5:
        return "Open Palm"

    return f"Custom ({count} fingers)"


# ============================================================
# MEDIAPIPE VIDEO PROCESSOR
# ============================================================

class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Mirror camera like a selfie camera
        image = cv2.flip(image, 1)

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        detected_gestures = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw MediaPipe landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                # Detect gesture
                gesture = get_gesture(
                    hand_landmarks.landmark
                )

                detected_gestures.append(gesture)

        # Display result
        if detected_gestures:

            text = " | ".join(detected_gestures)

            cv2.rectangle(
                image,
                (10, 10),
                (650, 70),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                text,
                (25, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        else:

            cv2.rectangle(
                image,
                (10, 10),
                (350, 70),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                "No Hand Detected",
                (25, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎮 Controls")

    st.write(
        """
        **Supported Gestures**

        👊 Fist  
        ☝️ Point  
        ✌️ Victory  
        👍 Thumbs Up  
        ✋ Open Palm
        """
    )

    st.divider()

    st.info(
        """
        Allow camera permission when your
        browser asks for it.
        """
    )


# ============================================================
# LIVE CAMERA
# ============================================================

st.subheader("📹 Live Gesture Detection")

st.write(
    "Click **START** and allow browser camera access."
)

webrtc_streamer(
    key="gesture-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
    async_processing=True,
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

tab1, tab2 = st.tabs(
    ["📋 Project Information", "ℹ️ How It Works"]
)

with tab1:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Language", "Python 3.12")

    with col2:
        st.metric("Vision", "MediaPipe")

    with col3:
        st.metric("WebRTC", "Enabled")

    with col4:
        st.metric("Framework", "Streamlit")


with tab2:

    st.markdown(
        """
        ### How the system works

        1. Browser requests webcam permission.
        2. WebRTC streams video frames.
        3. MediaPipe detects hand landmarks.
        4. Landmark coordinates are analyzed.
        5. Gesture classification runs in real time.
        6. Detected gesture is displayed on the video.

        ### Supported gestures

        - 👊 **Fist**
        - ☝️ **Point**
        - ✌️ **Victory**
        - 👍 **Thumbs Up**
        - ✋ **Open Palm**
        """
    )