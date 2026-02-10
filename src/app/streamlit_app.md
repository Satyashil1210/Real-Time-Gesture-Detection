# RT-Gesture3D – Streamlit App (Planned)

This document outlines how a future Streamlit-based UI could look:

- Live webcam preview using `cv2.VideoCapture` + `streamlit-webrtc` or similar.
- Real-time gesture recognition by calling:
  - `src/detection/mediapipe_wrapper.MediaPipeHandDetector`
  - `src/inference/predictor.detect_gesture_from_landmarks`
- Right panel showing:
  - Current gesture label and confidence.
  - Avatar image loaded from `assets/avatars/`.
  - Short description (meaning) from `datasets/gestures.csv`.

The goal is to keep the **core logic** in `src/inference/` and only
use Streamlit as a thin presentation layer.
