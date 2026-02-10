# Streamlit Placeholder

For now there is **no active Streamlit application** shipped with the project.

However, the project is organised so that adding one is easy:

- Input → webcam frames
- Detection → `src/detection/mediapipe_wrapper.py`
- Gesture logic → `src/inference/predictor.py`
- Visual mapping → `src/inference/mapping.py` + `assets/avatars/`

A future `streamlit_app.py` could simply:

```python
# pseudo-code example

from src.detection.mediapipe_wrapper import MediaPipeHandDetector
from src.inference.predictor import detect_gesture_from_landmarks
