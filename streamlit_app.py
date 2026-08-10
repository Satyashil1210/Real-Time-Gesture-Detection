import streamlit as st

st.set_page_config(page_title="RT-Gesture3D", layout="wide")

st.title("🎯 Real-Time Gesture Detection System")
st.subheader("Hand Gesture Recognition using MediaPipe & YOLOv8")

st.markdown("""
### 📋 Project Overview
A production-grade real-time hand gesture recognition system with:
- ✅ Static gesture recognition (current)
- 🔄 Dynamic gesture recognition (upcoming)
- 📊 Modular ML-ready architecture
""")

col1, col2 = st.columns(2)

with col1:
    st.info("### 🚀 Features")
    st.write("""
    - Real-time hand detection via MediaPipe
    - Landmark-based gesture classification
    - Live avatar rendering
    - CSV-based gesture registry
    """)

with col2:
    st.success("### 📁 Project Structure")
    st.write("""
    - `src/inference/` - Prediction pipeline
    - `src/detection/` - MediaPipe wrappers
    - `src/processing/` - Feature extraction
    - `datasets/` - Gesture registry
    """)

st.markdown("---")

st.subheader("💻 Tech Stack")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Language", "Python 3.10")
with col2:
    st.metric("Vision", "MediaPipe")
with col3:
    st.metric("ML", "TensorFlow Lite")
with col4:
    st.metric("Framework", "Streamlit")

st.markdown("---")

st.info("""
🔗 **GitHub Repository**: [Real-Time-Gesture-Detection](https://github.com/Satyashil1210/Real-Time-Gesture-Detection)

**Note**: For live webcam demo, run locally:
```bash
python src/inference/live_gesture_demo.py
```
""")