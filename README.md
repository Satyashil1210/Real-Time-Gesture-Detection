# RT-Gesture3D 🔹 Real-Time Hand Gesture Recognition

RT-Gesture3D is a real-time hand gesture recognition system built on top of
MediaPipe and OpenCV. The project uses a clean, production-style architecture
to separate detection, inference, capture, processing, and training logic.

Currently, the system runs a fast heuristic-based recognizer and maps detected
gestures to visual avatars. The architecture is future-ready for integrating
machine learning models.

---

## 🔹 Features

- Real-time hand detection using MediaPipe
- Gesture recognition using landmark-based heuristics
- Avatar and label visualization
- CSV-based gesture registry (`datasets/gestures.csv`)
- Easy extensibility for future ML training
- Modular code structure (capture, detection, processing, training, inference)
- Ready-to-use webcam demo

---

## 🔹 Project Structure

```text
RT-Gesture3D/
├── assets/avatars      # Gesture avatars (png / jpg)
├── datasets            # CSV based gesture definitions
├── data/raw            # Raw captured frames
├── data/processed      # Preprocessed landmarks / feature files
├── models/checkpoints  # Trained model snapshots (future)
├── src/
│   ├── inference       # Real-time inference pipeline
│   ├── detection       # MediaPipe abstraction
│   ├── capture         # Dataset recording tools
│   ├── processing      # Buffers & preprocessing
│   ├── training        # (Future) model training
│   └── app             # UI layer placeholders
🔹 How to Run (Live Demo)

From project root:

python src/inference/live_gesture_demo.py


Press q to exit.

🔹 Dataset & Mapping

Gestures are defined centrally inside:

datasets/gestures.csv


Structure:

id,label,meaning,avatar
0,neutral,Neutral,neutral.png
1,victory,Victory Sign,victory.jpg
2,ok,OK / Thumbs Up,ok.jpg
...


Avatars are loaded from:

assets/avatars/

🔹 Architecture

Pipeline:

Camera → MediaPipe → Landmarks
          ↓
    Rule-Based Predictor
          ↓
       Gesture ID
          ↓
   CSV Mapping → Avatar + Label

🔹 Roadmap

Future upgrades:

Replace rule-based logic with an ML classifier

Add temporal sequence modeling

Export model to ONNX / TFLite

Web UI (Streamlit / Flask)

Mobile or embedded deployment

🔹 Author

RT-Gesture3D – built by Satyashil , Arman Pal 