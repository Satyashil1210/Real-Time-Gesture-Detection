🚀 Real-Time Gesture Detection
✋ Static + 🔄 Dynamic Gesture Recognition Roadmap

A production-structured real-time hand gesture recognition system built using MediaPipe, OpenCV, and modular ML-ready architecture.

Currently, the system supports fast static gesture recognition using a landmark-based heuristic engine.
The architecture is designed to evolve into a deep learning-based dynamic gesture recognition system.

✨ Current Capabilities (v1 – Static Gesture Engine)

✅ Real-time hand detection using MediaPipe
✅ Landmark-based rule engine for gesture classification
✅ Avatar + label rendering system
✅ CSV-based gesture registry (datasets/gestures.csv)
✅ Modular & scalable code structure
✅ Webcam live demo ready
✅ Future-ready ML pipeline integration

🔮 Vision (v2 – ML + Dynamic Gesture Recognition)

Upcoming upgrades include:

🔄 Temporal gesture recognition (dynamic gestures)
🧠 LSTM / GRU / MLP based classifier
📊 Sliding window landmark buffering
📦 ONNX / TensorFlow Lite export
🌐 Streamlit interactive web UI
📁 Automatic dataset builder
⚡ Real-time FPS optimization
📱 Mobile / embedded deployment support

🏗 System Architecture
Camera Input 🎥
        ↓
MediaPipe Hand Landmarks ✋
        ↓
Feature Extraction 📊
        ↓
Static Rule Engine (Current Version)
        ↓
ML Classifier (Upcoming Version)
        ↓
Gesture ID
        ↓
CSV Mapping
        ↓
Avatar + Label Rendering 🖼

📂 Project Structure
Real-Time-Gesture-Detection/
├── assets/avatars        # Gesture avatars (png / jpg)
├── datasets              # CSV gesture definitions
├── data/raw              # Captured frames
├── data/processed        # Landmark feature files
├── models/checkpoints    # Trained models (future)
├── src/
│   ├── inference         # Real-time inference pipeline
│   ├── detection         # MediaPipe abstraction
│   ├── capture           # Dataset recording tools
│   ├── processing        # Buffers & preprocessing
│   ├── training          # ML training modules
│   └── app               # Streamlit UI layer

▶️ How to Run (Live Demo)
1️⃣ Activate Virtual Environment
.\venv\Scripts\Activate.ps1

2️⃣ Run Static Real-Time Demo
python src/inference/live_gesture_demo.py


Press q to exit.

3️⃣ Run Web UI (Streamlit)
streamlit run src/app/web_app_placeholder.py


Open in browser:

http://localhost:8501

🗂 Dataset & Gesture Registry

Gestures are centrally defined in:

datasets/gestures.csv


Example format:

id,label,meaning,avatar
0,neutral,Neutral,neutral.png
1,victory,Victory Sign,victory.jpg
2,ok,OK Gesture,ok.jpg


This ensures prediction logic is decoupled from UI rendering.

🛠 Tech Stack

🐍 Python 3.10
👁 MediaPipe
📸 OpenCV
📊 NumPy
⚙ TensorFlow Lite (ML-ready)
🌐 Streamlit
