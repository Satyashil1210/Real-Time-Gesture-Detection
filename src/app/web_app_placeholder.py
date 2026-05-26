from pathlib import Path
import sys

# =========================
# 🔧 PATH FIX
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVATARS_DIR = PROJECT_ROOT / "assets" / "avatars"

# =========================
# 🔥 IMPORTS
# =========================
try:
    from src.inference.live_gesture_main import main as run_live_demo
    from src.inference.mapping import GESTURES
except Exception as e:
    print("Import Error:", e)
    GESTURES = {}
    def run_live_demo():
        print("Demo not working")

# =========================
# 🧠 CLI MODE
# =========================
def run_cli():
    print("Launching RT-Gesture3D...")
    run_live_demo()

# =========================
# 🌐 STREAMLIT UI
# =========================
def run_streamlit_ui():
    import streamlit as st
    import time

    st.set_page_config(
        page_title="RT-Gesture3D",
        layout="wide",
        page_icon="🤖"
    )

    # =========================
    # 🎨 STYLE
    # =========================
    st.markdown("""
    <style>
    .main {background-color:#f4f6f8;}
    .card {
        padding:15px;
        border-radius:12px;
        background:white;
        box-shadow:0px 6px 16px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # 🔥 HEADER
    # =========================
    st.title("🤖 RT-Gesture3D System")
    st.caption("Computer Vision + Deep Learning + Human-Computer Interaction")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "✋ Gestures",
        "📦 Objects",
        "📚 Documentation"
    ])

    # =========================
    # 📊 DASHBOARD
    # =========================
    with tab1:

        col1, col2 = st.columns([2,1])

        with col1:
            st.markdown("### 🎬 Real-time AI System")

            st.markdown("""
This system performs:

✔ Gesture Recognition (Hand landmarks)  
✔ Object Detection (YOLOv8)  
✔ Avatar Feedback  
✔ Real-time processing  
""")

            if st.button("🚀 Start Live Demo"):
                run_live_demo()

        with col2:
            st.info(f"Gestures: {len(GESTURES)}")
            st.info("Models: MediaPipe + YOLOv8")
            st.info("Mode: Real-time")

        st.markdown("### 🔄 System Flow")

        st.code("""
Camera → Hand Detection → Gesture Logic → Avatar

Camera → YOLO → Object Detection → Bounding Box
""")

        st.markdown("### 🧠 AI Models (Present + Future)")

        st.success("""
Current:
✔ MediaPipe (Hand tracking)
✔ YOLOv8 (Object detection)

Future Ready:
✔ 3D CNN (Temporal feature extraction)
✔ Spatiotemporal CNN (ST-CNN)
""")

    # =========================
    # ✋ GESTURES
    # =========================
    with tab2:

        st.markdown("### ✋ Gesture Library")

        if not GESTURES:
            st.warning("No gestures found")
        else:
            cols = st.columns(3)

            for i, (key, info) in enumerate(GESTURES.items()):
                with cols[i % 3]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)

                    avatar_path = AVATARS_DIR / info.avatar_file

                    if avatar_path.exists():
                        st.image(str(avatar_path))

                    st.markdown(f"**{info.display_name}**")
                    st.caption(info.meaning)
                    st.code(f"id={info.id} | key={key}")

                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔍 Current Approach")
        st.info("""
- Landmark-based detection  
- Finger state logic  
- Static gesture recognition  
""")

        st.markdown("### 🚀 Future Upgrade")
        st.success("""
- Dynamic gesture recognition  
- Motion tracking  
- Deep learning-based classification  
""")

    # =========================
    # 📦 OBJECTS
    # =========================
    with tab3:

        st.markdown("### 📦 Object Detection System")

        st.markdown("""
Model Used: **YOLOv8**
""")

        st.success("""
✔ Person  
✔ Chair  
✔ Laptop  
✔ Mobile  
✔ Bottle  
✔ Book  
✔ TV  
✔ Cup  
""")

        st.markdown("### ⚙️ Working")

        st.code("""
Frame → YOLO → Detection → Filter → Bounding Box
""")

    # =========================
    # 📚 DOCUMENTATION
    # =========================
    with tab4:

        st.markdown("## 📚 Project Documentation")

        # Objective
        st.markdown("### 🎯 Objective")
        st.write("""
To develop a real-time system capable of recognizing gestures and objects  
using advanced deep learning architectures like:

- Spatiotemporal CNN (ST-CNN)  
- 3D CNN  

for capturing spatial + temporal features.
""")

        # Current system
        st.markdown("### 🧠 Current Implementation")
        st.success("""
✔ Real-time gesture detection (rule-based)  
✔ Object detection (YOLOv8)  
✔ Avatar UI system  
✔ Modular pipeline  
""")

        # Deep learning explanation
        st.markdown("### 🧠 Deep Learning Perspective")

        st.info("""
🔹 3D CNN:
- Extracts temporal features from multiple frames  
- Captures motion across time  

🔹 Spatiotemporal CNN:
- Combines spatial + temporal learning  
- Understands gesture sequences  

Currently simulated via rule-based logic,  
but architecture is ready for deep learning upgrade.
""")

        # Future scope
        st.markdown("### 🚀 Future Scope")

        st.success("""
🔥 NEXT PHASE: Dynamic Gesture Recognition

We will implement:

✔ 3D CNN models  
✔ Spatiotemporal CNN  
✔ Temporal sequence learning  

System will detect:

➡ Swipe gestures  
➡ Zoom gestures  
➡ Motion-based interaction  
➡ Continuous gesture tracking  

Transformation:

Static System → Intelligent Dynamic AI System
""")

        # Viva line
        st.markdown("### 🎤 Viva Answer")

        st.warning("""
Current system uses rule-based detection for real-time speed.

But architecture is designed to integrate:
- 3D CNN
- ST-CNN

making it scalable and future-ready.
""")

    # =========================
    # FOOTER
    # =========================
    st.markdown("---")
    st.caption(f"RT-Gesture3D • {time.strftime('%Y')} • Built with AI ❤️")

# =========================
# ENTRY
# =========================
if "streamlit" in sys.modules:
    run_streamlit_ui()
else:
    if __name__ == "__main__":
        run_cli()