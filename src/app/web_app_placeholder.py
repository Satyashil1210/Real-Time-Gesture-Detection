"""
Streamlit / CLI application layer for RT-Gesture3D.

Modes:
- CLI:
    python src/app/web_app_placeholder.py
  → Directly launches the OpenCV live gesture demo (no UI changes).

- Streamlit:
    streamlit run src/app/web_app_placeholder.py
  → Polished dashboard UI:
      - Dashboard tab (overview + launch button)
      - Gesture Library tab (avatars + ids)
      - Docs tab (architecture + viva points)

We do NOT change any detection logic.
"""

from pathlib import Path
import sys

# --------------------------------------------------
# Ensure project root is on PYTHONPATH (works for CLI + Streamlit)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVATARS_DIR = PROJECT_ROOT / "assets" / "avatars"

# --------------------------------------------------
# Import live demo + gesture mapping
# --------------------------------------------------
try:
    from src.inference.live_gesture_demo import main as run_live_demo
    from src.inference.mapping import GESTURES
except Exception as e:
    print("❌ Import error in web_app_placeholder.py")
    print("  PROJECT_ROOT:", PROJECT_ROOT)
    print("  Is PROJECT_ROOT in sys.path?:", str(PROJECT_ROOT) in sys.path)
    print("  Original error:", e)
    # Fallbacks so file at least doesn't explode on import
    GESTURES = {}
    def run_live_demo():
        print("⚠️ live_gesture_demo could not be imported. Check your src/inference folder.")
    # If you want Streamlit to crash instead of fallback, comment out the above
    # and uncomment this:
    # raise


# ------------------------------
# CLI mode (pure Python)
# ------------------------------
def run_cli():
    """Old behaviour: just run the live demo from the app layer."""
    print("RT-Gesture3D App Layer (CLI mode)")
    print("Launching live gesture demo from app module...\n")
    run_live_demo()


# ------------------------------
# Streamlit UI mode
# ------------------------------
def run_streamlit_ui():
    """
    Streamlit-based polished dashboard.

    Run with:
        streamlit run src/app/web_app_placeholder.py
    """
    import streamlit as st
    import time

    # --- Page config ---
    st.set_page_config(
        page_title="RT-Gesture3D – Real-time Gesture Recognition",
        page_icon="🖐️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Theme / styling ---
    PRIMARY = "#0B4F6C"        # deep teal
    ACCENT = "#F29E4C"         # warm amber
    MILD = "#F7F9FB"
    CARD_BG = "rgba(255,255,255,0.9)"
    TEXT = "#022B3A"

    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
        color: {TEXT};
    }}
    .stApp {{
      background: linear-gradient(180deg, #EBF5F7 0%, #FFFFFF 60%);
      min-height: 100vh;
    }}
    .block-container {{
      padding-top: 1.0rem;
      padding-bottom: 2.0rem;
    }}
    .main-card {{
      background: {CARD_BG};
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 6px 24px rgba(11,79,108,0.08);
    }}
    .side-panel {{
      background: linear-gradient(180deg, #FFFFFF 0%, #F7FBFC 100%);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }}
    .kpi-card {{
      background: rgba(255,255,255,0.9);
      padding: 10px 14px;
      border-radius: 10px;
      box-shadow: 0 3px 12px rgba(2,43,58,0.06);
    }}
    .small-muted {{font-size:12px; color:#5b6b72}}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # --- Header ---
    with st.container():
        left, right = st.columns([0.9, 0.1])
        with left:
            st.markdown(
                f"""
                <div style='display:flex;gap:14px;align-items:center'>
                  <div style='display:flex;flex-direction:column'>
                    <div style='font-weight:800;font-size:20px;color:{PRIMARY}'>
                      A Real-time Object & Gesture Recognition
                    </div>
                    <div style='font-size:13px;color:#4b636a'>
                      Advanced Synergistic Approach — Live hand gestures with avatar feedback
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div style='width:48px;height:48px;border-radius:12px;
                            background:radial-gradient(circle at 20% 20%, {ACCENT}, {PRIMARY});
                            box-shadow:0 4px 18px rgba(0,0,0,0.18);'>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Layout: sidebar (info + controls) + main area ---
    side, main = st.columns([1.2, 2.8])

    # ========== SIDEBAR ==========
    with side:
        st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
        st.subheader("Session Overview")

        # Static KPIs (placeholder – shows capability)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.metric("Input", "Webcam", "+ live")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.metric("Pipeline", "Rule-based", "real-time")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("**How the system works**")
        st.markdown(
            """
            - MediaPipe → 21 hand landmarks  
            - Custom rules → gesture class  
            - Mapping → avatar + description  
            """
        )

        st.divider()
        st.markdown("**Launch demo**")
        st.caption("Webcam-based live demo (same backend as CLI).")

        if st.button("🚀 Launch Live Camera Demo", use_container_width=True):
            st.info(
                "Launching OpenCV window... "
                "If you don't see it, check your taskbar and allow camera access."
            )
            run_live_demo()

        st.caption("Press **'q'** in the camera window to stop the demo.")

        st.divider()
        st.markdown("**Future scope**")
        st.caption(
            "- Collect dataset & train deep model\n"
            "- Integrate REST API / WebSocket\n"
            "- Add multi-hand / object support"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== MAIN AREA with TABS ==========
    with main:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)

        tab_dash, tab_gest, tab_docs = st.tabs(
            ["📊 Dashboard", "✋ Gesture Library", "📚 Docs / Viva Notes"]
        )

        # ----- Dashboard tab -----
        with tab_dash:
            upper = st.columns([2, 1.3])
            with upper[0]:
                st.markdown("### 🎬 Live Gesture Experience")

                st.markdown(
                    """
                    This interface is a **visual shell** around your existing
                    Python/OpenCV pipeline.

                    - Detection, gesture logic, and avatar overlay run in
                      `src/inference/live_gesture_demo.py`
                    - This page only provides UI, controls and documentation.
                    - Changing this UI does **not** affect model behaviour.
                    """
                )

                with st.expander("What exactly runs in the backend?"):
                    st.write(
                        """
                        - Capture frames from webcam using OpenCV  
                        - Run MediaPipe to get 21 landmarks for a single hand  
                        - Convert landmarks → finger state (extended / folded)  
                        - Apply rule-based mapping to classify gesture  
                        - Pick avatar & label from `datasets/gestures.csv`  
                        - Overlay text + avatar on the frame  
                        """
                    )

            with upper[1]:
                st.markdown("### 📌 Quick Facts")
                st.markdown(
                    f"""
                    - **Gestures supported:** {len(GESTURES) if GESTURES else 0}  
                    - **Mapping source:** `datasets/gestures.csv`  
                    - **Avatar folder:** `assets/avatars/`  
                    - **Engine:** MediaPipe + custom rule engine  
                    """
                )

                st.markdown("#### 🔁 Processing Flow")
                st.markdown(
                    """
                    1. Capture frame  
                    2. Detect hand landmarks  
                    3. Infer gesture class  
                    4. Render overlay + avatar  
                    """
                )

        # ----- Gesture Library tab -----
        with tab_gest:
            st.markdown("### ✋ Gesture Library & Avatars")

            if not GESTURES:
                st.warning("No gestures loaded. Check `datasets/gestures.csv` and mapping module.")
            else:
                cols = st.columns(3)
                idx = 0
                for key, info in GESTURES.items():
                    avatar_file = getattr(info, "avatar_file", None) or getattr(
                        info, "avatar", None
                    )
                    avatar_path = AVATARS_DIR / avatar_file if avatar_file else None

                    display_name = (
                        getattr(info, "label", None)
                        or getattr(info, "meaning", None)
                        or str(key)
                    )
                    meaning = getattr(info, "meaning", None)
                    gid = getattr(info, "id", "N/A")

                    with cols[idx % 3]:
                        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                        if avatar_path is not None and avatar_path.exists():
                            st.image(str(avatar_path), use_column_width=True)
                        else:
                            st.caption("No avatar image")

                        st.markdown(f"**{display_name}**")
                        if meaning:
                            st.caption(meaning)
                        st.code(f"id = {gid} | key = '{key}'", language="text")
                        st.markdown("</div>", unsafe_allow_html=True)

                    idx += 1

        # ----- Docs / Viva notes tab -----
        with tab_docs:
            st.markdown("### 📚 Project Documentation / Viva Pointers")

            st.markdown("#### 1. Problem Statement")
            st.markdown(
                """
                - Build a **real-time gesture recognition system**  
                - Uses a camera feed to understand human hand gestures  
                - Provides **visual feedback** via avatars and labels  
                """
            )

            st.markdown("#### 2. Tech Stack")
            st.markdown(
                """
                - **Language:** Python  
                - **Core libs:** OpenCV, MediaPipe, NumPy  
                - **Frontend / UI:** Streamlit  
                - **Architecture:** Modular (`src/` → capture, detection, processing, inference, app)  
                """
            )

            st.markdown("#### 3. Architecture (high level)")
            st.markdown(
                """
                1. **Capture layer** – reads frames from webcam  
                2. **Detection layer** – MediaPipe Hand module → 21 landmarks  
                3. **Processing layer** – converts landmarks to finger states  
                4. **Inference layer** – rule engine (`detect_gesture_from_landmarks`)  
                5. **Mapping layer** – `GESTURES` + `gestures.csv` map id → label → avatar  
                6. **App layer** – CLI / Streamlit UI for demo  
                """
            )

            st.markdown("#### 4. Viva friendly points")
            st.markdown(
                """
                - We intentionally kept the **model logic independent of UI**.  
                - The system is prepared for future **deep learning models** (ONNX / TFLite).  
                - Gesture configuration is **data-driven** via CSV instead of hard-coded.  
                - Codebase follows a **production-style folder structure**, so it can grow.  
                """
            )

            st.markdown("#### 5. Limitations & Future Work")
            st.markdown(
                """
                - Currently supports **single-hand** detection.  
                - Rule-based gestures; deep learned classifier can be added later.  
                - Works best in medium lighting; extreme lighting can affect detection.  
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ====== Footer ======
    st.markdown("---")
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        st.markdown(
            "Built with ❤️ using **Streamlit + OpenCV**. "
            "For production, containerize this app and put it behind a reverse proxy."
        )
    with fc2:
        st.markdown(
            f"<div class='small-muted'>RT-Gesture3D • v1.0 • {time.strftime('%Y')}</div>",
            unsafe_allow_html=True,
        )


# ------------------------------
# Entry behaviour
# ------------------------------
if "streamlit" in sys.modules:
    # Running under: streamlit run ...
    run_streamlit_ui()
else:
    # Normal Python execution:
    #   python src/app/web_app_placeholder.py
    if __name__ == "__main__":
        run_cli()
