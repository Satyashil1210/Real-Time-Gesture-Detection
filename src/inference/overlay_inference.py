from typing import Dict
import cv2
import numpy as np
from pathlib import Path

from .mapping import GESTURES, get_avatars_dir


# =========================
# 🖼 LOAD AVATARS
# =========================
def load_avatars(size=(150, 150)) -> Dict[str, np.ndarray]:

    avatars_dir = get_avatars_dir()
    avatars: Dict[str, np.ndarray] = {}

    print(f"🖼 Loading avatars from: {avatars_dir}")

    for key, info in GESTURES.items():
        path = avatars_dir / info.avatar_file

        if not path.exists():
            print(f"⚠️ Missing avatar: {path}")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"⚠️ Failed to load: {path}")
            continue

        avatars[key] = cv2.resize(img, size)

    print(f"✅ Loaded {len(avatars)} avatars")
    return avatars


# =========================
# 🎯 UI PANEL
# =========================
def draw_ui_panel(frame):

    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)

    cv2.putText(
        frame,
        "RT-Gesture3D | AI System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Left panel
    cv2.rectangle(frame, (0, 60), (160, h), (20, 20, 20), -1)

    # Right panel
    cv2.rectangle(frame, (w - 200, 60), (w, h), (20, 20, 20), -1)

    return frame


# =========================
# 🧍 AVATAR OVERLAY
# =========================
def overlay_avatar(frame, avatar_img):

    if avatar_img is None:
        return frame

    fh, fw = frame.shape[:2]
    ah, aw = avatar_img.shape[:2]

    x2 = fw - 20
    x1 = x2 - aw
    y1 = 80
    y2 = y1 + ah

    # Background
    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (40, 40, 40), -1)

    # Border
    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 255), 2)

    frame[y1:y2, x1:x2] = avatar_img
    return frame


# =========================
# ✋ GESTURE UI
# =========================
def overlay_gesture_text(frame, gesture_key: str, confidence: float):

    info = GESTURES.get(gesture_key)

    label = info.key.upper() if info else gesture_key.upper()
    meaning = info.meaning if info else ""

    # Label
    cv2.putText(
        frame,
        label,
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Meaning
    if meaning:
        cv2.putText(
            frame,
            meaning,
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Confidence bar
    bar_w = int(200 * confidence)

    cv2.rectangle(frame, (20, 220), (220, 240), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 220), (20 + bar_w, 240), (0, 255, 0), -1)

    return frame


# =========================
# 🎨 COLOR SYSTEM (NEW 🔥)
# =========================
def get_color(label):
    color_map = {
        "person": (0, 255, 0),
        "fan": (255, 0, 0),
        "book": (0, 255, 255),
        "mobile": (255, 255, 0),
        "laptop": (255, 0, 255),
        "bottle": (0, 128, 255),
        "chair": (0, 0, 255),
        "bag": (128, 0, 128),
        "projector": (255, 100, 0),
        "switch board": (200, 200, 0),
    }
    return color_map.get(label, (180, 180, 180))


# =========================
# 📦 OBJECT OVERLAY (UPDATED)
# =========================
def overlay_objects(frame, objects):

    if not objects:
        return frame

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        conf = obj["confidence"]

        if conf < 0.3:
            continue

        color = get_color(label)

        # BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # LABEL BG
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), color, -1)

        # TEXT
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


# =========================
# 📊 STATUS PANEL
# =========================
def draw_status(frame, gesture_enabled, object_enabled, paused):

    text = f"G:{gesture_enabled}  O:{object_enabled}  P:{paused}"

    cv2.putText(
        frame,
        text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    return frame