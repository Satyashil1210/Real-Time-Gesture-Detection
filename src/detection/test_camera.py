import cv2
import mediapipe as mp
import math
from pathlib import Path


# ==============================
# Gesture helper functions
# ==============================

def _dist(a, b):
    """Euclidean distance between two (x,y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended_states(landmarks):
    """
    landmarks: list of 21 (x,y,z) in pixel coords
    returns: dict -> thumb, index, middle, ring, pinky (True = extended)
    """
    tips_idx = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    pips_idx = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }

    states = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

    # Index..pinky: tip y < pip y => extended (camera upright)
    for name in ["index", "middle", "ring", "pinky"]:
        tip_id = tips_idx[name]
        pip_id = pips_idx[name]
        tip_y = landmarks[tip_id][1]
        pip_y = landmarks[pip_id][1]
        states[name] = (tip_y < pip_y - 5)

    # Thumb: horizontal distance from wrist + IP joint
    wrist_x = landmarks[0][0]
    thumb_tip_x = landmarks[tips_idx["thumb"]][0]
    thumb_ip_x = landmarks[pips_idx["thumb"]][0]

    if abs(thumb_tip_x - wrist_x) > 30:
        states["thumb"] = True
    else:
        states["thumb"] = abs(thumb_tip_x - thumb_ip_x) > 20

    return states


def detect_gesture_from_landmarks(pts, img_w, img_h):
    """
    Returns (gid, label, confidence)

    ID mapping:
      0 - neutral
      1 - victory (✌️)
      2 - ok (👍🏻)
      3 - perfect (👌🏻)
      4 - stop (🤚🏻)
      5 - rock (🤘🏻)
      6 - calm (☝🏻)
    """
    st = finger_extended_states(pts)
    ext_count = sum(st.values())

    tip = lambda idx: (pts[idx][0], pts[idx][1])
    thumb_tip = tip(4)
    index_tip = tip(8)

    # thumb–index distance for perfect
    d_thumb_index = _dist(thumb_tip, index_tip)
    scale_thresh = max(40, int(img_w * 0.07))   # ~7% of width or min 40px

    # PERFECT (👌)
    if d_thumb_index < scale_thresh and st["thumb"] and st["index"]:
        return 3, "perfect", 0.95

    # STOP (🤚🏻)
    non_thumb_ext = [st["index"], st["middle"], st["ring"], st["pinky"]]
    if sum(non_thumb_ext) >= 4:
        return 4, "stop", 0.9

    # ROCK (🤘)
    if st["index"] and st["pinky"] and (not st["middle"]) and (not st["ring"]):
        return 5, "rock", 0.9

    # VICTORY (✌️)
    if st["index"] and st["middle"] and (not st["ring"]) and (not st["pinky"]):
        return 1, "victory", 0.92

    # CALM (☝🏻)
    if st["index"] and (not st["middle"]) and (not st["ring"]) and (not st["pinky"]):
        return 6, "calm", 0.9

    # OK / thumbs-up (👍🏻)
    if st["thumb"] and (not st["index"]) and (not st["middle"]) and (not st["ring"]) and (not st["pinky"]):
        return 2, "ok", 0.9

    # fallback: agar sirf 1 finger extended hai
    if ext_count == 1:
        if st["thumb"]:
            return 2, "ok", 0.8
        if st["index"]:
            return 6, "calm", 0.8
        if st["pinky"]:
            return 5, "rock", 0.75

    return 0, "neutral", 0.5


# ==============================
# Avatar handling
# ==============================

AVATAR_FILES = {
    "neutral": "neutral.jpg",
    "victory": "victory.jpg",
    "ok": "ok.jpg",
    "perfect": "perfect.jpg",
    "stop": "stop.jpg",
    "rock": "rock.jpg",
    "calm": "calm.jpg",
}


def load_avatars(size=(150, 150)):
    """
    assets/avtar/ folder se avatars load karke dict[label] = image return karega.
    """
    # this file = src/inference/live_gesture_demo.py
    # parents[2] = project root (RT-Gesture3D)
    base_dir = Path(__file__).resolve().parents[2]

    # 👇 yahi main fix hai
    assets_dir = base_dir / "assets" / "avtar"

    avatars = {}
    print(f"🖼  Loading avatars from: {assets_dir}")

    for label, fname in AVATAR_FILES.items():
        path = assets_dir / fname
        if not path.exists():
            print(f"⚠️ Avatar file missing for '{label}': {path}")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Could not read avatar image for '{label}': {path}")
            continue

        img = cv2.resize(img, size)
        avatars[label] = img

    print(f"✅ Loaded {len(avatars)} avatar(s).")
    return avatars


def overlay_avatar(frame, avatar_img):
    """
    Avatar ko frame ke top-right corner me paste karega.
    """
    if avatar_img is None:
        return frame

    fh, fw = frame.shape[:2]
    ah, aw = avatar_img.shape[:2]

    # top-right corner with 10px margin
    x2 = fw - 10
    x1 = x2 - aw
    y1 = 10
    y2 = y1 + ah

    if x1 < 0 or y2 > fh:
        return frame

    frame[y1:y2, x1:x2] = avatar_img
    return frame


# ==============================
# Main live loop
# ==============================

def main():
    print("▶️ Starting RT-Gesture3D demo...")

    avatars = load_avatars(size=(150, 150))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Camera could not be opened. Check if another app is using it.")
        return

    print("✅ Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape
        label_text = "neutral"
        conf_text = 0.0

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # landmarks → pixel coords list
                pts = []
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append((x, y, lm.z))

                gid, label, conf = detect_gesture_from_landmarks(pts, w, h)
                label_text = label
                conf_text = conf

        # Text overlay
        cv2.putText(
            frame,
            f"{label_text} ({conf_text:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Avatar overlay
        avatar_img = avatars.get(label_text)
        if avatar_img is not None:
            frame = overlay_avatar(frame, avatar_img)

        cv2.imshow("RT-Gesture3D - Live Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Q pressed, exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")


if __name__ == "__main__":
    main()
