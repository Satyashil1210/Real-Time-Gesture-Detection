import math
from typing import List, Tuple

from .mapping import GESTURES, ID_TO_KEY


Point3D = Tuple[int, int, float]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended_states(landmarks: List[Point3D]):

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

    # 🔥 MORE STABLE (adaptive threshold)
    for name in ["index", "middle", "ring", "pinky"]:
        tip_id = tips_idx[name]
        pip_id = pips_idx[name]

        tip_y = landmarks[tip_id][1]
        pip_y = landmarks[pip_id][1]

        # adaptive margin
        states[name] = (tip_y < pip_y - 2)

    # 🔥 IMPROVED THUMB LOGIC
    wrist_x = landmarks[0][0]
    thumb_tip_x = landmarks[tips_idx["thumb"]][0]
    thumb_ip_x = landmarks[pips_idx["thumb"]][0]

    if abs(thumb_tip_x - wrist_x) > 25:
        states["thumb"] = True
    else:
        states["thumb"] = abs(thumb_tip_x - thumb_ip_x) > 12

    return states


def detect_gesture_from_landmarks(pts, img_w, img_h):

    if not pts or len(pts) < 21:
        return 0, ID_TO_KEY[0], 0.5

    st = finger_extended_states(pts)
    ext_count = sum(st.values())

    tip = lambda idx: (pts[idx][0], pts[idx][1])
    thumb_tip = tip(4)
    index_tip = tip(8)

    d_thumb_index = _dist(thumb_tip, index_tip)

    # 🔥 adaptive scaling (better for different distances)
    scale_thresh = max(35, int(img_w * 0.06))

    # =========================
    # 👌 PERFECT
    # =========================
    if d_thumb_index < scale_thresh and st["thumb"] and st["index"]:
        return 3, ID_TO_KEY[3], 0.95

    # =========================
    # ✋ STOP
    # =========================
    if sum([st["index"], st["middle"], st["ring"], st["pinky"]]) >= 4:
        return 4, ID_TO_KEY[4], 0.9

    # =========================
    # 🤘 ROCK
    # =========================
    if st["index"] and st["pinky"] and (not st["middle"]) and (not st["ring"]):
        return 5, ID_TO_KEY[5], 0.9

    # =========================
    # ✌️ VICTORY
    # =========================
    if st["index"] and st["middle"]:

        if not st["ring"] and not st["pinky"]:
            return 1, ID_TO_KEY[1], 0.95

        if not st["ring"] or not st["pinky"]:
            return 1, ID_TO_KEY[1], 0.9

    # =========================
    # ☝️ CALM
    # =========================
    if st["index"] and not st["middle"] and not st["ring"] and not st["pinky"]:
        return 6, ID_TO_KEY[6], 0.9

    # =========================
    # 👍 OK
    # =========================
    if st["thumb"] and not st["index"] and not st["middle"] and not st["ring"] and not st["pinky"]:
        return 2, ID_TO_KEY[2], 0.9

    # =========================
    # 🔁 FALLBACK (stable)
    # =========================
    if ext_count == 1:
        if st["thumb"]:
            return 2, ID_TO_KEY[2], 0.8
        if st["index"]:
            return 6, ID_TO_KEY[6], 0.8
        if st["pinky"]:
            return 5, ID_TO_KEY[5], 0.75

    return 0, ID_TO_KEY[0], 0.5