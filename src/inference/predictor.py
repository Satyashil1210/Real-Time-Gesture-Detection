import math
from typing import List, Tuple

from .mapping import GESTURES, ID_TO_KEY


Point3D = Tuple[int, int, float]  # (x, y, z)


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended_states(landmarks: List[Point3D]):
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


def detect_gesture_from_landmarks(
    pts: List[Point3D],
    img_w: int,
    img_h: int,
):
    """
    Core heuristic-based gesture detector.

    Returns:
        gesture_key: str   (e.g. "rock", "ok", "neutral")
        gesture_id: int    (0..6)
        confidence: float  (0..1)
    """
    st = finger_extended_states(pts)
    ext_count = sum(st.values())

    tip = lambda idx: (pts[idx][0], pts[idx][1])
    thumb_tip = tip(4)
    index_tip = tip(8)

    # thumb–index distance for perfect
    d_thumb_index = _dist(thumb_tip, index_tip)
    scale_thresh = max(40, int(img_w * 0.07))   # ~7% of width or min 40px

    # PERFECT (👌) => thumb & index extended and close
    if d_thumb_index < scale_thresh and st["thumb"] and st["index"]:
        key = ID_TO_KEY[3]
        return key, 3, 0.95

    # STOP (🤚🏻) => open palm
    non_thumb_ext = [st["index"], st["middle"], st["ring"], st["pinky"]]
    if sum(non_thumb_ext) >= 4:
        key = ID_TO_KEY[4]
        return key, 4, 0.9

    # ROCK (🤘) => index + pinky extended, middle & ring folded
    if st["index"] and st["pinky"] and (not st["middle"]) and (not st["ring"]):
        key = ID_TO_KEY[5]
        return key, 5, 0.9

    # VICTORY (✌️) => index + middle extended, ring & pinky folded
    if st["index"] and st["middle"] and (not st["ring"]) and (not st["pinky"]):
        key = ID_TO_KEY[1]
        return key, 1, 0.92

    # CALM (☝🏻) => only index (thumb allowed)
    if st["index"] and (not st["middle"]) and (not st["ring"]) and (not st["pinky"]):
        key = ID_TO_KEY[6]
        return key, 6, 0.9

    # OK / thumbs-up (👍🏻) => only thumb
    if st["thumb"] and (not st["index"]) and (not st["middle"]) and (not st["ring"]) and (not st["pinky"]):
        key = ID_TO_KEY[2]
        return key, 2, 0.9

    # fallback: agar sirf 1 finger extended hai
    if ext_count == 1:
        if st["thumb"]:
            key = ID_TO_KEY[2]
            return key, 2, 0.8
        if st["index"]:
            key = ID_TO_KEY[6]
            return key, 6, 0.8
        if st["pinky"]:
            key = ID_TO_KEY[5]
            return key, 5, 0.75

    # Neutral / fist
    key = ID_TO_KEY[0]
    return key, 0, 0.5
