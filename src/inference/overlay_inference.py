from typing import Dict

import cv2
import numpy as np

from .mapping import GESTURES, get_avatars_dir


def load_avatars(size=(150, 150)) -> Dict[str, np.ndarray]:
    """
    assets/avatars/ folder se avatars load karke dict[label_key] = image return karega.
    """
    avatars_dir = get_avatars_dir()
    avatars: Dict[str, np.ndarray] = {}

    print(f"🖼  Loading avatars from: {avatars_dir}")

    for key, info in GESTURES.items():
        path = avatars_dir / info.avatar_file
        if not path.exists():
            print(f"⚠️ Avatar file missing for '{key}': {path}")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Could not read avatar image for '{key}': {path}")
            continue

        img = cv2.resize(img, size)
        avatars[key] = img

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


def overlay_gesture_text(frame, gesture_key: str, confidence: float):
    """
    Gesture label + meaning text show karta hai.
    """
    info = GESTURES.get(gesture_key)
    if info is None:
        label_line = f"{gesture_key} ({confidence:.2f})"
        meaning_line = ""
    else:
        label_line = f"{info.key} ({confidence:.2f})"
        meaning_line = info.meaning

    # main label
    cv2.putText(
        frame,
        label_line,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # meaning (optional)
    if meaning_line:
        cv2.putText(
            frame,
            meaning_line,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame
