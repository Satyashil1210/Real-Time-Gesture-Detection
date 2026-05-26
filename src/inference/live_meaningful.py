"""
live_meaningful.py — RT-Gesture3D v2
======================================
Default: Simple gesture + object UI (purana wala)
[1/2/3]: Domain meaning panel on (Hospital/SignLang/SmartHome)
[t]    : Temporal 3D CNN on/off
[0]    : Wapis simple mode

Controls:
  [g] gesture on/off      [o] objects on/off    [t] temporal on/off
  [1] Hospital domain     [2] Sign Language      [3] Smart Home
  [0] Simple mode (domain off)
  [p] pause               [r] resume             [q] quit
"""

from __future__ import annotations
import sys, math
from collections import deque, Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.gesture_registry   import get_info, GESTURE_KEYS, ID_TO_KEY
from src.inference.meaningful_overlay import draw_full_ui, draw_objects, ALERT_SYS
from src.inference.temporal_model     import TemporalGestureModel
from src.processing.buffer            import FrameBuffer
from src.detection.object_detector    import ObjectDetector
from src.inference.overlay_inference  import (
    overlay_gesture_text, overlay_avatar,
    draw_ui_panel, draw_status, load_avatars,
)


# ── rule-based gesture predictor ─────────────────────────────────────────────
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_states(lm):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    st   = [False] * 5
    for i in range(1, 5):
        st[i] = lm[tips[i]][1] < lm[pips[i]][1] - 2
    wx  = lm[0][0]; ttx = lm[4][0]; tipx = lm[3][0]
    st[0] = abs(ttx - wx) > 25 or abs(ttx - tipx) > 12
    return st


def predict_rule(pts, w, h):
    if not pts or len(pts) < 21:
        return "neutral", 0.5
    st  = _finger_states(pts)
    d   = _dist((pts[4][0], pts[4][1]), (pts[8][0], pts[8][1]))
    thr = max(35, int(w * 0.06))

    if d < thr and st[0] and st[1]:                         return "perfect", 0.95
    if sum(st[1:]) >= 4:                                    return "stop",    0.90
    if st[1] and st[4] and not st[2] and not st[3]:         return "rock",    0.90
    if st[1] and st[2] and not st[3] and not st[4]:         return "victory", 0.95
    if st[1] and not st[2] and not st[3] and not st[4]:     return "calm",    0.90
    if st[0] and not st[1] and not st[2] and not st[3] and not st[4]:
                                                             return "ok",      0.90
    return "neutral", 0.50


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("▶️  RT-Gesture3D v2 — Meaningful System")
    print("   Default : Simple gesture + object (purana wala)")
    print("   [1/2/3] : Domain meaning panel on")
    print("   [0]     : Wapis simple mode")
    print("   [t]     : Temporal 3D CNN on/off\n")

    # ── init ─────────────────────────────────────────────────────────────
    avatars        = load_avatars(size=(150, 150))
    temporal_model = TemporalGestureModel()
    buf            = FrameBuffer(window=16)
    obj_detector   = ObjectDetector()

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ Camera not found"); return

    print(f"✅ Camera ready  |  Temporal: {temporal_model.model_name()}")
    print("🎮 [g/o/t] toggle  [1/2/3] domain  [0] simple  [p/r] pause  [q] quit\n")

    # ── state ─────────────────────────────────────────────────────────────
    gesture_enabled  = True
    object_enabled   = True
    temporal_enabled = False   # [t] se on karo
    domain_active    = False   # [1/2/3] se on, [0] se off

    paused = False
    domain = "hospital"

    fc          = 0
    cached_objs = []
    history     = deque(maxlen=7)
    slabel      = "neutral";   sconf = 0.0
    tlabel      = "buffering"; tconf = 0.0

    DOMAIN_MAP = {"1": "hospital", "2": "signlang", "3": "smarthome"}

    # ── main loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── keyboard ──────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF

        if   k == ord('q'): break
        elif k == ord('g'): gesture_enabled  = not gesture_enabled
        elif k == ord('o'): object_enabled   = not object_enabled
        elif k == ord('t'):
            temporal_enabled = not temporal_enabled
            print(f"   Temporal → {'ON ✅' if temporal_enabled else 'OFF ❌'}")
        elif k == ord('p'): paused = True
        elif k == ord('r'): paused = False
        elif k == ord('0'):
            domain_active = False
            print("   Mode → Simple UI")
        elif k < 128 and chr(k) in DOMAIN_MAP:
            domain        = DOMAIN_MAP[chr(k)]
            domain_active = True
            print(f"   Domain → {domain}")

        # ── pause ─────────────────────────────────────────────────────────
        if paused:
            _draw_paused(frame, w, h)
            cv2.imshow("RT-Gesture3D v2", frame)
            continue

        # ── object detection ──────────────────────────────────────────────
        if object_enabled:
            fc += 1
            if fc % 5 == 0:
                cached_objs = obj_detector.detect(frame)
            frame = draw_objects(frame, cached_objs)

        # ── hand landmarks + rule-based ───────────────────────────────────
        if gesture_enabled:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                for hlm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                    pts = [(int(lm.x * w), int(lm.y * h), lm.z)
                           for lm in hlm.landmark]
                    cl, cc = predict_rule(pts, w, h)
                    history.append(cl)
                slabel = Counter(history).most_common(1)[0][0]
                sconf  = cc

        # ── temporal model ────────────────────────────────────────────────
        buf.push(frame)
        if temporal_enabled and buf.is_ready() and buf.has_motion():
            clip_frames = [
                (buf.get_clip()[i] * 255).astype("uint8")
                for i in range(buf.get_clip().shape[0])
            ]
            tlabel, tconf = temporal_model.predict(clip_frames)

        # ── final label fusion ────────────────────────────────────────────
        _t_valid = tlabel not in (
            "buffering", "none", "error", "neutral",
            "processing", "dynamic_ready"
        )
        if temporal_enabled and tconf > 0.75 and _t_valid:
            final_label = tlabel
            final_conf  = tconf
        else:
            final_label = slabel
            final_conf  = sconf

        # ── UI ────────────────────────────────────────────────────────────
        if domain_active:
            # meaningful domain UI — hospital / signlang / smarthome
            frame = draw_full_ui(
                frame,
                final_label, final_conf,
                tlabel, tconf,
                cached_objs if object_enabled else [],
                domain,
                gesture_enabled, object_enabled, temporal_enabled, paused,
            )
        else:
            # simple UI — bilkul purana wala
            frame = draw_ui_panel(frame)
            if gesture_enabled:
                frame = overlay_gesture_text(frame, final_label, final_conf)
                av = avatars.get(final_label)
                if av is not None:
                    frame = overlay_avatar(frame, av)
            frame = draw_status(frame, gesture_enabled, object_enabled, paused)

        cv2.imshow("RT-Gesture3D v2", frame)

    # ── cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


# ── helpers ──────────────────────────────────────────────────────────────────
def _draw_paused(frame, w, h):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "PAUSED — press [r] to resume",
                (w // 2 - 230, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


if __name__ == "__main__":
    main()