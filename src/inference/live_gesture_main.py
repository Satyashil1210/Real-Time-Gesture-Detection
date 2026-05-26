"""
live_gesture_main.py  —  RT-Gesture3D
=======================================
Complete real-time pipeline integrating:
  • YOLOv8  object detection
  • MediaPipe  hand landmarks
  • Rule-based gesture predictor
  • 3D-CNN / ST-CNN  temporal model (auto-loads if trained)
  • FrameBuffer  sliding window

Controls
--------
  g  toggle gesture recognition
  o  toggle object detection
  t  toggle temporal model display
  p  pause
  r  resume
  q  quit
"""

from __future__ import annotations

import sys
from collections import deque, Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predictor       import detect_gesture_from_landmarks
from src.inference.overlay_inference import (
    load_avatars, overlay_avatar, overlay_gesture_text,
    overlay_objects, draw_ui_panel, draw_status,
)
from src.detection.object_detector  import ObjectDetector
from src.inference.temporal_model   import TemporalGestureModel
from src.processing.buffer          import FrameBuffer

import mediapipe as mp


# ── tunables ─────────────────────────────────────────────────────────────────
CAM_W      = 1280
CAM_H      = 720
MAX_SEQ    = 16          # frames fed to temporal model
SMOOTH_WIN = 7           # majority-vote window for rule-based gesture
OBJ_EVERY  = 5           # run YOLO every N frames


def main():
    print("▶️  Starting RT-Gesture3D …")

    # ── models ───────────────────────────────────────────────────────────────
    avatars        = load_avatars(size=(180, 180))
    obj_detector   = ObjectDetector()
    temporal_model = TemporalGestureModel()
    frame_buffer   = FrameBuffer(window=MAX_SEQ, size=(112, 112))

    # ── mediapipe ────────────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    # ── camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("❌ Camera failed"); return

    print("✅ Camera ready")
    print(f"   Temporal model : {temporal_model.model_name()}")
    print("🎮 Controls: [g] gesture  [o] objects  [t] temporal  [p] pause  [r] resume  [q] quit")

    # ── state ─────────────────────────────────────────────────────────────────
    gesture_enabled  = True
    object_enabled   = True
    temporal_enabled = True
    paused           = False

    frame_count    = 0
    cached_objects = []
    label_history  = deque(maxlen=SMOOTH_WIN)

    stable_label    = "neutral"
    rule_conf       = 0.0
    temporal_label  = "buffering"
    temporal_conf   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── UI chrome ─────────────────────────────────────────────────────
        frame = draw_ui_panel(frame)

        # ── keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('g'): gesture_enabled  = not gesture_enabled
        elif key == ord('o'): object_enabled   = not object_enabled
        elif key == ord('t'): temporal_enabled = not temporal_enabled
        elif key == ord('p'): paused = True
        elif key == ord('r'): paused = False

        if paused:
            _draw_paused(frame, w, h)
            cv2.imshow("RT-Gesture3D (HD)", frame)
            continue

        # ── object detection ──────────────────────────────────────────────
        if object_enabled:
            frame_count += 1
            if frame_count % OBJ_EVERY == 0:
                cached_objects = obj_detector.detect(frame)
            frame = overlay_objects(frame, cached_objects)

        # ── hand landmark + rule-based gesture ────────────────────────────
        if gesture_enabled:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                curr_label = "neutral"
                curr_conf  = 0.0

                for handLms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    pts = [(int(lm.x * w), int(lm.y * h), lm.z)
                           for lm in handLms.landmark]
                    _, curr_label, curr_conf = detect_gesture_from_landmarks(pts, w, h)

                label_history.append(curr_label)
                stable_label = Counter(label_history).most_common(1)[0][0]
                rule_conf    = curr_conf

        # ── frame buffer + temporal model ─────────────────────────────────
        frame_buffer.push(frame)

        if temporal_enabled and frame_buffer.is_ready():
            if frame_buffer.has_motion():
                sequence = _buffer_to_bgr_list(frame_buffer)
                temporal_label, temporal_conf = temporal_model.predict(sequence)

        # ── gesture overlay ───────────────────────────────────────────────
        if gesture_enabled:
            frame = overlay_gesture_text(frame, stable_label, rule_conf)
            avatar_img = avatars.get(stable_label)
            if avatar_img is not None:
                frame = overlay_avatar(frame, avatar_img)

        # ── temporal model overlay ─────────────────────────────────────────
        if temporal_enabled:
            frame = _draw_temporal_info(
                frame, temporal_label, temporal_conf,
                trained=temporal_model.is_trained(),
                model_name=temporal_model.model_name(),
            )

        # ── status bar ────────────────────────────────────────────────────
        frame = draw_status(frame, gesture_enabled, object_enabled, paused)

        cv2.imshow("RT-Gesture3D (HD)", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _buffer_to_bgr_list(buf: FrameBuffer):
    """Convert normalised float buffer back to uint8 BGR list."""
    import numpy as np
    clip = buf.get_clip()                          # (T, H, W, 3) float32
    return [(clip[i] * 255).astype("uint8") for i in range(clip.shape[0])]


def _draw_paused(frame, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "PAUSED — press [r] to resume",
                (w // 2 - 220, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


def _draw_temporal_info(frame, label, conf, trained, model_name):
    h, w = frame.shape[:2]
    panel_x = w - 195

    # background tile
    cv2.rectangle(frame, (panel_x, h - 120), (w - 5, h - 5), (20, 20, 20), -1)
    cv2.rectangle(frame, (panel_x, h - 120), (w - 5, h - 5), (80, 80, 80), 1)

    status_colour = (0, 200, 255) if trained else (100, 100, 100)

    cv2.putText(frame, f"Temporal ({model_name})",
                (panel_x + 5, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_colour, 1)

    cv2.putText(frame, label.upper(),
                (panel_x + 5, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_colour, 2)

    # confidence bar
    bar_w = int(180 * min(conf, 1.0))
    cv2.rectangle(frame, (panel_x + 5, h - 50), (panel_x + 185, h - 35), (40, 40, 40), -1)
    cv2.rectangle(frame, (panel_x + 5, h - 50), (panel_x + 5 + bar_w, h - 35), status_colour, -1)

    cv2.putText(frame, f"{conf:.0%}",
                (panel_x + 5, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if not trained:
        cv2.putText(frame, "train model first",
                    (panel_x + 5, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    return frame


if __name__ == "__main__":
    main()
