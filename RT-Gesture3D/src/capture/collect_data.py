"""
collect_data.py
================
Webcam-based gesture dataset collector for RT-Gesture3D.

Usage:
    python src/capture/collect_data.py

Controls (during recording):
    SPACE  → start / stop recording a clip
    n      → next gesture class
    p      → previous gesture class
    d      → delete last saved clip
    q      → quit

Output:
    data/raw/<gesture_name>/clip_XXXX.npy   (T, H, W, C)  uint8
    data/raw/<gesture_name>/clip_XXXX.mp4   (optional preview)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── config ──────────────────────────────────────────────────────────────────
GESTURE_CLASSES = ["neutral", "victory", "ok", "perfect", "stop", "rock", "calm"]
CLIPS_PER_CLASS = 60        # target clips per gesture
FRAMES_PER_CLIP = 16        # frames captured per clip
FRAME_SIZE      = (112, 112)  # resize before saving (matches 3D-CNN input)
FPS_CAP         = 15         # capture fps cap
DATA_DIR        = ROOT / "data" / "raw"
COUNTDOWN_SEC   = 3


# ── helpers ─────────────────────────────────────────────────────────────────

def _clip_path(gesture: str, idx: int) -> Path:
    d = DATA_DIR / gesture
    d.mkdir(parents=True, exist_ok=True)
    return d / f"clip_{idx:04d}.npy"


def _count_existing(gesture: str) -> int:
    d = DATA_DIR / gesture
    if not d.exists():
        return 0
    return len(list(d.glob("clip_*.npy")))


def _draw_overlay(frame, gesture, cls_idx, total_cls,
                  count, target, recording, countdown, msg=""):
    h, w = frame.shape[:2]

    # top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.putText(frame, f"RT-Gesture3D  |  Data Collector",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)

    # gesture label
    colour = (0, 255, 80) if recording else (200, 200, 200)
    cv2.putText(frame,
                f"[{cls_idx+1}/{total_cls}]  {gesture.upper()}",
                (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)

    # clip counter
    pct = min(count / target, 1.0)
    bar_w = int((w - 24) * pct)
    cv2.rectangle(frame, (12, 110), (w - 12, 128), (40, 40, 40), -1)
    cv2.rectangle(frame, (12, 110), (12 + bar_w, 128), (0, 200, 80), -1)
    cv2.putText(frame, f"{count}/{target} clips",
                (12, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    # recording indicator
    if recording:
        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC",
                    (w - 65, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # countdown
    if countdown > 0:
        cv2.putText(frame, str(countdown),
                    (w // 2 - 30, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 6)

    # status message
    if msg:
        cv2.putText(frame, msg,
                    (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # key hints
    hints = "[SPACE] record  [n/p] gesture  [d] delete  [q] quit"
    cv2.putText(frame, hints,
                (12, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not found"); return

    print("✅ Camera ready")
    print(f"📁 Saving to: {DATA_DIR}")
    print(f"🎯 Target: {CLIPS_PER_CLASS} clips × {len(GESTURE_CLASSES)} gestures")

    cls_idx   = 0
    recording = False
    buffer    = []
    countdown = 0
    msg       = ""
    last_save = ""

    frame_interval = 1.0 / FPS_CAP
    last_frame_t   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        gesture = GESTURE_CLASSES[cls_idx]
        count   = _count_existing(gesture)

        now = time.time()

        # ── capture frames while recording ──────────────────────────────────
        if recording and (now - last_frame_t) >= frame_interval:
            small = cv2.resize(frame, FRAME_SIZE)
            buffer.append(small.copy())
            last_frame_t = now

            if len(buffer) >= FRAMES_PER_CLIP:
                # save clip
                idx  = count
                path = _clip_path(gesture, idx)
                arr  = np.stack(buffer, axis=0)   # (T, H, W, C)
                np.save(str(path), arr)
                last_save = f"✅ Saved clip_{idx:04d}  ({gesture})"
                msg = last_save
                print(last_save)
                buffer.clear()
                recording = False
                count += 1

        # ── draw UI ─────────────────────────────────────────────────────────
        display = frame.copy()
        display = _draw_overlay(display, gesture, cls_idx,
                                len(GESTURE_CLASSES),
                                count, CLIPS_PER_CLASS,
                                recording, countdown, msg)
        cv2.imshow("RT-Gesture3D  |  Data Collector", display)

        # ── keyboard ────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if not recording:
                recording = True
                buffer.clear()
                msg = "⏺ Recording…"
            else:
                # manual stop — discard partial
                recording = False
                buffer.clear()
                msg = "⏹ Stopped (clip discarded)"

        elif key == ord('n'):
            cls_idx = (cls_idx + 1) % len(GESTURE_CLASSES)
            recording = False; buffer.clear()
            msg = f"➡ {GESTURE_CLASSES[cls_idx]}"

        elif key == ord('p'):
            cls_idx = (cls_idx - 1) % len(GESTURE_CLASSES)
            recording = False; buffer.clear()
            msg = f"⬅ {GESTURE_CLASSES[cls_idx]}"

        elif key == ord('d'):
            # delete last saved clip
            existing = sorted((DATA_DIR / gesture).glob("clip_*.npy"))
            if existing:
                existing[-1].unlink()
                msg = f"🗑 Deleted {existing[-1].name}"
            else:
                msg = "Nothing to delete"

    cap.release()
    cv2.destroyAllWindows()

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n📊 Dataset summary:")
    total = 0
    for g in GESTURE_CLASSES:
        c = _count_existing(g)
        total += c
        bar = "█" * c + "░" * max(0, CLIPS_PER_CLASS - c)
        print(f"  {g:<10} {bar}  {c}/{CLIPS_PER_CLASS}")
    print(f"\n  Total clips : {total}")
    print(f"  Total frames: {total * FRAMES_PER_CLIP}")


if __name__ == "__main__":
    main()
