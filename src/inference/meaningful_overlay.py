"""
meaningful_overlay.py — RT-Gesture3D v2
=========================================
Complete UI rendering with domain-aware meaningful output.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import List

import cv2
import numpy as np


# ── lazy import to avoid circular ────────────────────────────────────────────
def _get_registry():
    from src.inference.gesture_registry import (
        get_info, CRITICAL_GESTURES, WARNING_GESTURES
    )
    return get_info, CRITICAL_GESTURES, WARNING_GESTURES


# ── Alert system ──────────────────────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self._log          = deque(maxlen=4)
        self._last_gesture = "neutral"
        self._alert_start  = 0.0
        self._alert_active = False
        self._alert_level  = "normal"

    def update(self, gesture: str, conf: float):
        if gesture == self._last_gesture:
            return
        self._last_gesture = gesture

        get_info, _, _ = _get_registry()
        info = get_info(gesture)
        ts   = time.strftime("%H:%M:%S")
        self._log.appendleft(f"{ts}  {info.icon} {info.display_name}")

        if info.alert_level in ("critical", "warning"):
            self._alert_active = True
            self._alert_start  = time.time()
            self._alert_level  = info.alert_level
        elif time.time() - self._alert_start > 5:
            self._alert_active = False

    def is_critical(self) -> bool:
        return self._alert_active and self._alert_level == "critical"

    def is_warning(self) -> bool:
        return self._alert_active and self._alert_level == "warning"

    def log(self) -> List[str]:
        return list(self._log)


# global alert instance
ALERT_SYS = AlertSystem()

# ── domains ───────────────────────────────────────────────────────────────────
DOMAINS = ["hospital", "signlang", "smarthome"]
DOMAIN_LABELS = {
    "hospital":  "Hospital",
    "signlang":  "Sign Lang",
    "smarthome": "Smart Home",
}
DOMAIN_COLORS = {
    "hospital":  (50,  160, 255),
    "signlang":  (50,  220,  80),
    "smarthome": (255, 170,  30),
}

# ── object colors ─────────────────────────────────────────────────────────────
OBJ_COLORS = {
    "person":  (0, 255, 0),
    "mobile":  (255, 255, 0),
    "laptop":  (255, 0, 255),
    "bottle":  (0, 128, 255),
    "book":    (0, 255, 255),
    "fan":     (255, 80, 0),
}


# ── drawing helpers ───────────────────────────────────────────────────────────
def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.75):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub  = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def _word_wrap(text, max_chars=22):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# MASTER DRAW FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def draw_full_ui(frame, gesture_key, rule_conf,
                 temporal_label, temporal_conf,
                 objects, domain,
                 gesture_enabled, object_enabled,
                 temporal_enabled, paused):
    """
    Master UI function — call once per frame.
    Draws: alert flash, top bar, left panel, right panel,
           action log, status dots.
    """
    h, w = frame.shape[:2]

    # 1. alert flash (must be first)
    frame = _draw_alert_flash(frame, gesture_key, w, h)

    # 2. top bar
    frame = _draw_top_bar(frame, w, domain)

    # 3. left panel — gesture meaning
    frame = _draw_gesture_panel(frame, gesture_key, rule_conf, domain, h)

    # 4. right panel — temporal + objects
    frame = _draw_right_panel(frame, temporal_label, temporal_conf,
                               objects, object_enabled, temporal_enabled, w, h)

    # 5. bottom — action log
    frame = _draw_action_log(frame, gesture_key, rule_conf, w, h)

    # 6. status dots
    frame = _draw_status_dots(frame, gesture_enabled, object_enabled,
                               temporal_enabled, paused, w, h)

    return frame


# ── panel renderers ───────────────────────────────────────────────────────────

def _draw_alert_flash(frame, gesture_key, w, h):
    ALERT_SYS.update(gesture_key, 1.0)

    if ALERT_SYS.is_critical():
        alpha = 0.15 + 0.2 * abs(math.sin(time.time() * 4))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "EMERGENCY ALERT",
                    (w // 2 - 200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    elif ALERT_SYS.is_warning():
        alpha = 0.08 + 0.1 * abs(math.sin(time.time() * 2))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 140, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def _draw_top_bar(frame, w, active_domain):
    _alpha_rect(frame, 0, 0, w, 54, (15, 15, 15))
    cv2.putText(frame, "RT-Gesture3D v2  |  Meaningful AI",
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2)

    # domain tabs
    tab_w = 120
    start_x = w - (tab_w + 4) * len(DOMAINS) - 10
    for i, d in enumerate(DOMAINS):
        tx    = start_x + i * (tab_w + 4)
        color = DOMAIN_COLORS[d] if d == active_domain else (55, 55, 55)
        _alpha_rect(frame, tx, 4, tx + tab_w, 50, color,
                    0.9 if d == active_domain else 0.5)
        cv2.putText(frame, DOMAIN_LABELS[d],
                    (tx + 6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1)
    return frame


def _draw_gesture_panel(frame, gesture_key, conf, domain, h):
    get_info, _, _ = _get_registry()
    info = get_info(gesture_key)

    meaning = {
        "hospital":  info.hospital_meaning,
        "signlang":  info.signlang_meaning,
        "smarthome": info.smarthome_action,
    }.get(domain, info.description)

    color = info.color_bgr
    _alpha_rect(frame, 0, 54, 220, h, (18, 18, 18))

    # gesture name
    cv2.putText(frame, info.display_name,
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # confidence bar
    bar_w = int(190 * min(conf, 1.0))
    cv2.rectangle(frame, (10, 112), (200, 126), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 112), (10 + bar_w, 126), color, -1)
    cv2.putText(frame, f"{conf:.0%}",
                (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # domain meaning
    cv2.putText(frame, "Meaning:",
                (10, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)
    for li, line in enumerate(_word_wrap(meaning)[:4]):
        cv2.putText(frame, line,
                    (10, 192 + li * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    # alert badge
    al = info.alert_level
    if al == "critical":
        cv2.rectangle(frame, (10, h - 160), (210, h - 132), (0, 0, 180), -1)
        cv2.putText(frame, "CRITICAL ALERT",
                    (16, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    elif al == "warning":
        cv2.rectangle(frame, (10, h - 160), (210, h - 132), (0, 100, 200), -1)
        cv2.putText(frame, "WARNING",
                    (16, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    # smart home action
    if domain == "smarthome" and gesture_key != "neutral":
        cv2.rectangle(frame, (10, h - 120), (210, h - 92), (20, 70, 20), -1)
        cv2.putText(frame, "ACTION TRIGGERED",
                    (13, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 80), 1)
        cv2.putText(frame, " ".join(info.smarthome_action.split()[:4]),
                    (13, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

    return frame


def _draw_right_panel(frame, temporal_label, temporal_conf,
                       objects, object_enabled, temporal_enabled, w, h):
    _alpha_rect(frame, w - 210, 54, w, h, (18, 18, 18))

    if temporal_enabled:
        get_info, _, _ = _get_registry()
        t_info  = get_info(temporal_label)
        t_color = t_info.color_bgr

        cv2.putText(frame, "3D-CNN / ST-CNN",
                    (w - 205, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 120, 120), 1)
        cv2.putText(frame, t_info.display_name,
                    (w - 205, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, t_color, 2)

        bw = int(190 * min(temporal_conf, 1.0))
        cv2.rectangle(frame, (w - 205, 118), (w - 15, 132), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - 205, 118), (w - 205 + bw, 132), t_color, -1)
        cv2.putText(frame, f"{temporal_conf:.0%}",
                    (w - 205, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)

    if object_enabled and objects:
        cv2.putText(frame, "Objects:",
                    (w - 205, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 120, 120), 1)
        for oi, obj in enumerate(objects[:5]):
            lbl  = obj["label"]
            conf = obj["confidence"]
            cv2.putText(frame, f"  {lbl}  {conf:.0%}",
                        (w - 205, 202 + oi * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

    return frame


def _draw_action_log(frame, gesture_key, conf, w, h):
    ALERT_SYS.update(gesture_key, conf)
    log = ALERT_SYS.log()
    _alpha_rect(frame, 220, h - 68, w - 210, h, (12, 12, 12))
    cv2.putText(frame, "Recent:",
                (228, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1)
    for li, entry in enumerate(log[:3]):
        cv2.putText(frame, entry,
                    (228, h - 32 + li * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1)
    return frame


def _draw_status_dots(frame, g, o, t, p, w, h):
    items = [
        (f"[G]{'ON' if g else 'OFF'}", (0, 200, 0)  if g else (70, 70, 70)),
        (f"[O]{'ON' if o else 'OFF'}", (0, 200, 0)  if o else (70, 70, 70)),
        (f"[T]{'ON' if t else 'OFF'}", (0, 200, 0)  if t else (70, 70, 70)),
        (f"{'PAUSED' if p else 'LIVE'}",
         (0, 120, 255) if p else (0, 200, 0)),
    ]
    for i, (txt, col) in enumerate(items):
        cv2.putText(frame, txt,
                    (228 + i * 110, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    return frame


# ── object bounding boxes ─────────────────────────────────────────────────────
def draw_objects(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        lbl  = obj["label"]
        conf = obj["confidence"]
        if conf < 0.3:
            continue
        col = OBJ_COLORS.get(lbl, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.rectangle(frame, (x1, y1 - 24), (x1 + 160, y1), col, -1)
        cv2.putText(frame, f"{lbl} {conf:.0%}",
                    (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame