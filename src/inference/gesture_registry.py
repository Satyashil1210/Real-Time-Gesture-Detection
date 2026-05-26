"""
gesture_registry.py — RT-Gesture3D v2
=======================================
Central registry of ALL gestures with:
  - Real-world domain meanings
  - Alert levels
  - Actions to trigger
  - Display colors
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GestureInfo:
    id: int
    key: str
    display_name: str

    # Real-world meanings per domain
    hospital_meaning: str
    signlang_meaning: str
    smarthome_action: str

    # UI
    color_bgr: tuple          # OpenCV BGR
    alert_level: str          # "critical" | "warning" | "info" | "normal"
    icon: str                 # emoji for overlay
    description: str          # one-line description shown on screen

    # Dynamic gesture config (for synthetic data generation)
    motion_type: str          # "static" | "approach" | "wave" | "circular" | "push" | "pull"
    fingers: int
    motion_intensity: float


GESTURE_REGISTRY: Dict[str, GestureInfo] = {

    # ── STATIC GESTURES (rule-based + 3D CNN) ────────────────────────────────

    "neutral": GestureInfo(
        id=0, key="neutral", display_name="Neutral",
        hospital_meaning="Patient resting — no request",
        signlang_meaning="—",
        smarthome_action="No action",
        color_bgr=(150, 150, 150), alert_level="normal",
        icon="😐", description="No gesture detected",
        motion_type="static", fingers=0, motion_intensity=1.0,
    ),

    "victory": GestureInfo(
        id=1, key="victory", display_name="Victory / Yes",
        hospital_meaning="Patient says YES / feeling better",
        signlang_meaning="Yes / Agree",
        smarthome_action="Confirm last action",
        color_bgr=(0, 220, 0), alert_level="info",
        icon="✌️", description="YES — Patient confirms / agrees",
        motion_type="static", fingers=2, motion_intensity=2.0,
    ),

    "ok": GestureInfo(
        id=2, key="ok", display_name="OK / Thumbs Up",
        hospital_meaning="Patient is OK — no assistance needed",
        signlang_meaning="OK / Good",
        smarthome_action="Lights ON",
        color_bgr=(0, 200, 255), alert_level="info",
        icon="👍", description="OK — Patient is fine",
        motion_type="static", fingers=1, motion_intensity=2.0,
    ),

    "perfect": GestureInfo(
        id=3, key="perfect", display_name="Perfect / Understood",
        hospital_meaning="Patient understood instructions",
        signlang_meaning="Perfect / Understood",
        smarthome_action="Volume UP",
        color_bgr=(0, 255, 200), alert_level="info",
        icon="👌", description="PERFECT — Understood",
        motion_type="static", fingers=3, motion_intensity=2.0,
    ),

    "stop": GestureInfo(
        id=4, key="stop", display_name="STOP / No",
        hospital_meaning="⚠️ Patient says STOP — discomfort/pain",
        signlang_meaning="Stop / No / Wait",
        smarthome_action="All devices OFF",
        color_bgr=(0, 0, 255), alert_level="warning",
        icon="✋", description="STOP — Patient in discomfort",
        motion_type="push", fingers=5, motion_intensity=5.0,
    ),

    "rock": GestureInfo(
        id=5, key="rock", display_name="Rock / Call",
        hospital_meaning="Call family / phone request",
        signlang_meaning="Call me / Phone",
        smarthome_action="Make phone call",
        color_bgr=(200, 0, 200), alert_level="info",
        icon="🤘", description="CALL — Phone/family request",
        motion_type="static", fingers=2, motion_intensity=3.0,
    ),

    "calm": GestureInfo(
        id=6, key="calm", display_name="Point / One",
        hospital_meaning="Patient pointing — needs attention at location",
        signlang_meaning="One / Point / Look",
        smarthome_action="Fan ON",
        color_bgr=(255, 160, 0), alert_level="info",
        icon="☝️", description="POINT — Needs attention here",
        motion_type="static", fingers=1, motion_intensity=2.0,
    ),

    # ── DYNAMIC GESTURES (3D CNN / ST-CNN primary) ───────────────────────────

    "namaste": GestureInfo(
        id=7, key="namaste", display_name="Namaste / Hello",
        hospital_meaning="Patient greeting / feeling good",
        signlang_meaning="Hello / Namaste / Greet",
        smarthome_action="Welcome mode ON (lights warm)",
        color_bgr=(0, 215, 255), alert_level="info",
        icon="🙏", description="NAMASTE — Greeting / Hello",
        motion_type="approach",   # both hands coming together
        fingers=5, motion_intensity=4.0,
    ),

    "help": GestureInfo(
        id=8, key="help", display_name="🚨 HELP / Emergency",
        hospital_meaning="🚨 EMERGENCY — Nurse alert immediately",
        signlang_meaning="Help / Emergency",
        smarthome_action="Emergency alert — call security",
        color_bgr=(0, 0, 255), alert_level="critical",
        icon="🚨", description="HELP — EMERGENCY ALERT",
        motion_type="wave",       # frantic waving motion
        fingers=5, motion_intensity=9.0,
    ),

    "wave": GestureInfo(
        id=9, key="wave", display_name="Wave / Bye",
        hospital_meaning="Patient saying bye / wants to rest",
        signlang_meaning="Goodbye / See you",
        smarthome_action="Goodbye mode (dim lights, lower TV)",
        color_bgr=(255, 200, 0), alert_level="info",
        icon="👋", description="WAVE — Goodbye / Rest now",
        motion_type="wave",
        fingers=5, motion_intensity=6.0,
    ),

    "thumbs_down": GestureInfo(
        id=10, key="thumbs_down", display_name="No / Disagree",
        hospital_meaning="Patient says NO / disagrees with treatment",
        signlang_meaning="No / Disagree / Bad",
        smarthome_action="Cancel last action",
        color_bgr=(50, 50, 255), alert_level="warning",
        icon="👎", description="NO — Patient disagrees",
        motion_type="pull",
        fingers=1, motion_intensity=3.0,
    ),

    "clap": GestureInfo(
        id=11, key="clap", display_name="Clap / More",
        hospital_meaning="Patient wants more (water, food, medicine)",
        signlang_meaning="More / Again / Please repeat",
        smarthome_action="Repeat last action",
        color_bgr=(0, 255, 128), alert_level="info",
        icon="👏", description="MORE — Patient wants more",
        motion_type="circular",
        fingers=5, motion_intensity=5.0,
    ),
}

# ── Quick lookups ─────────────────────────────────────────────────────────────
ID_TO_KEY     = {info.id: info.key for info in GESTURE_REGISTRY.values()}
GESTURE_KEYS  = list(GESTURE_REGISTRY.keys())
NUM_CLASSES   = len(GESTURE_REGISTRY)

CRITICAL_GESTURES = [k for k, v in GESTURE_REGISTRY.items()
                     if v.alert_level == "critical"]
WARNING_GESTURES  = [k for k, v in GESTURE_REGISTRY.items()
                     if v.alert_level == "warning"]


def get_info(key: str) -> GestureInfo:
    return GESTURE_REGISTRY.get(key, GESTURE_REGISTRY["neutral"])


def get_domain_meaning(key: str, domain: str) -> str:
    """domain: 'hospital' | 'signlang' | 'smarthome'"""
    info = get_info(key)
    if domain == "hospital":   return info.hospital_meaning
    if domain == "signlang":   return info.signlang_meaning
    if domain == "smarthome":  return info.smarthome_action
    return info.description
