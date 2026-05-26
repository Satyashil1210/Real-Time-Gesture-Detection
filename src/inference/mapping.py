from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GestureInfo:
    """
    Basic info about each gesture.
    """
    id: int
    key: str
    display_name: str
    meaning: str
    avatar_file: str 


# ==============================
# 🎯 GESTURE REGISTRY
# ==============================
GESTURES = {
    "neutral": GestureInfo(
        id=0,
        key="neutral",
        display_name="Neutral",
        meaning="Neutral",
        avatar_file="neutral.png",
    ),
    "victory": GestureInfo(
        id=1,
        key="victory",
        display_name="Victory",
        meaning="Victory Sign",
        avatar_file="victory.jpg",
    ),
    "ok": GestureInfo(
        id=2,
        key="ok",
        display_name="OK",
        meaning="OK / Thumbs Up",
        avatar_file="ok.jpg",
    ),
    "perfect": GestureInfo(
        id=3,
        key="perfect",
        display_name="Perfect",
        meaning="Perfect Gesture",
        avatar_file="perfect.jpg",
    ),
    "stop": GestureInfo(
        id=4,
        key="stop",
        display_name="Stop",
        meaning="Stop / Open Palm",
        avatar_file="stop.jpg",
    ),
    "rock": GestureInfo(
        id=5,
        key="rock",
        display_name="Rock",
        meaning="Rock Sign",
        avatar_file="rock.jpg",
    ),
    "calm": GestureInfo(
        id=6,
        key="calm",
        display_name="Calm",
        meaning="Calm / Point Gesture",
        avatar_file="calm.jpg",
    ),
}


# 🔥 SAFE ID → KEY MAPPING
ID_TO_KEY = {info.id: info.key for info in GESTURES.values()}


def get_key_from_id(gid: int) -> str:
    """
    Safe getter (prevents crash if id not found)
    """
    return ID_TO_KEY.get(gid, "neutral")


# ==============================
# 📁 PATH HELPERS
# ==============================
def get_project_root() -> Path:
    """
    Returns RT-Gesture3D root directory.
    Assumes this file is at src/inference/mapping.py
    """
    return Path(__file__).resolve().parents[2]


def get_avatars_dir() -> Path:
    """
    Returns RT-Gesture3D/assets/avatars path.
    Auto-creates folder if missing.
    """
    path = get_project_root() / "assets" / "avatars"
    path.mkdir(parents=True, exist_ok=True)
    return path