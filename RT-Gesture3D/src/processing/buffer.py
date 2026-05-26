"""
buffer.py
=========
Sliding-window frame-sequence buffer for RT-Gesture3D.

Responsibilities
----------------
- Accumulate raw BGR frames from the live camera
- Resize + normalise frames on insertion (zero-copy output)
- Expose a fixed-length window ready for 3D-CNN inference
- Track a simple "motion score" to skip idle (no-hand) windows
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np


# ── defaults (match 3D-CNN input) ───────────────────────────────────────────
DEFAULT_WINDOW   = 16          # temporal depth  (T)
DEFAULT_SIZE     = (112, 112)  # spatial size    (H, W)
MOTION_THRESHOLD = 0.004       # mean abs-diff threshold to flag as "moving"


class FrameBuffer:
    """
    Sliding-window buffer.

    Usage
    -----
    buf = FrameBuffer(window=16, size=(112,112))

    # each camera tick:
    buf.push(bgr_frame)

    if buf.is_ready():
        clip = buf.get_clip()          # np.ndarray (T, H, W, 3)  float32 [0,1]
        if buf.has_motion():
            run_temporal_model(clip)
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        size: Tuple[int, int] = DEFAULT_SIZE,
    ) -> None:
        self.window = window
        self.size   = size

        self._q: deque[np.ndarray] = deque(maxlen=window)
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_scores: deque[float]     = deque(maxlen=window)

    # ── public API ───────────────────────────────────────────────────────────

    def push(self, bgr_frame: np.ndarray) -> None:
        """
        Insert one raw BGR frame.
        Resize → normalise → store.
        Motion score computed on the fly.
        """
        if bgr_frame is None:
            return

        # 1. spatial resize
        small = cv2.resize(bgr_frame, self.size, interpolation=cv2.INTER_LINEAR)

        # 2. motion score (grayscale diff)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_gray is not None:
            score = float(np.mean(np.abs(gray - self._prev_gray)))
        else:
            score = 0.0
        self._prev_gray = gray
        self._motion_scores.append(score)

        # 3. normalise to float32 [0, 1]
        norm = small.astype(np.float32) / 255.0
        self._q.append(norm)

    def is_ready(self) -> bool:
        """True once the buffer holds a full window of frames."""
        return len(self._q) >= self.window

    def has_motion(self, threshold: float = MOTION_THRESHOLD) -> bool:
        """
        Returns True if the mean motion over the current window
        exceeds *threshold*.  Use to skip inference on static scenes.
        """
        if len(self._motion_scores) < self.window:
            return False
        return float(np.mean(self._motion_scores)) > threshold

    def get_clip(self) -> np.ndarray:
        """
        Returns the current window as a numpy array.

        Shape : (T, H, W, 3)   float32   range [0, 1]
        """
        frames = list(self._q)[-self.window:]
        return np.stack(frames, axis=0)   # (T, H, W, 3)

    def get_clip_tensor_chw(self) -> np.ndarray:
        """
        Returns the clip in (C, T, H, W) layout,
        ready for a PyTorch 3D-CNN (no batch dim).

        Shape : (3, T, H, W)   float32
        """
        clip = self.get_clip()            # (T, H, W, 3)
        clip = clip.transpose(3, 0, 1, 2) # (C, T, H, W)
        return clip

    def clear(self) -> None:
        self._q.clear()
        self._motion_scores.clear()
        self._prev_gray = None

    def __len__(self) -> int:
        return len(self._q)

    # ── augmentation helpers (used during training, not inference) ───────────

    @staticmethod
    def augment_clip(clip: np.ndarray) -> np.ndarray:
        """
        Light augmentation for training.

        Input / output shape : (T, H, W, 3)   float32
        """
        T, H, W, C = clip.shape

        # 1. random horizontal flip
        if np.random.rand() < 0.5:
            clip = clip[:, :, ::-1, :].copy()

        # 2. random brightness / contrast
        alpha = np.random.uniform(0.8, 1.2)   # contrast
        beta  = np.random.uniform(-0.1, 0.1)  # brightness
        clip  = np.clip(clip * alpha + beta, 0.0, 1.0)

        # 3. random temporal jitter (drop 1-2 frames, repeat edge)
        if T > 4 and np.random.rand() < 0.3:
            drop = np.random.randint(1, 3)
            indices = sorted(np.random.choice(T, T - drop, replace=False).tolist())
            # pad back to T by repeating last frame
            sampled = clip[indices]
            pad = np.repeat(sampled[-1:], T - len(indices), axis=0)
            clip = np.concatenate([sampled, pad], axis=0)

        return clip
