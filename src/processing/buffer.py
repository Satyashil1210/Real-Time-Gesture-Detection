"""
buffer.py — RT-Gesture3D v2
===========================

Sliding-window temporal frame buffer.

Features
--------
✅ Real-time frame buffering
✅ Motion detection
✅ 3D-CNN ready tensor output
✅ ST-CNN compatible
✅ Temporal augmentation helpers
✅ Optimized for live inference
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_WINDOW = 16

DEFAULT_SIZE = (112, 112)

MOTION_THRESHOLD = 0.004

# ─────────────────────────────────────────────────────────────
# Frame Buffer
# ─────────────────────────────────────────────────────────────

class FrameBuffer:

    """
    Sliding temporal window buffer.

    Usage
    -----

    buf = FrameBuffer(window=16)

    buf.push(frame)

    if buf.is_ready():

        clip = buf.get_clip()

        if buf.has_motion():
            run_model(clip)
    """

    def __init__(

        self,

        window: int = DEFAULT_WINDOW,

        size: Tuple[int, int] = DEFAULT_SIZE,
    ):

        self.window = window

        self.size = size

        # frame queue
        self._frames = deque(
            maxlen=window
        )

        # motion scores
        self._motion_scores = deque(
            maxlen=window
        )

        # previous gray frame
        self._prev_gray: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────
    # Push Frame
    # ─────────────────────────────────────────────────────────

    def push(
        self,
        frame: np.ndarray
    ) -> None:

        if frame is None:
            return

        # resize
        small = cv2.resize(

            frame,

            self.size,

            interpolation=cv2.INTER_LINEAR
        )

        # grayscale for motion
        gray = cv2.cvtColor(

            small,

            cv2.COLOR_BGR2GRAY

        ).astype(np.float32) / 255.0

        # motion score
        if self._prev_gray is not None:

            motion_score = float(

                np.mean(
                    np.abs(
                        gray - self._prev_gray
                    )
                )
            )

        else:

            motion_score = 0.0

        self._prev_gray = gray

        self._motion_scores.append(
            motion_score
        )

        # normalize frame
        norm = small.astype(
            np.float32
        ) / 255.0

        self._frames.append(norm)

    # ─────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────

    def is_ready(self) -> bool:

        return len(self._frames) >= self.window

    # ─────────────────────────────────────────────────────────

    def has_motion(

        self,

        threshold: float = MOTION_THRESHOLD
    ) -> bool:

        if len(self._motion_scores) < self.window:
            return False

        avg_motion = float(
            np.mean(self._motion_scores)
        )

        return avg_motion > threshold

    # ─────────────────────────────────────────────────────────
    # Clip
    # ─────────────────────────────────────────────────────────

    def get_clip(self) -> np.ndarray:

        """
        Returns:
            (T,H,W,3) float32 [0,1]
        """

        frames = list(
            self._frames
        )[-self.window:]

        return np.stack(
            frames,
            axis=0
        )

    # ─────────────────────────────────────────────────────────
    # Tensor
    # ─────────────────────────────────────────────────────────

    def get_tensor(self):

        """
        Returns:
            (1,3,T,H,W)
        """

        import torch

        clip = self.get_clip()

        tensor = torch.from_numpy(
            clip
        )

        # (T,H,W,C) → (C,T,H,W)
        tensor = tensor.permute(
            3,
            0,
            1,
            2
        )

        # batch dim
        tensor = tensor.unsqueeze(0)

        return tensor

    # ─────────────────────────────────────────────────────────
    # CHW Tensor (No Batch)
    # ─────────────────────────────────────────────────────────

    def get_clip_tensor_chw(self):

        """
        Returns:
            (3,T,H,W)
        """

        clip = self.get_clip()

        return clip.transpose(
            3,
            0,
            1,
            2
        )

    # ─────────────────────────────────────────────────────────
    # Clear
    # ─────────────────────────────────────────────────────────

    def clear(self):

        self._frames.clear()

        self._motion_scores.clear()

        self._prev_gray = None

    # ─────────────────────────────────────────────────────────

    def __len__(self):

        return len(self._frames)

    # ─────────────────────────────────────────────────────────
    # Training Augmentation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def augment_clip(
        clip: np.ndarray
    ) -> np.ndarray:

        """
        Input:
            (T,H,W,3)

        Output:
            augmented clip
        """

        T, H, W, C = clip.shape

        # ─────────────────────────────────────────────────────
        # Horizontal Flip
        # ─────────────────────────────────────────────────────

        if np.random.rand() < 0.5:

            clip = clip[:, :, ::-1, :].copy()

        # ─────────────────────────────────────────────────────
        # Brightness + Contrast
        # ─────────────────────────────────────────────────────

        alpha = np.random.uniform(
            0.8,
            1.2
        )

        beta = np.random.uniform(
            -0.1,
            0.1
        )

        clip = np.clip(
            clip * alpha + beta,
            0.0,
            1.0
        )

        # ─────────────────────────────────────────────────────
        # Temporal Jitter
        # ─────────────────────────────────────────────────────

        if T > 4 and np.random.rand() < 0.3:

            drop = np.random.randint(
                1,
                3
            )

            indices = sorted(

                np.random.choice(
                    T,
                    T - drop,
                    replace=False
                ).tolist()
            )

            sampled = clip[indices]

            pad = np.repeat(

                sampled[-1:],

                T - len(indices),

                axis=0
            )

            clip = np.concatenate(
                [sampled, pad],
                axis=0
            )

        # ─────────────────────────────────────────────────────
        # Gaussian Blur
        # ─────────────────────────────────────────────────────

        if np.random.rand() < 0.2:

            blurred = []

            ksize = np.random.choice([3, 5])

            for frame in clip:

                f = (frame * 255).astype(
                    np.uint8
                )

                f = cv2.GaussianBlur(
                    f,
                    (ksize, ksize),
                    0
                )

                blurred.append(
                    f.astype(np.float32) / 255.0
                )

            clip = np.stack(
                blurred,
                axis=0
            )

        return clip