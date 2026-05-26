"""
temporal_model.py — RT-Gesture3D v2
===================================

Production-ready Temporal Gesture Model.

Features
--------
✅ Loads best_st_cnn.pt automatically
✅ Falls back to best_cnn3d.pt
✅ Falls back to heuristic if no model exists
✅ Supports 12 gestures
✅ Real-time optimized
✅ Dynamic gesture compatible
✅ Existing API compatible

predict(sequence)
-----------------
Input:
    list of BGR frames

Returns:
    (gesture_name, confidence)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.dataset import (
    GestureDataset,
    GESTURE_CLASSES,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────
# Optional Torch
# ─────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True

except ImportError:

    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

T_FRAMES = 16

FRAME_SIZE = (112, 112)


# ─────────────────────────────────────────────────────────────
# Temporal Gesture Model
# ─────────────────────────────────────────────────────────────

class TemporalGestureModel:

    def __init__(self, models_dir=None):

        self.min_frames = T_FRAMES

        self._model = None

        self._device = "cpu"

        self._model_name = "heuristic"

        if _TORCH_AVAILABLE:
            self._load_model(models_dir)

    # ─────────────────────────────────────────────────────────
    # Load Model
    # ─────────────────────────────────────────────────────────

    def _load_model(self, models_dir):

        from src.training.models import build_model

        if models_dir is None:
            models_dir = ROOT / "models"
        else:
            models_dir = Path(models_dir)

        candidates = [

            ("st_cnn",
             models_dir / "best_st_cnn.pt"),

            ("cnn3d",
             models_dir / "best_cnn3d.pt"),
        ]

        for arch, model_path in candidates:

            if model_path.exists():

                try:

                    self._model = build_model(
                        arch=arch,
                        num_classes=NUM_CLASSES,
                        pretrained_path=str(model_path),
                        device=self._device,
                    )

                    self._model.eval()

                    self._model_name = arch

                    print(
                        f"✅ Temporal Model Loaded:"
                    )

                    print(
                        f"   Architecture : {arch}"
                    )

                    print(
                        f"   Classes      : {NUM_CLASSES}"
                    )

                    print(
                        f"   Weights      : {model_path.name}"
                    )

                    return

                except Exception as e:

                    print(
                        f"⚠️ Failed loading "
                        f"{model_path.name}"
                    )

                    print(f"   Error: {e}")

        print(
            "ℹ️ No trained model found."
        )

        print(
            "   Falling back to heuristic mode."
        )

        print(
            "   Run training first:"
        )

        print(
            "   python src/training/train.py"
        )

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def predict(
        self,
        sequence: List[np.ndarray]
    ) -> Tuple[str, float]:

        # empty
        if not sequence:
            return "none", 0.0

        # not enough frames
        if len(sequence) < self.min_frames:
            return "buffering", 0.0

        # neural network
        if self._model is not None:
            return self._nn_predict(sequence)

        # fallback heuristic
        return self._heuristic_predict(sequence)

    # ─────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────────────────────

    def model_name(self) -> str:
        return self._model_name

    # ─────────────────────────────────────────────────────────
    # Neural Prediction
    # ─────────────────────────────────────────────────────────

    def _nn_predict(
        self,
        sequence: List[np.ndarray]
    ) -> Tuple[str, float]:

        try:

            tensor = self._preprocess(sequence)

            with torch.no_grad():

                logits = self._model(tensor)

                probs = F.softmax(
                    logits,
                    dim=1
                )[0]

                conf, pred = probs.max(0)

            label = GESTURE_CLASSES[
                int(pred.item())
            ]

            return (
                label,
                float(conf.item())
            )

        except Exception as e:

            print(
                f"⚠️ NN prediction error: {e}"
            )

            return "error", 0.0

    # ─────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────

    def _preprocess(
        self,
        sequence: List[np.ndarray]
    ):

        frames = []

        for frame in sequence:

            resized = cv2.resize(
                frame,
                FRAME_SIZE,
                interpolation=cv2.INTER_LINEAR
            )

            frames.append(resized)

        # stack
        arr = np.stack(
            frames,
            axis=0
        )

        # temporal sampling
        arr = GestureDataset._sample_frames(
            arr,
            T_FRAMES
        )

        # normalize
        arr = arr.astype(np.float32) / 255.0

        # tensor
        tensor = torch.from_numpy(arr)

        # (T,H,W,C) → (C,T,H,W)
        tensor = tensor.permute(
            3,
            0,
            1,
            2
        )

        # batch dim
        tensor = tensor.unsqueeze(0)

        return tensor.to(self._device)

    # ─────────────────────────────────────────────────────────
    # Heuristic Fallback
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_predict(
        sequence: List[np.ndarray]
    ) -> Tuple[str, float]:

        """
        Motion-energy heuristic.

        Only fallback when no trained model exists.
        """

        frames = []

        for frame in sequence[-T_FRAMES:]:

            gray = cv2.cvtColor(

                cv2.resize(
                    frame,
                    (64, 64)
                ),

                cv2.COLOR_BGR2GRAY

            ).astype(np.float32) / 255.0

            frames.append(gray)

        diffs = [

            np.mean(
                np.abs(
                    frames[i + 1] - frames[i]
                )
            )

            for i in range(len(frames) - 1)
        ]

        motion = (
            float(np.mean(diffs))
            if diffs else 0.0
        )

        # dynamic movement
        if motion > 0.08:

            confidence = min(
                0.5 + motion * 4,
                0.90
            )

            return "help", confidence

        # low movement
        return "neutral", 0.60