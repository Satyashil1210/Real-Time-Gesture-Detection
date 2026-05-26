"""
temporal_model.py  —  RT-Gesture3D
====================================
Production TemporalGestureModel.

- Loads GestureST_CNN (or GestureCNN3D) from models/best_*.pt if available.
- Falls back to a motion-heuristic when no trained model is present.
- Exposes the same predict(sequence) API the main loop already calls.

predict(sequence) 
    sequence : list of BGR frames (numpy uint8)  length >= 16
    returns  : (label_str, confidence_float)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.dataset import GESTURE_CLASSES, GestureDataset

# optional torch import
try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

FRAME_SIZE = (112, 112)
T_FRAMES   = 16


class TemporalGestureModel:
    """
    Real-time temporal gesture model.

    Hierarchy
    ---------
    1. If best_st_cnn.pt exists  → use GestureST_CNN
    2. Elif best_cnn3d.pt exists → use GestureCNN3D
    3. Else                      → motion heuristic (no training needed)
    """

    def __init__(self, models_dir: str | Path | None = None):
        self.min_frames  = T_FRAMES
        self._model      = None
        self._device     = "cpu"
        self._model_name = "heuristic"

        if _TORCH_OK:
            self._try_load_model(models_dir)

    # ── model loading ────────────────────────────────────────────────────────

    def _try_load_model(self, models_dir):
        from src.training.models import build_model

        if models_dir is None:
            models_dir = ROOT / "models"
        else:
            models_dir = Path(models_dir)

        candidates = [
            ("st_cnn",  models_dir / "best_st_cnn.pt"),
            ("cnn3d",   models_dir / "best_cnn3d.pt"),
        ]

        for arch, path in candidates:
            if path.exists():
                try:
                    self._model = build_model(
                        arch=arch,
                        num_classes=len(GESTURE_CLASSES),
                        pretrained_path=str(path),
                        device=self._device,
                    )
                    self._model.eval()
                    self._model_name = arch
                    print(f"✅ TemporalModel: loaded {arch} from {path.name}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {path.name}: {e}")

        print("ℹ️  TemporalModel: no trained weights found → motion heuristic active")
        print("   Run:  python src/training/train.py   to train a model.")

    # ── public API ───────────────────────────────────────────────────────────

    def predict(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        """
        Parameters
        ----------
        sequence : list of BGR uint8 frames (any size, any length)

        Returns
        -------
        (gesture_label, confidence)   e.g. ("stop", 0.91)
        """
        if not sequence:
            return "none", 0.0

        if len(sequence) < self.min_frames:
            return "buffering", 0.0

        if self._model is not None and _TORCH_OK:
            return self._nn_predict(sequence)
        else:
            return self._heuristic_predict(sequence)

    def is_trained(self) -> bool:
        return self._model is not None

    def model_name(self) -> str:
        return self._model_name

    # ── neural-net prediction ─────────────────────────────────────────────────

    def _nn_predict(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        try:
            clip = self._preprocess(sequence)              # (1, 3, T, H, W)
            with torch.no_grad():
                logits = self._model(clip)                 # (1, C)
                probs  = F.softmax(logits, dim=1)[0]       # (C,)
                conf, pred = probs.max(0)
            label = GESTURE_CLASSES[int(pred.item())]
            return label, float(conf.item())
        except Exception as e:
            print(f"⚠️ NN predict error: {e}")
            return "error", 0.0

    def _preprocess(self, sequence: List[np.ndarray]) -> "torch.Tensor":
        """Resize → sample → normalise → (1, 3, T, H, W) tensor."""
        frames = []
        for f in sequence:
            small = cv2.resize(f, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            frames.append(small)

        # uniform sample to T_FRAMES
        arr = np.stack(frames, axis=0)                     # (N, H, W, 3)
        arr = GestureDataset._sample_frames(arr, T_FRAMES) # (T, H, W, 3)
        arr = arr.astype(np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(3, 0, 1, 2)   # (3, T, H, W)
        return t.unsqueeze(0).to(self._device)             # (1, 3, T, H, W)

    # ── motion heuristic (no model needed) ────────────────────────────────────

    @staticmethod
    def _heuristic_predict(sequence: List[np.ndarray]) -> Tuple[str, float]:
        """
        Simple motion-energy heuristic.
        Returns "neutral" for low motion, "stop" for high motion.
        Not accurate — only a placeholder until the model is trained.
        """
        grays = []
        for f in sequence[-T_FRAMES:]:
            g = cv2.cvtColor(
                cv2.resize(f, (64, 64)),
                cv2.COLOR_BGR2GRAY
            ).astype(np.float32) / 255.0
            grays.append(g)

        diffs = [np.mean(np.abs(grays[i+1] - grays[i]))
                 for i in range(len(grays) - 1)]
        motion = float(np.mean(diffs)) if diffs else 0.0

        if motion > 0.05:
            return "stop", min(0.5 + motion * 4, 0.9)
        return "neutral", 0.6
