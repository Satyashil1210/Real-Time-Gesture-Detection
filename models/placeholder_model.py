
## 5️⃣ `models/placeholder_model.py`

"""
Placeholder model definition for RT-Gesture3D.

Currently, RT-Gesture3D uses heuristic rules in
`src/inference/predictor.py` and does NOT rely on a learned model.

This module defines a minimal interface so that a future ML model
can plug into the same inference pipeline without changing the app.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass
class GesturePrediction:
    label: str
    confidence: float


class PlaceholderGestureModel:
    """
    Example interface for a gesture classification model.

    In the future, you can replace this with:
        - a scikit-learn model
        - a PyTorch / TensorFlow model
        - an ONNXRuntime wrapper

    For now, this class does not perform any real learning-based inference.
    """

    def __init__(self) -> None:
        # Any model weights or config would be loaded here.
        self._loaded = False

    def load_from_checkpoint(self, path: str) -> None:
        """
        Load model weights from the given path (placeholder).
        """
        # TODO: implement real loading once a model is trained.
        print(f"[PlaceholderGestureModel] load_from_checkpoint called with: {path}")
        self._loaded = True

    def predict(self, features: np.ndarray) -> GesturePrediction:
        """
        Predict gesture from a feature vector.

        For now this returns a fixed dummy value.
        In a real model, `features` must be the output of the same
        preprocessing used during training.
        """
        _ = features  # unused for now
        return GesturePrediction(label="neutral", confidence=0.5)
