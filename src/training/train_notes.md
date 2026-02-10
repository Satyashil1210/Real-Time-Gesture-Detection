# RT-Gesture3D – Training Notes

Current version of RT-Gesture3D uses a **heuristic-based recognizer**
(see `src/inference/predictor.py`) built directly on top of MediaPipe
hand landmarks.

This folder is intentionally reserved for a future **learned model**
pipeline, for example:

- Load images or landmark sequences from `data/raw` and `data/processed`.
- Use utilities from `src/processing/` to build feature vectors.
- Train a classifier:
  - Classical ML (SVM / RandomForest / XGBoost), or
  - Lightweight neural network.
- Export the trained model into `models/checkpoints/` or ONNX/TFLite
  for fast runtime inference.

The idea: the **project architecture is already split** into `capture`,
`processing`, `training`, and `inference`, so swapping the heuristic
predictor with a learned model later will be straightforward.
