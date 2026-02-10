# RT-Gesture3D – Capture Module

This folder contains utilities for recording raw data from the webcam
that can later be used for training a learned gesture recognizer.

- `recorder.py`
  - Opens the webcam.
  - Saves individual frames into `data/raw/<label>/` when you press `s`.
  - Useful for collecting samples of a specific gesture class (e.g., `ok`, `rock`, `stop`).

Recommended workflow (future):
1. Decide a label name (must match your dataset convention, e.g., `ok`, `rock`, `stop`).
2. Update `label = "custom"` in `recorder.py`.
3. Run `python src/capture/recorder.py` from project root.
4. Perform gesture in front of the camera and press `s` multiple times.
5. Repeat for each label you want to support.
