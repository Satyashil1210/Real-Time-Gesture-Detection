# RT-Gesture3D – Processing Module

This folder is responsible for **transforming raw inputs** into
model-ready features.

Typical responsibilities (future-ready):

- Convert raw MediaPipe landmarks into numeric feature vectors.
- Normalize coordinates (e.g., relative to the wrist, scale-invariant).
- Build short temporal windows (sequences of frames) for RNN/Temporal models.
- Apply simple augmentations (noise, slight scaling, etc.).

Currently implemented:
- `buffer.py`: a generic `RingBuffer` used for smoothing predictions or
  accumulating the last N frames.

If you later add a learned model:
- Implement a `preprocess.py` that takes `List[(x,y,z)]` landmarks and
  returns a `np.ndarray` feature vector.
- Reuse the same preprocessing during both **training** and **inference**.
