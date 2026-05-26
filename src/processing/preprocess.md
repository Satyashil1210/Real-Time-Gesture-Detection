# RT-Gesture3D v2 — Processing Module
# ===================================

This module is responsible for transforming raw camera inputs
into model-ready temporal features for:

✅ Rule-based gesture recognition
✅ 3D-CNN gesture classification
✅ ST-CNN temporal understanding
✅ Real-world meaningful AI interactions

# ────────────────────────────────────────────────────────────
# Responsibilities
# ────────────────────────────────────────────────────────────

The processing module handles:

• Frame buffering
• Temporal sequence creation
• Motion estimation
• Landmark normalization
• Feature extraction
• Temporal smoothing
• Clip augmentation
• Preprocessing for training + inference

# ────────────────────────────────────────────────────────────
# Current Components
# ────────────────────────────────────────────────────────────

processing/
│
├── buffer.py
│
├── preprocess.py        (future-ready)
│
├── temporal_utils.py    (optional future)
│
└── augment.py           (optional future)

# ────────────────────────────────────────────────────────────
# buffer.py
# ────────────────────────────────────────────────────────────

Current live temporal processing system.

Responsibilities:
-----------------
✅ Sliding temporal window
✅ Frame accumulation
✅ Motion detection
✅ 3D-CNN tensor generation
✅ ST-CNN clip preparation
✅ Real-time optimized buffering

Output:
-------
(T, H, W, 3) float32 clips

Example:
--------
(16, 112, 112, 3)

Used by:
--------
- temporal_model.py
- live_meaningful.py
- training pipeline

# ────────────────────────────────────────────────────────────
# preprocess.py (Future Ready)
# ────────────────────────────────────────────────────────────

Will convert MediaPipe hand landmarks into
normalized feature vectors.

Future responsibilities:
------------------------
✅ Wrist-relative normalization
✅ Scale-invariant coordinates
✅ Angle extraction
✅ Finger-state extraction
✅ Temporal feature stacking
✅ Gesture embeddings

Input:
------
List[(x, y, z)]

Output:
-------
np.ndarray feature vector

Example:
--------
[thumb_angle,
 index_angle,
 palm_distance,
 motion_energy,
 ...]

IMPORTANT:
-----------
Use the SAME preprocessing during:

✅ Training
✅ Validation
✅ Inference

to avoid train/inference mismatch.

# ────────────────────────────────────────────────────────────
# Temporal Processing Pipeline
# ────────────────────────────────────────────────────────────

Camera Frame
    ↓
MediaPipe Hands
    ↓
Landmark Extraction
    ↓
FrameBuffer
    ↓
Temporal Window (16 Frames)
    ↓
Motion Detection
    ↓
Temporal Model
    ↓
ST-CNN / 3D-CNN
    ↓
Meaningful Gesture Output

# ────────────────────────────────────────────────────────────
# Motion Detection
# ────────────────────────────────────────────────────────────

The buffer system computes motion using:

Mean Absolute Difference
between consecutive grayscale frames.

Purpose:
--------
✅ Skip unnecessary inference
✅ Improve FPS
✅ Reduce CPU load
✅ Ignore idle/no-hand frames

# ────────────────────────────────────────────────────────────
# Current Supported Models
# ────────────────────────────────────────────────────────────

Static:
--------
- Rule-based gestures

Temporal:
----------
- GestureCNN3D
- GestureST_CNN

Fallback:
----------
- Motion heuristic model

# ────────────────────────────────────────────────────────────
# Augmentations
# ────────────────────────────────────────────────────────────

Current augmentations:
----------------------
✅ Horizontal flip
✅ Brightness jitter
✅ Contrast jitter
✅ Temporal jitter
✅ Gaussian blur

Future augmentations:
---------------------
- Random crop
- Rotation
- Synthetic motion blur
- Background randomization

# ────────────────────────────────────────────────────────────
# Real-Time Optimization
# ────────────────────────────────────────────────────────────

Optimizations:
--------------
✅ Sliding deque buffer
✅ Zero-copy temporal windows
✅ Motion-gated inference
✅ Reduced YOLO frequency
✅ Lightweight preprocessing
✅ Temporal smoothing

# ────────────────────────────────────────────────────────────
# Recommended Future Additions
# ────────────────────────────────────────────────────────────

Future modules:
---------------
- preprocess.py
- temporal_utils.py
- gesture_features.py
- hand_tracking.py
- motion_estimator.py

Potential future models:
------------------------
- LSTM
- GRU
- Transformer
- TCN
- ActionFormer

# ────────────────────────────────────────────────────────────
# Best Practices
# ────────────────────────────────────────────────────────────

✅ Keep preprocessing identical across:
   - training
   - validation
   - inference

✅ Normalize landmarks consistently

✅ Use motion gating before temporal inference

✅ Keep temporal window fixed (16 frames)

✅ Avoid excessive augmentations during inference

# ────────────────────────────────────────────────────────────
# Used By
# ────────────────────────────────────────────────────────────

Main integration:
-----------------
- live_meaningful.py
- temporal_model.py
- train.py
- dataset.py

# ────────────────────────────────────────────────────────────
# RT-Gesture3D v2
# Meaningful AI Gesture System
# ────────────────────────────────────────────────────────────