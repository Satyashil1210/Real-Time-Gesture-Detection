# RT-Gesture3D v2 Dataset Format
# ==============================

Dataset supports:
✅ Static gestures
✅ Dynamic gestures
✅ 3D-CNN / ST-CNN
✅ Temporal clips
✅ Real-world meaningful AI system

data/
│
├── raw/
│   │
│   ├── neutral/
│   │   ├── clip_0001.npy
│   │   ├── clip_0002.npy
│   │   └── ...
│   │
│   ├── victory/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── ok/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── perfect/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── stop/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── rock/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── calm/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── namaste/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── help/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── wave/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   ├── thumbs_down/
│   │   ├── clip_0001.npy
│   │   └── ...
│   │
│   └── clap/
│       ├── clip_0001.npy
│       └── ...
│
├── processed/
│   │
│   ├── landmarks/
│   │   ├── neutral_landmarks.npz
│   │   ├── victory_landmarks.npz
│   │   ├── ok_landmarks.npz
│   │   ├── perfect_landmarks.npz
│   │   ├── stop_landmarks.npz
│   │   ├── rock_landmarks.npz
│   │   ├── calm_landmarks.npz
│   │   ├── namaste_landmarks.npz
│   │   ├── help_landmarks.npz
│   │   ├── wave_landmarks.npz
│   │   ├── thumbs_down_landmarks.npz
│   │   └── clap_landmarks.npz
│   │
│   └── features/
│       ├── gesture_embeddings.npy
│       ├── temporal_features.npy
│       └── metadata.json
│
├── train/
├── val/
└── test/

# ────────────────────────────────────────────────────────────
# Raw Clip Format
# ────────────────────────────────────────────────────────────

Each .npy clip:

Shape:
    (T, H, W, 3)

Example:
    (16, 112, 112, 3)

Type:
    uint8

Meaning:
    T = temporal frames
    H = height
    W = width
    3 = RGB/BGR channels

# ────────────────────────────────────────────────────────────
# Supported Gestures
# ────────────────────────────────────────────────────────────

Static:
- neutral
- victory
- ok
- perfect
- stop
- rock
- calm

Dynamic:
- namaste
- help
- wave
- thumbs_down
- clap

# ────────────────────────────────────────────────────────────
# Recommended Dataset Size
# ────────────────────────────────────────────────────────────

Minimum:
    30 clips per gesture

Recommended:
    100+ clips per gesture

Best:
    Multiple:
    - lighting conditions
    - backgrounds
    - hand angles
    - camera distances
    - users

# ────────────────────────────────────────────────────────────
# Training Pipeline
# ────────────────────────────────────────────────────────────

1. Create folders

2. Generate clips:
    python generate_dataset.py

3. Train:
    python src/training/train.py

4. Run live system:
    python src/inference/live_meaningful.py