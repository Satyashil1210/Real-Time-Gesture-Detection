"""
dataset.py — RT-Gesture3D v2
============================
PyTorch Dataset for loading gesture clips.

Supports:
- 7 static gestures
- 5 dynamic gestures
- 3D CNN / ST-CNN
- Temporal augmentation
- Weighted sampling
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

T_FRAMES = 16
FRAME_H  = 112
FRAME_W  = 112

# 12 gesture classes
GESTURE_CLASSES = [
    # static gestures
    "neutral",
    "victory",
    "ok",
    "perfect",
    "stop",
    "rock",
    "calm",

    # dynamic gestures
    "namaste",
    "help",
    "wave",
    "thumbs_down",
    "clap",
]

CLASS_TO_IDX = {
    g: i for i, g in enumerate(GESTURE_CLASSES)
}

NUM_CLASSES = len(GESTURE_CLASSES)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class GestureDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split="train",
        val_frac=0.15,
        test_frac=0.10,
        augment=False,
        t_frames=T_FRAMES,
        frame_size=(FRAME_H, FRAME_W),
    ):

        self.data_dir   = Path(data_dir)
        self.split      = split
        self.augment    = augment
        self.t_frames   = t_frames
        self.frame_size = frame_size

        self.samples: List[Tuple[Path, int]] = []

        self._collect_samples(
            val_frac,
            test_frac
        )

        print(
            f"[GestureDataset] "
            f"{split:5s} → {len(self.samples)} clips"
        )

    # ─────────────────────────────────────────────────────────

    def _collect_samples(
        self,
        val_frac,
        test_frac
    ):

        for gesture in GESTURE_CLASSES:

            cls_dir = self.data_dir / gesture

            if not cls_dir.exists():
                continue

            clips = sorted(
                cls_dir.glob("clip_*.npy")
            )

            n = len(clips)

            if n == 0:
                continue

            n_test  = max(1, int(n * test_frac))
            n_val   = max(1, int(n * val_frac))
            n_train = n - n_test - n_val

            if self.split == "train":
                chosen = clips[:n_train]

            elif self.split == "val":
                chosen = clips[n_train:n_train+n_val]

            else:
                chosen = clips[n_train+n_val:]

            idx = CLASS_TO_IDX[gesture]

            self.samples.extend(
                (p, idx) for p in chosen
            )

    # ─────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    # ─────────────────────────────────────────────────────────

    def __getitem__(self, index):

        path, label = self.samples[index]

        clip = self._load_clip(path)

        if self.augment:
            clip = self._augment(clip)

        tensor = self._to_tensor(clip)

        return tensor, label

    # ─────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────

    def _load_clip(self, path):

        arr = np.load(str(path))

        arr = self._sample_frames(
            arr,
            self.t_frames
        )

        if arr.shape[1:3] != (
            self.frame_size[0],
            self.frame_size[1]
        ):

            resized = []

            for frame in arr:

                frame = cv2.resize(
                    frame,
                    (
                        self.frame_size[1],
                        self.frame_size[0]
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                resized.append(frame)

            arr = np.stack(
                resized,
                axis=0
            )

        return arr.astype(np.float32) / 255.0

    # ─────────────────────────────────────────────────────────
    # Frame Sampling
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _sample_frames(arr, t):

        total = arr.shape[0]

        if total == t:
            return arr

        if total < t:

            pad = np.repeat(
                arr[-1:],
                t - total,
                axis=0
            )

            return np.concatenate(
                [arr, pad],
                axis=0
            )

        idx = np.linspace(
            0,
            total - 1,
            t,
            dtype=int
        )

        return arr[idx]

    # alias
    _sample = _sample_frames

    # ─────────────────────────────────────────────────────────
    # Tensor
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(clip):

        tensor = torch.from_numpy(clip)

        # (T,H,W,C) → (C,T,H,W)
        tensor = tensor.permute(
            3,
            0,
            1,
            2
        )

        return tensor

    # ─────────────────────────────────────────────────────────
    # Augmentation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _augment(clip):

        # horizontal flip
        if random.random() < 0.5:

            clip = clip[:, :, ::-1, :].copy()

        # brightness + contrast
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-0.08, 0.08)

        clip = np.clip(
            clip * alpha + beta,
            0.0,
            1.0
        )

        # temporal crop
        T = clip.shape[0]

        if T > 6 and random.random() < 0.4:

            start = random.randint(0, 2)
            end   = T - random.randint(0, 2)

            clip = GestureDataset._sample_frames(
                clip[start:end],
                T
            )

        # gaussian blur
        if random.random() < 0.2:

            blurred = []

            ksize = random.choice([3, 5])

            for frame in clip:

                f = (frame * 255).astype(np.uint8)

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

    # ─────────────────────────────────────────────────────────
    # Class Balancing
    # ─────────────────────────────────────────────────────────

    def class_weights(self):

        counts = torch.zeros(NUM_CLASSES)

        for _, idx in self.samples:
            counts[idx] += 1

        counts = counts.clamp(min=1)

        weights = 1.0 / counts

        weights /= weights.sum()

        return weights

    # ─────────────────────────────────────────────────────────

    def sample_weights(self):

        cw = self.class_weights()

        return [
            float(cw[idx])
            for _, idx in self.samples
        ]


# ─────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────

def make_loaders(
    data_dir,
    batch_size=8,
    num_workers=0,
    val_frac=0.15,
    test_frac=0.10,
    t_frames=T_FRAMES,
    frame_size=(FRAME_H, FRAME_W),
):

    train_ds = GestureDataset(
        data_dir,
        split="train",
        val_frac=val_frac,
        test_frac=test_frac,
        augment=True,
        t_frames=t_frames,
        frame_size=frame_size,
    )

    val_ds = GestureDataset(
        data_dir,
        split="val",
        val_frac=val_frac,
        test_frac=test_frac,
        augment=False,
        t_frames=t_frames,
        frame_size=frame_size,
    )

    test_ds = GestureDataset(
        data_dir,
        split="test",
        val_frac=val_frac,
        test_frac=test_frac,
        augment=False,
        t_frames=t_frames,
        frame_size=frame_size,
    )

    sample_weights = train_ds.sample_weights()

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )