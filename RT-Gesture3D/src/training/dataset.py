"""
dataset.py  —  RT-Gesture3D
============================
PyTorch Dataset for loading gesture clips saved by collect_data.py.

Each clip is a .npy file of shape (T, H, W, 3)  uint8.
The loader:
  - Pads/crops to fixed temporal length
  - Resizes to target spatial size
  - Normalises to float32 [0, 1]
  - Applies optional augmentation
  - Returns tensor (3, T, H, W)  — PyTorch 3D-CNN convention
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── default constants (match model input) ────────────────────────────────────
T_FRAMES = 16
FRAME_H  = 112
FRAME_W  = 112

GESTURE_CLASSES = ["neutral", "victory", "ok", "perfect", "stop", "rock", "calm"]
CLASS_TO_IDX    = {g: i for i, g in enumerate(GESTURE_CLASSES)}


# ════════════════════════════════════════════════════════════════════════════
class GestureDataset(Dataset):
    """
    Parameters
    ----------
    data_dir   : path to data/raw/
    split      : 'train' | 'val' | 'test'
    val_frac   : fraction of each class used for val  (default 0.15)
    test_frac  : fraction of each class used for test (default 0.10)
    augment    : apply random augmentation (training only)
    t_frames   : temporal clip length
    frame_size : (H, W) spatial size
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        augment: bool = False,
        t_frames: int = T_FRAMES,
        frame_size: Tuple[int, int] = (FRAME_H, FRAME_W),
    ) -> None:
        super().__init__()
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.augment    = augment
        self.t_frames   = t_frames
        self.frame_size = frame_size

        self.samples: List[Tuple[Path, int]] = []
        self._collect_samples(val_frac, test_frac)

        print(f"[GestureDataset] {split:5s} → {len(self.samples)} clips")

    # ── internal ─────────────────────────────────────────────────────────────

    def _collect_samples(self, val_frac, test_frac):
        for gesture in GESTURE_CLASSES:
            cls_dir = self.data_dir / gesture
            if not cls_dir.exists():
                continue

            clips = sorted(cls_dir.glob("clip_*.npy"))
            n     = len(clips)
            if n == 0:
                continue

            # deterministic split by index
            n_test = max(1, int(n * test_frac))
            n_val  = max(1, int(n * val_frac))
            n_train = n - n_test - n_val

            if self.split == "train":
                chosen = clips[:n_train]
            elif self.split == "val":
                chosen = clips[n_train: n_train + n_val]
            else:  # test
                chosen = clips[n_train + n_val:]

            idx = CLASS_TO_IDX[gesture]
            self.samples.extend((p, idx) for p in chosen)

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        clip = self._load_clip(path)               # (T, H, W, 3)  float32
        if self.augment:
            clip = self._augment(clip)
        tensor = self._to_tensor(clip)             # (3, T, H, W)  float32
        return tensor, label

    # ── clip loading + preprocessing ─────────────────────────────────────────

    def _load_clip(self, path: Path) -> np.ndarray:
        """Load .npy clip and normalise to float32 [0,1]."""
        arr = np.load(str(path))                   # (T, H, W, 3)  uint8

        # temporal sampling / padding
        arr = self._sample_frames(arr, self.t_frames)

        # spatial resize
        if arr.shape[1:3] != (self.frame_size[0], self.frame_size[1]):
            resized = []
            for f in arr:
                f = cv2.resize(f, (self.frame_size[1], self.frame_size[0]),
                               interpolation=cv2.INTER_LINEAR)
                resized.append(f)
            arr = np.stack(resized, axis=0)

        return arr.astype(np.float32) / 255.0

    @staticmethod
    def _sample_frames(arr: np.ndarray, t: int) -> np.ndarray:
        """
        Uniformly sample exactly *t* frames from a clip of any length.
        If clip is shorter than t, loop-pad the last frame.
        """
        T_orig = arr.shape[0]
        if T_orig == t:
            return arr
        if T_orig < t:
            pad = np.repeat(arr[-1:], t - T_orig, axis=0)
            return np.concatenate([arr, pad], axis=0)
        # uniform subsample
        indices = np.linspace(0, T_orig - 1, t, dtype=int)
        return arr[indices]

    @staticmethod
    def _to_tensor(clip: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) float32 → (3, T, H, W) float32 tensor."""
        t = torch.from_numpy(clip)                 # (T, H, W, 3)
        t = t.permute(3, 0, 1, 2)                 # (3, T, H, W)
        return t

    # ── augmentation ─────────────────────────────────────────────────────────

    @staticmethod
    def _augment(clip: np.ndarray) -> np.ndarray:
        """
        In-place-safe augmentation for gesture clips.
        clip : (T, H, W, 3)  float32
        """
        # horizontal flip
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()

        # brightness / contrast jitter
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-0.08, 0.08)
        clip  = np.clip(clip * alpha + beta, 0.0, 1.0)

        # random temporal crop (drop 1-2 frames and re-sample)
        T = clip.shape[0]
        if T > 6 and random.random() < 0.4:
            start = random.randint(0, 2)
            end   = T - random.randint(0, 2)
            clip  = GestureDataset._sample_frames(clip[start:end], T)

        # random gaussian blur (simulate motion blur)
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            blurred = []
            for f in clip:
                f_u8 = (f * 255).astype(np.uint8)
                f_u8 = cv2.GaussianBlur(f_u8, (ksize, ksize), 0)
                blurred.append(f_u8.astype(np.float32) / 255.0)
            clip = np.stack(blurred, axis=0)

        return clip

    # ── class weights (for imbalanced datasets) ───────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """Returns per-class weights inversely proportional to frequency."""
        counts = torch.zeros(len(GESTURE_CLASSES))
        for _, idx in self.samples:
            counts[idx] += 1
        counts = counts.clamp(min=1)
        weights = 1.0 / counts
        weights /= weights.sum()
        return weights

    def sample_weights(self) -> List[float]:
        cw = self.class_weights()
        return [float(cw[idx]) for _, idx in self.samples]


# ════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ════════════════════════════════════════════════════════════════════════════

def make_loaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    t_frames: int = T_FRAMES,
    frame_size: Tuple[int, int] = (FRAME_H, FRAME_W),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Training uses a WeightedRandomSampler to handle class imbalance.
    """
    train_ds = GestureDataset(data_dir, "train", val_frac, test_frac,
                              augment=True, t_frames=t_frames, frame_size=frame_size)
    val_ds   = GestureDataset(data_dir, "val",   val_frac, test_frac,
                              augment=False, t_frames=t_frames, frame_size=frame_size)
    test_ds  = GestureDataset(data_dir, "test",  val_frac, test_frac,
                              augment=False, t_frames=t_frames, frame_size=frame_size)

    # weighted sampler for training
    sw = train_ds.sample_weights()
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
