"""
models.py  —  RT-Gesture3D
==========================
Two model families:

1. GestureCNN3D   — pure 3D-CNN (C3D-style)
   Input : (B, 3, T, H, W)   T=16, H=W=112
   Output: (B, num_classes)

2. GestureST_CNN  — Spatiotemporal CNN
   Spatial  stream: 2D-CNN on centre frame
   Temporal stream: 3D-CNN on optical-flow stack
   Fusion: late fusion (concat → FC)

Both are lightweight enough to run in real time on CPU/MPS/CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GestureCNN3D  (C3D-inspired, trimmed for real-time)
# ══════════════════════════════════════════════════════════════════════════════

class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GestureCNN3D(nn.Module):
    """
    Lightweight C3D variant.

    Architecture summary
    --------------------
    Input  : (B, 3, 16, 112, 112)
    conv1  : (B, 32,  16, 112, 112)
    pool1  : (B, 32,  8,  56,  56)
    conv2  : (B, 64,  8,  56,  56)
    pool2  : (B, 64,  4,  28,  28)
    conv3a : (B, 128, 4,  28,  28)
    conv3b : (B, 128, 4,  28,  28)
    pool3  : (B, 128, 2,  14,  14)
    conv4a : (B, 256, 2,  14,  14)
    conv4b : (B, 256, 2,  14,  14)
    pool4  : (B, 256, 1,  7,   7)
    GAP    : (B, 256)
    fc1    : (B, 512)
    fc2    : (B, num_classes)
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        self.conv1 = _Conv3dBnRelu(3,   32,  (3,3,3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv2 = _Conv3dBnRelu(32,  64,  (3,3,3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv3a = _Conv3dBnRelu(64,  128, (3,3,3), padding=1)
        self.conv3b = _Conv3dBnRelu(128, 128, (3,3,3), padding=1)
        self.pool3  = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv4a = _Conv3dBnRelu(128, 256, (3,3,3), padding=1)
        self.conv4b = _Conv3dBnRelu(256, 256, (3,3,3), padding=1)
        self.pool4  = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # Global Average Pool → removes spatial dims
        self.gap = nn.AdaptiveAvgPool3d(1)   # (B, 256, 1, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, T, H, W)"""
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.gap(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 256-d feature vector (before classifier)."""
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.gap(x)
        return x.flatten(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GestureST_CNN  —  Spatiotemporal two-stream network
# ══════════════════════════════════════════════════════════════════════════════

class _SpatialStream(nn.Module):
    """2D-CNN on the centre frame — captures appearance."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(3,  32, 3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                 # 56×56
            # block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                 # 28×28
            # block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                 # 14×14
            # block 4
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                         # 1×1
            nn.Flatten(),                                    # 256
        )

    def forward(self, x):
        return self.net(x)


class _TemporalStream(nn.Module):
    """
    3D-CNN on the full clip — captures motion.
    Input: (B, 3, T, H, W)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv3d(3,  32, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),                     # T×56×56

            # block 2
            nn.Conv3d(32, 64, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),                     # T/2×28×28

            # block 3
            nn.Conv3d(64, 128, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),                     # T/4×14×14

            # block 4
            nn.Conv3d(128, 256, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),                   # 1×1×1
            nn.Flatten(),                              # 256
        )

    def forward(self, x):
        return self.net(x)


class GestureST_CNN(nn.Module):
    """
    Two-stream Spatiotemporal CNN.

    Spatial  stream → 256-d
    Temporal stream → 256-d
    Concat          → 512-d
    FC              → num_classes
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        self.spatial_stream  = _SpatialStream()
        self.temporal_stream = _TemporalStream()

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _centre_frame(self, clip: torch.Tensor) -> torch.Tensor:
        """Extract the centre frame from (B, 3, T, H, W)."""
        T = clip.shape[2]
        return clip[:, :, T // 2, :, :]   # (B, 3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, T, H, W)"""
        sp = self.spatial_stream(self._centre_frame(x))   # (B, 256)
        tp = self.temporal_stream(x)                       # (B, 256)
        fused = torch.cat([sp, tp], dim=1)                 # (B, 512)
        return self.fusion(fused)                          # (B, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.spatial_stream(self._centre_frame(x))
        tp = self.temporal_stream(x)
        return torch.cat([sp, tp], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    arch: str = "st_cnn",
    num_classes: int = 7,
    pretrained_path: str | None = None,
    device: str = "cpu",
) -> nn.Module:
    """
    arch: 'cnn3d'  → GestureCNN3D
          'st_cnn' → GestureST_CNN  (default, recommended)
    """
    arch = arch.lower()

    if arch == "cnn3d":
        model = GestureCNN3D(num_classes=num_classes)
    elif arch == "st_cnn":
        model = GestureST_CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'cnn3d' or 'st_cnn'.")

    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Weights loaded from {pretrained_path}")

    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# quick sanity-check
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 16, 112, 112
    x = torch.randn(B, C, T, H, W)

    print("── GestureCNN3D ──")
    m1 = GestureCNN3D(num_classes=7)
    y1 = m1(x)
    params1 = sum(p.numel() for p in m1.parameters()) / 1e6
    print(f"  output : {y1.shape}")
    print(f"  params : {params1:.2f}M")

    print("\n── GestureST_CNN ──")
    m2 = GestureST_CNN(num_classes=7)
    y2 = m2(x)
    params2 = sum(p.numel() for p in m2.parameters()) / 1e6
    print(f"  output : {y2.shape}")
    print(f"  params : {params2:.2f}M")
