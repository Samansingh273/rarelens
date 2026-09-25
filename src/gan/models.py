"""A small DCGAN for 128x128 images, sized for a data-starved setting (71 real images).

Generator     : z (128-d noise) -> 4x4 -> 8 -> 16 -> 32 -> 64 -> 128 RGB.
                Upsample + 3x3 conv (instead of transposed conv) avoids checkerboard artefacts, and
                mirror ("reflect") padding avoids a light frame along the image border. That frame
                would appear only on synthetic (rare-class) images and could become a shortcut
                for the classifier on Day 4.
Discriminator : 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> real/fake score.
                Spectral normalisation on every layer keeps it from overpowering the
                generator — the main stability trick besides DiffAugment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, width: int = 64, img_size: int = 128):
        super().__init__()
        assert img_size in (64, 128), "img_size must be 64 or 128"
        self.z_dim, self.img_size = z_dim, img_size
        n_up = {64: 4, 128: 5}[img_size]
        chans = [width * 8 // (2 ** i) for i in range(n_up + 1)]          # 512,256,128,64,32,16
        chans = [max(c, 16) for c in chans]
        self.fc = nn.Sequential(nn.Linear(z_dim, chans[0] * 16), nn.BatchNorm1d(chans[0] * 16), nn.ReLU(True))
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(cin, cout, 3, 1, 1, bias=False, padding_mode="reflect"),
                       nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(nn.Conv2d(chans[-1], 3, 3, 1, 1, padding_mode="reflect"), nn.Tanh())
        self.c0 = chans[0]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.c0, 4, 4)
        return self.to_rgb(self.blocks(x))                                # values in [-1, 1]


class Discriminator(nn.Module):
    def __init__(self, width: int = 32, img_size: int = 128):
        super().__init__()
        n_down = {64: 4, 128: 5}[img_size]
        chans = [3] + [min(width * (2 ** i), 512) for i in range(n_down)]  # 3,32,64,128,256,512
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        self.body = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Linear(chans[-1] * 16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).flatten(1)).squeeze(1)              # raw score (hinge loss)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
