"""Differentiable augmentation for data-efficient GAN training.

Idea from Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training"
(NeurIPS 2020). With only ~70 real images the discriminator quickly memorises them
and training collapses. Randomly augmenting BOTH real and generated images every time
the discriminator sees them — with augmentations that gradients can flow through —
stops that memorisation without the augmentations leaking into the generated images.

Policy used here: colour (brightness / saturation / contrast), translation, cutout.
Inputs and outputs are image tensors in [-1, 1], shape (B, 3, H, W).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    for p in policy.split(","):
        for fn in AUGMENT_FNS.get(p.strip(), []):
            x = fn(x)
    return x.contiguous()


def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + mean


def _contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean


def _translation(x, ratio: float = 0.125):
    """Shift by up to 1/8 of the image size, filling with zeros (padding shared by real and fake)."""
    b, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    xp = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio: float = 0.5):
    """Blank out one random square covering up to a quarter of the image."""
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [_brightness, _saturation, _contrast],
    "translation": [_translation],
    "cutout": [_cutout],
}


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random flip + 90-degree turn per image (valid for dermoscopy: no 'up').

    Applied to REAL images only, before DiffAugment. Unlike DiffAugment these
    symmetries are genuine properties of the data, so it is fine for the
    generator to learn them — it effectively turns 71 images into 8 x 71 views.
    """
    out = []
    for img in x:
        k = int(torch.randint(4, (1,)))
        img = torch.rot90(img, k, dims=(1, 2))
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=(2,))
        out.append(img)
    return torch.stack(out)
