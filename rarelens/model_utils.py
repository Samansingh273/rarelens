"""Shared by prepare.py and engine.py so the app scores images exactly as its numbers were measured.

* App model  = the two GAN-augmented ResNet-18s (DF study + VASC study, seed 42) averaged, with
               8-way test-time augmentation (4 rotations x mirror; skin lesions have no "up").
* Unusual-image check = deep nearest-neighbour distance (Sun et al., ICML 2022) in the feature space of
               a general-purpose ImageNet ResNet-18: distance to the k-th nearest TRAINING image, flagged when
               larger than for 99 % of validation images. The classifier's own features are NOT used: they map
               every input into one of the 7 disease clusters, so a sunset can land "close" to the moles.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

KNN_K = 10


def dihedral(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (8B, 3, H, W): the 8 rotations/mirrors, grouped by transform."""
    outs = []
    for k in range(4):
        r = torch.rot90(x, k, (2, 3))
        outs += [r, torch.flip(r, (3,))]
    return torch.cat(outs)


@torch.no_grad()
def ensemble_probs(models, x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    b = x.shape[0]
    xs = dihedral(x) if tta else x
    p = 0
    for m in models:
        q = torch.softmax(m(xs).float(), 1)
        p = p + (q.view(8, b, -1).mean(0) if tta else q)
    return p / len(models)


@torch.no_grad()
def penultimate(m, x: torch.Tensor) -> torch.Tensor:
    """L2-normalised 512-d features before the final layer of a torchvision ResNet."""
    h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    h = m.layer4(m.layer3(m.layer2(m.layer1(h))))
    return F.normalize(m.avgpool(h).flatten(1).float(), dim=1)


def knn_distance(ref: torch.Tensor, f: torch.Tensor, k: int = KNN_K, exclude_self: bool = False) -> torch.Tensor:
    """Cosine distance from each row of f to its k-th nearest row of ref."""
    sims = f @ ref.T
    kk = k + 1 if exclude_self else k
    return 1 - sims.topk(kk, dim=1).values[:, -1]


def imagenet_backbone(device):
    """ImageNet ResNet-18 (torchvision; loaded from the local cache after the first download)."""
    from torchvision import models
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
