"""Day 2 — rare-class augmentation baselines.

Experimental design ("equal extra budget"):
    training set = ALL real training images (unchanged)
                 + N_EXTRA additional rare-class samples produced by method X

Every method adds the SAME number of extra rare-class samples, so the
comparison is only about *how* those samples are made:

    oversample_only : N copies of real rare images, no change      (control: pure rebalancing)
    classic_aug     : N copies of real rare images, randomly rotated/flipped/cropped/jittered
                      (a fresh random version every epoch)
    smote_interp    : N new images, each a pixel blend of two real rare images from
                      *different* lesions that are near-neighbours in CNN feature space
    gan (Day 4)     : N images from the GAN trained on the rare class

`oversample_only` separates "the rare class is simply seen more often" from
"the extra samples carry new visual variation" — without it, any gain from
augmentation (or the GAN) could just be a rebalancing effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD
from .data.dataset import HAM10000Dataset, eval_transform


# --------------------------------------------------------------------------- #
# Classic augmentation (rotation / flip / crop / brightness jitter)
# --------------------------------------------------------------------------- #
class RandomRot90:
    """Rotate by 0/90/180/270 degrees — lossless, no interpolation, no empty corners."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = int(torch.randint(4, (1,)))
        return img if k == 0 else img.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])


class RandomRotateReflect:
    """Small random rotation with MIRROR padding instead of black fill.

    A plain RandomRotation fills the corners with black. Because only rare-class
    images are augmented, black corners would appear on rare-class images only and
    the classifier could learn "black corners => rare class" — a shortcut that does
    not exist in the (un-augmented) test images. Mirror padding avoids that.
    """

    def __init__(self, degrees: float = 20):
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        pad = int(np.ceil(max(h, w) * (np.sqrt(2) - 1) / 2)) + 2   # enough for any angle
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        rot = Image.fromarray(padded).rotate(angle, resample=Image.BILINEAR)
        return rot.crop((pad, pad, pad + w, pad + h))


def classic_rare_transform(size: int = 224):
    """uint8 HWC array -> augmented, normalised tensor.

    Dermoscopy has no 'up', so flips and 90-degree turns are all valid (together they
    cover all 8 orientations); a further small rotation, random crop and brightness /
    contrast jitter add variation. Hue is barely touched: colour is diagnostic.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90(),
        RandomRotateReflect(20),
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Real training set + N extra rare-class samples
# --------------------------------------------------------------------------- #
class RealPlusExtra(Dataset):
    """All real training images (no augmentation) followed by `n_extra` rare-class samples.

    Give EITHER
      extra_cache_idx : indices into `images` (re-used real rare images; oversampling / classic aug)
      extra_array     : (n, H, W, 3) uint8 array of new images (SMOTE today, GAN on Day 4)
    extra_transform is applied to the extra samples only (default: no augmentation).
    """

    def __init__(self, images: np.ndarray, train_df: pd.DataFrame, rare_idx: int,
                 extra_cache_idx=None, extra_array=None, extra_transform=None):
        assert (extra_cache_idx is None) != (extra_array is None), "give exactly one extra source"
        self.real = HAM10000Dataset(images, train_df)
        self.images = images
        self.extra_cache_idx = None if extra_cache_idx is None else np.asarray(extra_cache_idx)
        self.extra_array = extra_array
        self.n_extra = len(self.extra_cache_idx) if extra_array is None else len(extra_array)
        self.extra_transform = extra_transform or eval_transform()
        self.rare_idx = rare_idx
        self.labels = np.concatenate([self.real.labels, np.full(self.n_extra, rare_idx, dtype=np.int64)])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        n_real = len(self.real)
        if i < n_real:
            return self.real[i]
        j = i - n_real
        img = (np.array(self.images[self.extra_cache_idx[j]]) if self.extra_array is None
               else np.array(self.extra_array[j]))
        return self.extra_transform(img), self.rare_idx


def sample_rare_indices(train_df: pd.DataFrame, rare_idx: int, n: int, seed: int = 42) -> np.ndarray:
    """n cache indices re-using the real rare training images as EVENLY as possible.

    With 71 real images and n=500, every image is used 7 times and 3 of them 8 times
    (rather than random draws where some would be used 12x and others 2x).
    """
    pool = train_df.loc[train_df["label_idx"] == rare_idx, "cache_idx"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(pool) for _ in range(int(np.ceil(n / len(pool))))])[:n]
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------- #
# SMOTE-style interpolation for images
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _features(x_uint8: np.ndarray, feature_space: str, device) -> np.ndarray:
    """L2-normalised embeddings used only to decide which rare images are neighbours."""
    if feature_space == "resnet":
        try:
            from torchvision import models
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = torch.nn.Identity()
            net = net.to(device).eval()
            tf = eval_transform()
            feats = []
            for k in range(0, len(x_uint8), 64):
                batch = torch.stack([tf(np.array(im)) for im in x_uint8[k:k + 64]]).to(device)
                feats.append(net(batch).float().cpu())
            f = torch.cat(feats).numpy()
        except Exception as e:  # e.g. no internet for the weights
            print(f"WARNING: ResNet features unavailable ({e}); falling back to pixel space.")
            feature_space = "pixel"
    if feature_space == "pixel":
        t = torch.from_numpy(np.asarray(x_uint8)).permute(0, 3, 1, 2).float()
        f = torch.nn.functional.adaptive_avg_pool2d(t, 16).flatten(1).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True).clip(min=1e-8)


def smote_rare_images(images: np.ndarray, train_df: pd.DataFrame, rare_idx: int, n_new: int,
                      k: int = 5, seed: int = 42, feature_space: str = "resnet", device=None):
    """Create `n_new` synthetic rare-class images by SMOTE-style interpolation.

    For each new image: pick a random real rare image A, pick B at random from A's k nearest
    rare neighbours that belong to a DIFFERENT lesion (two photos of the same lesion would
    add almost nothing), draw lam ~ U(0, 1), and output  A + lam * (B - A)  in pixel space.

    Returns (array uint8 (n_new, H, W, 3), provenance DataFrame with image ids and lam).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    rare = train_df[train_df["label_idx"] == rare_idx].reset_index(drop=True)
    X = np.stack([np.array(images[c]) for c in rare["cache_idx"]])
    lesions = rare["lesion_id"].to_numpy()

    feats = _features(X, feature_space, device)
    sim = feats @ feats.T
    sim[lesions[:, None] == lesions[None, :]] = -np.inf           # exclude self and same-lesion photos
    kk = min(k, int(np.isfinite(sim).sum(1).min()))
    if kk < 1:
        raise ValueError("Every rare image would need a neighbour from a different lesion.")
    neighbours = np.argsort(-sim, axis=1)[:, :kk]

    out = np.empty((n_new, *X.shape[1:]), dtype=np.uint8)
    rows = []
    for m in range(n_new):
        a = int(rng.integers(len(X)))
        b = int(neighbours[a, rng.integers(kk)])
        lam = float(rng.uniform(0, 1))
        out[m] = np.clip(X[a].astype(np.float32) + lam * (X[b].astype(np.float32) - X[a]), 0, 255).round()
        rows.append({"a_image_id": rare.at[a, "image_id"], "b_image_id": rare.at[b, "image_id"],
                     "lam": round(lam, 4), "similarity": round(float(sim[a, b]), 4)})
    return out, pd.DataFrame(rows)
