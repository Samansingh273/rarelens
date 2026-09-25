"""Step 3b — PyTorch Dataset / DataLoader over the cached image array.

The Dataset takes an optional `transform` so Day 2 can plug in augmentation
(e.g. only for the rare class) without touching anything else here. Day 1 uses
`eval_transform()` for every split: ToTensor + ImageNet normalisation, nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD


def eval_transform():
    """uint8 HWC array -> normalised float CHW tensor. No augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),                       # HWC uint8 [0,255] -> CHW float [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


class HAM10000Dataset(Dataset):
    """Rows of `split_df` index into `images` via the `cache_idx` column.

    images    : (N, H, W, 3) uint8 array/memmap (the cache from preprocess.py)
    split_df  : rows for ONE split, with columns image_id, label_idx, cache_idx
    transform : callable(uint8 HWC ndarray) -> tensor
    class_transforms : optional {label_idx: transform} overriding `transform`
                       for specific classes (used on Day 2 for rare-class-only aug)
    """

    def __init__(self, images: np.ndarray, split_df: pd.DataFrame, transform=None, class_transforms=None):
        self.images = images
        self.df = split_df.reset_index(drop=True)
        self.cache_idx = self.df["cache_idx"].to_numpy()
        self.labels = self.df["label_idx"].to_numpy().astype(np.int64)
        self.transform = transform or eval_transform()
        self.class_transforms = class_transforms or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = np.array(self.images[self.cache_idx[i]])  # writable copy out of the memmap
        y = int(self.labels[i])
        tf = self.class_transforms.get(y, self.transform)
        return tf(img), y


def attach_cache_index(split: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Map each split row to its row in the image cache (== row order of metadata)."""
    pos = pd.Series(np.arange(len(meta)), index=meta["image_id"].values)
    split = split.copy()
    split["cache_idx"] = pos.loc[split["image_id"]].values
    return split


def make_loaders(images: np.ndarray, split: pd.DataFrame, batch_size: int = 64, num_workers: int = 2,
                 train_transform=None, train_class_transforms=None, seed: int = 42):
    """Return {'train','val','test'} DataLoaders. Only the train loader shuffles."""
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for s in ("train", "val", "test"):
        ds = HAM10000Dataset(
            images, split[split["split"] == s],
            transform=train_transform if s == "train" else None,
            class_transforms=train_class_transforms if s == "train" else None,
        )
        loaders[s] = DataLoader(
            ds, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
            generator=g if s == "train" else None,
            persistent_workers=num_workers > 0,
        )
    return loaders
