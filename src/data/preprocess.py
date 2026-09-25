"""Step 3a — resize every image once and cache the lot as a single uint8 .npy array.

Decoding 10,015 JPEGs (600x450) every epoch is the slowest part of training on
Colab. Instead we resize once to 224x224 and store an array of shape
(N, 224, 224, 3) uint8 (~1.5 GB) on Google Drive. Later days reuse it, so the
2.6 GB raw download is only needed once.

Resize choice: the full 600x450 frame is resized to 224x224 (a mild 4:3 -> 1:1
squash) rather than centre-cropped, so no part of a large lesion is cut off.
Row i of the cache corresponds to row i of data/processed/metadata.csv.

Colab note: Google Drive is slow for random reads, so the cache is built on
local disk, copied to Drive once, and loaded fully into RAM for training.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from ..config import Paths


def _load_resized(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)


def _load(path, in_memory: bool) -> np.ndarray:
    if in_memory:
        print(f"Loading {path.name} into RAM ...")
        return np.load(path)
    return np.load(path, mmap_mode="r")


def build_image_cache(meta: pd.DataFrame, paths: Paths, size: int = 224, workers: int = 8,
                      overwrite: bool = False, in_memory: bool = True) -> np.ndarray:
    """Create (or reuse) the resized image cache and return it as an array."""
    out = paths.image_cache(size)
    if out.exists() and not overwrite:
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(meta):
            print(f"[reuse] image cache {out}  shape={arr.shape}")
            del arr
            return _load(out, in_memory)
        print(f"Cache has {arr.shape[0]} rows but metadata has {len(meta)} — rebuilding.")
        del arr
    if "path" not in meta:
        raise RuntimeError("Raw images are needed to build the cache — run the download step first.")

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    local = paths.raw_dir / f"_building_{out.name}"          # fast local disk
    arr = np.lib.format.open_memmap(local, mode="w+", dtype=np.uint8, shape=(len(meta), size, size, 3))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, img in enumerate(tqdm(ex.map(lambda p: _load_resized(p, size), meta["path"]),
                                     total=len(meta), desc=f"resize→{size}px")):
            arr[i] = img
    arr.flush()
    del arr

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npy")
    shutil.copyfile(local, tmp)
    tmp.replace(out)       # atomic rename: a half-copied cache is never mistaken for a finished one
    local.unlink()
    print(f"Saved {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return _load(out, in_memory)
