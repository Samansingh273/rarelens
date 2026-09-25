"""Step 1a — download HAM10000 (ISIC 2018 Task 3 training set) from the public ISIC bucket.

HAM10000 is the ISIC 2018 Task 3 training set and also the "HAM10000" portion of
ISIC 2019. We use the 2018 release because it ships the lesion-grouping file
(`lesion_id`), which we need for a leakage-free split (several images in
HAM10000 show the *same* lesion).

No login or API key needed. Downloads resume if the connection drops.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from ..config import DOWNLOADS, Paths


def _remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers.get("Content-Length", 0)) or None
    except requests.RequestException:
        return None


def download_file(url: str, dest: Path, chunk: int = 1 << 20, retries: int = 5) -> Path:
    """Stream `url` to `dest`, resuming a partial file if one exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"[skip] {dest.name} already downloaded ({total / 1e6:.1f} MB)")
        return dest

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:  # range not satisfiable -> already complete
                    return dest
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                with dest.open(mode) as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, desc=dest.name
                ) as bar:
                    for block in r.iter_content(chunk_size=chunk):
                        f.write(block)
                        bar.update(len(block))
            if total is None or dest.stat().st_size == total:
                return dest
        except requests.RequestException as e:
            print(f"[retry {attempt}/{retries}] {dest.name}: {e}")
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _extract(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"unzip {zip_path.name}", unit="file"):
            zf.extract(m, out_dir)


def download_ham10000(paths: Paths, keep_zips: bool = False) -> None:
    """Download + extract everything needed into `paths.raw_dir`. Idempotent."""
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    already_there = {
        "ISIC2018_Task3_Training_Input.zip": lambda: len(list(paths.raw_images_dir.glob("*.jpg"))) >= 10_015,
        "ISIC2018_Task3_Training_GroundTruth.zip": lambda: paths.raw_labels_csv.exists(),
        "ISIC2018_Task3_Training_LesionGroupings.csv": lambda: paths.raw_lesion_csv.exists(),
    }

    for name, url in DOWNLOADS.items():
        if already_there[name]():
            print(f"[skip] {name}: already extracted in {paths.raw_dir}")
            continue
        dest = paths.raw_dir / name
        download_file(url, dest)
        if dest.suffix == ".zip":
            _extract(dest, paths.raw_dir)
            if not keep_zips:
                dest.unlink()  # free ~2.6 GB of local disk

    n = len(list(paths.raw_images_dir.glob("*.jpg")))
    print(f"Done. {n} images in {paths.raw_images_dir}")
    if n != 10_015:
        print(f"WARNING: expected 10,015 HAM10000 images, found {n}.")
