"""Step 1b — build a tidy metadata table, inspect the class distribution, pick the rare class.

Rare-class rule (from the blueprint): a class with well under 2% of all images.
Among those candidates we pick the one with the *fewest distinct lesions*,
because that is the true number of independent examples a classifier (and later
the GAN) can learn from — HAM10000 often has several photos of the same lesion.
Ties are broken by fewest images.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..config import CLASS_CODES, CLASS_NAMES, RARE_SHARE_THRESHOLD, Paths

# Chart colours (validated reference palette): slot 1 for ordinary classes,
# slot 2 to highlight rare-class candidates. Text stays in neutral ink.
C_COMMON, C_RARE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _find(root: Path, pattern: str) -> Path:
    hits = sorted(Path(root).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}. Run the download step first.")
    return hits[0]


def build_metadata(paths: Paths) -> pd.DataFrame:
    """One row per image: image_id, lesion_id, label, label_idx, confirm_type, path.

    If the raw download is gone (e.g. a fresh Colab runtime on a later day) but
    metadata.csv was saved on Drive, that saved table is returned instead (without
    the `path` column — which later days don't need because they use the cache).
    """
    raw_present = paths.raw_dir.exists() and (
        paths.raw_labels_csv.exists() or any(paths.raw_dir.rglob("*Task3_Training_GroundTruth.csv")))
    if not raw_present and paths.metadata_csv.exists():
        print(f"[reuse] raw data not on disk; loading saved {paths.metadata_csv}")
        return pd.read_csv(paths.metadata_csv)

    labels_csv = paths.raw_labels_csv if paths.raw_labels_csv.exists() else _find(paths.raw_dir, "*Task3_Training_GroundTruth.csv")
    lesion_csv = paths.raw_lesion_csv if paths.raw_lesion_csv.exists() else _find(paths.raw_dir, "*LesionGroupings.csv")

    gt = pd.read_csv(labels_csv)
    onehot = gt[CLASS_CODES].to_numpy()
    if not np.allclose(onehot.sum(1), 1):
        raise ValueError("Ground-truth rows are not one-hot — unexpected file format.")
    meta = pd.DataFrame({
        "image_id": gt["image"],
        "label_idx": onehot.argmax(1).astype(int),
    })
    meta["label"] = [CLASS_CODES[i] for i in meta["label_idx"]]

    lesions = pd.read_csv(lesion_csv).rename(columns={"image": "image_id", "diagnosis_confirm_type": "confirm_type"})
    meta = meta.merge(lesions, on="image_id", how="left", validate="one_to_one")
    missing = meta["lesion_id"].isna().sum()
    if missing:
        # Should never happen for HAM10000; fall back to "each image is its own lesion".
        print(f"WARNING: {missing} images have no lesion_id; treating each as a separate lesion.")
        meta.loc[meta["lesion_id"].isna(), "lesion_id"] = "NOLESION_" + meta["image_id"]

    img_dir = paths.raw_images_dir if paths.raw_images_dir.exists() else _find(paths.raw_dir, "ISIC_*.jpg").parent
    meta["path"] = [str(img_dir / f"{i}.jpg") for i in meta["image_id"]]
    n_missing_files = sum(not Path(p).exists() for p in meta["path"])
    if n_missing_files:
        raise FileNotFoundError(f"{n_missing_files} image files listed in the labels CSV are missing on disk.")

    # Sanity: one diagnosis per lesion (true in HAM10000; required for a clean grouped split).
    per_lesion = meta.groupby("lesion_id")["label"].nunique()
    if (per_lesion > 1).any():
        raise ValueError(f"{(per_lesion > 1).sum()} lesions carry more than one diagnosis.")

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    meta.drop(columns=["path"]).to_csv(paths.metadata_csv, index=False)
    return meta


def class_distribution(meta: pd.DataFrame) -> pd.DataFrame:
    total = len(meta)
    g = meta.groupby("label")
    df = pd.DataFrame({
        "name": [CLASS_NAMES[c] for c in CLASS_CODES],
        "images": g.size().reindex(CLASS_CODES).fillna(0).astype(int).values,
        "lesions": g["lesion_id"].nunique().reindex(CLASS_CODES).fillna(0).astype(int).values,
    }, index=pd.Index(CLASS_CODES, name="class"))
    df["pct_images"] = 100 * df["images"] / total
    df["imgs_per_lesion"] = df["images"] / df["lesions"].clip(lower=1)
    df["under_2pct"] = df["pct_images"] < 100 * RARE_SHARE_THRESHOLD
    return df.sort_values("images", ascending=False)


def pick_rare_class(dist: pd.DataFrame, override: str | None = None) -> str:
    if override:
        if override not in dist.index:
            raise ValueError(f"Unknown class {override!r}; choose from {list(dist.index)}")
        return override
    cands = dist[dist["under_2pct"]]
    if cands.empty:
        raise ValueError("No class is under the 2% threshold.")
    return cands.sort_values(["lesions", "images"]).index[0]


def plot_class_distribution(dist: pd.DataFrame, rare: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    colors = [C_RARE if u else C_COMMON for u in dist["under_2pct"]]
    bars = ax.bar(dist.index, dist["images"], color=colors, width=0.6, edgecolor="white", linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("Images (log scale)", color=INK_MUTED)
    ax.set_title("HAM10000 class distribution", color=INK, loc="left", fontsize=12, fontweight="bold")
    for b, (_, r) in zip(bars, dist.iterrows()):
        ax.annotate(f"{r['images']:,}\n{r['pct_images']:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(top=dist["images"].max() * 4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_COMMON, label="≥ 2% of images"),
                       Patch(color=C_RARE, label=f"< 2% of images (chosen rare class: {rare})")],
              frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()


def plot_examples(meta: pd.DataFrame, classes: list[str], out: Path, per_class: int = 8, seed: int = 0) -> None:
    """Grid of random example images per class (one lesion per image, to show real variety)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.4, len(classes) * 1.5), dpi=110)
    axes = np.atleast_2d(axes)
    for r, c in enumerate(classes):
        sub = meta[meta["label"] == c].drop_duplicates("lesion_id")
        pick = sub.sample(min(per_class, len(sub)), random_state=int(rng.integers(1e9)))
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(Image.open(pick.iloc[k]["path"]).convert("RGB"))
        axes[r, 0].set_title(f"{c} — {CLASS_NAMES[c]}", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()


def save_dataset_info(paths: Paths, dist: pd.DataFrame, rare: str) -> dict:
    info = {
        "dataset": "HAM10000 (ISIC 2018 Task 3 training set)",
        "n_images": int(dist["images"].sum()),
        "n_lesions": int(dist["lesions"].sum()),
        "rare_class": rare,
        "rare_class_name": CLASS_NAMES[rare],
        "rare_selection_rule": "share < 2% of images; among those, fewest distinct lesions (ties: fewest images)",
        "class_distribution": dist.reset_index().to_dict(orient="records"),
    }
    paths.dataset_info_json.write_text(json.dumps(info, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return info
