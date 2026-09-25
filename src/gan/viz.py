"""Plots for Day 3: sample grids, training curves, memorisation check."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

C1, C2 = "#2a78d6", "#eb6834"           # validated categorical slots 1 and 2
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def show_image(img: np.ndarray, title: str = "", out: Path | None = None, width: float = 12) -> None:
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def image_rows(rows: list[np.ndarray], row_titles: list[str], out: Path | None = None, size: float = 1.6) -> None:
    """Several rows of images, one label per row (e.g. real vs synthetic)."""
    ncol = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * size, len(rows) * (size + 0.3)), dpi=110)
    axes = np.atleast_2d(axes)
    for r, (imgs, t) in enumerate(zip(rows, row_titles)):
        for c in range(ncol):
            axes[r, c].axis("off")
            if c < len(imgs):
                axes[r, c].imshow(imgs[c])
        axes[r, 0].set_title(t, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def numbered_grid(images: np.ndarray, title: str = "", ncol: int = 8, out: Path | None = None,
                  size: float = 1.5) -> None:
    """Images with their position number on top — used to pick images to reject by hand."""
    n = len(images)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * (size + 0.25)), dpi=100)
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k])
            ax.set_title(str(k), fontsize=8, color=INK_MUTED)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def plot_gan_history(hist, out: Path | None = None) -> None:
    """Three small panels on separate axes: losses, r_t, FID."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), dpi=110)
    a1, a2, a3 = axes
    a1.plot(hist["iter"], hist["loss_d"], color=C1, lw=2, label="discriminator")
    a1.plot(hist["iter"], hist["loss_g"], color=C2, lw=2, label="generator")
    a1.legend(frameon=False, fontsize=8)
    _style(a1, "Hinge losses")
    a2.plot(hist["iter"], hist["r_t"], color=C1, lw=2)
    a2.set_ylim(0, 1.05)
    _style(a2, "r_t: share of real images judged real")
    f = hist.dropna(subset=["fid"])
    a3.plot(f["iter"], f["fid"], color=C1, lw=2, marker="o", ms=5)
    if len(f):
        b = f.loc[f["fid"].idxmin()]
        a3.annotate(f"best {b['fid']:.1f}", (b["iter"], b["fid"]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    _style(a3, "FID vs real training images (lower = better)")
    for a in axes:
        a.set_xlabel("iteration", color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def plot_nn_distances(d_rr: np.ndarray, d_gr: np.ndarray, thresholds: dict, out: Path | None = None) -> None:
    """How close generated images sit to the real ones, compared with real lesions to each other."""
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=120)
    bins = np.linspace(0, max(d_rr.max(), d_gr.max()) * 1.05, 40)
    ax.hist(d_gr, bins=bins, density=True, color=C1, alpha=0.55, label="generated → nearest real image")
    ax.hist(d_rr, bins=bins, density=True, color=C2, alpha=0.55,
            label="real → nearest real image of another lesion")
    for key, txt in [("copy_threshold", "copy threshold"), ("outlier_threshold", "outlier threshold")]:
        v = thresholds.get(key)
        if v == v and v is not None:
            ax.axvline(v, color=INK_MUTED, ls="--", lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.95, f" {txt}", fontsize=8, color=INK_MUTED, va="top")
    ax.set_xlabel("feature distance (Inception, normalised)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("density", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.85))
    _style(ax, "Memorisation check: is the GAN copying its training images?")
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def show_nn_pairs(gen: np.ndarray, real: np.ndarray, nn_idx: np.ndarray, dist: np.ndarray,
                  out: Path | None = None, n: int = 8) -> None:
    """Top row: generated images. Bottom row: the nearest real training image to each."""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 4.0), dpi=110)
    for k in range(n):
        axes[0, k].imshow(gen[k])
        axes[0, k].set_title(f"distance {dist[k]:.2f}", fontsize=8, color=INK_MUTED)
        axes[1, k].imshow(real[nn_idx[k]])
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("Top: generated images   ·   Bottom: the most similar real training image",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()
