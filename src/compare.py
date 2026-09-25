"""Comparison table + chart built from results/results_log.csv (one row per run)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preferred display order of the experiment arms.
CONFIG_ORDER = ["baseline_no_aug", "oversample_only", "classic_aug", "smote_interp", "gan_aug"]
LABELS = {
    "baseline_no_aug": "Baseline (no aug)",
    "oversample_only": "Oversample only",
    "classic_aug": "Classic aug",
    "smote_interp": "SMOTE interp.",
    "gan_aug": "GAN aug",
}
# Validated categorical slots 1-3; text in neutral ink.
SERIES = [("rare_precision", "Precision", "#2a78d6"),
          ("rare_recall", "Recall", "#eb6834"),
          ("rare_f1", "F1", "#1baf7a")]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (well-behaved for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def comparison_table(log_path: Path, seed: int | None = 42, rare: str = "DF") -> pd.DataFrame:
    """Latest run per configuration (for one seed), with change vs the baseline.

    `recall_95ci` is the uncertainty from the small rare-class test set alone
    (e.g. 9/20 -> 0.26-0.66). Configurations whose intervals overlap heavily are
    not distinguishable from one run — that is what the multi-seed Days 6-7 address.
    """
    log = pd.read_csv(log_path)
    if "rare_class" in log.columns:
        log = log[log["rare_class"].fillna("DF") == rare]
    if seed is not None:
        log = log[log["seed"] == seed]
    log = log.sort_values("timestamp").groupby("config", as_index=False).last()
    order = [c for c in CONFIG_ORDER if c in set(log["config"])] + \
            [c for c in log["config"] if c not in CONFIG_ORDER]
    t = log.set_index("config").loc[order]
    cols = ["n_synthetic", "n_train_rare", "test_accuracy", "test_macro_f1",
            "rare_precision", "rare_recall", "rare_f1", "rare_auroc", "best_epoch"]
    cis = [wilson_ci(int(round(r * s)), int(s)) for r, s in zip(t["rare_recall"], t["rare_support"])]
    t = t[cols].rename(columns={"n_synthetic": "n_extra"})
    t.insert(t.columns.get_loc("rare_recall") + 1, "recall_95ci", [f"{lo:.2f}–{hi:.2f}" for lo, hi in cis])
    if "baseline_no_aug" in t.index:
        base = t.loc["baseline_no_aug"]
        for c in ["rare_recall", "rare_f1"]:
            t[f"Δ{c}"] = (t[c] - base[c]).round(3)
    t.index = [LABELS.get(i, i) for i in t.index]
    return t.round(3)


def plot_rare_metrics(table: pd.DataFrame, rare: str, out: Path) -> None:
    """Grouped bars: rare-class precision / recall / F1 per configuration (one shared 0-1 axis)."""
    n = len(table)
    x = np.arange(n)
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(6.5, 1.9 * n + 2), 4.2), dpi=130)
    for s, (col, name, color) in enumerate(SERIES):
        bars = ax.bar(x + (s - 1) * w, table[col], w, label=name, color=color, edgecolor="white", linewidth=2)
        if col == "rare_f1":  # label only the headline metric
            for b, v in zip(bars, table[col]):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, table.index, color=INK_MUTED)
    ax.set_ylim(0, 1.16)                      # headroom so the legend + labels never cover a bar
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel(f"{rare} test score", color=INK_MUTED)
    ax.set_title(f"Rare class ({rare}) on the held-out test set", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()


def show_augmented_examples(images: np.ndarray, cache_idx: int, transform, out: Path, n: int = 7) -> None:
    """One real rare image followed by n random classic-augmentation draws of it."""
    from .data.dataset import denormalize
    base = np.array(images[cache_idx])
    fig, axes = plt.subplots(1, n + 1, figsize=(1.8 * (n + 1), 2.1), dpi=110)
    axes[0].imshow(base)
    axes[0].set_title("original", fontsize=9, color=INK)
    for ax in axes[1:]:
        ax.imshow(denormalize(transform(base)).permute(1, 2, 0).numpy())
        ax.set_title("augmented", fontsize=9, color=INK_MUTED)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()


def show_smote_examples(images: np.ndarray, id_to_cache: dict, synth: np.ndarray, prov: pd.DataFrame,
                        out: Path, n: int = 4) -> None:
    """Rows of (real A, real B, synthetic blend) so the interpolation can be judged by eye."""
    fig, axes = plt.subplots(n, 3, figsize=(6, 2.1 * n), dpi=110)
    for r in range(n):
        p = prov.iloc[r]
        for c, (img, title) in enumerate([(images[id_to_cache[p.a_image_id]], "real A"),
                                          (images[id_to_cache[p.b_image_id]], "real B (neighbour)"),
                                          (synth[r], f"synthetic, λ={p.lam:.2f}")]):
            axes[r, c].imshow(np.array(img))
            axes[r, c].set_title(title, fontsize=9, color=INK if c == 2 else INK_MUTED)
            axes[r, c].axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()
