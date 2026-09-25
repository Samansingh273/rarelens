"""Day 4 analysis: multi-seed comparison, where each method succeeds / fails, shortcut checks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_CODES, Paths
from .experiments import CONFIG_ORDER, CONFIGS, run_dir_for

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = [("test_accuracy", "Accuracy"), ("test_macro_f1", "Macro-F1"), ("rare_precision", "{r} precision"),
           ("rare_recall", "{r} recall"), ("rare_f1", "{r} F1"), ("rare_auroc", "{r} AUROC")]


def load_log(paths: Paths, rare: str = "DF") -> pd.DataFrame:
    """Results log for one rare class, one row per (config, seed) — the latest run wins if re-run."""
    log = pd.read_csv(paths.results_log)
    log = log[(log["rare_class"].fillna("DF") == rare) & log["config"].isin(CONFIG_ORDER)].sort_values("timestamp")
    log = log.drop_duplicates(["config", "seed"], keep="last")
    log["order"] = log["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return log.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)


def seed_summary(log: pd.DataFrame, rare: str) -> pd.DataFrame:
    """Mean ± sd over seeds for every configuration (display strings)."""
    rows = []
    for cfg in CONFIG_ORDER:
        g = log[log["config"] == cfg]
        if g.empty:
            continue
        row = {"method": CONFIGS[cfg]["label"], "seeds": len(g)}
        for col, name in METRICS:
            m, s = g[col].mean(), g[col].std(ddof=1) if len(g) > 1 else 0.0
            row[name.format(r=rare)] = f"{m:.3f} ± {s:.3f}" if len(g) > 1 else f"{m:.3f}"
        caught = (g["rare_recall"] * g["rare_support"]).round().sum()
        row[f"{rare} caught (all seeds)"] = f"{int(caught)} / {int(g['rare_support'].sum())}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def paired_vs_baseline(log: pd.DataFrame, metric: str = "rare_f1") -> pd.DataFrame:
    """Per-seed difference from the baseline trained with the SAME seed, plus wins/losses."""
    base = log[log["config"] == "baseline_no_aug"].set_index("seed")[metric]
    rows = []
    for cfg in CONFIG_ORDER[1:]:
        g = log[log["config"] == cfg].set_index("seed")[metric]
        common = sorted(set(g.index) & set(base.index))
        if not common:
            continue
        d = (g.loc[common] - base.loc[common])
        row = {"method": CONFIGS[cfg]["label"]}
        row.update({f"seed {s}": round(float(d.loc[s]), 3) for s in common})
        row.update({"mean change": round(float(d.mean()), 3), "better": int((d > 0).sum()),
                    "same": int((d == 0).sum()), "worse": int((d < 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def plot_seed_comparison(log: pd.DataFrame, rare: str, out: Path | None = None) -> None:
    """Three panels on a shared 0-1 scale: bar = mean over seeds, dots = individual seeds."""
    cfgs = [c for c in CONFIG_ORDER if (log["config"] == c).any()]
    labels = [CONFIGS[c]["label"].replace(" (no aug)", "\n(no aug)").replace(" interp.", "\ninterp.")
              for c in cfgs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120, sharey=True)
    for ax, (col, name) in zip(axes, [("rare_precision", "precision"), ("rare_recall", "recall"), ("rare_f1", "F1")]):
        for i, c in enumerate(cfgs):
            vals = log.loc[log["config"] == c, col].to_numpy()
            color = ORANGE if c == "gan_aug" else BLUE
            ax.bar(i, vals.mean(), width=0.62, color=color, alpha=0.85, edgecolor="white", linewidth=2)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), vals, s=22, color=INK, zorder=3, linewidths=0)
            ax.annotate(f"{vals.mean():.2f}", (i + 0.33, vals.mean()), xytext=(2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK)          # label sits beside the bar top
        ax.set_xticks(range(len(cfgs)), labels, fontsize=8, color=INK_MUTED)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{rare} {name}", loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    n_seeds = log.groupby("config").size().max()
    fig.suptitle(f"Rare class on the held-out test set — bars: mean over {n_seeds} seeds, dots: individual seeds "
                 f"(GAN in orange)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def load_predictions(paths: Paths, log: pd.DataFrame) -> dict:
    """{(config, seed): per-image test predictions} for every run in the log."""
    preds = {}
    for _, r in log.iterrows():
        f = run_dir_for(paths, r["config"], r["seed"], r["rare_class"]) / "test_predictions.csv"
        if f.exists():
            preds[(r["config"], int(r["seed"]))] = pd.read_csv(f)
    return preds


def per_image_hits(preds: dict, rare: str) -> pd.DataFrame:
    """For every real rare-class TEST image: the share of seeds in which each method caught it."""
    any_df = next(iter(preds.values()))
    rare_rows = any_df[any_df["label"] == rare][["image_id", "lesion_id"]].reset_index(drop=True)
    out = rare_rows.set_index("image_id")
    for cfg in CONFIG_ORDER:
        runs = [p for (c, _), p in preds.items() if c == cfg]
        if not runs:
            continue
        hits = np.mean([p.set_index("image_id").loc[out.index, "pred"].eq(rare).to_numpy() for p in runs], axis=0)
        out[CONFIGS[cfg]["label"]] = hits.round(2)
    return out


def false_positive_sources(preds: dict, rare: str) -> pd.DataFrame:
    """Average number of test images of each OTHER class wrongly labelled as the rare class."""
    table = {}
    for cfg in CONFIG_ORDER:
        runs = [p for (c, _), p in preds.items() if c == cfg]
        if not runs:
            continue
        counts = [p[(p["pred"] == rare) & (p["label"] != rare)]["label"].value_counts() for p in runs]
        table[CONFIGS[cfg]["label"]] = pd.concat(counts, axis=1).fillna(0).mean(axis=1)
    df = pd.DataFrame(table).reindex([c for c in CLASS_CODES if c != rare]).fillna(0).round(1)
    df.loc["TOTAL false alarms"] = df.sum()
    return df


def _laplacian_var(gray: np.ndarray) -> float:
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


def image_stats(images: np.ndarray) -> dict:
    """Colour and sharpness summary of a (N, H, W, 3) uint8 set."""
    x = images.astype(np.float32)
    r, g, b = x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean()
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    sharp = [_laplacian_var(im) for im in gray]
    return {"n": len(images), "mean R": round(float(r), 1), "mean G": round(float(g), 1), "mean B": round(float(b), 1),
            "redness (R−G)": round(float(r - g), 1), "sharpness (Laplacian var)": round(float(np.median(sharp)), 1)}


def shortcut_table(sets: dict) -> pd.DataFrame:
    """Compare colour/sharpness of several image sets. A big gap between the synthetic DF set and
    the real DF set is a cue the classifier could latch onto instead of the lesion itself."""
    df = pd.DataFrame({name: image_stats(arr) for name, arr in sets.items()}).T
    df["n"] = df["n"].astype(int)
    return df


def plot_two_diseases(logs: dict, out: Path | None = None, metric: str = "rare_f1", name: str = "F1") -> None:
    """Day 5: the same five methods on two rare diseases. logs = {"DF": log_df, "VASC": log_vasc}.
    Bars = mean over seeds, dots = seeds; GAN in orange, one panel per disease, shared 0-1 scale."""
    fig, axes = plt.subplots(1, len(logs), figsize=(6.2 * len(logs), 4.2), dpi=120, sharey=True, squeeze=False)
    for ax, (rare, log) in zip(axes[0], logs.items()):
        cfgs = [c for c in CONFIG_ORDER if (log["config"] == c).any()]
        base = log.loc[log["config"] == "baseline_no_aug", metric].mean()
        for i, c in enumerate(cfgs):
            vals = log.loc[log["config"] == c, metric].to_numpy()
            ax.bar(i, vals.mean(), width=0.62, color=ORANGE if c == "gan_aug" else BLUE, alpha=0.85,
                   edgecolor="white", linewidth=2)
            ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=20, color=INK, zorder=3, linewidths=0)
            ax.annotate(f"{vals.mean():.2f}", (i + 0.33, vals.mean()), xytext=(2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK)
        if base == base:
            ax.axhline(base, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(range(len(cfgs)), [CONFIGS[c]["label"].replace(" (no aug)", "").replace(" interp.", "")
                                         for c in cfgs], fontsize=8, color=INK_MUTED)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{rare} — test {name} ({int(log.groupby('config').size().max())} seeds)", loc="left",
                     fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(f"Same methods, two rare diseases — dashed line: baseline mean (GAN in orange)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()
