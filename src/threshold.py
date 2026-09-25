"""Day 5: the threshold baseline.

The question a reviewer asks first: *"Do you need new images at all, or would simply lowering the
decision threshold for the rare class do the same job?"*

Normally a model predicts the class with the highest probability (argmax). Here the rare class is
predicted whenever its probability is at least `t`; otherwise the most likely of the other classes is used.
`t` is chosen on the VALIDATION set (max rare-class F1) and then applied once to the test set — the same
"choose on val, report on test" rule as every other result in the project.

Every trained model (5 methods × 3 seeds) is re-loaded from its saved `best.pt`, so nothing is retrained.
Validation/test probabilities are cached next to each checkpoint as `val_test_probs.npz` (the app reuses them).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_recall_fscore_support

from .config import CLASS_CODES, NUM_CLASSES, Paths
from .data.dataset import make_loaders
from .experiments import CONFIG_ORDER, CONFIGS, run_dir_for
from .models.classifier import build_classifier
from .train import predict

GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
BLUE, TEAL, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK_MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# Probabilities from saved checkpoints
# --------------------------------------------------------------------------- #
def run_probabilities(paths: Paths, config: str, seed: int, rare: str, images: np.ndarray, split: pd.DataFrame,
                      device, batch_size: int = 128, arch: str = "resnet18") -> dict | None:
    """Val + test softmax probabilities of one trained run (cached). None if its best.pt is missing."""
    run_dir = run_dir_for(paths, config, seed, rare)
    cache = run_dir / "val_test_probs.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        return None
    model = build_classifier(arch, NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    loaders = make_loaders(images, split, batch_size, num_workers=0)
    amp = device.type == "cuda"
    y_val, _, p_val, _ = predict(model, loaders["val"], device, amp)
    y_test, _, p_test, _ = predict(model, loaders["test"], device, amp)
    out = {"y_val": y_val, "p_val": p_val.astype(np.float32), "y_test": y_test, "p_test": p_test.astype(np.float32)}
    np.savez_compressed(cache, **out)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #
def predict_with_threshold(probs: np.ndarray, rare_idx: int, t: float) -> np.ndarray:
    """Rare class if p_rare >= t, otherwise the most likely OTHER class."""
    others = probs.copy()
    others[:, rare_idx] = -1.0
    return np.where(probs[:, rare_idx] >= t, rare_idx, others.argmax(1))


def rare_scores(y: np.ndarray, pred: np.ndarray, rare_idx: int) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y == rare_idx, pred == rare_idx, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "caught": int(((pred == rare_idx) & (y == rare_idx)).sum()),
            "false_alarms": int(((pred == rare_idx) & (y != rare_idx)).sum()),
            "macro_f1": float(f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))}


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray, rare_idx: int, grid: np.ndarray = GRID) -> tuple[float, float]:
    """Threshold with the best validation rare-class F1 (middle of the best plateau if several tie)."""
    f1s = np.array([rare_scores(y_val, predict_with_threshold(p_val, rare_idx, t), rare_idx)["f1"] for t in grid])
    best = np.flatnonzero(np.isclose(f1s, f1s.max()))
    t = float(grid[best[len(best) // 2]])
    return t, float(f1s.max())


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #
def threshold_study(paths: Paths, log: pd.DataFrame, images: np.ndarray, split: pd.DataFrame, rare: str,
                    device) -> pd.DataFrame:
    """One row per trained run: default (argmax) vs val-tuned threshold, both on the test set."""
    rare_idx = CLASS_CODES.index(rare)
    rows, missing = [], []
    for _, r in log.iterrows():
        cfg, seed = r["config"], int(r["seed"])
        pr = run_probabilities(paths, cfg, seed, rare, images, split, device)
        if pr is None:
            missing.append(f"{cfg} seed {seed}")
            continue
        default = rare_scores(pr["y_test"], pr["p_test"].argmax(1), rare_idx)
        t, val_f1 = tune_threshold(pr["y_val"], pr["p_val"], rare_idx)
        tuned = rare_scores(pr["y_test"], predict_with_threshold(pr["p_test"], rare_idx, t), rare_idx)
        ap = average_precision_score(pr["y_test"] == rare_idx, pr["p_test"][:, rare_idx])
        row = {"config": cfg, "method": CONFIGS[cfg]["label"], "seed": seed, "threshold": t, "val_f1_tuned": val_f1,
               "test_AP": float(ap), "logged_f1": float(r["rare_f1"])}
        row.update({f"default_{k}": v for k, v in default.items()})
        row.update({f"tuned_{k}": v for k, v in tuned.items()})
        rows.append(row)
        print(f"{CONFIGS[cfg]['label']:18s} seed {seed}: threshold {t:.2f} | test {rare} F1 "
              f"{default['f1']:.2f} (default) -> {tuned['f1']:.2f} (tuned) | caught {default['caught']} -> {tuned['caught']}")
    if missing:
        print(f"\n⚠️ No saved model (best.pt) for: {', '.join(missing)} — left out of this table.")
    df = pd.DataFrame(rows)
    if len(df):
        gap = (df["default_f1"] - df["logged_f1"]).abs().max()
        print(f"\nCheck: re-loaded models reproduce the logged test F1 (largest difference {gap:.3f}).")
    return df


def threshold_summary(study: pd.DataFrame, rare: str) -> pd.DataFrame:
    """Mean ± sd over seeds, per method."""
    def ms(g, c):
        return f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" if len(g) > 1 else f"{g[c].mean():.3f}"
    rows = []
    for cfg in CONFIG_ORDER:
        g = study[study["config"] == cfg]
        if g.empty:
            continue
        rows.append({"method": CONFIGS[cfg]["label"], "seeds": len(g),
                     f"{rare} F1 — default": ms(g, "default_f1"),
                     f"{rare} F1 — tuned threshold": ms(g, "tuned_f1"),
                     "recall — tuned": ms(g, "tuned_recall"),
                     "precision — tuned": ms(g, "tuned_precision"),
                     "false alarms / run — tuned": round(g["tuned_false_alarms"].mean(), 1),
                     "threshold (mean)": round(g["threshold"].mean(), 2),
                     f"{rare} AP (threshold-free)": ms(g, "test_AP")})
    return pd.DataFrame(rows).set_index("method")


def paired_tuned(study: pd.DataFrame, metric: str = "tuned_f1") -> pd.DataFrame:
    """Each method with a TUNED threshold vs the baseline with a TUNED threshold, same seed."""
    base = study[study["config"] == "baseline_no_aug"].set_index("seed")[metric]
    rows = []
    for cfg in CONFIG_ORDER[1:]:
        g = study[study["config"] == cfg].set_index("seed")[metric]
        common = sorted(set(g.index) & set(base.index))
        if not common:
            continue
        d = g.loc[common] - base.loc[common]
        rows.append({"method": CONFIGS[cfg]["label"], **{f"seed {s}": round(float(d.loc[s]), 3) for s in common},
                     "mean change": round(float(d.mean()), 3), "better": int((d > 0).sum()),
                     "same": int((d == 0).sum()), "worse": int((d < 0).sum())})
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_threshold_effect(study: pd.DataFrame, paths: Paths, rare: str, out: Path | None = None) -> None:
    """Left: F1 with the default rule vs a val-tuned threshold, per method.
    Right: test precision-recall curves, baseline vs GAN (one line per seed)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4), dpi=120, gridspec_kw={"width_ratios": [1.5, 1]})
    cfgs = [c for c in CONFIG_ORDER if (study["config"] == c).any()]
    w = 0.36
    for i, c in enumerate(cfgs):
        g = study[study["config"] == c]
        for k, (col, color, name) in enumerate([("default_f1", BLUE, "default rule (argmax)"),
                                                ("tuned_f1", TEAL, "threshold tuned on validation")]):
            x = i + (k - 0.5) * w
            v = g[col].to_numpy()
            a1.bar(x, v.mean(), w, color=color, alpha=0.9, edgecolor="white", lw=1.5, label=name if i == 0 else None)
            a1.scatter(x + np.linspace(-0.07, 0.07, len(v)), v, s=14, color=INK, zorder=3, linewidths=0)
            a1.annotate(f"{v.mean():.2f}", (x, v.mean()), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    a1.set_xticks(range(len(cfgs)), [CONFIGS[c]["label"] for c in cfgs], fontsize=8, color=INK_MUTED)
    a1.set_ylim(0, 1.12)
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style(a1, f"{rare} F1 on the test set — does a better threshold close the gap?")

    for c, color in [("baseline_no_aug", BLUE), ("gan_aug", ORANGE)]:
        for j, (_, r) in enumerate(study[study["config"] == c].iterrows()):
            pr = np.load(run_dir_for(paths, c, r["seed"], rare) / "val_test_probs.npz")
            y = pr["y_test"] == CLASS_CODES.index(rare)
            prec, rec, _ = precision_recall_curve(y, pr["p_test"][:, CLASS_CODES.index(rare)])
            a2.plot(rec, prec, color=color, lw=1.4, alpha=0.85,
                    label=f"{CONFIGS[c]['label']} (AP {study[study.config == c].test_AP.mean():.2f})" if j == 0 else None)
    a2.set_xlabel(f"{rare} recall", color=INK_MUTED, fontsize=9)
    a2.set_ylabel(f"{rare} precision", color=INK_MUTED, fontsize=9)
    a2.set_xlim(0, 1.02)
    a2.set_ylim(0, 1.05)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    _style(a2, "Precision–recall trade-off at every threshold (one line per seed)")
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    plt.show()
