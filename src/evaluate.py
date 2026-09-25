"""Metrics + plots. The headline numbers are the RARE-CLASS precision / recall / F1.

Overall accuracy is reported but is misleading under imbalance: a model that
never predicts the rare class still scores ~98.9% on it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)

from .config import CLASS_CODES

INK, INK_MUTED = "#0b0b0b", "#52514e"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, rare_idx: int) -> dict:
    labels = list(range(len(CLASS_CODES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rare_true = (y_true == rare_idx).astype(int)
    try:
        auroc = float(roc_auc_score(rare_true, probs[:, rare_idx]))
    except ValueError:  # only one class present (tiny smoke-test sets)
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "rare_precision": float(p[rare_idx]),
        "rare_recall": float(r[rare_idx]),
        "rare_f1": float(f[rare_idx]),
        "rare_support": int(s[rare_idx]),
        "rare_auroc": auroc,               # threshold-free: can the model *rank* rare cases at all?
        "rare_n_predicted": int((y_pred == rare_idx).sum()),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=list(range(len(CLASS_CODES))),
                                target_names=CLASS_CODES, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, rare_idx: int, out: Path, title: str) -> np.ndarray:
    """Row-normalised confusion matrix (recall on the diagonal) with raw counts in each cell."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_CODES))))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=130)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_CODES)):
        for j in range(len(CLASS_CODES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else INK)
    ticks = [f"{c} ◆" if k == rare_idx else c for k, c in enumerate(CLASS_CODES)]
    ax.set_xticks(range(len(CLASS_CODES)), ticks, fontsize=8, color=INK_MUTED)
    ax.set_yticks(range(len(CLASS_CODES)), ticks, fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row share (recall)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.show()
    return cm


def plot_history(history: pd.DataFrame, out: Path, title: str) -> None:
    """Two panels (never two y-axes): loss, then validation F1 scores."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120)
    a1.plot(history["epoch"], history["train_loss"], color="#2a78d6", lw=2, label="train")
    a1.plot(history["epoch"], history["val_loss"], color="#eb6834", lw=2, label="val")
    a1.set_title("Cross-entropy loss", loc="left", fontsize=10, color=INK)
    a2.plot(history["epoch"], history["val_macro_f1"], color="#2a78d6", lw=2, label="val macro-F1")
    a2.plot(history["epoch"], history["val_rare_f1"], color="#eb6834", lw=2, label="val rare-class F1")
    a2.set_ylim(0, 1)
    a2.set_title("Validation F1", loc="left", fontsize=10, color=INK)
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_xlabel("epoch", color=INK_MUTED)
        a.grid(color="#e4e3df", lw=0.8)
        a.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.show()
