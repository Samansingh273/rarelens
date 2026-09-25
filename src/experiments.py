"""Day 4 — one entry point to (re)run any configuration with any seed.

Every configuration uses the Day 1 recipe (ResNet-18, 15 epochs, AdamW 1e-4, checkpoint
chosen on validation macro-F1, test used once). Only the 500 extra rare-class training
samples differ:

    baseline_no_aug : no extras
    oversample_only : 500 copies of real DF training images
    classic_aug     : the same copies, randomly flipped / rotated / cropped / jittered
    smote_interp    : the 500 SMOTE blends saved on Day 2
    gan_aug         : the 500 filtered GAN images saved on Day 3

The seed changes the classifier's training randomness (head initialisation, batch order,
augmentation draws). The synthetic sets themselves are fixed — one SMOTE set, one trained
GAN — so the spread across seeds measures classifier variance, not GAN-training variance.

`run_config` skips any (config, seed) already in results_log.csv, so after a Colab
disconnect the notebook can simply be run again and picks up where it stopped.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .augment import RealPlusExtra, classic_rare_transform, sample_rare_indices
from .config import CLASS_CODES, Paths, TrainConfig
from .train import run_experiment

CONFIGS = {
    "baseline_no_aug": dict(run_name="day1_baseline_noaug", label="Baseline (no aug)",
                            notes="Control: real data only, no augmentation, plain cross-entropy"),
    "oversample_only": dict(run_name="day2_oversample_only", label="Oversample only",
                            notes="Control: + {n} un-augmented {rare} copies"),
    "classic_aug": dict(run_name="day2_classic_aug", label="Classic aug",
                        notes="+ {n} classic-augmented {rare} copies (flip/rot90/rot-reflect/crop/jitter)"),
    "smote_interp": dict(run_name="day2_smote_interp", label="SMOTE interp.",
                         notes="+ {n} SMOTE-style {rare} blends (Day 2 set)"),
    "gan_aug": dict(run_name="day4_gan_aug", label="GAN aug",
                    notes="+ {n} filtered GAN {rare} images (Day 3 set)"),
}
CONFIG_ORDER = list(CONFIGS)


def run_name_for(config: str, rare: str) -> str:
    """Run-folder name. DF (the main study) keeps the Day 1-4 names; any other rare class gets a
    prefix (e.g. `vasc_day4_gan_aug`) so its runs never overwrite the DF runs."""
    base = CONFIGS[config]["run_name"]
    return base if rare == "DF" else f"{rare.lower()}_{base}"


def run_dir_for(paths: Paths, config: str, seed: int, rare: str = "DF") -> Path:
    return paths.runs_dir / f"{run_name_for(config, rare)}_seed{int(seed)}"


def synthetic_path(paths: Paths, kind: str, rare: str, n: int = 500) -> Path:
    return paths.project_dir / "data" / "synthetic" / f"{kind}_{rare}_{n}.npy"


def make_train_dataset(config: str, images: np.ndarray, train_df: pd.DataFrame, rare: str,
                       paths: Paths, n_extra: int = 500, seed: int = 42):
    """Return (training dataset or None for the baseline, number of extra samples)."""
    rare_idx = CLASS_CODES.index(rare)
    if config == "baseline_no_aug":
        return None, 0
    if config in ("oversample_only", "classic_aug"):
        idx = sample_rare_indices(train_df, rare_idx, n_extra, seed=seed)
        tf = classic_rare_transform(images.shape[1]) if config == "classic_aug" else None
        ds = RealPlusExtra(images, train_df, rare_idx, extra_cache_idx=idx, extra_transform=tf)
        return ds, ds.n_extra
    kind = {"smote_interp": "smote", "gan_aug": "gan"}[config]
    f = synthetic_path(paths, kind, rare, n_extra)
    if not f.exists():
        raise FileNotFoundError(f"{f} not found — run the Day {2 if kind == 'smote' else 3} notebook first.")
    extra = np.load(f)
    if len(extra) != n_extra:
        print(f"WARNING: {f.name} holds {len(extra)} images, expected {n_extra}.")
    ds = RealPlusExtra(images, train_df, rare_idx, extra_array=extra)
    return ds, ds.n_extra


def completed_runs(paths: Paths, rare: str = "DF") -> set:
    """{(config, seed)} already in results_log.csv for this rare class."""
    if not paths.results_log.exists():
        return set()
    log = pd.read_csv(paths.results_log)
    if "rare_class" in log.columns:
        log = log[log["rare_class"].fillna("DF") == rare]
    return set(zip(log["config"], log["seed"].astype(int)))


@contextlib.contextmanager
def quiet_plots():
    """Figures are still saved to the run folder, just not drawn in the notebook."""
    show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = show
        plt.close("all")


def run_config(config: str, seed: int, paths: Paths, images: np.ndarray, split: pd.DataFrame, rare: str,
               n_extra: int = 500, epochs: int = 15, pretrained: bool = True, force: bool = False,
               show_plots: bool = False):
    """Train + evaluate one (config, seed). Returns the run summary, or None if skipped."""
    if not force and (config, seed) in completed_runs(paths, rare):
        print(f"[skip] {rare} {config} seed {seed} already done")
        return None
    spec = CONFIGS[config]
    train_df = split[split["split"] == "train"].reset_index(drop=True)
    ds, n = make_train_dataset(config, images, train_df, rare, paths, n_extra, seed)
    cfg = TrainConfig(run_name=run_name_for(config, rare), config_name=config, arch="resnet18", pretrained=pretrained,
                      epochs=epochs, batch_size=64, lr=1e-4, weight_decay=1e-4, seed=seed, split_seed=42,
                      rare_class=rare, notes=spec["notes"].format(n=n, rare=rare))
    print(f"\n=== {spec['label']} | seed {seed} | extra {rare} samples: {n} ===")
    ctx = contextlib.nullcontext() if show_plots else quiet_plots()
    with ctx:
        out = run_experiment(cfg, paths, images, split, train_dataset=ds, n_synthetic=n)
    t = out["summary"]["test"]
    print(f"--> TEST {rare}: recall {t['rare_recall']:.2f}  precision {t['rare_precision']:.2f}  "
          f"F1 {t['rare_f1']:.2f} | macro-F1 {t['macro_f1']:.3f}")
    return out["summary"]
