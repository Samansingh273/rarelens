"""Small shared helpers: seeding, device selection, CSV results log."""
from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # benchmark=True is faster; tiny run-to-run GPU nondeterminism remains, which
    # is exactly why Days 6-7 repeat every configuration over several seeds.
    torch.backends.cudnn.benchmark = True


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


RESULTS_COLUMNS = [
    "timestamp", "run_name", "config", "arch", "seed", "split_seed", "rare_class",
    "epochs", "best_epoch", "n_train", "n_train_rare", "n_synthetic",
    "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
    "rare_precision", "rare_recall", "rare_f1", "rare_auroc", "rare_support",
    "val_rare_f1", "notes",
]


def append_results_row(log_path: Path, row: dict) -> None:
    """Append one experiment's summary to the shared results table (CSV).

    Every configuration (Day 1 baseline, Day 2 classic/Mixup, Day 4 GAN) writes
    one row per run here, so the final comparison table is just this file.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    new_file = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: _fmt(row.get(k, "")) for k in RESULTS_COLUMNS})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return v
