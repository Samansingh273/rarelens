"""Training / evaluation loop shared by every experiment configuration.

`run_experiment` is the single entry point. Day 1 calls it with no augmentation.
Days 2 and 4 call the *same* function, changing only the training data or the
training transforms, so any difference in results comes from the data, not the recipe.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CLASS_CODES, NUM_CLASSES, Paths, TrainConfig
from .data.dataset import make_loaders
from .evaluate import compute_metrics, per_class_report, plot_confusion, plot_history
from .models.classifier import build_classifier
from .utils import append_results_row, get_device, set_seed


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool) -> float:
    model.train()
    crit = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    """Return (y_true, y_pred, probs, mean_loss) over a loader."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    ys, ps, loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss += crit(logits.float(), y).item()
        ps.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y.cpu())
    y_true = torch.cat(ys).numpy()
    probs = torch.cat(ps).numpy()
    return y_true, probs.argmax(1), probs, loss / max(len(y_true), 1)


def run_experiment(cfg: TrainConfig, paths: Paths, images: np.ndarray, split: pd.DataFrame,
                   train_transform=None, train_class_transforms=None, train_dataset=None,
                   n_synthetic: int = 0) -> dict:
    """Train on `train`, select the checkpoint on `val`, report once on `test`.

    train_transform / train_class_transforms : Day 2 hooks (augmentation)
    train_dataset                            : Day 4 hook (real + synthetic dataset)
    """
    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.amp and device.type == "cuda"
    rare_idx = CLASS_CODES.index(cfg.rare_class)
    run_dir = paths.runs_dir / f"{cfg.run_name}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")

    loaders = make_loaders(images, split, cfg.batch_size, cfg.num_workers,
                           train_transform, train_class_transforms, seed=cfg.seed)
    if train_dataset is not None:
        loaders["train"] = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                                      generator=torch.Generator().manual_seed(cfg.seed))
    train_labels = np.asarray(loaders["train"].dataset.labels) if hasattr(loaders["train"].dataset, "labels") else None
    n_train = len(loaders["train"].dataset)
    n_train_rare = int((train_labels == rare_idx).sum()) if train_labels is not None else -1

    model = build_classifier(cfg.arch, NUM_CLASSES, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = _grad_scaler(amp)

    print(f"Device: {device} | AMP: {amp} | train={n_train} (rare {cfg.rare_class}: {n_train_rare}) "
          f"| val={len(loaders['val'].dataset)} | test={len(loaders['test'].dataset)}")
    history, best_score, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, scaler, device, amp)
        scheduler.step()
        yv, pv, prv, v_loss = predict(model, loaders["val"], device, amp)
        vm = compute_metrics(yv, pv, prv, rare_idx)
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss, "val_accuracy": vm["accuracy"],
               "val_macro_f1": vm["macro_f1"], "val_rare_precision": vm["rare_precision"],
               "val_rare_recall": vm["rare_recall"], "val_rare_f1": vm["rare_f1"],
               "val_rare_n_pred": vm["rare_n_predicted"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        history.append(row)
        score = row[cfg.select_metric]
        flag = ""
        if score > best_score:
            best_score, best_epoch, flag = score, epoch, "  *best*"
            best_state = copy.deepcopy(model.state_dict())
        print(f"ep {epoch:02d}/{cfg.epochs} | loss {tr_loss:.4f} / val {v_loss:.4f} | val acc {vm['accuracy']:.3f} "
              f"macroF1 {vm['macro_f1']:.3f} | {cfg.rare_class} P {vm['rare_precision']:.2f} "
              f"R {vm['rare_recall']:.2f} F1 {vm['rare_f1']:.2f} (pred {vm['rare_n_predicted']}) "
              f"| {row['seconds']:.0f}s{flag}")

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    torch.save(best_state, run_dir / "best.pt")
    plot_history(hist, run_dir / "training_curves.png", f"{cfg.run_name} (seed {cfg.seed})")

    # ---- final, one-shot test evaluation with the checkpoint chosen on val ---- #
    model.load_state_dict(best_state)
    yv, pv, prv, _ = predict(model, loaders["val"], device, amp)
    val_m = compute_metrics(yv, pv, prv, rare_idx)
    yt, pt, prt, _ = predict(model, loaders["test"], device, amp)
    test_m = compute_metrics(yt, pt, prt, rare_idx)

    report = per_class_report(yt, pt)
    report.to_csv(run_dir / "test_classification_report.csv")
    cm = plot_confusion(yt, pt, rare_idx, run_dir / "test_confusion_matrix.png",
                        f"Test confusion matrix — {cfg.run_name}")
    pd.DataFrame(cm, index=CLASS_CODES, columns=CLASS_CODES).to_csv(run_dir / "test_confusion_matrix.csv")

    test_rows = split[split["split"] == "test"].reset_index(drop=True)
    preds = test_rows[["image_id", "lesion_id", "label"]].copy()
    preds["pred"] = [CLASS_CODES[i] for i in pt]
    for k, c in enumerate(CLASS_CODES):
        preds[f"p_{c}"] = prt[:, k].round(5)
    preds.to_csv(run_dir / "test_predictions.csv", index=False)  # used for Day 8-9 error analysis

    summary = {"config": asdict(cfg), "best_epoch": best_epoch, "n_train": n_train,
               "n_train_rare": n_train_rare, "val": val_m, "test": test_m}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    append_results_row(paths.results_log, {
        "run_name": cfg.run_name, "config": cfg.config_name, "arch": cfg.arch, "seed": cfg.seed,
        "split_seed": cfg.split_seed, "rare_class": cfg.rare_class, "epochs": cfg.epochs,
        "best_epoch": best_epoch, "n_train": n_train, "n_train_rare": n_train_rare, "n_synthetic": n_synthetic,
        "test_accuracy": test_m["accuracy"], "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"], "rare_precision": test_m["rare_precision"],
        "rare_recall": test_m["rare_recall"], "rare_f1": test_m["rare_f1"], "rare_auroc": test_m["rare_auroc"],
        "rare_support": test_m["rare_support"], "val_rare_f1": val_m["rare_f1"], "notes": cfg.notes,
    })
    print(f"\nSaved run artefacts to {run_dir}\nAppended results row to {paths.results_log}")
    return {"summary": summary, "history": hist, "report": report, "confusion": cm, "run_dir": run_dir}
