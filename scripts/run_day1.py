#!/usr/bin/env python
"""Day 1 end-to-end: download -> inspect -> split -> cache -> train baseline -> evaluate -> log.

Same steps as notebooks/01_day1_baseline.ipynb, as a single command:

    python scripts/run_day1.py                       # Colab defaults
    python scripts/run_day1.py --epochs 15 --seed 42
    python scripts/run_day1.py --rare-class VASC     # force a rare class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_CODES, Paths, TrainConfig, default_paths  # noqa: E402
from src.data.dataset import attach_cache_index  # noqa: E402
from src.data.download import download_ham10000  # noqa: E402
from src.data.explore import (build_metadata, class_distribution, pick_rare_class,  # noqa: E402
                              plot_class_distribution, plot_examples, save_dataset_info)
from src.data.preprocess import build_image_cache  # noqa: E402
from src.data.split import make_split, split_summary  # noqa: E402
from src.train import run_experiment  # noqa: E402


def main() -> None:
    d = default_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=str(d.project_dir))
    ap.add_argument("--raw-dir", default=str(d.raw_dir))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--rare-class", default=None, choices=CLASS_CODES, help="override automatic choice")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only: random init")
    ap.add_argument("--no-show", action="store_true", help="don't open plot windows")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    paths = Paths(Path(args.project_dir), Path(args.raw_dir))
    paths.make_dirs()

    if not args.skip_download:
        download_ham10000(paths)

    meta = build_metadata(paths)
    dist = class_distribution(meta)
    rare = pick_rare_class(dist, args.rare_class)
    print("\nClass distribution:\n", dist.round(2).to_string())
    print(f"\nRare class -> {rare}")
    save_dataset_info(paths, dist, rare)
    plot_class_distribution(dist, rare, paths.figures_dir / "class_distribution.png")
    plot_examples(meta, [rare] + [c for c in dist[dist["under_2pct"]].index if c != rare],
                  paths.figures_dir / "rare_class_examples.png")

    split = make_split(meta, paths, seed=args.split_seed)
    print("\nSplit summary:\n", split_summary(split, rare).to_string())
    split = attach_cache_index(split, meta)

    images = build_image_cache(meta, paths, size=args.image_size)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                      split_seed=args.split_seed, image_size=args.image_size, num_workers=args.num_workers,
                      pretrained=not args.no_pretrained, rare_class=rare,
                      notes="Day 1 control: real data only, no augmentation, plain cross-entropy")
    out = run_experiment(cfg, paths, images, split)
    t = out["summary"]["test"]
    print(f"\nTEST  acc {t['accuracy']:.3f} | bal-acc {t['balanced_accuracy']:.3f} | macro-F1 {t['macro_f1']:.3f}")
    print(f"      {rare}: precision {t['rare_precision']:.3f} recall {t['rare_recall']:.3f} "
          f"F1 {t['rare_f1']:.3f} AUROC {t['rare_auroc']:.3f} (support {t['rare_support']})")


if __name__ == "__main__":
    main()
