"""Memorisation check + automatic filtering of GAN candidates.

Every image is described by its Inception features (L2-normalised), and we measure
how far each generated image is from its NEAREST real training image.

Reference scale: for each real image, the distance to its nearest real image from a
DIFFERENT lesion. That tells us how far apart two genuinely different real lesions are.

Two-sided filter (defaults):
  * "copy"    : closer to a real image than 95% of real lesions are to each other
                (below the 5th percentile of the real-real distances) -> likely a memorised
                copy that adds no new information. Removed.
  * "outlier" : among the rest, the 25% farthest from any real image -> most likely
                broken / unrealistic. Removed.
  * the required number (e.g. 500) is then drawn at random from what is left, so the
    final set keeps the natural variety of the generator rather than only its "safest" outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _l2n(f: np.ndarray) -> np.ndarray:
    return f / np.linalg.norm(f, axis=1, keepdims=True).clip(min=1e-12)


def real_real_nn(real_feats: np.ndarray, lesion_ids) -> np.ndarray:
    """Distance from each real image to its nearest real image of a different lesion."""
    r = _l2n(real_feats)
    d = np.sqrt(np.clip(2 - 2 * r @ r.T, 0, None))
    lesions = np.asarray(lesion_ids)
    d[lesions[:, None] == lesions[None, :]] = np.inf
    return d.min(1)


def gen_real_nn(gen_feats: np.ndarray, real_feats: np.ndarray):
    """For each generated image: distance to, and index of, the nearest real image."""
    g, r = _l2n(gen_feats), _l2n(real_feats)
    d = np.sqrt(np.clip(2 - 2 * g @ r.T, 0, None))
    return d.min(1), d.argmin(1)


def select_synthetic(gen_feats: np.ndarray, real_feats: np.ndarray, lesion_ids, n_keep: int = 500,
                     copy_pct: float = 5, drop_far_frac: float = 0.25, seed: int = 42):
    """Return (indices of the selected candidates, per-candidate DataFrame, thresholds dict)."""
    d_rr = real_real_nn(real_feats, lesion_ids)
    d_gr, nn_idx = gen_real_nn(gen_feats, real_feats)
    copy_thr = float(np.percentile(d_rr, copy_pct))
    is_copy = d_gr < copy_thr
    rest = np.where(~is_copy)[0]
    far_thr = float(np.quantile(d_gr[rest], 1 - drop_far_frac)) if len(rest) else float("nan")
    ok = np.where(~is_copy & (d_gr <= far_thr))[0]

    status = np.where(is_copy, "copy", np.where(d_gr > far_thr, "outlier", "kept")).astype(object)
    rng = np.random.default_rng(seed)
    if len(ok) < n_keep:
        print(f"WARNING: only {len(ok)} candidates passed the filter (< {n_keep}); generate more candidates.")
    chosen = rng.choice(ok, size=min(n_keep, len(ok)), replace=False)
    status[chosen] = "selected"
    info = pd.DataFrame({"candidate": np.arange(len(gen_feats)), "nn_distance": d_gr.round(4),
                         "nearest_real_pos": nn_idx, "status": status})
    thresholds = {"copy_threshold": copy_thr, "outlier_threshold": far_thr,
                  "real_real_median": float(np.median(d_rr)), "real_real_distances": d_rr}
    return chosen, info, thresholds


def replace_rejected(chosen: np.ndarray, info: pd.DataFrame, reject_positions, seed: int = 0):
    """Swap manually rejected images (positions within `chosen`) for other kept candidates."""
    reject_positions = sorted(set(int(p) for p in reject_positions))
    if not reject_positions:
        return chosen, info
    pool = info.index[info["status"] == "kept"].to_numpy()
    rng = np.random.default_rng(seed)
    if len(pool) < len(reject_positions):
        raise ValueError("Not enough spare candidates; generate more.")
    subs = rng.choice(pool, size=len(reject_positions), replace=False)
    chosen = chosen.copy()
    info = info.copy()
    for pos, new in zip(reject_positions, subs):
        info.loc[chosen[pos], "status"] = "rejected_manual"
        info.loc[new, "status"] = "selected"
        chosen[pos] = new
    return chosen, info
