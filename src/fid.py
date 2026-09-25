"""FID (Fréchet Inception Distance) with pytorch-fid, plus the feature extractor
reused for the memorisation check and the synthetic-image filter.

FID compares two sets of images through the statistics of their Inception-v3
features: lower = the two sets look statistically more alike.

Small-sample warning: FID is biased upwards when a set is small, and here the real
rare-class set has only 71 images. Absolute FID numbers are therefore NOT comparable
with values in GAN papers (which use 10k-50k images). What IS meaningful is comparing
methods under the SAME sample sizes — hence `matched_fid` and a real-vs-real reference.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

_MODEL = None


def _inception(device):
    global _MODEL
    if _MODEL is None:
        from pytorch_fid.inception import InceptionV3
        _MODEL = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    return _MODEL.to(device)


@torch.no_grad()
def inception_features(images_uint8: np.ndarray, device=None, batch: int = 50) -> np.ndarray:
    """(N, H, W, 3) uint8 images -> (N, 2048) pool-3 features (inputs resized to 299 internally)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _inception(device)
    feats = []
    for i in range(0, len(images_uint8), batch):
        x = torch.from_numpy(np.ascontiguousarray(images_uint8[i:i + batch])).permute(0, 3, 1, 2)
        x = x.float().div(255).to(device)
        feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(feats).numpy().astype(np.float64)


def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    """Exact Fréchet distance between the Gaussians fitted to two feature sets.

    FID = |mu1 - mu2|^2 + Tr(S1) + Tr(S2) - 2 Tr( sqrt(S1 S2) ).

    The usual implementation takes a 2048x2048 matrix square root (~1 min on a 2-core CPU).
    With n images per set (n << 2048) the covariances are low-rank, and
        Tr( sqrt(S1 S2) ) = sum of singular values of  X1 X2^T / sqrt((n1-1)(n2-1)),
    where X1, X2 are the mean-centred feature matrices. That is an exact identity (the
    non-zero eigenvalues of S1 S2 are the squared singular values of that small n1 x n2
    matrix), so this gives the same number in milliseconds. Features come from the
    standard pytorch-fid Inception network, so values follow the usual FID definition.
    """
    f1, f2 = np.asarray(f1, np.float64), np.asarray(f2, np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    x1, x2 = f1 - mu1, f2 - mu2
    n1, n2 = len(f1), len(f2)
    tr1 = (x1 ** 2).sum() / (n1 - 1)
    tr2 = (x2 ** 2).sum() / (n2 - 1)
    m = (x1 @ x2.T) / np.sqrt((n1 - 1) * (n2 - 1))
    tr_sqrt = np.linalg.svd(m, compute_uv=False).sum()
    return float(((mu1 - mu2) ** 2).sum() + tr1 + tr2 - 2 * tr_sqrt)


def fid_reference_impl(f1: np.ndarray, f2: np.ndarray) -> float:
    """pytorch-fid's own (slow) implementation — used only to verify fid_from_features."""
    from pytorch_fid.fid_score import calculate_frechet_distance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(calculate_frechet_distance(f1.mean(0), np.cov(f1, rowvar=False),
                                                f2.mean(0), np.cov(f2, rowvar=False)))


def matched_fid(f_candidate: np.ndarray, f_ref: np.ndarray, n: int, draws: int = 10, seed: int = 0):
    """FID using a random subset of n candidate images (mean, sd over `draws` subsets).

    Lets a 500-image synthetic set be compared fairly with a small real set of size n.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(f_candidate))
    vals = [fid_from_features(f_candidate[rng.choice(len(f_candidate), n, replace=False)], f_ref)
            for _ in range(draws)]
    return float(np.mean(vals)), float(np.std(vals))
