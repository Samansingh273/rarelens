"""GAN training loop: hinge loss + spectral norm + DiffAugment + EMA generator.

Trained ONLY on the rare class's TRAINING images — the GAN never sees validation or
test images, so the Day 4 comparison stays leak-free.

Checkpoints go to Drive every `eval_every` iterations and training resumes from the
last one if Colab disconnects. At each evaluation a fixed-noise sample grid is shown
and FID (vs the real training images) is computed; the EMA generator with the lowest
FID is kept as `G_best.pt`.

Balance: with only ~70 images plus DiffAugment, a plain 1:1 setup lets the generator out-run
the discriminator (D's real/fake gap collapses to ~0 within ~100 iterations and D stops giving
useful feedback; freezing G makes D recover at once). TTUR (D learning rate 4x G's) plus 2 D
steps per G step keeps D engaged — the defaults below.

What to watch while it trains:
  * loss_D stuck near 0 and loss_G climbing  -> discriminator has won, samples stop improving
  * loss_D stuck at exactly 2.00 (D(real) = D(fake) = 0) -> discriminator has given up
  * r_t = share of real images D scores as real. Stuck at 1.00 while FID gets worse
    -> D is memorising the 71 images (DiffAugment is there to prevent this)
  * the sample grid showing the SAME image many times -> mode collapse
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .diffaug import diff_augment, random_dihedral
from .models import Discriminator, Generator, count_params


@dataclass
class GANConfig:
    img_size: int = 128
    z_dim: int = 128
    g_width: int = 64
    d_width: int = 32
    iters: int = 20000
    batch_size: int = 32
    lr_g: float = 1e-4             # TTUR: the discriminator learns 4x faster than the generator ...
    lr_d: float = 4e-4
    betas: tuple = (0.0, 0.99)
    d_steps: int = 2               # ... and takes 2 steps per generator step, so it keeps up
    ema_beta: float = 0.999
    diffaug_policy: str = "color,translation,cutout"
    dihedral_real: bool = True
    log_every: int = 250
    eval_every: int = 2000
    fid_samples: int = 500
    seed: int = 42


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] (B, 3, H, W) tensor -> (B, H, W, 3) uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def resize_uint8(images: np.ndarray, size: int) -> np.ndarray:
    """Resize a (N, H, W, 3) uint8 stack with high-quality resampling."""
    return np.stack([np.asarray(Image.fromarray(im).resize((size, size), Image.BICUBIC if size > im.shape[0]
                                                           else Image.LANCZOS)) for im in images])


@torch.no_grad()
def generate(G: Generator, n: int, device, seed: int | None = None, batch: int = 100) -> np.ndarray:
    """n images from generator G as (n, S, S, 3) uint8."""
    G.eval()
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        z = torch.randn(min(batch, n - i), G.z_dim, generator=g).to(device)
        out.append(to_uint8(G(z)))
    return np.concatenate(out)


def load_generator(path: Path, device="cpu") -> Generator:
    """Load a saved generator (used on Day 4 and by the web app)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    G = Generator(cfg["z_dim"], cfg["g_width"], cfg["img_size"]).to(device)
    G.load_state_dict(ck["state_dict"])
    return G.eval()


def _save_grid(images: np.ndarray, path: Path, ncol: int = 8) -> np.ndarray:
    n, h, w, _ = images.shape
    nrow = int(np.ceil(n / ncol))
    grid = np.full((nrow * (h + 2) + 2, ncol * (w + 2) + 2, 3), 255, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, ncol)
        grid[2 + r * (h + 2): 2 + r * (h + 2) + h, 2 + c * (w + 2): 2 + c * (w + 2) + w] = im
    Image.fromarray(grid).save(path)
    return grid


def train_gan(cfg: GANConfig, real_uint8: np.ndarray, out_dir: Path, device=None,
              fid_ref_features: np.ndarray | None = None, resume: bool = True, show=None) -> pd.DataFrame:
    """Train the GAN. real_uint8: (N, S, S, 3) rare-class training images at cfg.img_size.

    show(grid_uint8, title) is called at each evaluation (the notebook passes a plotting fn).
    Returns the training history (one row per log step).
    """
    from ..fid import fid_from_features, inception_features

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    real = torch.from_numpy(real_uint8).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
    G = Generator(cfg.z_dim, cfg.g_width, cfg.img_size).to(device)
    D = Discriminator(cfg.d_width, cfg.img_size).to(device)
    G_ema = copy.deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    fixed_z = torch.randn(32, cfg.z_dim, generator=torch.Generator().manual_seed(123)).to(device)

    start, history, best_fid = 0, [], float("inf")
    last_ckpt = out_dir / "checkpoint_last.pt"
    if resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, history, best_fid = ck["iter"], ck["history"], ck["best_fid"]
        print(f"Resumed from iteration {start} (best FID so far {best_fid:.1f})")
    (out_dir / "gan_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"Real images: {len(real)} | G params {count_params(G) / 1e6:.2f}M | D params {count_params(D) / 1e6:.2f}M "
          f"| device {device} | iterations {start} -> {cfg.iters}")
    aug = lambda x: diff_augment(x, cfg.diffaug_policy)
    run = {"d": 0.0, "g": 0.0, "rt": 0.0, "dr": 0.0, "df": 0.0, "n": 0}
    t0 = time.time()

    for it in range(start + 1, cfg.iters + 1):
        G.train()
        # ---------------- discriminator step(s) ----------------
        for _ in range(cfg.d_steps):
            idx = torch.randint(len(real), (cfg.batch_size,), device=device)
            x_real = real[idx]
            if cfg.dihedral_real:
                x_real = random_dihedral(x_real)
            z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
            with torch.no_grad():
                x_fake = G(z)
            d_real, d_fake = D(aug(x_real)), D(aug(x_fake))
            loss_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
        # ---------------- generator step ----------------
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        loss_g = -D(aug(G(z))).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        # ---------------- EMA of generator weights ----------------
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - cfg.ema_beta)
            for be, b in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b)

        run["d"] += loss_d.item(); run["g"] += loss_g.item(); run["n"] += 1
        run["rt"] += (d_real > 0).float().mean().item()
        run["dr"] += d_real.mean().item(); run["df"] += d_fake.mean().item()

        if it % cfg.log_every == 0 or it == cfg.iters:
            n = run["n"]
            row = {"iter": it, "loss_d": run["d"] / n, "loss_g": run["g"] / n, "r_t": run["rt"] / n,
                   "d_real": run["dr"] / n, "d_fake": run["df"] / n, "fid": np.nan}
            run = {k: 0.0 for k in run} | {"n": 0}
            if it % cfg.eval_every == 0 or it == cfg.iters:
                imgs = generate(G_ema, cfg.fid_samples, device, seed=0)
                if fid_ref_features is not None:
                    row["fid"] = fid_from_features(inception_features(imgs, device), fid_ref_features)
                with torch.no_grad():
                    grid = _save_grid(to_uint8(G_ema.eval()(fixed_z)), out_dir / "samples" / f"iter_{it:06d}.png")
                ck = {"iter": it, "G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                      "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "history": history + [row],
                      "best_fid": best_fid, "config": asdict(cfg)}
                if row["fid"] == row["fid"] and row["fid"] < best_fid:       # not NaN and improved
                    best_fid = ck["best_fid"] = row["fid"]
                    torch.save({"state_dict": G_ema.state_dict(), "config": asdict(cfg), "iter": it, "fid": best_fid},
                               out_dir / "G_best.pt")
                torch.save(ck, out_dir / "checkpoint_tmp.pt")
                (out_dir / "checkpoint_tmp.pt").replace(last_ckpt)
                torch.save({"state_dict": G_ema.state_dict(), "config": asdict(cfg), "iter": it, "fid": row["fid"]},
                           out_dir / "G_last.pt")
                if show is not None:
                    show(grid, f"iteration {it}   FID {row['fid']:.1f}" if row["fid"] == row["fid"] else f"iteration {it}")
            history.append(row)
            el = time.time() - t0
            eta = el / (it - start) * (cfg.iters - it)
            fid_txt = f" | FID {row['fid']:.1f}" if row["fid"] == row["fid"] else ""
            print(f"it {it:6d}/{cfg.iters} | D {row['loss_d']:.3f} G {row['loss_g']:.3f} | r_t {row['r_t']:.2f}"
                  f"{fid_txt} | {el / 60:.1f} min, ETA {eta / 60:.1f} min")

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "gan_history.csv", index=False)
    if not (out_dir / "G_best.pt").exists():                      # no FID reference given
        (out_dir / "G_last.pt").replace(out_dir / "G_best.pt")
    return hist
