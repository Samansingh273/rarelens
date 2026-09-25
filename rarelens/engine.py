"""RareLens inference engine: classifiers, Grad-CAM, the 'outside the training data' check, and the GAN."""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES  # noqa: E402
from src.models.classifier import build_classifier  # noqa: E402
from rarelens.model_utils import ensemble_probs, imagenet_backbone, knn_distance, penultimate  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
# Grad-CAM colours: transparent -> amber -> deep red (matches the app's rare-class accent)
_CAM_STOPS = np.array([[0.00, 255, 214, 102, 0], [0.30, 255, 214, 102, 70], [0.55, 247, 150, 40, 165],
                       [0.80, 230, 72, 30, 215], [1.00, 190, 20, 40, 240]], dtype=np.float32)


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cam_rgba(cam: np.ndarray) -> np.ndarray:
    out = np.zeros((*cam.shape, 4), np.float32)
    for c in range(4):
        out[..., c] = np.interp(cam, _CAM_STOPS[:, 0], _CAM_STOPS[:, c + 1])
    return out.round().astype(np.uint8)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    an, bn = a / a.norm(), b / b.norm()
    omega = torch.acos((an * bn).sum().clamp(-1, 1))
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


class Engine:
    def __init__(self):
        meta_file = DATA / "app_meta.json"
        if not meta_file.exists():
            raise SystemExit("rarelens/data/app_meta.json is missing - run  python prepare_app.py  first.")
        self.meta = json.loads(meta_file.read_text())
        self.evidence = json.loads((DATA / "evidence.json").read_text())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.meta["classes"]
        self.models = {}
        for key, m in self.meta["models"].items():
            net = build_classifier("resnet18", NUM_CLASSES, pretrained=False)
            net.load_state_dict(torch.load(ROOT / m["path"], map_location="cpu", weights_only=True))
            self.models[key] = net.to(self.device).eval()
        self.main_key = self.meta["main_model"]
        am = self.meta.get("app_model", {})
        self.ens_keys = [k for k in am.get("members", [self.main_key]) if k in self.models] or [self.main_key]
        self.ens = [self.models[k] for k in self.ens_keys]
        self.tta = bool(am.get("tta", False))
        self.model_name = am.get("name", "GAN model (DF study)")
        self.knn_thr = am.get("knn_threshold")
        self.knn_k = am.get("knn_k", 10)
        self.low_conf = am.get("low_confidence", 0.5)
        kf = DATA / "knn_train_features.npz"
        self.knn_ref = torch.from_numpy(np.load(kf)["feats"].astype(np.float32)).to(self.device) if kf.exists() else None
        self.backbone = imagenet_backbone(self.device) if self.knn_ref is not None else None
        self.gans, self.real_feats = {}, {}
        from src.gan.trainer import load_generator
        for rare, g in self.meta.get("gan", {}).items():
            self.gans[rare] = load_generator(ROOT / g["generator"], self.device)
            f = DATA / f"gan_{rare}_real.npz"
            if f.exists():
                z = np.load(f, allow_pickle=True)
                self.real_feats[rare] = (z["feats"], float(z["copy_threshold"]))
        self._inception_ok = None

    # ------------------------------------------------------------------ utils
    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "").replace(" Laptop GPU", "")
        return "CPU"

    @staticmethod
    def load_image(data: bytes) -> np.ndarray:
        """Same preprocessing as the training cache: full frame -> 224x224, bicubic."""
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.uint8)

    def _tensor(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        return ((x - MEAN) / STD).to(self.device)

    # ------------------------------------------------------------- classify
    def _gradcam(self, net, x: torch.Tensor, cls: int) -> np.ndarray:
        feats = {}
        h = net.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
        try:
            with torch.enable_grad():
                logits = net(x)
                feats["a"].retain_grad()
                net.zero_grad(set_to_none=True)
                logits[0, cls].backward()
                a, g = feats["a"], feats["a"].grad
                cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
                cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        finally:
            h.remove()
        cam = cam.detach().cpu().numpy()
        rng = cam.max() - cam.min()
        return (cam - cam.min()) / rng if rng > 0 else np.zeros_like(cam)

    def predict(self, img: np.ndarray) -> dict:
        x = self._tensor(img)
        probs = ensemble_probs(self.ens, x, tta=self.tta)[0].cpu().numpy()   # configuration chosen on validation
        top = int(probs.argmax())
        cam = np.mean([self._gradcam(net, x, top) for net in self.ens], axis=0)
        cam = cam / cam.max() if cam.max() > 0 else cam

        flags, dist = [], None
        if self.knn_ref is not None and self.knn_thr is not None:
            with torch.no_grad():
                dist = float(knn_distance(self.knn_ref, penultimate(self.backbone, x), self.knn_k)[0])
            if dist > self.knn_thr:
                flags.append({"kind": "unusual", "text": "This image looks unlike the dermoscopy photos the model was "
                                                         "trained on (it is further from every training image than 99% of "
                                                         "real validation images). Treat the result as unreliable."})
        if probs.max() < self.low_conf:
            flags.append({"kind": "uncertain", "text": f"Low confidence: the top class has only {probs.max():.0%}. "
                                                       f"The model is unsure."})

        per_method = {}
        with torch.no_grad():
            for key, net in self.models.items():
                rare, cfg = key.split("/")
                p = torch.softmax(net(x).float(), 1)[0].cpu().numpy()
                per_method.setdefault(rare, []).append({"config": cfg, "label": self.meta["models"][key]["label"],
                                                        "p_rare": round(float(p[self.classes.index(rare)]), 4)})
        return {"probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
                "top": self.classes[top], "cam": png_b64(_cam_rgba(cam)),
                "knn_distance": None if dist is None else round(dist, 4), "knn_threshold": self.knn_thr,
                "flags": flags, "per_method": per_method}

    # ------------------------------------------------------------------ GAN
    def _z(self, rare: str, seed: int, n: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(n, self.gans[rare].z_dim, generator=g)

    @torch.no_grad()
    def _decode(self, rare: str, z: torch.Tensor) -> np.ndarray:
        G = self.gans[rare]
        x = G(z.to(self.device))
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    def generate(self, rare: str, seed: int, n: int = 12) -> dict:
        imgs = self._decode(rare, self._z(rare, seed, n))
        checks = self.copy_check(rare, imgs)
        return {"images": [png_b64(im) for im in imgs], "checks": checks}

    def morph(self, rare: str, seed: int, a: int, b: int, t: float, n: int = 12) -> str:
        z = self._z(rare, seed, n)
        return png_b64(self._decode(rare, _slerp(z[a], z[b], float(t)).unsqueeze(0))[0])

    def grid_png(self, rare: str, seed: int, n: int = 12) -> bytes:
        imgs = self._decode(rare, self._z(rare, seed, n))
        cols = 6
        rows = int(np.ceil(n / cols))
        s = imgs.shape[1]
        grid = np.full((rows * (s + 4) + 4, cols * (s + 4) + 4, 3), 15, np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            grid[4 + r * (s + 4): 4 + r * (s + 4) + s, 4 + c * (s + 4): 4 + c * (s + 4) + s] = im
        buf = io.BytesIO()
        Image.fromarray(grid).save(buf, format="PNG")
        return buf.getvalue()

    def copy_check(self, rare: str, imgs: np.ndarray):
        """Distance from each generated image to its closest REAL training image (Inception features),
        compared with the copy threshold used on Day 3. None if pytorch-fid is not installed."""
        if rare not in self.real_feats or self._inception_ok is False:
            return None
        try:
            from src.fid import inception_features
            from src.gan.filtering import gen_real_nn
            f = inception_features(imgs, self.device)
            self._inception_ok = True
        except Exception as e:  # pragma: no cover
            print("copy check disabled:", e)
            self._inception_ok = False
            return None
        real, thr = self.real_feats[rare]
        d, _ = gen_real_nn(f, real)
        return [{"distance": round(float(x), 3), "copy": bool(x < thr)} for x in d]
