"""One-off step before the app runs: collect everything RareLens shows into rarelens/data/.

    python prepare_app.py

Nothing is trained here. It reads the finished experiments (results/, models/, data/) and writes:

    data/app_meta.json    which saved models the app loads, example images, calibration for the
                          "outside the training data" warning, GAN facts
    data/evidence.json    every number on the Evidence tab, computed from results_log.csv and the
                          saved per-run predictions (nothing typed in by hand)
    data/examples/        test-set images to try in the Examine tab
    data/cases/           thumbnails of every rare test image (who catches what)
    data/provenance/      generated images next to their closest real training image
    data/gan_<RARE>_real.npz   Inception features of the real training images (live copy check)

Model choice (fixed rule, decided before looking at the app): every method uses its seed-42 run,
the same run reported on Days 1-4; the main model is the GAN-augmented DF classifier.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_CODES, CLASS_NAMES, NUM_CLASSES, Paths  # noqa: E402
from src.data.dataset import attach_cache_index  # noqa: E402
from src.data.explore import build_metadata  # noqa: E402
from src.data.split import make_split  # noqa: E402
from src.experiments import CONFIG_ORDER, CONFIGS, run_dir_for  # noqa: E402
from src.threshold import predict_with_threshold, rare_scores  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
RARES = ["DF", "VASC"]
APP_SEED = 42
MAIN = ("DF", "gan_aug")


def _rel(p: Path) -> str:
    return Path(os.path.relpath(p, ROOT)).as_posix()


def _save_jpg(arr: np.ndarray, path: Path, size: int | None = None, quality: int = 92) -> None:
    im = Image.fromarray(np.asarray(arr))
    if size:
        im = im.resize((size, size), Image.LANCZOS if size < im.width else Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=quality)


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# --------------------------------------------------------------------------- #
def load_everything(paths: Paths):
    meta = build_metadata(paths)
    split = attach_cache_index(make_split(meta, paths, seed=42), meta)
    images = np.load(paths.image_cache(224), mmap_mode="r")
    return meta, split, images


def collect_models(paths: Paths) -> dict:
    models = {}
    for rare in RARES:
        for cfg in CONFIG_ORDER:
            ck = run_dir_for(paths, cfg, APP_SEED, rare) / "best.pt"
            if ck.exists():
                models[f"{rare}/{cfg}"] = {"rare": rare, "config": cfg, "label": CONFIGS[cfg]["label"],
                                           "seed": APP_SEED, "path": _rel(ck)}
    if f"{MAIN[0]}/{MAIN[1]}" not in models:
        raise SystemExit("The main model (DF, GAN aug, seed 42) was not found - run the Day 4 notebook first.")
    return models


def run_stats(paths: Paths, rare: str) -> dict:
    """Per method: per-seed metrics from the results log + false alarms from saved predictions
    + AP / tuned-threshold numbers from the Day 5 threshold study (if present)."""
    from src.analysis import load_log, load_predictions, per_image_hits

    log = load_log(paths, rare)
    preds = load_predictions(paths, log)
    thr_file = paths.results_dir / f"day5_threshold_{rare}_runs.csv"
    thr = pd.read_csv(thr_file) if thr_file.exists() else pd.DataFrame()
    methods = []
    for cfg in CONFIG_ORDER:
        g = log[log["config"] == cfg].sort_values("seed")
        if g.empty:
            continue
        seeds = []
        for _, r in g.iterrows():
            s = int(r["seed"])
            p = preds.get((cfg, s))
            fa = int(((p["pred"] == rare) & (p["label"] != rare)).sum()) if p is not None else None
            row = {"seed": s, "f1": _r(r["rare_f1"]), "recall": _r(r["rare_recall"]),
                   "precision": _r(r["rare_precision"]), "macro_f1": _r(r["test_macro_f1"]),
                   "auroc": _r(r["rare_auroc"]), "caught": int(round(r["rare_recall"] * r["rare_support"])),
                   "support": int(r["rare_support"]), "false_alarms": fa}
            if len(thr):
                t = thr[(thr["config"] == cfg) & (thr["seed"] == s)]
                if len(t):
                    t = t.iloc[0]
                    row.update({"ap": _r(t["test_AP"]), "tuned_f1": _r(t["tuned_f1"]),
                                "tuned_false_alarms": int(t["tuned_false_alarms"]), "threshold": _r(t["threshold"], 2)})
            seeds.append(row)
        methods.append({"config": cfg, "label": CONFIGS[cfg]["label"], "seeds": seeds})

    # same-seed change vs the baseline
    base = {s["seed"]: s for m in methods if m["config"] == "baseline_no_aug" for s in m["seeds"]}
    for m in methods:
        d = [s["f1"] - base[s["seed"]]["f1"] for s in m["seeds"] if s["seed"] in base]
        m["paired_f1"] = {"mean": _r(np.mean(d)) if d else None, "better": int(sum(x > 0 for x in d)),
                          "same": int(sum(x == 0 for x in d)), "worse": int(sum(x < 0 for x in d))}

    hits = per_image_hits(preds, rare) if preds else pd.DataFrame()
    return {"methods": methods, "hits": hits}


def threshold_curve(paths: Paths, rare: str) -> dict | None:
    f = run_dir_for(paths, "gan_aug", APP_SEED, rare) / "val_test_probs.npz"
    if not f.exists():
        return None
    z = np.load(f)
    k = CLASS_CODES.index(rare)
    pts = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        s = rare_scores(z["y_test"], predict_with_threshold(z["p_test"], k, t), k)
        pts.append({"t": float(t), "caught": s["caught"], "false_alarms": s["false_alarms"],
                    "precision": _r(s["precision"], 3), "recall": _r(s["recall"], 3), "f1": _r(s["f1"], 3)})
    default = rare_scores(z["y_test"], z["p_test"].argmax(1), k)
    return {"rare": rare, "n_rare": int((z["y_test"] == k).sum()), "points": pts,
            "default": {"caught": default["caught"], "false_alarms": default["false_alarms"], "f1": _r(default["f1"], 3)}}


# --------------------------------------------------------------------------- #
def export_examples(split: pd.DataFrame, images, hits: dict) -> list:
    """A handful of TEST-set images (never seen in training) for the Examine tab."""
    test = split[split["split"] == "test"]
    rng = np.random.default_rng(7)
    chosen = []
    h = hits.get("DF")
    gan, base = CONFIGS["gan_aug"]["label"], CONFIGS["baseline_no_aug"]["label"]
    if h is not None and len(h) and gan in h and base in h:
        helped = h.index[h[gan] > h[base]].tolist()
        always = h.index[(h[[c for c in h.columns if c != "lesion_id"]] == 1).all(axis=1)].tolist()
        never = h.index[(h[[c for c in h.columns if c != "lesion_id"]] == 0).all(axis=1)].tolist()
        for lst, note in [(helped, "GAN-trained model catches this more often than the baseline"),
                          (always, "every method catches this one"), (never, "no method ever catches this one")]:
            if lst:
                chosen.append((lst[0], note))
    want = {"DF": 3, "VASC": 2, "NV": 1, "MEL": 1, "BKL": 1, "BCC": 1, "AKIEC": 1}
    for code, n in want.items():
        have = sum(1 for i, _ in chosen if test.set_index("image_id").at[i, "label"] == code)
        pool = test.loc[(test["label"] == code) & ~test["image_id"].isin([i for i, _ in chosen]), "image_id"].to_numpy()
        for i in rng.choice(pool, size=min(max(n - have, 0), len(pool)), replace=False):
            chosen.append((i, ""))
    out = []
    t = test.set_index("image_id")
    for img_id, note in chosen:
        _save_jpg(images[int(t.at[img_id, "cache_idx"])], DATA / "examples" / f"{img_id}.jpg")
        out.append({"id": img_id, "label": t.at[img_id, "label"], "name": CLASS_NAMES[t.at[img_id, "label"]],
                    "note": note, "src": f"data/examples/{img_id}.jpg"})
    return out


def export_cases(split: pd.DataFrame, images, hits: pd.DataFrame, rare: str) -> list:
    if hits is None or not len(hits):
        return []
    t = split.set_index("image_id")
    rows = []
    cols = [c for c in hits.columns if c != "lesion_id"]
    for img_id, r in hits.iterrows():
        _save_jpg(images[int(t.at[img_id, "cache_idx"])], DATA / "cases" / f"{img_id}.jpg", size=112, quality=88)
        rows.append({"id": img_id, "lesion": r["lesion_id"], "src": f"data/cases/{img_id}.jpg",
                     "hits": {c: _r(r[c], 2) for c in cols}})
    return rows


# --------------------------------------------------------------------------- #
def gan_facts(paths: Paths, split: pd.DataFrame, images, device) -> dict:
    """Best-FID generator per disease, FID tables, and the real features for the live copy check."""
    from src.gan.filtering import gen_real_nn, real_real_nn
    from src.gan.trainer import resize_uint8
    try:
        from src.fid import inception_features
    except Exception as e:  # pytorch-fid missing
        print("  (pytorch-fid not available - live copy check disabled):", e)
        inception_features = None

    out = {}
    syn = paths.project_dir / "data" / "synthetic"
    for rare in RARES:
        g = paths.project_dir / "models" / f"gan_{rare}" / "G_best.pt"
        if not g.exists():
            continue
        ck = torch.load(g, map_location="cpu", weights_only=False)
        info = {"generator": _rel(g), "best_iter": int(ck.get("iter", -1)), "best_fid": _r(ck.get("fid"), 1)}
        fid_csv = paths.results_dir / ("gan_fid.csv" if rare == "DF" else f"day5_{rare.lower()}_fid.csv")
        if fid_csv.exists():
            f = pd.read_csv(fid_csv)
            info["fid_table"] = [{"set": r.iloc[0], "fid": _r(r.get("FID (size-matched)"), 1),
                                  "sd": _r(r.get("± sd"), 1)} for _, r in f.iterrows()]
        if rare == "DF" and (paths.results_dir / "gan_summary.json").exists():
            s = json.loads((paths.results_dir / "gan_summary.json").read_text())
            info["filter_counts"] = s.get("filter_counts", {})
            info["candidates"] = s.get("candidates")
        train = split[(split["split"] == "train") & (split["label"] == rare)].reset_index(drop=True)
        info["train_images"] = int(len(train))
        real128 = resize_uint8(np.stack([images[int(i)] for i in train["cache_idx"]]), 128)

        if inception_features is not None:
            f_real = inception_features(real128, device)
            d_rr = real_real_nn(f_real, train["lesion_id"].to_numpy())
            copy_thr = float(np.percentile(d_rr, 5))
            np.savez_compressed(DATA / f"gan_{rare}_real.npz", feats=f_real.astype(np.float32),
                                image_ids=train["image_id"].to_numpy(), copy_threshold=copy_thr)
            info.update({"copy_threshold": _r(copy_thr), "real_real_median": _r(np.median(d_rr))})
            sel = syn / f"gan_{rare}_500_128.npy"
            if sel.exists():
                d_gr, _ = gen_real_nn(inception_features(np.load(sel), device), f_real)
                info["gen_real_median"] = _r(np.median(d_gr))

        prov_csv = syn / f"gan_{rare}_500_provenance.csv"
        sel = syn / f"gan_{rare}_500_128.npy"
        if prov_csv.exists() and sel.exists():
            prov = pd.read_csv(prov_csv)
            gen = np.load(sel)
            t = split.set_index("image_id")
            pairs = []
            order = np.argsort(prov["nn_distance"].to_numpy())
            for j in [order[len(order) // 10], order[len(order) // 2], order[int(len(order) * 0.9)]]:
                rid = prov.at[j, "nearest_real_image_id"]
                gp, rp = DATA / "provenance" / f"{rare}_gen_{j}.jpg", DATA / "provenance" / f"{rare}_real_{rid}.jpg"
                _save_jpg(gen[j], gp, size=256)
                _save_jpg(images[int(t.at[rid, "cache_idx"])], rp, size=256)
                pairs.append({"gen": f"data/provenance/{gp.name}", "real": f"data/provenance/{rp.name}",
                              "real_id": rid, "distance": _r(prov.at[j, "nn_distance"])})
            info["pairs"] = pairs
        out[rare] = info
        print(f"  GAN {rare}: best FID {info['best_fid']} at iteration {info['best_iter']}")
    return out


def _load_clf(paths: Paths, cfg: str, rare: str, device):
    from src.models.classifier import build_classifier
    m = build_classifier("resnet18", NUM_CLASSES, pretrained=False)
    m.load_state_dict(torch.load(run_dir_for(paths, cfg, APP_SEED, rare) / "best.pt", map_location="cpu", weights_only=True))
    return m.to(device).eval()


def _batches(split_part: pd.DataFrame, images, device, bs: int = 64):
    from src.data.dataset import eval_transform
    tf = eval_transform()
    idx = split_part["cache_idx"].to_numpy()
    for k in range(0, len(idx), bs):
        yield torch.stack([tf(np.array(images[int(i)])) for i in idx[k:k + bs]]).to(device)


def _summary(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    pred = p.argmax(1)
    out = {"accuracy": _r(accuracy_score(y, pred)),
           "macro_f1": _r(f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))}
    for rare in RARES:
        s = rare_scores(y, pred, CLASS_CODES.index(rare))
        out[rare] = {k: (_r(v) if isinstance(v, float) else v) for k, v in s.items() if k != "macro_f1"}
    return out


def _curve(y: np.ndarray, p: np.ndarray, rare: str) -> dict:
    k = CLASS_CODES.index(rare)
    pts = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        s = rare_scores(y, predict_with_threshold(p, k, t), k)
        pts.append({"t": float(t), "caught": s["caught"], "false_alarms": s["false_alarms"],
                    "precision": _r(s["precision"], 3), "recall": _r(s["recall"], 3), "f1": _r(s["f1"], 3)})
    d = rare_scores(y, p.argmax(1), k)
    return {"rare": rare, "n_rare": int((y == k).sum()), "points": pts,
            "default": {"caught": d["caught"], "false_alarms": d["false_alarms"], "f1": _r(d["f1"], 3)}}


def app_model(paths: Paths, split: pd.DataFrame, images, models: dict, device) -> tuple[dict, dict]:
    """Choose the app's model on the VALIDATION set, then score it once on TEST.

    Candidates: the DF-study GAN classifier alone, or averaged with the VASC-study GAN classifier,
    each with or without 8-way rotation/mirror test-time augmentation. Rule (fixed in advance):
    highest mean of DF F1 and VASC F1 on validation - the app exists for the rare classes.
    Also calibrates the unusual-image check on training/validation features."""
    from rarelens.model_utils import KNN_K, ensemble_probs, imagenet_backbone, knn_distance, penultimate

    keys_all = [k for k in ("DF/gan_aug", "VASC/gan_aug") if k in models]
    nets = {k: _load_clf(paths, models[k]["config"], models[k]["rare"], device) for k in keys_all}
    cands = []
    for members in ([keys_all[0]], keys_all) if len(keys_all) > 1 else ([keys_all[0]],):
        for tta in (False, True):
            cands.append({"members": members, "tta": tta})
    parts = {s_: split[split["split"] == s_] for s_ in ("train", "val", "test")}
    ys = {s_: np.array([CLASS_CODES.index(c) for c in parts[s_]["label"]]) for s_ in ("val", "test")}
    with torch.no_grad():
        for c in cands:
            ens = [nets[k] for k in c["members"]]
            for s_ in ("val", "test"):
                c[f"p_{s_}"] = torch.cat([ensemble_probs(ens, x, tta=c["tta"]).cpu()
                                          for x in _batches(parts[s_], images, device)]).numpy()
            c["val"] = _summary(ys["val"], c["p_val"])
            c["test"] = _summary(ys["test"], c["p_test"])
            c["score"] = (c["val"]["DF"]["f1"] + c["val"]["VASC"]["f1"]) / 2
            c["name"] = ("GAN model (DF study)" if len(c["members"]) == 1 else "Both GAN models, averaged") + \
                        (" + 8 views" if c["tta"] else "")
            print(f"  candidate {c['name']:38s} val rare-F1 {c['score']:.3f} | test DF F1 {c['test']['DF']['f1']:.3f}, "
                  f"VASC F1 {c['test']['VASC']['f1']:.3f}")
        best = max(cands, key=lambda c: c["score"])
        feat_model = imagenet_backbone(device)
        f_train = torch.cat([penultimate(feat_model, x) for x in _batches(parts["train"], images, device)])
        d_val = knn_distance(f_train, torch.cat([penultimate(feat_model, x) for x in _batches(parts["val"], images, device)])).cpu().numpy()
        d_test = knn_distance(f_train, torch.cat([penultimate(feat_model, x) for x in _batches(parts["test"], images, device)])).cpu().numpy()
    thr = float(np.percentile(d_val, 99))
    np.savez_compressed(DATA / "knn_train_features.npz", feats=f_train.cpu().numpy().astype(np.float16))

    y, pt = ys["test"], best["p_test"]
    ev = {"members": best["members"], "tta": 8 if best["tta"] else 1, "name": best["name"], "test": best["test"],
          "candidates": [{"name": c["name"], "val_rare_f1": _r(c["score"]), "test": c["test"], "chosen": c is best}
                         for c in cands],
          "curves": {r: _curve(y, pt, r) for r in RARES}, "low_confidence_share_test": _r((pt.max(1) < 0.5).mean(), 3),
          "knn": {"k": KNN_K, "threshold": _r(thr), "val_median": _r(np.median(d_val)),
                  "test_flagged_share": _r((d_test > thr).mean(), 3)}}
    meta = {"members": best["members"], "tta": best["tta"], "name": best["name"], "knn_k": KNN_K,
            "knn_threshold": thr, "low_confidence": 0.5}
    t = ev["test"]
    print(f"  chosen on validation: {best['name']} -> test accuracy {t['accuracy']:.3f} | macro-F1 {t['macro_f1']:.3f} | "
          f"DF F1 {t['DF']['f1']:.3f} | VASC F1 {t['VASC']['f1']:.3f}")
    print(f"  unusual-image check: threshold {thr:.3f}; flags {ev['knn']['test_flagged_share']:.1%} of real test images")
    return ev, meta


# --------------------------------------------------------------------------- #
def main() -> None:
    paths = Paths(ROOT, ROOT / "data" / "raw")
    DATA.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RareLens prepare | project {ROOT} | device {device}")

    meta, split, images = load_everything(paths)
    print("1/6 models")
    models = collect_models(paths)
    print(f"  {len(models)} saved classifiers found (seed {APP_SEED})")

    print("2/6 results")
    evidence = {"dataset": {"images": int(len(split)), "classes": {c: CLASS_NAMES[c] for c in CLASS_CODES},
                            "counts": {c: int((split["label"] == c).sum()) for c in CLASS_CODES},
                            "test_counts": {c: int(((split["label"] == c) & (split["split"] == "test")).sum())
                                            for c in CLASS_CODES}},
                "diseases": {}}
    hits = {}
    for rare in RARES:
        try:
            st = run_stats(paths, rare)
        except (KeyError, StopIteration, FileNotFoundError) as e:
            print(f"  {rare}: no finished runs ({e})")
            continue
        if not st["methods"]:
            continue
        hits[rare] = st["hits"]
        evidence["diseases"][rare] = {"name": CLASS_NAMES[rare], "methods": st["methods"]}
        print(f"  {rare}: {sum(len(m['seeds']) for m in st['methods'])} runs")

    print("3/6 example images")
    examples = export_examples(split, images, hits)
    for rare in list(evidence["diseases"]):
        evidence["diseases"][rare]["cases"] = export_cases(split, images, hits.get(rare), rare)
    print(f"  {len(examples)} examples")

    print("4/6 GAN facts + live copy check features (Inception)")
    gans = gan_facts(paths, split, images, device)
    evidence["gan"] = gans

    print("5/6 choosing the app's model on validation + unusual-image check")
    app_ev, app_meta_part = app_model(paths, split, images, models, device)
    evidence["app_model"] = app_ev
    for rare in list(evidence["diseases"]):
        evidence["diseases"][rare]["threshold_curve"] = app_ev["curves"].get(rare)

    print("6/6 writing files")
    app_meta = {"main_model": f"{MAIN[0]}/{MAIN[1]}", "app_model": app_meta_part, "models": models, "classes": CLASS_CODES,
                "class_names": CLASS_NAMES, "examples": examples, "rares": list(evidence["diseases"]),
                "gan": {r: {"generator": g["generator"], "copy_threshold": g.get("copy_threshold")} for r, g in gans.items()}}
    (DATA / "app_meta.json").write_text(json.dumps(app_meta, indent=2))
    (DATA / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(f"\nDone. Files in {DATA}\nNow start the app:  python app.py")


if __name__ == "__main__":
    main()
