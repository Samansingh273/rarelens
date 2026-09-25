"""Step 2 — stratified, lesion-grouped train/val/test split (70 / 15 / 15).

Why grouped by lesion: HAM10000 contains ~7,470 distinct lesions photographed
~10,015 times. If two photos of the same lesion land in train and test, the test
score is inflated (the model has effectively "seen" the test lesion). So we split
*lesions*, stratified by diagnosis, and every image follows its lesion.

Because each lesion has exactly one diagnosis, stratifying at the lesion level
keeps the class mix — including the rare class — realistic in every split.

This split is created ONCE (split_seed=42) and reused unchanged by every later
experiment, so all four configurations are evaluated on the same test images.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import CLASS_CODES, Paths


def make_split(meta: pd.DataFrame, paths: Paths, seed: int = 42,
               val_frac: float = 0.15, test_frac: float = 0.15, overwrite: bool = False) -> pd.DataFrame:
    out = paths.split_csv(seed)
    if out.exists() and not overwrite:
        print(f"[reuse] existing split {out}")
        split = pd.read_csv(out)
        check_split(split)
        return split

    lesions = meta.groupby("lesion_id", as_index=False)["label"].first()
    train_l, temp_l = train_test_split(
        lesions, test_size=val_frac + test_frac, stratify=lesions["label"], random_state=seed)
    val_l, test_l = train_test_split(
        temp_l, test_size=test_frac / (val_frac + test_frac), stratify=temp_l["label"], random_state=seed)

    which = pd.concat([
        pd.DataFrame({"lesion_id": train_l["lesion_id"], "split": "train"}),
        pd.DataFrame({"lesion_id": val_l["lesion_id"], "split": "val"}),
        pd.DataFrame({"lesion_id": test_l["lesion_id"], "split": "test"}),
    ])
    split = meta.drop(columns=[c for c in ["path"] if c in meta]).merge(which, on="lesion_id", how="left")
    split = split[["image_id", "lesion_id", "label", "label_idx", "split"]]
    check_split(split)
    paths.splits_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(out, index=False)
    print(f"Saved split to {out}")
    return split


def check_split(split: pd.DataFrame) -> None:
    """Hard guarantees: every image assigned, no lesion in two splits, every class in every split."""
    assert split["split"].notna().all(), "some images have no split"
    leak = split.groupby("lesion_id")["split"].nunique()
    assert (leak == 1).all(), f"{(leak > 1).sum()} lesions leak across splits"
    for s in ("train", "val", "test"):
        present = set(split.loc[split["split"] == s, "label"])
        missing = set(CLASS_CODES) - present
        assert not missing, f"split {s!r} is missing classes {missing}"


def split_summary(split: pd.DataFrame, rare: str | None = None) -> pd.DataFrame:
    """Images per class per split, with the share of each split in brackets."""
    counts = pd.crosstab(split["label"], split["split"]).reindex(CLASS_CODES)[["train", "val", "test"]]
    counts.loc["TOTAL"] = counts.sum()
    pct = 100 * counts / counts.loc["TOTAL"]
    table = counts.astype(str) + " (" + pct.round(1).astype(str) + "%)"
    lesions = pd.crosstab(split["label"], split["split"], values=split["lesion_id"], aggfunc="nunique")
    lesions = lesions.reindex(CLASS_CODES)[["train", "val", "test"]]
    lesions.loc["TOTAL"] = lesions.sum()
    for s in ("train", "val", "test"):
        table[f"{s}_lesions"] = lesions[s].astype(int)
    if rare:
        table.index = [f"{i}  ← rare" if i == rare else i for i in table.index]
    return table
