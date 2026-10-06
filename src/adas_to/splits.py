"""
Train / validation / test split.

Design
------
* Group key is ``dongle_id`` (a driver). The ADAS-TO paper mandates driver-disjoint
  splits to prevent within-driver leakage, and the group structure is severe:
  208 drivers, a median of 2 clips each, and the largest driver holding 13.4% of
  the rows. Every clip of a driver therefore lands in exactly one split.
* Sizes are balanced by **clip count**, not driver count, because driver sizes run
  1..140 clips. A greedy least-filled assignment over drivers (seed-shuffled, then
  processed in descending clip count) keeps the three splits at the target
  fractions without letting one large driver perturb the balance.
* Deterministic: same seed -> same assignment, independent of row order.

Typical use::

    from adas_to import splits
    assign = splits.load_assignments()          # dongle_id -> 'train'|'val'|'test'
    d = splits.add_split(model_table, assign)
    train = d[d.split == "train"]; val = d[d.split == "val"]; test = d[d.split == "test"]

``scripts/build_splits.py`` writes ``data/derived/splits.parquet`` and
``data/derived/split_meta.json`` from the model table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ORDER = ("train", "val", "test")


def assign_group_splits(df: pd.DataFrame, group_col: str = C.GROUP_COL,
                        *, fractions: dict | None = None,
                        seed: int = C.SPLIT_SEED) -> pd.DataFrame:
    """Assign every group (driver) to one of train/val/test, balanced by row count.

    Returns a frame with columns ``[group_col, split, n_rows]``.
    """
    fractions = dict(fractions or C.SPLIT_FRACTIONS)
    if set(fractions) != set(ORDER):
        raise ValueError(f"fractions must cover exactly {ORDER}: {fractions}")

    size = df.groupby(group_col).size().sort_index()
    if size.empty:
        raise ValueError("no rows to split")
    total = int(size.sum())
    target = {k: fractions[k] * total for k in ORDER}

    groups = size.index.to_numpy()
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]          # seed-shuffled tie-break
    counts = size.loc[groups].to_numpy()
    order = np.argsort(-counts, kind="stable")             # biggest drivers first

    fill = {k: 0 for k in ORDER}
    split_of: dict = {}
    for i in order:
        g = groups[i]
        k = min(ORDER, key=lambda s: fill[s] / target[s])  # least-filled split, by ratio
        split_of[g] = k
        fill[k] += int(counts[i])

    out = pd.DataFrame({group_col: list(split_of.keys()),
                        "split": list(split_of.values())})
    out["n_rows"] = out[group_col].map(size).astype(int)
    return out[[group_col, "split", "n_rows"]]


def add_split(df: pd.DataFrame, assign: pd.DataFrame | None = None,
              group_col: str = C.GROUP_COL) -> pd.DataFrame:
    """Return ``df`` with a ``split`` column joined on the group key."""
    if assign is None:
        assign = load_assignments()
    if assign is None:
        raise FileNotFoundError(
            f"no split assignments at {C.SPLIT_ASSIGNMENTS}; run scripts/build_splits.py")
    mapping = assign.set_index(group_col)["split"]
    out = df.copy()
    out["split"] = out[group_col].map(mapping)
    return out


def load_assignments(path: Path | None = None) -> pd.DataFrame | None:
    p = Path(path or C.SPLIT_ASSIGNMENTS)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def split_groups(assign: pd.DataFrame | None = None,
                 group_col: str = C.GROUP_COL) -> dict[str, set]:
    assign = assign if assign is not None else load_assignments()
    if assign is None:
        return {k: set() for k in ORDER}
    return {k: set(assign.loc[assign.split == k, group_col]) for k in ORDER}


def assert_disjoint(assign: pd.DataFrame, group_col: str = C.GROUP_COL) -> None:
    """Every group must appear in exactly one split."""
    per = assign.groupby(group_col)["split"].nunique()
    bad = per[per != 1]
    assert bad.empty, f"groups in more than one split: {bad.index.tolist()[:10]}"
    assert set(assign["split"]) <= set(ORDER), f"unknown split labels: {set(assign['split'])}"


def summary(df: pd.DataFrame, assign: pd.DataFrame, *, target: str | None = None,
            brand_col: str | None = None, group_col: str = C.GROUP_COL) -> dict:
    """Per-split driver/row counts, share, class balance and brand coverage."""
    d = add_split(df, assign, group_col)
    out: dict = {"fractions": C.SPLIT_FRACTIONS, "seed": C.SPLIT_SEED,
                 "n_rows": int(len(d)), "n_groups": int(d[group_col].nunique()),
                 "splits": {}}
    for k in ORDER:
        s = d[d.split == k]
        rec = {"n_groups": int(s[group_col].nunique()),
               "n_rows": int(len(s)),
               "row_share": round(len(s) / len(d), 4) if len(d) else 0.0}
        if target and target in s.columns:
            rec["class_balance"] = {str(a): int(b) for a, b in
                                    s[target].value_counts().items()}
        if brand_col and brand_col in s.columns:
            rec["n_brands"] = int(s[brand_col].nunique())
        out["splits"][k] = rec
    return out


def write_meta(meta: dict, path: Path | None = None) -> Path:
    p = Path(path or C.SPLIT_META)
    p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return p
