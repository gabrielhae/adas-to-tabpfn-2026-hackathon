"""
Model factory + preprocessing.

TabPFN is the headline model, but v9+ requires a one-time license acceptance
(`TABPFN_TOKEN`). We therefore expose a uniform interface over several models so the
pipeline runs today and upgrades transparently the moment a token is available.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
def tabpfn_available() -> tuple[bool, str]:
    """Return (available, reason). Checks import + token presence."""
    try:
        import tabpfn  # noqa: F401
    except Exception as e:  # pragma: no cover
        return False, f"tabpfn not importable: {e}"
    tok = C.tabpfn_token()
    if not tok:
        # weights may already be cached from a previous licensed run
        if _tabpfn_weights_cached():
            return True, "tabpfn installed; weights cached locally"
        return False, (
            "tabpfn installed but no TABPFN_TOKEN and no cached weights. "
            "One-time: accept the license at https://ux.priorlabs.ai/account/licenses "
            "then set TABPFN_TOKEN=<api key>."
        )
    return True, "tabpfn installed with TABPFN_TOKEN"


def _tabpfn_weights_cached() -> bool:
    import glob
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("APPDATA") or os.path.expanduser("~")
    pats = [
        os.path.join(base, "**", "tabpfn*", "**", "*.ckpt"),
        os.path.join(base, "**", "tabpfn*", "**", "*.safetensors"),
        os.path.expanduser("~/.cache/**/tabpfn*/**/*.ckpt"),
    ]
    for p in pats:
        if glob.glob(p, recursive=True):
            return True
    return False


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray
    feature_names: list[str]
    categorical: list[str]


def prepare(df: pd.DataFrame, schema: dict, target: str) -> Prepared:
    """Assemble X, y and group vector. Drops rows with a missing target."""
    d = df[df[target].notna()].copy()
    num = [c for c in schema["features_numeric"] if c in d.columns]
    cat = [c for c in schema["features_categorical"] if c in d.columns]
    X = d[num + cat].copy()
    # one-hot the low-cardinality categoricals only; car_model left as category code
    for c in cat:
        X[c] = X[c].astype("string").fillna("<NA>")
    return Prepared(X=X, y=d[target].astype("string"), groups=d["dongle_id"].to_numpy(),
                    feature_names=num + cat, categorical=cat)


def encode(Xtr: pd.DataFrame, Xte: pd.DataFrame, cat: list[str]):
    """One-hot categoricals using training categories; median-impute numerics."""
    combined = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    keep_cat = [c for c in cat if combined[c].nunique(dropna=False) <= 30]
    high_card = [c for c in cat if c not in keep_cat]
    if high_card:
        # frequency-encode high-cardinality fields (car_model etc.) to avoid blow-up
        for c in high_card:
            freq = Xtr[c].value_counts(normalize=True)
            combined[c] = combined[c].map(freq).fillna(0.0).astype(float)
    n = len(Xtr)
    if keep_cat:
        combined = pd.get_dummies(combined, columns=keep_cat, dummy_na=False, dtype=float)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    med = combined.iloc[:n].median(numeric_only=True)
    combined = combined.fillna(med).fillna(0.0)
    return combined.iloc[:n].reset_index(drop=True), combined.iloc[n:].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _kv_kwargs(cls, kv_cache: bool) -> dict:
    """Add ``fit_mode='fit_with_cache'`` when the installed tabpfn supports it.

    Caches the training-side attention state so repeated ``predict`` calls against
    an unchanged training set skip recomputing it. TabPFN-3+ only.
    """
    if not kv_cache:
        return {}
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    return {"fit_mode": "fit_with_cache"} if "fit_mode" in params else {}


def make_model(kind: str = "auto", *, task: str = "classification", seed: int = 0,
               kv_cache: bool = False):
    """Return an unfitted sklearn-compatible estimator.

    kind: 'tabpfn' | 'lightgbm' | 'logistic' | 'auto'
    kv_cache: enable TabPFN ``fit_with_cache`` (reuse when re-predicting against
              the same training set). Ignored by the non-tabpfn models.
    """
    if kind == "auto":
        ok, _ = tabpfn_available()
        kind = "tabpfn" if ok else "lightgbm"

    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        tok = C.tabpfn_token()
        if tok:
            os.environ["TABPFN_TOKEN"] = tok
        if task == "classification":
            return TabPFNClassifier(device="auto", random_state=seed,
                                    balance_probabilities=True,
                                    **_kv_kwargs(TabPFNClassifier, kv_cache))
        return TabPFNRegressor(device="auto", random_state=seed,
                               **_kv_kwargs(TabPFNRegressor, kv_cache))

    if kind == "lightgbm":
        import lightgbm as lgb
        if task == "classification":
            return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                      num_leaves=31, subsample=0.8,
                                      colsample_bytree=0.8, random_state=seed,
                                      verbose=-1)
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, verbose=-1)

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if task == "classification":
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0,
                                                    class_weight="balanced"))
        return make_pipeline(StandardScaler(), RidgeCV())

    raise ValueError(f"unknown model kind: {kind}")


def collapse_rare(y: pd.Series, min_count: int = 10) -> pd.Series:
    """Merge classes with fewer than `min_count` members into 'other'.

    `turn_ramp` has a single example in this sample, which makes macro-F1 meaningless.
    """
    vc = y.value_counts()
    rare = set(vc[vc < min_count].index)
    if rare:
        y = y.where(~y.isin(rare), "other")
    return y
