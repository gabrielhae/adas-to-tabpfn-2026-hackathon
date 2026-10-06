"""
Build the driver-disjoint train / validation / test split.

Reads data/derived/model_table.parquet, assigns every driver (dongle_id) to exactly
one of train/val/test (60/20/20 by clip count), and writes:

  * data/derived/splits.parquet   dongle_id, split, n_rows
  * data/derived/split_meta.json  per-split counts, class balance, brand coverage

The explorer (app/server.py) shows only the test split; the /api/predict endpoint
fits on train+val and predicts the held-out test clip.

Usage:
  python scripts/build_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C
from adas_to import splits


def main() -> None:
    if not C.MODEL_TABLE.exists():
        raise SystemExit(f"missing {C.MODEL_TABLE}; run scripts/build_dataset.py first")

    df = pd.read_parquet(C.MODEL_TABLE)
    assign = splits.assign_group_splits(df)
    splits.assert_disjoint(assign)
    assign.to_parquet(C.SPLIT_ASSIGNMENTS, index=False)

    meta = splits.summary(df, assign, target="post_maneuver_type", brand_col="brand")
    splits.write_meta(meta)

    print(f"groups (drivers): {meta['n_groups']}   rows: {meta['n_rows']:,}")
    print(f"fractions {meta['fractions']}  seed={meta['seed']}")
    print(f"{'split':<6}{'drivers':>9}{'rows':>8}{'share':>9}{'brands':>8}")
    for k in splits.ORDER:
        r = meta["splits"][k]
        print(f"{k:<6}{r['n_groups']:>9}{r['n_rows']:>8}{r['row_share']:>9.3f}"
              f"{r.get('n_brands', 0):>8}")
    print(f"\nwrote {C.SPLIT_ASSIGNMENTS}")
    print(f"wrote {C.SPLIT_META}")

    if C.FORECAST_TABLE.exists():
        fc = pd.read_parquet(C.FORECAST_TABLE, columns=[C.GROUP_COL])
        missing = set(fc[C.GROUP_COL].unique()) - set(assign[C.GROUP_COL])
        if missing:
            print(f"\nWARNING: {len(missing)} forecast drivers not in split "
                  f"(unlabelled clips?) e.g. {list(missing)[:3]}")


if __name__ == "__main__":
    main()
